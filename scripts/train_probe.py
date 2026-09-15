"""
=========================================================================
The M1 probe — one arm, one target, one seed.
=========================================================================

Design record and pre-registered readings: ``notes/m1_probe.md``. This is the
supervised half of WP-C: per-site regression of one of the three probe targets
from the thin spatial plaquettes of a 3D timeslice, with the architecture
chosen by ``PROBE_ARM``.

Everything except the block comes from ``probe_common.py`` — ensemble, splits,
inputs, standardisation, the R² sufficient statistics — so arms cannot drift
apart. Compare ``train_glueball.py``, where the same discipline puts the L-CNN
behind a switch inside one script rather than in a sibling.

    PROBE_ARM=gelt   PROBE_TARGET=T2 PROBE_INIT_SEED=0 python scripts/train_probe.py
    PROBE_ARM=frozen PROBE_TARGET=T2 PROBE_INIT_SEED=0 python scripts/train_probe.py
    PROBE_NULL=1 PROBE_ARM=gelt PROBE_TARGET=T2 python scripts/train_probe.py

Arms: gelt, frozen, frozen_matched, lcnn, lcnn_norm (``probe_common.ARMS``).
``PROBE_NULL=1`` freezes the equivariant stack at initialisation and trains only
the readout head — the random-features null of reading R-E. It is *not* "no
training at all": every arm's head is zero-initialised, so an untrained network
predicts exactly the training mean and scores R² = 0 by construction, which
would measure nothing.

The dump is kilobytes: per test configuration, the six sums an R² is made of
(``probe_common.STAT_KEYS``). ``probe_readings.py`` turns a directory of them
into the pre-registered table.
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_common import (  # noqa: E402
    ARMS,
    JACK_BLOCK,
    N_CONFIGS,
    N_SLICES,
    TARGETS,
    accumulate_stats,
    build_all_targets,
    build_arm,
    env_flag,
    env_float,
    env_int,
    env_str,
    jackknife,
    load_timeslices,
    probe_inputs,
    r2_from_stats,
    real_dofs,
    splits,
    standardize,
)

ARM = env_str("PROBE_ARM", "gelt")
TARGET = env_str("PROBE_TARGET", "T2")
INIT_SEED = env_int("PROBE_INIT_SEED", 0)
ENSEMBLE_SEED = env_int("PROBE_ENSEMBLE_SEED", 0)
NULL = env_flag("PROBE_NULL", False)

LR = env_float("PROBE_LR", 3e-3)
WEIGHT_DECAY = env_float("PROBE_WEIGHT_DECAY", 1e-3)
EPOCHS = env_int("PROBE_EPOCHS", 20)
PATIENCE = env_int("PROBE_PATIENCE", 5)
BATCH_CONFIGS = env_int("PROBE_BATCH", 8)
GRAD_CHECKPOINT = env_flag("PROBE_GRAD_CHECKPOINT", True)
RUN_TAG = env_str("PROBE_RUN_TAG", "")

if ARM not in ARMS:
    raise SystemExit(f"PROBE_ARM must be one of {sorted(ARMS)} (got {ARM!r})")
if TARGET not in TARGETS:
    raise SystemExit(f"PROBE_TARGET must be one of {TARGETS} (got {TARGET!r})")
if RUN_TAG and not RUN_TAG.startswith("_"):
    raise SystemExit(f"PROBE_RUN_TAG must start with '_' (got {RUN_TAG!r})")

# A run whose *best* validation loss is this much worse than the trivial
# predictor has diverged. The target is standardised, so predicting the training
# mean scores exactly 1.0 and the threshold is absolute, not relative to some
# other run — which is what makes it a statistic rather than a ranking.
# Recorded per run because M2 is a failure-rate property, not an accuracy one
# (notes/where_attention_can_win.md §7); reading R-F in notes/m1_probe.md.
DIVERGENCE_VAL = env_float("PROBE_DIVERGENCE_VAL", 10.0)

OUT_DIR = "results/m1_probe"
# Every knob that changes the result is in the name, so no run can overwrite
# another — the same rule train_glueball.py's artifact names follow.
STEM = (
    f"probe_{ARM}_{TARGET}_ens{ENSEMBLE_SEED}_init{INIT_SEED}"
    + ("_null" if NULL else "")
    + ("" if N_CONFIGS == 200 else f"_n{N_CONFIGS}")
    + RUN_TAG
)


def batches(idx, size):
    for lo in range(0, len(idx), size):
        yield idx[lo : lo + size]


def run_split(model, arch, U3, y, idx, device, optimizer=None, per_config=False):
    """One pass over ``idx``. Trains when ``optimizer`` is given, else evaluates.

    With ``per_config`` every configuration is scored on its own so the R²
    sufficient statistics come out per configuration — which is what makes the
    correlated jackknife of ΔR² possible offline.
    """
    train = optimizer is not None
    model.train(train)
    total, n_seen, stats = 0.0, 0, []
    size = 1 if per_config else BATCH_CONFIGS
    for chunk in batches(idx, size):
        U = U3[chunk].reshape(-1, *U3.shape[2:])  # (b·slices, 3, L,L,L, nc,nc)
        t = y[chunk].reshape(-1, *y.shape[2:]).to(device, torch.float32)
        with torch.set_grad_enabled(train):
            W, T = probe_inputs(U, arch, device)
            pred = model(W, T)
            loss = torch.nn.functional.mse_loss(pred, t)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += loss.item() * t.shape[0]
        n_seen += t.shape[0]
        if per_config:
            stats.append(accumulate_stats(t.detach().cpu(), pred.detach().cpu()))
    return total / max(n_seen, 1), (torch.stack(stats) if per_config else None)


def main():
    # PROBE_DEVICE overrides the cuda → mps → cpu order; the smoke tests use it
    # to stay on the CPU, where the complex kernels are all available.
    forced = env_str("PROBE_DEVICE", "")
    device = torch.device(
        forced or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    print("=" * 78)
    print(f"M1 probe — arm {ARM}  target {TARGET}  ensemble {ENSEMBLE_SEED}  "
          f"init {INIT_SEED}" + ("  [NULL: frozen stack]" if NULL else ""))
    print(f"device: {device}   lr {LR:g}  epochs {EPOCHS}  batch {BATCH_CONFIGS} configs")
    print("=" * 78)

    U3 = load_timeslices(seed=ENSEMBLE_SEED)
    _, targets = build_all_targets(U3, names=(TARGET,))
    tr, va, te = splits()
    y, mu, sigma = standardize(targets[TARGET], tr)
    y = y.float()
    print(f"splits: train {len(tr)}  val {len(va)}  test {len(te)} configurations "
          f"× {N_SLICES} timeslices")

    model, arch = build_arm(ARM, seed=INIT_SEED, grad_checkpoint=GRAD_CHECKPOINT)
    model = model.to(device)
    dofs = real_dofs(model)
    print(f"arm {ARM}: {arch}, {dofs} real DOFs, {ARMS[ARM]}")

    if NULL:
        # Random features: the equivariant stack stays at its initialisation and
        # only the per-site readout learns. Nothing in the stack requires grad,
        # so the graph starts at the head and the pass is forward-only through
        # the expensive part.
        head = "head_fc1" if arch == "lcnn" else "mlp"
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith(head) or name.startswith("head_fc2"))
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable: {sum(p.numel() * (2 if p.is_complex() else 1) for p in params)}"
          f" of {dofs} real DOFs")

    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    os.makedirs(OUT_DIR, exist_ok=True)
    checkpoint = f"{OUT_DIR}/{STEM}.pth"

    best_val, best_epoch, since = float("inf"), -1, 0
    history = []
    t_start = time.time()
    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss, _ = run_split(model, arch, U3, y, tr, device, optimizer)
        scheduler.step()
        val_loss, _ = run_split(model, arch, U3, y, va, device)
        history.append((train_loss, val_loss))
        mark = ""
        if val_loss < best_val:
            best_val, best_epoch, since = val_loss, epoch, 0
            torch.save(model.state_dict(), checkpoint)
            mark = "  *"
        else:
            since += 1
        print(f"  epoch {epoch + 1:3d}/{EPOCHS}  train {train_loss:.5f}  "
              f"val {val_loss:.5f}  ({time.time() - t0:.1f}s){mark}")
        if since >= PATIENCE:
            print(f"  early stop: {PATIENCE} epochs without improvement")
            break

    # The reported number is always the selected checkpoint's, never the last
    # epoch's — selection is on val, evaluation on test, and the two never mix.
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    test_loss, stats = run_split(model, arch, U3, y, te, device, per_config=True)
    r2, err, _ = jackknife(stats, r2_from_stats)
    diverged = not (best_val < DIVERGENCE_VAL)  # not (<) also catches nan
    print(f"\nbest epoch {best_epoch + 1} (val {best_val:.5f})"
          + (f"  ** DIVERGED (> {DIVERGENCE_VAL:g}, the trivial predictor scores "
             f"1.0) **" if diverged else ""))
    print(f"test MSE {test_loss:.5f}   R² = {r2:+.4f} ± {err:.4f} "
          f"(blocked jackknife, block {JACK_BLOCK} configs)")

    dump = f"{OUT_DIR}/{STEM}_stats.pt"
    torch.save({
        "stats": stats, "r2": r2, "r2_err": err, "test_mse": test_loss,
        "arm": ARM, "arch": arch, "target": TARGET, "spec": ARMS[ARM],
        "real_dofs": dofs, "null": NULL,
        "ensemble_seed": ENSEMBLE_SEED, "init_seed": INIT_SEED, "run_tag": RUN_TAG,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "epochs_run": len(history),
        "best_epoch": best_epoch, "best_val": best_val, "history": history,
        "diverged": diverged, "divergence_val": DIVERGENCE_VAL,
        "target_mu": mu, "target_sigma": sigma,
        "n_configs": N_CONFIGS, "n_slices": N_SLICES,
        "test_configs": te.tolist(), "jack_block": JACK_BLOCK,
        "wall_seconds": time.time() - t_start,
    }, dump)
    print(f"wrote {dump}")
    print(json.dumps({"arm": ARM, "target": TARGET, "init": INIT_SEED,
                      "null": NULL, "r2": round(r2, 5), "err": round(err, 5),
                      "diverged": diverged}))


if __name__ == "__main__":
    main()
