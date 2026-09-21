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
    validate_argv,
    BETA,
    Z2_LCNN_CONV_INIT,
    Z2_LCNN_REF_INIT_W,
    GROUP,
    IS_Z2,
    JACK_BLOCK,
    N_CONFIGS,
    N_SLICES,
    TARGETS,
    accumulate_stats,
    available_arms,
    build_all_targets,
    build_mask,
    arm_transport,
    build_arm,
    env_flag,
    env_float,
    env_int,
    env_str,
    jackknife,
    load_samples,
    probe_inputs,
    r2_from_stats,
    real_dofs,
    splits,
    standardize,
)

ARM = env_str("PROBE_ARM", "gelt")
TARGET = env_str("PROBE_TARGET", "V1" if IS_Z2 else "T2")
INIT_SEED = env_int("PROBE_INIT_SEED", 0)
ENSEMBLE_SEED = env_int("PROBE_ENSEMBLE_SEED", 0)
NULL = env_flag("PROBE_NULL", False)

LR = env_float("PROBE_LR", 3e-3)
WEIGHT_DECAY = env_float("PROBE_WEIGHT_DECAY", 1e-3)
EPOCHS = env_int("PROBE_EPOCHS", 20)
# **Off by default, and that is not an oversight.** The schedule below is a
# cosine annealed to zero at ``EPOCHS``, so the low-rate epochs at the end are
# precisely where it converges; stopping early on a patience counter throws away
# the half of the run that does the converging, and it does so *at different
# epochs for different arms*, which turns a matched budget into an unmatched
# one. Measured: the 40-epoch gelt gate stopped at epoch 16 of 40 on a val
# oscillation and never saw its own anneal. Set PROBE_PATIENCE > 0 to re-enable.
PATIENCE = env_int("PROBE_PATIENCE", 0)
# A Z₂ configuration is 27 648 sites against an SU(2) timeslice's 1 728, so the
# same batch would be 2.7× the activations even before the slice axis is
# counted. Halved by default; the knob is the same one.
BATCH_CONFIGS = env_int("PROBE_BATCH", 4 if IS_Z2 else 8)
GRAD_CHECKPOINT = env_flag("PROBE_GRAD_CHECKPOINT", True)
RUN_TAG = env_str("PROBE_RUN_TAG", "")
# Read here rather than inside main() so validate_argv() sees the flag: the
# smoke tests pass --device=cpu, and an option the validator does not know about
# would be rejected as a typo.
DEVICE = env_str("PROBE_DEVICE", "")

if ARM not in available_arms():
    raise SystemExit(
        f"PROBE_ARM must be one of {sorted(available_arms())} with "
        f"PROBE_GROUP={GROUP} (got {ARM!r})"
    )
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

# …and the failure the threshold above cannot see. A run that learns, then blows
# up, then settles back at the trivial predictor has a *best* val from before
# the catastrophe, so `diverged` stays False while the run is plainly a failure.
# Measured on the two tuning checks: lcnn at 3e-3 reached val 0.414 at epoch 19,
# hit train 1.5e4 at epoch 21 and ended at 0.995 — worthless — with best_val
# 0.414. `collapsed` catches that shape: it ended no better than predicting the
# mean, having previously done materially better. The excursion peaks are kept
# alongside because they separate the two families qualitatively — GELT recovers
# from an excursion and the L-CNN does not (notes/m1_probe.md §4.1).
COLLAPSE_VAL = env_float("PROBE_COLLAPSE_VAL", 1.0)  # the trivial predictor

OUT_DIR = "results/z2_vortex/probe" if IS_Z2 else "results/m1_probe"
# Every knob that changes the result is in the name, so no run can overwrite
# another — the same rule train_glueball.py's artifact names follow.
# The ensemble's identity differs by study: SU(2) has seeded chains, Z₂ keys its
# cache on β alone (probe_common.cache_path), so β is what must be in the name.
_ensemble_tag = f"b{BETA}" if IS_Z2 else f"ens{ENSEMBLE_SEED}"
_default_n = 100 if IS_Z2 else 200
STEM = (
    f"probe_{ARM}_{TARGET}_{_ensemble_tag}_init{INIT_SEED}"
    + ("_null" if NULL else "")
    + ("" if N_CONFIGS == _default_n else f"_n{N_CONFIGS}")
    + RUN_TAG
)


def batches(idx, size):
    for lo in range(0, len(idx), size):
        yield idx[lo : lo + size]


def run_split(model, arch, U3, y, idx, device, optimizer=None, per_config=False,
              transport="average", mask=None):
    """One pass over ``idx``. Trains when ``optimizer`` is given, else evaluates.

    With ``per_config`` every configuration is scored on its own so the R²
    sufficient statistics come out per configuration — which is what makes the
    correlated jackknife of ΔR² possible offline.

    ``mask`` restricts **both the loss and the statistics** to the sites the
    study supervises on. The Z₂ vortex task masks to sites that carry a vortex:
    ~90% of sites have V1 exactly 0, and training on them spends the gradient on
    "is there a vortex here" rather than on the cluster size that is the
    question (``notes/where_attention_can_win.md`` §9.4). Masking the loss but
    not the statistics — or the other way round — would make the reported R² a
    different quantity from the one optimised, so they move together here.
    """
    train = optimizer is not None
    model.train(train)
    total, n_seen, stats = 0.0, 0, []
    size = 1 if per_config else BATCH_CONFIGS
    for chunk in batches(idx, size):
        U = U3[chunk].reshape(-1, *U3.shape[2:])  # (b·slices, 3, *Λ, nc, nc)
        t = y[chunk].reshape(-1, *y.shape[2:]).to(device, torch.float32)
        m = None if mask is None else mask[chunk].reshape(-1, *y.shape[2:]).to(device)
        with torch.set_grad_enabled(train):
            W, T = probe_inputs(U, arch, device, transport=transport)
            pred = model(W, T)
            loss = (
                torch.nn.functional.mse_loss(pred, t) if m is None
                else torch.nn.functional.mse_loss(pred[m], t[m])
            )
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += loss.item() * t.shape[0]
        n_seen += t.shape[0]
        if per_config:
            p_cpu, t_cpu = pred.detach().cpu(), t.detach().cpu()
            if m is not None:
                m_cpu = m.cpu()
                p_cpu, t_cpu = p_cpu[m_cpu], t_cpu[m_cpu]
            stats.append(accumulate_stats(t_cpu, p_cpu))
    return total / max(n_seen, 1), (torch.stack(stats) if per_config else None)


def main():
    validate_argv()
    # PROBE_DEVICE overrides the cuda → mps → cpu order; the smoke tests use it
    # to stay on the CPU, where the complex kernels are all available.
    forced = DEVICE
    device = torch.device(
        forced or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    print("=" * 78)
    study = "Z₂ vortex geometry" if IS_Z2 else "M1 probe"
    ensemble = f"β {BETA}" if IS_Z2 else f"ensemble {ENSEMBLE_SEED}"
    print(f"{study} — arm {ARM}  target {TARGET}  {ensemble}  "
          f"init {INIT_SEED}" + ("  [NULL: frozen stack]" if NULL else ""))
    print(f"device: {device}   lr {LR:g}  epochs {EPOCHS}  batch {BATCH_CONFIGS} configs")
    print("=" * 78)

    U3 = load_samples(seed=ENSEMBLE_SEED)
    _, targets = build_all_targets(U3, names=(TARGET,))
    mask = build_mask(U3)
    tr, va, te = splits()
    # The standardisation is computed over the **supervised** sites, so the
    # trivial predictor still scores exactly R² = 0 and the divergence and
    # collapse thresholds below stay absolute rather than drifting with the
    # mask's occupancy.
    y, mu, sigma = standardize(targets[TARGET], tr, mask=mask)
    y = y.float()
    print(f"splits: train {len(tr)}  val {len(va)}  test {len(te)} configurations "
          f"× {N_SLICES} " + ("configuration(s) fed whole" if IS_Z2 else "timeslices"))

    model, arch = build_arm(ARM, seed=INIT_SEED, grad_checkpoint=GRAD_CHECKPOINT)
    transport = arm_transport(ARM)
    model = model.to(device)
    dofs = real_dofs(model)
    # None for every arm the scale does not reach, so a GELT dump does not
    # claim a setting it never had.
    conv_init = Z2_LCNN_CONV_INIT if (IS_Z2 and arch == "lcnn") else None
    # Their init knob is a different number against a different fan-in, so it
    # is recorded under its own key rather than folded into conv_init.
    init_w = Z2_LCNN_REF_INIT_W if (IS_Z2 and arch == "lcnn_ref") else None
    print(f"arm {ARM}: {arch}, {dofs} real DOFs, {ARMS[ARM]}")

    if NULL:
        # Random features: the equivariant stack stays at its initialisation and
        # only the per-site readout learns. Nothing in the stack requires grad,
        # so the graph starts at the head and the pass is forward-only through
        # the expensive part.
        head = "head_fc1" if arch in ("lcnn", "lcnn_ref") else "mlp"
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith(head) or name.startswith("head_fc2"))
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable: {sum(p.numel() * (2 if p.is_complex() else 1) for p in params)}"
          f" of {dofs} real DOFs")

    # Refuse to overwrite a dump that was produced with different settings.
    # The learning rate is **not** in the stem — the batch puts it in the run
    # tag, a hand-written loop easily does not — so three rates under one tag
    # silently overwrite each other and the survivor is whichever finished last.
    # Nothing downstream could tell; the dump records the rate it was run with,
    # so the collision is invisible in the output as well as in the filenames.
    if os.path.exists(dump_path := f"{OUT_DIR}/{STEM}_stats.pt"):
        prev = torch.load(dump_path, map_location="cpu", weights_only=False)
        differs = {
            k: (prev.get(k), cur)
            for k, cur in (("lr", LR), ("epochs_run", EPOCHS), ("null", NULL),
                           ("transport", transport), ("batch", BATCH_CONFIGS),
                           ("conv_init_scale", conv_init),
                           ("init_w", init_w))
            if prev.get(k) is not None and prev.get(k) != cur
            # epochs_run is what the previous run *reached*, which is ≤ EPOCHS
            # when it stopped early, so only a larger value is a real conflict
            and not (k == "epochs_run" and prev.get(k) <= cur)
        }
        if differs:
            raise SystemExit(
                f"{dump_path} already exists and was produced with different "
                f"settings: "
                + ", ".join(f"{k} {a} → {b}" for k, (a, b) in differs.items())
                + f"\nGive this run its own --run-tag (the rate is not in the "
                f"stem), or delete the dump if it is superseded."
            )

    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    os.makedirs(OUT_DIR, exist_ok=True)
    checkpoint = f"{OUT_DIR}/{STEM}.pth"

    best_val, best_epoch, since = float("inf"), -1, 0
    history = []
    t_start = time.time()
    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss, _ = run_split(model, arch, U3, y, tr, device, optimizer,
                                  transport=transport, mask=mask)
        scheduler.step()
        val_loss, _ = run_split(model, arch, U3, y, va, device,
                                transport=transport, mask=mask)
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
        if PATIENCE > 0 and since >= PATIENCE:
            print(f"  early stop: {PATIENCE} epochs without improvement")
            break

    # The reported number is always the selected checkpoint's, never the last
    # epoch's — selection is on val, evaluation on test, and the two never mix.
    # Unless nothing was ever selected: a run whose val loss is nan from the
    # first epoch never improves on ``inf``, so no checkpoint is written. Score
    # the weights as they stand instead of dying on a file that does not exist —
    # the runs that reach this branch are exactly R-F's most extreme points, and
    # crashing here would delete the evidence of divergence rather than record
    # it. (``train_glueball.py``'s checkpoint comment warns about the same trap.)
    if best_epoch < 0:
        print("  ** no epoch improved on the initialisation — no checkpoint was "
              "written, scoring the final weights and recording the run as "
              "diverged")
    else:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    test_loss, stats = run_split(model, arch, U3, y, te, device, per_config=True,
                                 transport=transport, mask=mask)
    r2, err, _ = jackknife(stats, r2_from_stats)
    diverged = not (best_val < DIVERGENCE_VAL)  # not (<) also catches nan
    final_val = history[-1][1]
    max_train = max(h[0] for h in history)
    max_val = max(h[1] for h in history)
    collapsed = not (final_val < COLLAPSE_VAL) and best_val < 0.9 * COLLAPSE_VAL
    print(f"\nbest epoch {best_epoch + 1} of {len(history)} "
          f"(val {best_val:.5f}, last {history[-1][1]:.5f})"
          + (f"  ** DIVERGED (> {DIVERGENCE_VAL:g}, the trivial predictor scores "
             f"1.0) **" if diverged else "")
          + (f"  ** COLLAPSED (ended at {final_val:.3f}, no better than the "
             f"trivial predictor, after reaching {best_val:.3f}) **"
             if collapsed else ""))
    if max_train > DIVERGENCE_VAL:
        print(f"  ** excursion: train peaked at {max_train:.4g}, val at "
              f"{max_val:.4g}"
              + ("  — and the run did not come back **" if collapsed else
                 "  — and the run came back **"))
    print(f"test MSE {test_loss:.5f}   R² = {r2:+.4f} ± {err:.4f} "
          f"(blocked jackknife, block {JACK_BLOCK} configs)")

    dump = f"{OUT_DIR}/{STEM}_stats.pt"
    torch.save({
        "stats": stats, "r2": r2, "r2_err": err, "test_mse": test_loss,
        "arm": ARM, "arch": arch, "target": TARGET, "spec": ARMS[ARM],
        "transport": transport, "group": GROUP, "beta": BETA,
        "masked": mask is not None,
        "real_dofs": dofs, "null": NULL,
        "ensemble_seed": ENSEMBLE_SEED, "init_seed": INIT_SEED, "run_tag": RUN_TAG,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "batch": BATCH_CONFIGS,
        "conv_init_scale": conv_init, "init_w": init_w,
        "epochs_run": len(history),
        "best_epoch": best_epoch, "best_val": best_val, "history": history,
        "diverged": diverged, "divergence_val": DIVERGENCE_VAL,
        "collapsed": collapsed, "collapse_val": COLLAPSE_VAL,
        "final_val": final_val, "max_train": max_train, "max_val": max_val,
        "target_mu": mu, "target_sigma": sigma,
        "n_configs": N_CONFIGS, "n_slices": N_SLICES,
        "test_configs": te.tolist(), "jack_block": JACK_BLOCK,
        "wall_seconds": time.time() - t_start,
    }, dump)
    print(f"wrote {dump}")
    print(json.dumps({"arm": ARM, "target": TARGET, "init": INIT_SEED,
                      "null": NULL, "r2": round(r2, 5), "err": round(err, 5),
                      "diverged": diverged, "collapsed": collapsed}))


if __name__ == "__main__":
    main()
