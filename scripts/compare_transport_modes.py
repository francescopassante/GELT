"""A/B/C the three transport modes on accuracy *and* speed.

The experiment `notes/new_architecture.md` (2026-05-26) assumed the answer to
but never ran: does shortest-path averaging hurt, help, or cost anything
relative to a single canonical path?

Background — why this is worth measuring rather than reasoning about:

  * ``"average"``   T_Δx = mean over all shortest paths. Rotation symmetric,
                    but a sum of group elements is NOT a group element:
                    T·T† = 𝟙 fails on every multi-path (diagonal) offset. For
                    two paths T·T† = (𝟙 + Re W)/2 with W the enclosed Wilson
                    loop. This is the "dumbbell pollution" that note raised and
                    then ruled out on the false premise that the code was on
                    single-path mode. It was not.
  * ``"single"``    one canonical shortest path (lowest-index nonzero direction
                    at each DP step). T·T† = 𝟙 exactly, so each bilinear is a
                    clean rectangular Wilson-loop primitive — but 90° rotation
                    equivariance is broken.
  * ``"projected"`` average, then mapped back onto the group. Keeps rotation
                    symmetry *and* restores T·T† = 𝟙. Well defined for SU(N)
                    only: Z₂ averages cancel to exactly 0, where the nearest
                    group element is a tie and any tie-break breaks gauge
                    covariance (build_transport_average raises there).

The target is per-site Wilson-loop regression, sweeping loop size. That is the
task the 3×3 wall was observed on, and the size sweep is the discriminator: if
averaging dilutes specific-path loop content, the gap should *grow* with loop
size rather than sit at a constant offset.

Run (from the repo root):

    python scripts/compare_transport_modes.py

Env overrides: TRANSPORT_L, TRANSPORT_BETA, TRANSPORT_N, TRANSPORT_R,
TRANSPORT_EPOCHS, TRANSPORT_GROUP (su2|z2), TRANSPORT_LOOPS (e.g. "1x1,2x2,3x3").
Writes results/wilson_regression/transport_mode_comparison.{png,json}
and prints a summary table.
"""

import functools
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from gelt.blocks_rope import GELT
from gelt.lattice import SU, Z2, build_transport_average, plaquette_tensor, rectangular_wilson_loop
from gelt.sampler import heatbath_overrelaxation_sweep, mcmc_ensemble, metropolis_sweep

# Output artifacts are grouped by study under results/; create the dirs the
# first time this runs in a fresh clone (they hold generated files only).
for _d in ("results/sampler", "results/glueball", "results/attention",
          "results/wilson_regression", "datasets"):
    os.makedirs(_d, exist_ok=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROUP_NAME = os.environ.get("TRANSPORT_GROUP", "su2").lower()
L = int(os.environ.get("TRANSPORT_L", 8))
D = 2
R = int(os.environ.get("TRANSPORT_R", 2))
N = int(os.environ.get("TRANSPORT_N", 2000))
BETA = float(os.environ.get("TRANSPORT_BETA", 2.4 if GROUP_NAME == "su2" else 0.7))
EPOCHS = int(os.environ.get("TRANSPORT_EPOCHS", 40))
BATCH = int(os.environ.get("TRANSPORT_BATCH", 32))
LR = 2e-3

LOOPS = [
    tuple(int(v) for v in spec.lower().split("x"))
    for spec in os.environ.get("TRANSPORT_LOOPS", "1x1,2x2,3x3").split(",")
]

# Model capacity: fixed across modes, so the only difference is the transport.
NHEAD = 4
LAYERS = 4
D_QKV = 4
MLP_HIDDEN = 32

INIT_SEED = 0  # identical model init across modes
DATA_SEED = 1  # identical ensemble across modes

gaugegroup = SU(2) if GROUP_NAME == "su2" else Z2()
IS_Z2 = isinstance(gaugegroup, Z2)
model_dtype = torch.float32 if IS_Z2 else torch.complex64
data_dtype = torch.float32

# Z₂ has no well-defined projection where averages vanish, so it gets two arms.
MODES = ["average", "single"] if IS_Z2 else ["average", "single", "projected"]

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def sample_configs():
    """One ensemble, shared by every mode and every loop size."""
    torch.manual_seed(DATA_SEED)
    if IS_Z2:
        sweep = metropolis_sweep
    else:
        sweep = functools.partial(heatbath_overrelaxation_sweep, n_or=4)
    configs, _ = mcmc_ensemble(
        L,
        D,
        gaugegroup,
        BETA,
        N,
        n_therm=200,
        n_skip=5,
        dtype=data_dtype,
        sweep_fn=sweep,
    )
    return configs


def build_arrays(configs, mode, loop):
    """(W, T, y) for one mode and one loop size, plus the transport build time."""
    Rl, Tl = loop
    W = plaquette_tensor(configs, gaugegroup).to(model_dtype)
    y = rectangular_wilson_loop(configs, gaugegroup, R=Rl, T=Tl, mu=0, nu=1)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    T = build_transport_average(configs, R=R, gaugegroup=gaugegroup, mode=mode)
    if device == "cuda":
        torch.cuda.synchronize()
    build_s = time.perf_counter() - t0

    return W, T.to(model_dtype), y.to(torch.float32), build_s


def split_three(n, frac=(0.7, 0.15, 0.15)):
    g = torch.Generator().manual_seed(DATA_SEED)
    perm = torch.randperm(n, generator=g)
    n_tr = int(frac[0] * n)
    n_va = int(frac[1] * n)
    return perm[:n_tr], perm[n_tr : n_tr + n_va], perm[n_tr + n_va :]


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------


def make_model():
    torch.manual_seed(INIT_SEED)
    return GELT(
        gaugegroup=gaugegroup,
        L=L,
        D=D,
        R=R,
        nhead=NHEAD,
        gemhsa_layers=LAYERS,
        d_qkv=D_QKV,
        dtype=model_dtype,
        mlp_hidden=MLP_HIDDEN,
        reduction="none",  # per-site Wilson-loop supervision
        mlp_zero_init=False,
    ).to(device)


def run_one(W, T, y, tag):
    """Train once; return test R², best val loss, mean step time, peak memory."""
    idx_tr, idx_va, idx_te = split_three(len(y))

    # Standardise the target on the train split only.
    mu = y[idx_tr].mean()
    sd = y[idx_tr].std().clamp_min(1e-8)
    yz = (y - mu) / sd

    def loader(idx, shuffle):
        return DataLoader(
            torch.utils.data.TensorDataset(W[idx], T[idx], yz[idx]),
            batch_size=BATCH,
            shuffle=shuffle,
        )

    train_loader = loader(idx_tr, True)
    val_loader = loader(idx_va, False)
    test_loader = loader(idx_te, False)

    model = make_model()
    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
    crit = nn.MSELoss()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    step_times, best_val, best_state = [], float("inf"), None
    for epoch in range(EPOCHS):
        model.train()
        for Xb, Tb, yb in train_loader:
            Xb, Tb, yb = Xb.to(device), Tb.to(device), yb.to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt.zero_grad()
            loss = crit(model(Xb, Tb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if device == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for Xb, Tb, yb in val_loader:
                Xb, Tb, yb = Xb.to(device), Tb.to(device), yb.to(device)
                vl += crit(model(Xb, Tb), yb).item() * len(yb)
                vn += len(yb)
        vl /= vn
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        sched.step()
        print(f"    [{tag}] epoch {epoch + 1:3d}/{EPOCHS}  val {vl:.5f}", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    se, n_seen = 0.0, 0
    with torch.no_grad():
        for Xb, Tb, yb in test_loader:
            Xb, Tb, yb = Xb.to(device), Tb.to(device), yb.to(device)
            se += ((model(Xb, Tb) - yb) ** 2).sum().item()
            n_seen += yb.numel()
    mse = se / n_seen
    # Targets are standardised, so Var(y) = 1 and R² = 1 − MSE.
    r2 = 1.0 - mse

    peak_gb = (
        torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else float("nan")
    )
    mean_step_ms = 1000 * sum(step_times) / len(step_times)
    return r2, best_val, mean_step_ms, peak_gb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(
        f"device={device}  group={gaugegroup}  L={L} D={D} R={R} beta={BETA} "
        f"N={N} epochs={EPOCHS}\nmodes={MODES}  loops={LOOPS}",
        flush=True,
    )

    configs = sample_configs()
    print(f"ensemble: {tuple(configs.shape)}", flush=True)

    results = {}
    for loop in LOOPS:
        for mode in MODES:
            tag = f"{loop[0]}x{loop[1]}/{mode}"
            print(f"\n=== {tag} ===", flush=True)
            W, T, y, build_s = build_arrays(configs, mode, loop)
            r2, val, step_ms, peak = run_one(W, T, y, tag)
            results[tag] = dict(
                loop=f"{loop[0]}x{loop[1]}",
                mode=mode,
                r2=r2,
                val=val,
                build_s=build_s,
                step_ms=step_ms,
                peak_gb=peak,
            )
            print(
                f"  -> R²={r2:.4f}  val={val:.5f}  build={build_s:.1f}s  "
                f"step={step_ms:.1f}ms  peak={peak:.2f}GB",
                flush=True,
            )
            del W, T, y
            if device == "cuda":
                torch.cuda.empty_cache()

    print("\n" + "=" * 74)
    print(f"{'loop':>6} {'mode':>10} {'R²':>9} {'build(s)':>10} {'step(ms)':>10} {'peak(GB)':>10}")
    print("-" * 74)
    for k, r in results.items():
        print(
            f"{r['loop']:>6} {r['mode']:>10} {r['r2']:>9.4f} {r['build_s']:>10.1f} "
            f"{r['step_ms']:>10.1f} {r['peak_gb']:>10.2f}"
        )
    print("=" * 74)

    with open("results/wilson_regression/transport_mode_comparison.json", "w") as fh:
        json.dump(results, fh, indent=2)

    # --- figure: R² vs loop size per mode, and the speed cost ---------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    loops_lbl = [f"{a}x{b}" for a, b in LOOPS]
    for mode in MODES:
        ys = [results[f"{lb}/{mode}"]["r2"] for lb in loops_lbl]
        axes[0].plot(loops_lbl, ys, "o-", label=mode)
    axes[0].set_xlabel("Wilson loop size")
    axes[0].set_ylabel("test R²")
    axes[0].set_title(f"Accuracy vs transport mode ({gaugegroup}, L={L}, R={R})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    width = 0.8 / len(MODES)
    for i, mode in enumerate(MODES):
        ys = [results[f"{lb}/{mode}"]["step_ms"] for lb in loops_lbl]
        xs = [j + i * width for j in range(len(loops_lbl))]
        axes[1].bar(xs, ys, width=width, label=mode)
    axes[1].set_xticks([j + 0.4 - width / 2 for j in range(len(loops_lbl))])
    axes[1].set_xticklabels(loops_lbl)
    axes[1].set_xlabel("Wilson loop size")
    axes[1].set_ylabel("mean train step (ms)")
    axes[1].set_title("Step time (transport build reported separately)")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig("results/wilson_regression/transport_mode_comparison.png", dpi=150)
    print("wrote transport_mode_comparison.png and transport_mode_comparison.json")
