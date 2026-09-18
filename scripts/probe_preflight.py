"""
=========================================================================
The M1 probe's pre-flight — the best fixed-offset-weight method at reach.
=========================================================================

``notes/where_attention_can_win.md`` §5 makes this the first gate of every
future A/B, and it cost ~4 h instead of ~60–100 the last time it was run:

  > **measure the best classical method at the architecture's own reach**
  > *before* building any training code.

Here that method is the **best linear filter over the L1-ball of radius 4** —
which is exactly the M1-free family. A convolution has one weight per offset,
fixed for every site and every configuration; that is what "input-independent
offset weighting" means when you strip the nonlinearity away. Its R² is
therefore the natural ceiling to beat, and ``1 − R²`` is the headroom
input-dependent weighting has to work with. (It is a *lower* bound on what the
fixed-weight *architectures* reach — ``frozen`` and the L-CNN are nonlinear and
deep — which is the right direction for a gate: if the linear filter already
saturates a target, nothing downstream of it can separate.)

Four readings, and the gate, are fixed in ``notes/m1_probe.md`` §3. Run:

    python scripts/probe_preflight.py                 # seed-0 ensemble
    PROBE_ENSEMBLE_SEED=1 python scripts/probe_preflight.py

It samples nothing, loads no checkpoints and needs no GPU.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_common import (  # noqa: E402
    BALL_RADIUS,
    N_CONFIGS,
    N_SLICES,
    TARGETS,
    accumulate_stats,
    build_all_targets,
    env_flag,
    env_int,
    jackknife,
    load_samples,
    r2_from_stats,
    splits,
    standardize,
)
from gelt.probe_targets import ball_features, ball_reduce  # noqa: E402

# The gate, written down before the numbers exist. A linear filter that already
# explains this much of a target has shown the target to be a convolution in
# disguise: no architecture whose only extra ingredient is input-dependent
# offset weighting can separate in the remainder, and the target is dropped.
LINEAR_CEILING_GATE = 0.90
# …and the mirror condition on T0, which must be *exactly* linear or it is not
# a calibration arm.
CALIBRATION_FLOOR = 0.999
# A target whose per-site variation is this small relative to its mean is
# degenerate — concentration has flattened it and the regression would be
# fitting round-off.
MIN_RELATIVE_SPREAD = 1e-3

CHUNK = env_int("PROBE_PREFLIGHT_CHUNK", 10)  # configs per streaming chunk
VERBOSE = env_flag("PROBE_VERBOSE", True)


def _features(f_slice, kind):
    """``(n_sites, n_features)`` design matrix for one chunk of slices.

    ``"ball"`` is one column per offset in the L1-ball (the general
    convolution at matched reach); ``"radial"`` is one column per shell (the
    convolution that is blind to direction). Both get an intercept column, so
    the fit is affine and the trivial predictor is nested inside it.
    """
    # (configs, slices, *Λ) → one batch axis, the layout the ball reductions
    # take and the same site ordering ``y.reshape(-1)`` produces.
    f_slice = f_slice.reshape(-1, *f_slice.shape[2:])
    if kind == "ball":
        X = ball_features(f_slice, BALL_RADIUS)  # (n_ball, b, *Λ)
    else:
        S, _ = ball_reduce(f_slice, BALL_RADIUS)
        X = torch.cat([S[:1], S[1:] - S[:-1]], dim=0)  # shells
    X = X.reshape(X.shape[0], -1).t().double()  # (n_sites, n_feat)
    return torch.cat([X, torch.ones(X.shape[0], 1, dtype=torch.float64)], dim=1)


def fit_linear_filter(f, y, train_idx, test_idx, kind, ridge=1e-8):
    """OLS over ``kind`` features; returns per-config test stats and the fit.

    The normal equations are accumulated in float64 over configuration chunks,
    so the (n_sites × n_features) design matrix is never materialised whole —
    at the production size it would be 8 M × 130.
    """
    n_feat = None
    XtX = Xty = None
    for lo in range(0, len(train_idx), CHUNK):
        idx = train_idx[lo : lo + CHUNK]
        X = _features(f[idx], kind)
        t = y[idx].reshape(-1).double()
        if XtX is None:
            n_feat = X.shape[1]
            XtX = torch.zeros(n_feat, n_feat, dtype=torch.float64)
            Xty = torch.zeros(n_feat, dtype=torch.float64)
        XtX += X.t() @ X
        Xty += X.t() @ t
    # A ridge proportional to the trace keeps the solve well posed when two
    # offsets carry near-identical information; at 1e-8 it is conditioning, not
    # regularisation, and it cannot flatter the fit.
    XtX += ridge * (XtX.diagonal().mean()) * torch.eye(n_feat, dtype=torch.float64)
    beta = torch.linalg.solve(XtX, Xty)

    stats = []
    for i in test_idx.tolist():
        idx = torch.tensor([i])
        pred = _features(f[idx], kind) @ beta
        stats.append(accumulate_stats(y[idx], pred))
    return torch.stack(stats), beta


def main():
    torch.manual_seed(0)
    print("=" * 78)
    print("M1 probe — pre-flight: the best fixed-offset-weight method at reach")
    print("=" * 78)
    U3 = load_samples(verbose=VERBOSE)
    print(f"\nTargets on the L1-ball of radius {BALL_RADIUS} "
          f"({N_CONFIGS} configs × {N_SLICES} timeslices):")
    f, targets = build_all_targets(U3, verbose=VERBOSE)
    del U3
    tr, va, te = splits()
    print(f"splits: train {len(tr)}  val {len(va)}  test {len(te)} configurations")

    rows, verdicts = [], {}
    for name in TARGETS:
        y_raw = targets[name]
        spread = (y_raw.std() / y_raw.abs().mean().clamp_min(1e-30)).item()
        y, mu, sigma = standardize(y_raw, tr)
        print(f"\n── {name} " + "─" * 66)
        print(f"  P1  dynamic range: std/|mean| = {spread:.3e}   "
              f"(μ = {mu:+.5f}, σ = {sigma:.5f})")

        res = {}
        for kind, label in (("ball", "P2  full ball"), ("radial", "P3  radial only")):
            stats, beta = fit_linear_filter(f, y, tr, te, kind)
            r2, err, _ = jackknife(stats, r2_from_stats)
            res[kind] = (r2, err)
            print(f"  {label:18s} ({beta.numel() - 1:3d} offsets + intercept): "
                  f"R² = {r2:+.4f} ± {err:.4f}")
        print(f"  P4  trivial (predict the test mean):           R² = +0.0000")

        r2_ball = res["ball"][0]
        if name == "T0":
            ok = r2_ball >= CALIBRATION_FLOOR
            verdicts[name] = ok
            print(f"  gate: calibration arm needs R² ≥ {CALIBRATION_FLOOR} "
                  f"→ {'PASS' if ok else 'FAIL'}")
        else:
            ok = (r2_ball <= LINEAR_CEILING_GATE) and (spread >= MIN_RELATIVE_SPREAD)
            verdicts[name] = ok
            print(f"  gate: needs R² ≤ {LINEAR_CEILING_GATE} and spread ≥ "
                  f"{MIN_RELATIVE_SPREAD:.0e} → {'PASS' if ok else 'FAIL'}")
            print(f"        headroom over the best convolution: "
                  f"{1 - r2_ball:+.4f}")
        rows.append((name, spread, res["ball"], res["radial"]))

    print("\n" + "=" * 78)
    print(f"{'target':8s} {'std/|mean|':>12s} {'R² ball':>18s} {'R² radial':>18s}"
          f" {'headroom':>10s}")
    for name, spread, ball, radial in rows:
        print(f"{name:8s} {spread:12.3e} {ball[0]:+11.4f} ± {ball[1]:.4f}"
              f" {radial[0]:+11.4f} ± {radial[1]:.4f} {1 - ball[0]:+10.4f}")

    os.makedirs("results/m1_probe", exist_ok=True)
    out = f"results/m1_probe/preflight_ens{env_int('PROBE_ENSEMBLE_SEED', 0)}.pt"
    torch.save({"rows": rows, "verdicts": verdicts,
                "gate": dict(linear_ceiling=LINEAR_CEILING_GATE,
                             calibration_floor=CALIBRATION_FLOOR,
                             min_spread=MIN_RELATIVE_SPREAD),
                "n_configs": N_CONFIGS, "n_slices": N_SLICES}, out)
    print(f"\nwrote {out}")

    failed = [k for k, v in verdicts.items() if not v]
    print("\nVERDICT: " + ("all targets clear the pre-flight — proceed to WP-C"
                           if not failed else
                           f"targets {failed} FAIL the gate; do not train them"))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
