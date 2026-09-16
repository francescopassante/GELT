"""
=========================================================================
The transport arms' pre-flight — do the two transports differ at all?
=========================================================================

``notes/where_attention_can_win.md`` §5 requires a measurement before any
training code runs. For the M1 arms that measurement was a ceiling (the best
linear filter at matched reach, ``probe_preflight.py``). For the **transport**
arms of ``notes/m1_probe.md`` §4.4 the right gate is a different shape, and
weaker on purpose: a *necessary condition*. ``gelt`` and ``gelt_single`` are
the same network to the byte and differ only in the tensor ``T`` they are fed,
so if the two ``T`` are numerically close on production configurations the arms
must tie and six GPU-hours would buy a tautology.

What it measures, per L1 shell, on the cached ensemble the probe already uses:

* how many offsets in the shell have more than one shortest path — those are
  the only entries ``mode="single"`` can change;
* ``‖T_avg − T_single‖ / ‖T_single‖``, the relative size of the change;
* ``|T T† − 𝟙|`` for the average, which is *why* the change is not a rounding
  detail: for a two-path offset ``T_avg T_avg† = (𝟙 + Re W)/2`` with ``W`` the
  Wilson loop the two paths enclose, so the defect is a direct readout of the
  loop content the averaging carries and a single path does not;
* the same two numbers for ``mode="projected"``, which keeps the average's
  rotation symmetry and puts it back on the group — the arm that separates
  "the average carries loop content" from "the average is not a group element".

And the cost, which is the secondary reading: the transport is 62.8% of a GELT
step (``notes/performance_audit.md`` §5.0(v)), so a cheaper mode that ties on
accuracy is a result about the 3.9× step-cost gap against the L-CNN.

    python scripts/probe_transport_gate.py
    PROBE_ENSEMBLE_SEED=1 python scripts/probe_transport_gate.py

Samples nothing, loads no checkpoints, and runs on the CPU in a couple of
minutes. ``PROBE_GATE_CONFIGS`` (default 8) sets how many configurations are
averaged over — the spread over sites is enormous and the mean converges long
before the ensemble does.
"""

import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_common import (  # noqa: E402
    R,
    env_int,
    env_str,
    gaugegroup,
    load_timeslices,
)
from gelt.lattice import build_transport_average, l1_ball_offsets  # noqa: E402

# The gate, written down before the numbers exist. It is a floor, not a
# ceiling: below it the transport arms cannot separate and must not be run.
# 2% is the scale at which a difference in the inputs could plausibly survive
# four layers and a zero-initialised readout; anything smaller is noise the
# optimiser will not find.
MIN_RELATIVE_DIFFERENCE = 0.02

N_GATE = env_int("PROBE_GATE_CONFIGS", 8)
DEVICE = env_str("PROBE_DEVICE", "cpu")


def n_shortest_paths(dx):
    """Multinomial ``|Δx|₁! / Π_μ |Δx_μ|!`` — how many shortest paths Δx has."""
    n = math.factorial(sum(abs(d) for d in dx))
    for d in dx:
        n //= math.factorial(abs(d))
    return n


def shell_stats(T_ref, T_alt, offsets, multi_only):
    """``(mean relative ‖ΔT‖, mean unitarity defect of T_alt)`` over a subset.

    The relative norm is per offset and per site — ``‖T_alt − T_ref‖_F`` over
    ``‖T_ref‖_F``, both taken on the colour block — so it is scale-free and a
    value of 0 means the two modes built the same tensor.
    """
    rel, defect, nc = [], [], T_ref.shape[-1]
    eye = torch.eye(nc, dtype=T_ref.dtype, device=T_ref.device)
    for i, dx in enumerate(offsets):
        if multi_only and n_shortest_paths(dx) == 1:
            continue
        a, b = T_ref[:, i], T_alt[:, i]
        num = (b - a).abs().pow(2).sum(dim=(-2, -1)).sqrt()
        den = a.abs().pow(2).sum(dim=(-2, -1)).sqrt().clamp_min(1e-30)
        rel.append((num / den).mean().item())
        bb = b @ gaugegroup.dagger(b)
        defect.append((bb - eye).abs().amax(dim=(-2, -1)).mean().item())
    if not rel:
        return float("nan"), float("nan")
    return sum(rel) / len(rel), sum(defect) / len(defect)


def main():
    U3 = load_timeslices(n_configs=N_GATE, n_slices=1)
    U = U3[:, 0].to(DEVICE)  # (n, 3, L,L,L, nc,nc)
    offsets = l1_ball_offsets(D=3, R=R)
    print(f"\nR = {R}, D = 3: {len(offsets)} offsets, "
          f"{sum(1 for dx in offsets if n_shortest_paths(dx) > 1)} of them "
          f"multi-path — those are the only entries 'single' can change.\n")

    modes, T, secs = ("average", "single", "projected"), {}, {}
    for mode in modes:
        t0 = time.time()
        T[mode] = build_transport_average(U, R, gaugegroup, mode=mode)
        secs[mode] = time.time() - t0

    print(f"{'shell':>5} {'offsets':>8} {'multi':>6} "
          f"{'‖ΔT‖/‖T‖ single':>16} {'‖ΔT‖/‖T‖ proj':>14} "
          f"{'|TT†−1| avg':>12} {'|TT†−1| proj':>13}")
    for shell in range(1, R + 1):
        idx = [i for i, dx in enumerate(offsets)
               if sum(abs(d) for d in dx) == shell]
        sub = [offsets[i] for i in idx]
        n_multi = sum(1 for dx in sub if n_shortest_paths(dx) > 1)
        sel = {m: T[m][:, idx] for m in modes}
        rel_s, _ = shell_stats(sel["average"], sel["single"], sub, False)
        rel_p, def_p = shell_stats(sel["average"], sel["projected"], sub, False)
        _, def_a = shell_stats(sel["average"], sel["average"], sub, False)
        print(f"{shell:>5} {len(idx):>8} {n_multi:>6} "
              f"{rel_s:>16.4f} {rel_p:>14.4f} {def_a:>12.4f} {def_p:>13.2e}")

    # `shell_stats` reports the defect of its *second* argument, so the average's
    # own defect is the (average, average) call — the one number that says how
    # far off the group the averaging takes T, and the reason the difference
    # above is loop content rather than a rounding detail.
    rel_multi, def_single = shell_stats(T["average"], T["single"], offsets, True)
    _, def_multi = shell_stats(T["average"], T["average"], offsets, True)
    print(f"\nOver the multi-path offsets only: ‖T_avg − T_single‖/‖T_single‖ = "
          f"{rel_multi:.4f}, mean |T_avg T_avg† − 𝟙| = {def_multi:.4f}"
          f"  (single: {def_single:.1e}, a link product — it never left)")
    total = sum(secs.values())
    print(f"Build cost at this shape ({tuple(U.shape)}, {DEVICE}): "
          + ", ".join(f"{m} {secs[m]:.3f}s" for m in modes)
          + f"  →  single is {secs['average'] / secs['single']:.2f}× cheaper "
            f"than average")
    if total < 1.0:
        print("  ** those timings are too small to trust — for the cost reading "
              "use PROBE_GATE_CONFIGS=64\n     on the GPU, at the shape a "
              "training step actually builds.")

    print()
    if rel_multi < MIN_RELATIVE_DIFFERENCE:
        print(f"GATE FAILED: the two transports agree to {rel_multi:.4f} < "
              f"{MIN_RELATIVE_DIFFERENCE} on the offsets that can differ at "
              f"all.\n  `gelt` and `gelt_single` are the same network fed the "
              f"same numbers; they must tie.\n  Do not run the transport arms.")
        return 1
    print(f"GATE PASSED: {rel_multi:.4f} ≥ {MIN_RELATIVE_DIFFERENCE}. The two "
          f"transports are different objects on\n  the production ensemble, so "
          f"a tie between the arms would be a result rather than a tautology.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
