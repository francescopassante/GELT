"""Does the GELT arm's field survive four blocks at D = 2, L = 8?

The same gate as ``scripts/z2_init_gate.py``, for the other architecture and
the other geometry. It exists for the reason that one does: an init scale
measured somewhere else is not a measurement here. ``INIT_SCALE = 10.0`` is
this repo's value for GELT, but it was set at D = 3, L = 12 with a 3-channel
plaquette input; the 1+1D Wilson-loop arm is D = 2, L = 8, one input channel
and an L1-ball of radius 3, and the field through an attention stack depends on
all of those.

**Read the field, never the output.** ``mlp_zero_init=True`` makes the model's
output exactly zero whatever the residual stream is doing, so a stack one step
from ``inf`` looks perfectly healthy at the readout. This reports
``max |W|`` over sites and channels after each block, on real configurations at
the production volume.

Forward-only, no training, seconds on a CPU.

    python scripts/wilson_regression_init_gate.py
    WR_GATE_SCALES=1,3,10,30 WR_GATE_SEEDS=4 python scripts/wilson_regression_init_gate.py

Env: ``WR_GATE_SCALES`` (init_scale grid), ``WR_GATE_QK`` (qk_init_scale),
``WR_GATE_SEEDS``, ``WR_GATE_CONFIGS``, ``WR_TARGET``, ``WR_SIZE``, ``WR_L``.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wilson_regression_common import (
    GELT_ARCHS,
    GELT_INIT_SCALE,
    GELT_QK_INIT_SCALE,
    build_gelt,
    build_transport,
    cfg,
    load_split,
    pick_device,
    validate_argv,
)


def field_profile(model, W, T):
    """``max |W_i(x)|`` after each GEMHSA block, under no_grad.

    Reaches into the stack rather than calling ``forward``: the block outputs
    are not returned by the model and the readout cannot see them.
    """
    out = []
    with torch.no_grad():
        # Mirrors GELT.forward + GELT.attn exactly, up to keeping the field
        # after each block instead of only the last one.
        first = model.gemhsa_models[0]
        w_dtype = first.w_QKV.dtype
        x = model.lift(W.to(w_dtype))
        T = T.to(w_dtype)
        T_dag = first.gaugegroup.dagger(T)
        if T.shape[1] == first.n_offsets - 1:
            T, T_dag = first.prepend_self_offset(T, T_dag)
        for blk in model.gemhsa_models:
            x = blk(x, T, T_dag)
            out.append(x.abs().max().item())
    return out


def main():
    validate_argv()
    target = str(cfg("WR_TARGET", "W44")).upper()
    size = str(cfg("WR_SIZE", "matched")).lower()
    L = int(cfg("WR_L", 8))
    n_cfg = int(cfg("WR_GATE_CONFIGS", 8))
    n_seeds = int(cfg("WR_GATE_SEEDS", 4))
    qk = float(cfg("WR_GATE_QK", GELT_QK_INIT_SCALE))
    scales = [float(s) for s in
              str(cfg("WR_GATE_SCALES", "1,3,10,30")).split(",") if s]
    device = pick_device(cfg("WR_DEVICE", None))

    if GELT_INIT_SCALE not in scales:
        raise SystemExit(
            f"the arm's configured init_scale {GELT_INIT_SCALE} is not in the "
            f"scanned grid {scales} — the verdict would pass vacuously. Add it "
            f"to WR_GATE_SCALES, or change GELT_INIT_SCALE."
        )

    R = GELT_ARCHS[(target, size)]["R"]
    U, W, _y, _b, _m = load_split("train", L=L, target=target)
    U, W = U[:n_cfg], W[:n_cfg]
    print(f"{target} {size}: R={R}, {n_cfg} configurations at L={L}, "
          f"{n_seeds} seeds, qk_init_scale={qk}, device={device}")
    T = build_transport(U, R=R, device=device, progress=False)
    W, T = W.to(device), T.to(device)

    print(f"\n{'init_scale':>10} | max |W| after block 1..n   | verdict")
    print("-" * 68)
    for sc in scales:
        worst = None
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            model, _n, _s = build_gelt(target, size, L=L, init_scale=sc,
                                       qk_init_scale=qk)
            prof = field_profile(model.to(device), W, T)
            if worst is None or prof[-1] > worst[-1]:
                worst = prof
        tail = worst[-1]
        # The failure this gate is for is multiplicative growth through depth,
        # so the verdict is on the *last* block, and the whole profile is
        # printed because a stack that grows and then saturates is a different
        # animal from one that grows monotonically.
        ok = "ok" if 1e-3 < tail < 1e3 else ("VANISHES" if tail <= 1e-3
                                             else "BLOWS UP")
        mark = "  <- configured" if sc == GELT_INIT_SCALE else ""
        print(f"{sc:10.3g} | " + "  ".join(f"{v:8.2e}" for v in worst)
              + f"  | {ok}{mark}")


if __name__ == "__main__":
    main()
