"""
=========================================================================
The Z₂ arms' initialisation gate — does the L-CNN's forward survive?
=========================================================================

``notes/where_attention_can_win.md`` §9.5.1: at ``nc = 1`` the L-Act gate
``g(Re Tr W/nc)·W`` multiplies by its own argument instead of damping it, and
the L-Bilin squares the result again, so the field grows doubly exponentially
with depth. The field entering the head, Z₂, ``c_hidden = 6``, three seeds:

    layers          1         2         3         4
    8³            10 … 24   150 … 770  2e6 … 5e8  4e22 … 6e29
    48 × 24 × 24  29 … 51   1e3 … 3e4  2e10 … 3e15   inf

**Read the field, never the output.** ``build_arm`` zero-initialises the head,
so the output is identically 0 whatever the field is: a model one gradient step
from ``inf`` looks perfectly healthy from outside. That is how the first
end-to-end Z₂ run came back NaN after an epoch rather than at construction.

``conv_init_scale`` tames it, and **the value the arms carry was chosen on an
8³ box while the study runs at 48 × 24 × 24 with 54× the sites**. The growth is
multiplicative in depth and the statistic is a maximum over sites, so the small
box is not a proxy for the large one — finding that out inside an 84-run grid
would cost the grid.

So this is the gate, and it is cheap because it is **forward-only, at
initialisation, on configurations that already exist**: no training, no
gradients, no sampling. It reports, per (β, scale, seed), the magnitude of the
L-CNN's output and of the field entering its head, with GELT on the same
configurations as the reference — GELT's aggregation is a convex combination,
so its field is the control that says the box is not the problem.

    python scripts/z2_init_gate.py

Environment overrides (each also ``--name=value`` in argv):
    Z2GATE_BETAS     couplings to check; default §9.6's two chosen ones.
    Z2GATE_SCALES    L-Conv init scales; must contain the arms' configured
                     value or the gate refuses to run (it would pass vacuously).
    Z2GATE_SEEDS     initialisation seeds per cell (default 3).
    Z2GATE_CONFIGS   configurations per cell (default 3).
    Z2GATE_MAX       the gate: the largest field magnitude that counts as O(1)
                     (default 10.0 — the SU(2) reference is 0.05 … 0.19, so
                     this is two decades of slack, not a tight bound).

Writes ``results/z2_vortex/init_gate.pt``. Exit status is 1 if the scale the
arms are configured with fails, so a batch can stop on it.
"""

import os
import sys

import torch

# This script is Z₂-only: it exists to check the Z₂ arms' initialisation, and
# importing probe_common under the SU(2) default would silently build the wrong
# model. Set before the import rather than asking the caller to remember.
os.environ["PROBE_GROUP"] = "z2"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe_common import (  # noqa: E402
    LATTICE,
    LAYERS,
    Z2_LCNN_CONV_INIT,
    arm_transport,
    build_arm,
    cache_path,
    env_float,
    env_int,
    env_str,
    gaugegroup,
    validate_argv,
    probe_inputs,
)
import probe_common as pc  # noqa: E402

BETAS = [float(b) for b in env_str("Z2GATE_BETAS", "0.7520,0.7450").split(",")]
SCALES = [float(x) for x in env_str("Z2GATE_SCALES", "0.1,0.2,0.3,0.5").split(",")]
SEEDS = env_int("Z2GATE_SEEDS", 3)
N_CONFIGS = env_int("Z2GATE_CONFIGS", 3)
GATE_MAX = env_float("Z2GATE_MAX", 10.0)


DEVICE = env_str("Z2GATE_DEVICE", "")


def device():
    forced = DEVICE
    return torch.device(
        forced or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    )


def lcnn_layer_profile(model, W, T):
    """``|W|max`` after each L-CB + L-Act, and the model's output magnitude.

    Walking the stack rather than reading the output alone, because the output
    passes through a zero-initialised head: a head that has not learned yet maps
    any finite field to 0, so the *output* can look healthy while the field that
    produced it is 1e20 and one gradient step away from inf.
    """
    prof = []
    with torch.no_grad():
        x = W
        for lcb, lact in zip(model.lcb_blocks, model.l_acts):
            x = lact(lcb(x, T))
            prof.append(x.abs().max().item())
        out = model(W, T)
    return prof, out.abs().max().item()


def gelt_layer_profile(model, W, T):
    """The same walk for GELT — the control that says the box is not at fault."""
    prof = []
    with torch.no_grad():
        x = model.lift(W)
        T_all = torch.cat([torch.ones_like(T[:, :1]), T], dim=1)
        T_dag = gaugegroup.dagger(T_all)
        for blk in model.gemhsa_models:
            x = blk(x, T_all, T_dag)
            prof.append(x.abs().max().item())
    return prof


def main():
    validate_argv()
    dev = device()
    print("=" * 78)
    print("Z₂ arms — initialisation gate at the production volume")
    print("notes/where_attention_can_win.md §9.5.1")
    print("=" * 78)
    print(f"device: {dev} | lattice {LATTICE} ({LATTICE[0] * LATTICE[1] * LATTICE[2]} "
          f"sites) | {LAYERS} layers | {N_CONFIGS} configs × {SEEDS} seeds")
    print(f"the arms are configured with conv_init_scale = {Z2_LCNN_CONV_INIT}")
    print(f"gate: the field entering the head must stay below {GATE_MAX}")

    if Z2_LCNN_CONV_INIT not in SCALES:
        # Otherwise the verdict below is vacuous: it would report PASS having
        # never tested the scale the arms actually use.
        raise SystemExit(
            f"the arms' conv_init_scale ({Z2_LCNN_CONV_INIT}) is not in "
            f"Z2GATE_SCALES ({SCALES}) — this gate exists to test *that* value, "
            f"so add it to the grid or point PROBE_Z2_LCNN_CONV_INIT at one of "
            f"the scales being scanned."
        )

    rows, worst = [], {}
    for beta in BETAS:
        pc.BETA = beta  # the cache key is β for Z₂ (probe_common.cache_path)
        path = cache_path()
        if not os.path.exists(path):
            raise SystemExit(
                f"Ensemble cache {path} not found — this gate samples nothing. "
                f"Run scripts/train_z2_glueball.py {beta} first."
            )
        print(f"\n── β = {beta}  ({path})")
        configs = torch.load(path, map_location="cpu")[:N_CONFIGS].to(pc.MODEL_DTYPE)

        # GELT once per β: it has no scale knob, and it is the reference.
        g_model, g_arch = build_arm("gelt", seed=0, grad_checkpoint=False)
        g_model = g_model.to(dev)
        W, T = probe_inputs(configs[:1], g_arch, dev, transport=arm_transport("gelt"))
        g_prof = gelt_layer_profile(g_model, W, T)
        print("   reference  GELT           field by layer: "
              + "  ".join(f"{v:.3g}" for v in g_prof))
        del W, T

        for scale in SCALES:
            os.environ["PROBE_Z2_LCNN_CONV_INIT"] = str(scale)
            pc.Z2_LCNN_CONV_INIT = scale
            for seed in range(SEEDS):
                model, arch = build_arm("lcnn", seed=seed, grad_checkpoint=False)
                model = model.to(dev)
                for c in range(N_CONFIGS):
                    W, T = probe_inputs(configs[c : c + 1], arch, dev,
                                        transport=arm_transport("lcnn"))
                    prof, out = lcnn_layer_profile(model, W, T)
                    finite = all(torch.isfinite(torch.tensor(v)) for v in prof)
                    rows.append(dict(beta=beta, scale=scale, seed=seed, config=c,
                                     profile=prof, out=out, finite=finite))
                    key = (beta, scale)
                    worst[key] = max(worst.get(key, 0.0),
                                     max(prof) if finite else float("inf"))
                    del W, T
                print(f"   scale {scale:4.2f}  seed {seed}  field by layer: "
                      + "  ".join(f"{v:.3g}" for v in prof)
                      + f"   |out| {out:.3g}"
                      + ("" if finite else "   ** NOT FINITE **"))

    print("\n" + "=" * 78)
    print(f"{'β':>8s} {'scale':>7s} {'worst field over seeds × configs':>34s}   verdict")
    ok_configured = True
    for (beta, scale), v in sorted(worst.items()):
        passed = v < GATE_MAX
        if scale == Z2_LCNN_CONV_INIT and not passed:
            ok_configured = False
        print(f"{beta:8.4f} {scale:7.2f} {v:34.4g}   "
              f"{'PASS' if passed else '** FAIL **'}"
              + ("   ← the arms' setting" if scale == Z2_LCNN_CONV_INIT else ""))

    os.makedirs("results/z2_vortex", exist_ok=True)
    out = "results/z2_vortex/init_gate.pt"
    torch.save({"rows": rows, "worst": {f"{b}_{s}": v for (b, s), v in worst.items()},
                "configured": Z2_LCNN_CONV_INIT, "gate_max": GATE_MAX,
                "betas": BETAS, "scales": SCALES, "seeds": SEEDS,
                "n_configs": N_CONFIGS, "lattice": LATTICE}, out)
    print(f"\nwrote {out}")

    if ok_configured:
        print(f"\nVERDICT: conv_init_scale = {Z2_LCNN_CONV_INIT} holds at the "
              f"production volume — the sweep may run.")
        return 0
    survivors = sorted({s for (b, s), v in worst.items() if v < GATE_MAX})
    print(f"\nVERDICT: conv_init_scale = {Z2_LCNN_CONV_INIT} FAILS at the "
          f"production volume. 8³ was not a proxy for 48 × 24 × 24, and §9.5.1 "
          f"has to be re-stated.\n"
          + (f"  Scales that survive here: {survivors} — set "
             f"PROBE_Z2_LCNN_CONV_INIT and re-run before anything trains."
             if survivors else
             "  No scale in the grid survives. Widen Z2GATE_SCALES downward "
             "before concluding the arm is unusable."))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
