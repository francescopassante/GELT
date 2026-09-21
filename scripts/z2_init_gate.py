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
    Z2GATE_ARM       which implementation to gate: ``lcnn`` (ours, the
                     default) or ``lcnn_ref`` (the authors'). They carry
                     different knobs — conv_init_scale and init_w — written
                     against different fan-ins, so a value measured for one
                     says nothing about the other.
    Z2GATE_BETAS     couplings to check; default §9.6's two chosen ones.
    Z2GATE_SCALES    L-Conv init scales; must contain the arms' configured
                     value or the gate refuses to run (it would pass vacuously).
    Z2GATE_SEEDS     initialisation seeds per cell (default 3).
    Z2GATE_CONFIGS   configurations per cell (default 3).
    Z2GATE_MAX       the gate: the largest field magnitude that counts as O(1)
                     (default 10.0 — the SU(2) reference is 0.05 … 0.19, so
                     this is two decades of slack, not a tight bound).

Writes ``results/z2_vortex/init_gate_<arm>.pt``. Exit status is 1 if the scale the
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
    Z2_LCNN_REF_INIT_W,
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

# Which implementation's initialisation is being gated. The two are *different
# parametrisations of the same layer family* — ours factors the bilinear kernel
# through L-Conv's c_out, theirs carries it whole — and their init variances are
# written against different fan-ins, so the 0.2 measured for ours says nothing
# about theirs. Each carries its own knob and its own grid.
ARM = env_str("Z2GATE_ARM", "lcnn")
if ARM not in ("lcnn", "lcnn_ref"):
    raise SystemExit(f"Z2GATE_ARM must be 'lcnn' or 'lcnn_ref' (got {ARM!r})")
_IS_REF = ARM == "lcnn_ref"
_SCALE_KNOB = "init_w" if _IS_REF else "conv_init_scale"
_SCALE_ENV = (
    "PROBE_Z2_LCNN_REF_INIT_W" if _IS_REF else "PROBE_Z2_LCNN_CONV_INIT"
)
_CONFIGURED = Z2_LCNN_REF_INIT_W if _IS_REF else Z2_LCNN_CONV_INIT
_DEFAULT_SCALES = "0.2,0.5,1.0,1.5" if _IS_REF else "0.1,0.2,0.3,0.5"
SCALES = [float(x) for x in env_str("Z2GATE_SCALES", _DEFAULT_SCALES).split(",")]
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


def lcnn_ref_layer_profile(model, W, U):
    """The same walk through the authors' stack, in their layout.

    Their block keeps the links and the field in one tensor and transports
    internally, so the walk unpacks the field after each layer rather than
    threading a separate transport. The magnitude is the complex modulus, not
    the max over the split re/im components.
    """
    from gelt.lcnn_reference import reference_layers, to_ref_layout

    ref = reference_layers()
    prof = []
    with torch.no_grad():
        x = ref.repack_x(to_ref_layout(U), to_ref_layout(W))
        for conv, act in zip(model.convs, model.acts):
            x = act(conv(x))
            w = ref.unpack_x(x, len(model.dims))[1]
            prof.append(torch.view_as_complex(w.contiguous()).abs().max().item())
        out = model(W, U)
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
    print(f"arm: {ARM}  ({'the authors' + chr(39) + ' implementation' if _IS_REF else 'our implementation'})")
    print(f"the arms are configured with {_SCALE_KNOB} = {_CONFIGURED}")
    print(f"gate: the field entering the head must stay below {GATE_MAX}")

    if _CONFIGURED not in SCALES:
        # Otherwise the verdict below is vacuous: it would report PASS having
        # never tested the scale the arms actually use.
        raise SystemExit(
            f"the {ARM} arm's {_SCALE_KNOB} ({_CONFIGURED}) is not in "
            f"Z2GATE_SCALES ({SCALES}) — this gate exists to test *that* value, "
            f"so add it to the grid or point {_SCALE_ENV} at one of "
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
            os.environ[_SCALE_ENV] = str(scale)
            if _IS_REF:
                pc.Z2_LCNN_REF_INIT_W = scale
            else:
                pc.Z2_LCNN_CONV_INIT = scale
            for seed in range(SEEDS):
                model, arch = build_arm(ARM, seed=seed, grad_checkpoint=False)
                model = model.to(dev)
                for c in range(N_CONFIGS):
                    W, T = probe_inputs(configs[c : c + 1], arch, dev,
                                        transport=arm_transport(ARM))
                    prof, out = (
                        lcnn_ref_layer_profile(model, W, T) if _IS_REF
                        else lcnn_layer_profile(model, W, T)
                    )
                    finite = all(torch.isfinite(torch.tensor(v)) for v in prof)
                    rows.append(dict(arm=ARM, beta=beta, scale=scale,
                                     seed=seed, config=c,
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
        if scale == _CONFIGURED and not passed:
            ok_configured = False
        print(f"{beta:8.4f} {scale:7.2f} {v:34.4g}   "
              f"{'PASS' if passed else '** FAIL **'}"
              + ("   ← the arms' setting" if scale == _CONFIGURED else ""))

    os.makedirs("results/z2_vortex", exist_ok=True)
    # Per arm: the two implementations' gates are different measurements and
    # must not overwrite each other.
    out = f"results/z2_vortex/init_gate_{ARM}.pt"
    torch.save({"rows": rows, "worst": {f"{b}_{s}": v for (b, s), v in worst.items()},
                "arm": ARM, "knob": _SCALE_KNOB,
                "configured": _CONFIGURED, "gate_max": GATE_MAX,
                "betas": BETAS, "scales": SCALES, "seeds": SEEDS,
                "n_configs": N_CONFIGS, "lattice": LATTICE}, out)
    print(f"\nwrote {out}")

    if ok_configured:
        print(f"\nVERDICT: {ARM} {_SCALE_KNOB} = {_CONFIGURED} holds at the "
              f"production volume — the sweep may run.")
        return 0
    survivors = sorted({s for (b, s), v in worst.items() if v < GATE_MAX})
    print(f"\nVERDICT: {ARM} {_SCALE_KNOB} = {_CONFIGURED} FAILS at the "
          f"production volume. 8³ was not a proxy for 48 × 24 × 24, and §9.5.1 "
          f"has to be re-stated.\n"
          + (f"  Scales that survive here: {survivors} — set "
             f"{_SCALE_ENV} and re-run before anything trains."
             if survivors else
             "  No scale in the grid survives. Widen Z2GATE_SCALES downward "
             "before concluding the arm is unusable."))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
