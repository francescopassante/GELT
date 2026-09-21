"""
=========================================================================
The glueball arms' initialisation gate — does the forward survive 4 layers?
=========================================================================

``scripts/z2_init_gate.py``'s sibling for the SU(2) glueball task, and it exists
for the same reason: an L-CB stack is a matrix polynomial whose degree doubles
per layer, so the field entering the head grows multiplicatively with depth and
the growth rate is set by the L-Conv weight scale. On Z₂ that turned out to be a
sharp, seed-dependent cliff, and the first value chosen there (on a small box)
was falsified on this gate's first run.

Two things make it necessary again for ``GLUEBALL_ARCH=lcnn_ref``:

* their initialisation is **not** ours. Both divide by a fan-in, but a different
  one — ours by ``c_prime·n_shifts``, theirs by ``w_in_size·t_w_size`` — and
  their activation is ``LActPoly`` (a polynomial in ``Re Tr W`` with random
  coefficients), not our ``LAct``. Nothing measured for ``conv_init_scale``
  transfers to ``init_w``.
* **read the field, never the output.** The Rayleigh loss needs a nonzero
  readout so no arm here is zero-initialised, but the head is still a small
  linear map: a stack sitting at 1e12 can produce a perfectly ordinary-looking
  Ō and then overflow on the first optimiser step.

Forward-only, at initialisation, on configurations that already exist — no
training, no gradients, no sampling. GELT runs on the same configurations as the
control: its aggregation is a convex combination, so if *its* field is O(1) the
box and the inputs are not what is wrong.

    python scripts/glueball_init_gate.py
    GIG_ARCHS=lcnn_ref GIG_INIT_WS=0.3,0.5,1.0 python scripts/glueball_init_gate.py

Environment overrides (each also ``--name=value`` in argv, via train_glueball):
    GIG_ARCHS      architectures to walk (default gelt,lcnn,lcnn_ref).
    GIG_INIT_WS    init_w values for lcnn_ref (default 0.3,0.5,1.0,1.5).
    GIG_CONV_INITS conv_init_scale values for lcnn (default 1.0 — the value the
                   published runs used; widen it if this gate ever fails).
    GIG_SEEDS      initialisation seeds per cell (default 3).
    GIG_CONFIGS    configurations per cell (default 2).
    GIG_MAX        the gate: largest field magnitude that counts as O(1)
                   (default 100.0 — GELT's own is the printed reference).

Writes ``results/glueball/init_gate.pt``. Exit status is 1 if any scanned arm's
default setting fails, so a batch can stop on it.
"""

import importlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gelt.lattice import random_links  # noqa: E402

import train_glueball as tg  # noqa: E402


def _env(name, default):
    flag = f"--{name.lower().replace('gig_', '').replace('_', '-')}="
    for a in sys.argv[1:]:
        if a.startswith(flag):
            return a.split("=", 1)[1]
    return os.environ.get(name, default)


ARCHS = [a.strip() for a in _env("GIG_ARCHS", "gelt,lcnn,lcnn_ref").split(",")]
INIT_WS = [float(x) for x in _env("GIG_INIT_WS", "0.3,0.5,1.0,1.5").split(",")]
CONV_INITS = [float(x) for x in _env("GIG_CONV_INITS", "1.0").split(",")]
SEEDS = int(_env("GIG_SEEDS", "3"))
N_CONFIGS = int(_env("GIG_CONFIGS", "2"))
GATE_MAX = float(_env("GIG_MAX", "100.0"))


def device():
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def reload_tg(arch, **env):
    """``train_glueball`` re-imported with a different ARCH.

    It reads its configuration once at module level — which is what makes a run
    unable to drift from its own command line — so switching architecture in one
    process means reloading it. Nothing heavy happens at import.
    """
    os.environ["GLUEBALL_ARCH"] = arch
    for k, v in env.items():
        os.environ[k] = str(v)
    return importlib.reload(tg)


def batch(mod, b):
    if os.path.exists(mod.CACHE):
        return torch.load(mod.CACHE)[:b].to(mod.MODEL_DTYPE)
    print(f"   cache absent ({mod.CACHE}) — Haar-random configs instead. The "
          f"field at init is input-dependent, so read this as indicative only.")
    return torch.stack(
        [random_links(mod.L, mod.D, mod.gaugegroup, Lt=mod.LT) for _ in range(b)]
    ).to(mod.MODEL_DTYPE)


# ── The three stack walks ────────────────────────────────────────────────────
# Deliberately duplicated from z2_init_gate.py rather than shared: they are
# five-line diagnostics over three different block APIs, and the thing that must
# not drift between the two gates is the *question*, not the traversal.

def walk_gelt(model, W, T):
    prof = []
    with torch.no_grad():
        x = model.lift(W)
        T_all = torch.cat([torch.ones_like(T[:, :1]), T], dim=1)
        # GELT itself keeps no group reference — GEMHSA does, and the dagger is
        # computed once at the GELT level and threaded in (blocks.py §attn).
        T_dag = model.gemhsa_models[0].gaugegroup.dagger(T_all)
        for blk in model.gemhsa_models:
            x = blk(x, T_all, T_dag)
            prof.append(x.abs().max().item())
    return prof


def walk_lcnn(model, W, T):
    prof = []
    with torch.no_grad():
        x = W
        for lcb, lact in zip(model.lcb_blocks, model.l_acts):
            x = lact(lcb(x, T))
            prof.append(x.abs().max().item())
    return prof


def walk_lcnn_ref(model, W, U):
    from gelt.lcnn_reference import reference_layers, to_ref_layout

    ref = reference_layers()
    prof = []
    with torch.no_grad():
        x = ref.repack_x(to_ref_layout(U), to_ref_layout(W))
        for conv, act in zip(model.convs, model.acts):
            x = act(conv(x))
            w = ref.unpack_x(x, len(model.dims))[1]
            prof.append(torch.view_as_complex(w.contiguous()).abs().max().item())
    return prof


WALK = {"gelt": walk_gelt, "lcnn": walk_lcnn, "lcnn_ref": walk_lcnn_ref}


def cells(arch):
    """(label, env) pairs: the scale knob this architecture actually has."""
    if arch == "lcnn_ref":
        return [(w, {"GLUEBALL_LCNN_REF_INIT_W": w}) for w in INIT_WS]
    if arch == "lcnn":
        return [(s, {"GLUEBALL_LCNN_INIT_SCALE": 1.0}) for s in CONV_INITS]
    return [(None, {})]


def main():
    dev = device()
    print("=" * 78)
    print("Glueball arms — initialisation gate at the production shape")
    print("notes/lcnn_reference_switch.md §6, step 1")
    print("=" * 78)
    print(f"device: {dev} | {N_CONFIGS} configs × {SEEDS} seeds | "
          f"gate: field < {GATE_MAX}")

    rows, failed = [], []
    for arch in ARCHS:
        for scale, env in cells(arch):
            mod = reload_tg(arch, **env)
            label = f"{arch}" + ("" if scale is None else f"  scale {scale}")
            print(f"\n── {label}   levels {list(mod.INPUT_SMEAR_LEVELS)}   "
                  f"{mod._geometry_label()}")
            configs = batch(mod, N_CONFIGS).to(dev)
            worst = 0.0
            for seed in range(SEEDS):
                torch.manual_seed(seed)
                model = mod._build_model().to(dev)
                n_real = sum(
                    p.numel() * (2 if p.is_complex() else 1)
                    for p in model.parameters()
                )
                for c in range(N_CONFIGS):
                    W, T = mod.config_inputs(configs[c : c + 1], dev)
                    prof = WALK[arch](model, W, T)
                    with torch.no_grad():
                        obar = mod.network_obar(model, configs[c : c + 1], dev)
                    finite = all(v == v and v != float("inf") for v in prof)
                    worst = max(worst, max(prof) if finite else float("inf"))
                    rows.append(dict(arch=arch, scale=scale, seed=seed, config=c,
                                     profile=prof, obar=obar.abs().max().item(),
                                     finite=finite, real_dofs=n_real))
                    del W, T
                print(f"   seed {seed}  field by layer: "
                      + "  ".join(f"{v:.3g}" for v in prof)
                      + f"   |Ō|max {rows[-1]['obar']:.3g}"
                      + ("" if finite else "   ** NOT FINITE **"))
            if worst >= GATE_MAX:
                failed.append(label)
            print(f"   worst field over seeds × configs: {worst:.4g}   "
                  f"{'PASS' if worst < GATE_MAX else '** FAIL **'}")

    os.makedirs("results/glueball", exist_ok=True)
    out = "results/glueball/init_gate.pt"
    torch.save({"rows": rows, "archs": ARCHS, "init_ws": INIT_WS,
                "conv_inits": CONV_INITS, "seeds": SEEDS,
                "n_configs": N_CONFIGS, "gate_max": GATE_MAX}, out)
    print(f"\nwrote {out}")

    if failed:
        print(f"\nVERDICT: {failed} exceed the gate. Pick a surviving scale "
              f"before training — a run that starts here does not fail at "
              f"construction, it fails after the first optimiser step.")
        return 1
    print("\nVERDICT: every scanned setting holds. The sweep may run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
