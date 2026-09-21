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
* **the gate is on C(0), not on the field.** The Rayleigh loss is
  ``−mean_Δ C(Δ)/max(C(0), ε)`` and C(0) is the *connected* variance of Ō over
  configurations — so what matters is the operator's **fluctuation**, not its
  magnitude. On thermalised, smeared configurations W ≈ 𝟙 and the field's
  magnitude is dominated by that constant part: an earlier version of this gate
  reported ``|W|max`` and would have passed the run that then failed on every
  single batch. C(0) is reported here alongside the scale-free ratio
  ``C(0)/⟨Ō⟩²``, which is what separates the two cures — if the ratio is healthy
  the operator is merely *small* and ``--lcnn-init-scale`` (a head multiplier,
  C(0) ∝ its square) fixes it; if the ratio is itself ~0 the operator is
  genuinely constant and the stack's own scale (``init_w``) has to move.
* the field walk is kept as the diagnostic for the *other* end: a stack at 1e12
  can still produce an ordinary-looking Ō and overflow on the first step.

Forward-only, at initialisation, on configurations that already exist — no
training, no gradients, no sampling. GELT runs on the same configurations as the
control: its aggregation is a convex combination, so if *its* field is O(1) the
box and the inputs are not what is wrong.

    python scripts/glueball_init_gate.py
    GIG_ARCHS=lcnn_ref GIG_INIT_WS=1.0,4.0 python scripts/glueball_init_gate.py

Environment overrides (each also ``--name=value`` in argv, via train_glueball):
    GIG_ARCHS      architectures to walk (default gelt,lcnn,lcnn_ref).
    GIG_INIT_WS    init_w values for lcnn_ref (default 1.0,2.0,4.0,8.0 —
                   upward; see the note on the constant below).
    GIG_CONV_INITS conv_init_scale values for lcnn (default 1.0 — the value the
                   published runs used; widen it if this gate ever fails).
    GIG_ACTS       lcnn_ref: 0/1, whether their LActPoly is applied (default 0,
                   the paper's architecture — Tables V-VI carry no L-Act).
    GIG_HEADS      lcnn_ref: mlp (ours, what the matched widths assume) or
                   linear (theirs: one Linear per site, no hidden layer).
    GIG_SEEDS      initialisation seeds per cell (default 3).
    GIG_CONFIGS    configurations per cell (default 2).
    GIG_MAX        largest field magnitude that counts as O(1) (default 100.0 —
                   GELT's own is the printed reference). Diagnostic only; the
                   gate proper is on C(0), below.

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
# Upward, not downward. The first version of this grid was centred below 1 on
# the assumption that the risk was the Z₂ one — an exploding field. At nc = 2
# with their fan-in the stack *decays* instead, and the failure is the opposite
# one: C(0) collapses onto the Rayleigh loss's floor and every batch comes back
# non-finite (measured 2026-09-21, C(0) = 1.2e-10 at init_w = 1.0).
INIT_WS = [float(x) for x in _env("GIG_INIT_WS", "1.0,2.0,4.0,8.0").split(",")]
CONV_INITS = [float(x) for x in _env("GIG_CONV_INITS", "1.0").split(",")]
# lcnn_ref only: the paper has no activation layer and a single linear head.
ACTS = [x.strip() == "1" for x in _env("GIG_ACTS", "0").split(",")]
HEADS = [h.strip() for h in _env("GIG_HEADS", "mlp").split(",")]
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
        # The activation and the head are scanned too, because both were
        # measured to decide whether the stack has a usable scale at all
        # (notes/lcnn_reference_switch.md §7-§8) and the paper's settings are
        # act=off, head=linear.
        return [
            (f"init_w {w} act {a} head {h}",
             {"GLUEBALL_LCNN_REF_INIT_W": w,
              "GLUEBALL_LCNN_REF_ACT": int(a),
              "GLUEBALL_LCNN_REF_HEAD": h})
            for w in INIT_WS for a in ACTS for h in HEADS
        ]
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
            label = f"{arch}" + ("" if scale is None else f"  {scale}")
            print(f"\n── {label}   levels {list(mod.INPUT_SMEAR_LEVELS)}   "
                  f"{mod._geometry_label()}")
            configs = batch(mod, N_CONFIGS).to(dev)
            bad = False
            for seed in range(SEEDS):
                torch.manual_seed(seed)
                model = mod._build_model().to(dev)
                n_real = sum(
                    p.numel() * (2 if p.is_complex() else 1)
                    for p in model.parameters()
                )
                # The field walk, on one configuration — a per-layer diagnostic.
                W, T = mod.config_inputs(configs[:1], dev)
                prof = WALK[arch](model, W, T)
                del W, T
                finite = all(v == v and v != float("inf") for v in prof)

                # The gate: Ō over the WHOLE batch at once, because C(0) is a
                # variance across configurations and one configuration cannot
                # have one. train_glueball's own rayleigh_loss, so this cannot
                # drift from what the run divides by.
                with torch.no_grad():
                    obar = mod.network_obar(model, configs, dev)
                    loss, C0, R = mod.rayleigh_loss(obar)
                C0 = C0.item()
                mean = obar.double().mean().item()
                rel = C0 / mean**2 if mean != 0 else float("inf")
                ok = (
                    finite and C0 == C0 and C0 > 10 * mod.EPS
                    and max(prof) < GATE_MAX
                )
                bad = bad or not ok
                rows.append(dict(arch=arch, scale=scale, seed=seed,
                                 profile=prof, C0=C0, mean_obar=mean,
                                 rel_fluctuation=rel, loss=loss.item(),
                                 finite=finite, real_dofs=n_real, ok=ok))
                print(f"   seed {seed}  field by layer: "
                      + "  ".join(f"{v:.3g}" for v in prof)
                      + f"\n            C(0) {C0:.3g}   ⟨Ō⟩ {mean:.3g}   "
                      + f"C(0)/⟨Ō⟩² {rel:.3g}   loss {loss.item():.4g}   "
                      + ("ok" if ok else "** FAIL **"))
                if C0 <= 10 * mod.EPS:
                    print(f"            ↳ C(0) is at or below the loss floor "
                          f"({10 * mod.EPS:.1e}): every batch will be "
                          f"non-finite. "
                          + ("The operator is merely small — raise "
                             "--lcnn-init-scale (C(0) ∝ its square)."
                             if rel > 1e-6 else
                             "The operator is genuinely constant — the stack's "
                             "own scale must move, not the head's."))
            if bad:
                failed.append(label)

    os.makedirs("results/glueball", exist_ok=True)
    out = "results/glueball/init_gate.pt"
    torch.save({"rows": rows, "archs": ARCHS, "init_ws": INIT_WS,
                "conv_inits": CONV_INITS, "seeds": SEEDS,
                "n_configs": N_CONFIGS, "gate_max": GATE_MAX}, out)
    print(f"\nwrote {out}")

    if failed:
        print(f"\nVERDICT: {failed} fail the gate. Pick a surviving scale "
              f"before training — a run that starts here does not fail at "
              f"construction, it fails on the first epoch, having spent the "
              f"ensemble load and the smearing ladder first.")
        return 1
    print("\nVERDICT: every scanned setting holds. The sweep may run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
