"""S4 — is the learned operator in A₁⁺⁺? Forward passes under all 48 elements of O_h.

Supervisor's S4 (``notes/prof_notes.md``). Gauge invariance and zero momentum do
not make Ō_GELT(t) a 0⁺⁺ operator: it must also be a scalar under the spatial
cubic group with parity. Nothing in the network enforces that — ChannelLift
mixes the three plane channels with free weights, RoPE rotates per axis with
learned frequencies — so its A₁⁺⁺ content is unmeasured.

Method. For each g ∈ O_h (``gelt.cubic``: the 48 signed permutations of the
spatial axes; time untouched), transform every test configuration, rebuild the
whole input pipeline from the transformed links (``train_glueball.config_inputs``:
APE ladder, plaquettes, transport — so the smearing is checked to commute with
g too) and record Ō_g(t) = Ō(gU). The ensemble measure is O_h-invariant, so
``(1/48) Σ_g Ō_g`` is the A₁⁺⁺ projection, and with the character table every
other irrep's part comes for free (``gelt.cubic.irrep_projections``).

Gates, in the order they run; the first two abort the run before the hours are
spent, the third aborts at the element that fails:

  anchor     g = identity must reproduce the source dump: the network's Ō
             (every net in the group) and the classical Ō at every APE level the
             dump's basis has, to ``ANCHOR_TOL`` of the operator's sd. A
             mismatch means the checkpoint, the ensemble or the split is not the
             one the dump was made from.
  action     each g preserves the anisotropic Wilson action per configuration
             (complex128, ``ACTION_TOL`` relative).
  classical  the classical 0⁺⁺ operator — a scalar by construction — is
             invariant under every g at every input level, to ``ANCHOR_TOL`` of
             its sd (float32 pipeline). This is "check the action first": a map
             that is not a symmetry fails here, not in the physics.

Outputs, under ``OUT_DIR`` (per net, once all 48 elements are done):

  ``cubic_<stem>.pt``            Ō_g for every g (48, B, Lt), the element
                                 labels, the gate values, the per-g
                                 non-invariance, the irrep fractions of C(0).
  ``<stem>_a1_test_obars.pt``    the A₁⁺⁺-projected operator in the standard
                                 dump format (``gelt_obar`` = Ō_A1g, the source
                                 dump's ``Obar_basis``, its meta plus
                                 ``cubic_projection``). The ``_ens<k>`` tag is
                                 kept, so ``fit_window_scan.py``,
                                 ``su2_fair_fight.py`` and
                                 ``operator_decomposition.py`` read it unchanged.

``--elements=generators`` (identity, one axis swap, one reflection) runs the
gates and prints ν(g) on the real ensemble in minutes, and writes only
``cubic_<stem>_gen.pt``.

Progress is saved after every element (``…partial.pt``) and a restart resumes
from it, so a lost job costs one element, not the run. Dumps sharing an
ensemble and an input pipeline are evaluated together: the inputs are built
once per (batch, g) and fed to every net.

Printed readings, orientation only — S4's readings (the ΔA₀ comparison) are
read offline on the ``_a1`` dumps with the existing scripts:
  non-invariance  ν(g) = ‖δŌ_g − δŌ_e‖ / ‖δŌ_e‖, δ = VEV-subtracted;
  irrep fractions f_Γ, each irrep's share of C(0) on the O_h-augmented
                  sample (``gelt.cubic.irrep_fractions``: exactly additive,
                  Σ_Γ f_Γ = 1; the naive C_Γ(0)/C(0) are not on a finite
                  sample). 1 − f_A1g is "the fraction of C(0) outside A₁⁺⁺";
  m_eff(1)        of Ō_e and Ō_A1g, blocked jackknife on the test split.

Cost: 48 × one eval pass over the 400 test configurations per group. **V100
only** — refuses to run without CUDA unless ``--smoke=1``, which runs the
generators on one random configuration (Lt = 2) on any device to check the
plumbing, with no ensemble, no anchor and no output.

Run:
    python scripts/cubic_projection.py [dump.pt ...] [--elements=all|generators]
    python scripts/cubic_projection.py --smoke=1     # CPU, seconds
Driver: ``scripts/cubic_batch.sh``.
"""

import os
import sys
import time

# Refuse a flag this script does not read, before anything else sees argv (the
# probe's validate_argv discipline); then hide argv from train_glueball, whose
# import parses its own --name=value flags.
_KNOWN_FLAGS = ("--elements=", "--smoke=", "--out-dir=")
_bad = [a for a in sys.argv[1:] if a.startswith("--") and not a.startswith(_KNOWN_FLAGS)]
if _bad:
    raise SystemExit(f"unknown flag(s) {_bad}; known: {list(_KNOWN_FLAGS)}")


def _flag(name, default):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{name}=")),
                default)


ELEMENTS_MODE = _flag("elements", "all")
SMOKE = _flag("smoke", "0") not in ("0", "false", "False", "")
OUT_DIR = _flag("out-dir", "results/glueball/cubic")
_POSITIONAL = [a for a in sys.argv[1:] if not a.startswith("--")]
if ELEMENTS_MODE not in ("all", "generators"):
    raise SystemExit(f"--elements must be 'all' or 'generators' (got {ELEMENTS_MODE!r})")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ARGV = sys.argv
sys.argv = sys.argv[:1]
import train_glueball as tg  # noqa: E402  (path fix must precede the imports)

sys.argv = _ARGV

from gelt.cubic import (  # noqa: E402
    IRREPS,
    CubicElement,
    apply_cubic,
    cubic_elements,
    irrep_fractions,
    irrep_projections,
)
from gelt.glueball import connected_correlator  # noqa: E402
from gelt.lattice import action, random_links  # noqa: E402

# ── Tunables ──────────────────────────────────────────────────────────────────
DEFAULT_DUMPS = [
    "dumps/best_glueball_gelt_sm0-2-4-6_test_obars.pt",
    "dumps/best_glueball_gelt_sm0-2-4-6_ens1_test_obars.pt",
    "dumps/best_glueball_gelt_sm0-2-4-6_ens2_test_obars.pt",
]
ANCHOR_TOL = 1e-3  # max |ΔŌ| / sd(Ō): float32 pipeline, same batching as the dump
ACTION_TOL = 1e-10  # relative, complex128 — the transform is index ops and daggers
BATCH = tg.BATCH_CONFIGS  # the dump's own batching, so the anchor compares like with like
# The smoke test's elements: identity, one axis swap, one reflection.
GENERATORS = [CubicElement((1, 2, 3), ()), CubicElement((2, 1, 3), ()),
              CubicElement((1, 2, 3), (1,))]
group = tg.gaugegroup


def cache_path(seed):
    """``train_glueball.CACHE`` for another ensemble seed (drift-guarded below)."""
    return (
        f"datasets/glueball_configs_L{tg.L}_Lt{tg.LT}_b{tg.BETA}_xi{tg.XI}_N{tg.N_CONFIGS}"
        + ("" if seed == 0 else f"_seed{seed}")
        + ".pt"
    )


if cache_path(tg.ENSEMBLE_SEED) != tg.CACHE:
    raise SystemExit("cache_path() no longer matches train_glueball.CACHE — update it")


def _stem(dump_path):
    return os.path.basename(dump_path).replace("_test_obars.pt", "")


def _load_dump(path):
    """The dump, its meta normalised, and its architecture.

    The oldest dumps (Run 5, and ens1's 4-level net) predate two meta fields:
    Run 5 has no ``ensemble_seed`` and both name their checkpoint without the
    ``results/glueball/`` directory it was later moved to. The seed is read off
    the filename's ``_ens<k>`` tag — the pairing rule every offline script
    uses — and must agree with the meta wherever the meta has one.
    """
    d = torch.load(path, weights_only=False)
    meta = dict(d["meta"])
    arch = meta.get("arch", "gelt")
    if arch not in ("gelt", "lcnn"):
        raise SystemExit(f"{path}: arch {arch!r} not supported (gelt, lcnn)")
    b = os.path.basename(path)
    seed = int(b.split("_ens")[1].split("_")[0]) if "_ens" in b else 0
    if meta.get("ensemble_seed") is not None and meta["ensemble_seed"] != seed:
        raise SystemExit(f"{path}: meta ensemble_seed {meta['ensemble_seed']} "
                         f"disagrees with the filename's tag ({seed})")
    meta["ensemble_seed"] = seed
    ckpt = meta["checkpoint"]
    if not os.path.exists(ckpt) and os.path.dirname(ckpt) == "":
        ckpt = os.path.join("results/glueball", ckpt)
    # The L-CNN's p2 rerun keeps its (gitignored) checkpoint beside its dump in
    # dumps/p2/, not where its meta says; results/ is root-owned on the V100, so
    # beside the dump is the one place a copied checkpoint can go.
    beside = os.path.join(os.path.dirname(path), os.path.basename(ckpt))
    if not os.path.exists(ckpt) and os.path.exists(beside):
        ckpt = beside
    meta["checkpoint"] = ckpt
    return d, meta, arch


def _build(meta, arch, device):
    """The dump's network, from its checkpoint, through ``train_glueball._build_model``.

    Module-level settings are pointed at the dump's before the one constructor
    runs, the way su2_attention_correlator.py does it. The width is read off the
    checkpoint rather than the meta: the two 7-level nets are d_model 24 and
    neither their name nor their meta says so.
    """
    ckpt = meta["checkpoint"]
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint {ckpt} not found (named by the dump's meta)")
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    tg.ARCH = arch
    tg.INPUT_SMEAR_LEVELS = tuple(meta["input_smear_levels"])
    tg.GRAD_CHECKPOINT = False
    if arch == "gelt":
        tg.D_MODEL = sd["lift.weight"].shape[0]
    else:
        geo = meta.get("lcnn_geometry")
        if not geo:
            raise SystemExit(f"{ckpt}: an lcnn dump without lcnn_geometry")
        tg.LCNN_K, tg.LCNN_C_HIDDEN = geo["K"], geo["c_hidden"]
        tg.LCNN_LAYERS, tg.LCNN_INIT_SCALE = geo["n_layers"], geo["init_scale"]
    model = tg._build_model().to(device)
    # Checkpoints older than _OffsetGather carry no `_nbr_idx_inv`: a geometric
    # index buffer (x − Δx mod L) the model builds for itself. That key alone
    # may be missing; everything else loads strictly.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("._nbr_idx_inv")]
    if missing or unexpected:
        raise SystemExit(f"{ckpt}: state_dict mismatch — missing {missing}, "
                         f"unexpected {unexpected}")
    model.eval()
    return model


def _group_key(meta, arch):
    geo = meta.get("lcnn_geometry") or {}
    return (meta["ensemble_seed"], tuple(meta["input_smear_levels"]), arch,
            geo.get("K"))


def _test_split(seed):
    path = cache_path(seed)
    if not os.path.exists(path):
        raise SystemExit(f"ensemble cache {path} absent — this script samples nothing")
    configs = torch.load(path).to(tg.MODEL_DTYPE)
    N = configs.shape[0]
    n_train = int(round(tg.TRAIN_FRACTION * N))
    n_val = int(round(tg.VAL_FRACTION * N))
    return configs[n_train + n_val:]


def _classical_from_W(W, b, Lt, n_levels):
    """The classical 0⁺⁺ Ō at every input level, read off the network's own input.

    ``W`` is ``(b·Lt, 3·n_levels, L,L,L, nc,nc)``: per level, the three spatial
    plaquette planes. Σ_planes Σ_x Re Tr P / nc is ``smearing_operator_basis``'s
    operator at that level. Returns ``(n_levels, b, Lt)``.
    """
    tr = W.diagonal(dim1=-2, dim2=-1).sum(-1).real / tg.NC
    ob = tr.sum(dim=(2, 3, 4)).double()  # (b·Lt, 3·n_levels)
    return ob.view(b, Lt, n_levels, 3).sum(-1).permute(2, 0, 1).cpu()


@torch.no_grad()
def evaluate_element(g, configs, models, device, n_levels):
    """Ō(gU) for every net, the classical Ō(gU) per level, and the action check."""
    obars = [[] for _ in models]
    classical, worst_action = [], 0.0
    for i in range(0, configs.shape[0], BATCH):
        U = configs[i:i + BATCH].to(device)
        b, Lt = U.shape[0], U.shape[2]
        gU = apply_cubic(U, group, g, batch_dims=1)
        S0 = action(U.to(torch.complex128), group, beta=tg.BETA, xi=tg.XI)
        Sg = action(gU.to(torch.complex128), group, beta=tg.BETA, xi=tg.XI)
        worst_action = max(worst_action, ((Sg - S0).abs() / S0.abs()).max().item())
        W, T = tg.config_inputs(gU, device)
        for k, model in enumerate(models):
            obars[k].append(model(W, T).sum(dim=(1, 2, 3)).view(b, Lt).double().cpu())
        classical.append(_classical_from_W(W, b, Lt, n_levels))
        del W, T
    return ([torch.cat(o) for o in obars], torch.cat(classical, dim=1), worst_action)


def _rel_dev(x, ref):
    return ((x - ref).abs().max() / ref.std()).item()


def _noninvariance(og, oe):
    dg, de = og - og.mean(), oe - oe.mean()
    return ((dg - de).norm() / de.norm()).item()


def _c0(obar):
    return connected_correlator(obar)[0].item()


def summarise(stem, elements, obar_g, labels):
    """ν(g), the irrep fractions of C(0), m_eff(1) of Ō_e and Ō_A1g."""
    e = 0
    nu = {labels[i]: _noninvariance(obar_g[i], obar_g[e]) for i in range(len(labels))}
    worst = max(nu, key=nu.get)
    print(f"  {stem}: non-invariance ν(g) max {nu[worst]:.3e} ({worst}), "
          f"median {sorted(nu.values())[len(nu) // 2]:.3e}")
    out = {"noninvariance": nu}
    if len(elements) == 48:
        proj = irrep_projections(obar_g, elements)
        frac = irrep_fractions(obar_g, elements)
        out["irrep_fraction_C0"] = frac
        print("    share of C(0): " + "  ".join(f"{r} {frac[r]:.4f}" for r in IRREPS))
        print(f"    outside A1g: {1 - frac['A1g']:.4f}   "
              f"(C(0) of Ō_A1g / C(0) of Ō_e = {_c0(proj['A1g']) / _c0(obar_g[e]):.4f})")
        for name, ob in (("Ō_e", obar_g[e]), ("Ō_A1g", proj["A1g"])):
            if ob.shape[1] < 3:
                break
            m, s = tg.blocked_jackknife_meff(ob, tg.JACK_BLOCK)
            print(f"    m_eff(1) {name}: {m[1].item():.4f} ± {s[1].item():.4f}"
                  f"   m_eff(2): {m[2].item():.4f} ± {s[2].item():.4f}")
        out["A1g"] = proj["A1g"]
    return out


def smoke(device):
    """Generators on one random Lt = 2 configuration: plumbing, not physics."""
    dump = (_POSITIONAL or DEFAULT_DUMPS)[0]
    d, meta, arch = _load_dump(dump)
    model = _build(meta, arch, device)
    torch.manual_seed(0)
    U = random_links(tg.L, tg.D, group, dtype=tg.MODEL_DTYPE, Lt=2).unsqueeze(0)
    n_lv = len(meta["input_smear_levels"])
    res = []
    for g in GENERATORS:
        t0 = time.time()
        (ob,), cl, wa = evaluate_element(g, U, [model], device, n_lv)
        res.append((g, ob, cl))
        print(f"  {g.label:12s} action rel {wa:.1e}  "
              f"classical max|Δ| {(cl - res[0][2]).abs().max().item():.2e} "
              f"(|Ō_cl| ~ {res[0][2].abs().max().item():.1f})  "
              f"net Ō {ob.flatten().tolist()}  [{time.time() - t0:.1f} s]")
    print("smoke done: no ensemble, no anchor, nothing written.")


def run_group(dumps, device):
    loaded = [(p, *_load_dump(p)) for p in dumps]
    seed = loaded[0][2]["ensemble_seed"]
    levels = list(loaded[0][2]["input_smear_levels"])
    stems = [_stem(p) for p, *_ in loaded]
    finals = [os.path.join(OUT_DIR, f"cubic_{s}.pt") for s in stems]
    if ELEMENTS_MODE == "all" and all(os.path.exists(f) for f in finals):
        print(f"group ens seed {seed}: every output exists — skipped")
        return
    elements = cubic_elements() if ELEMENTS_MODE == "all" else GENERATORS
    labels = [g.label for g in elements]
    # The progress file. For the generators it is also the result (three
    # elements are not a projection, so there is nothing else to write).
    partial = os.path.join(OUT_DIR, f"cubic_{'+'.join(stems)}" + (
        ".partial.pt" if ELEMENTS_MODE == "all" else "_gen.pt"))

    print(f"── group: ensemble seed {seed}, input levels {levels}, "
          f"{len(loaded)} net(s): {', '.join(stems)}")
    models = [_build(meta, arch, device) for _, _, meta, arch in loaded]
    for (_, _, meta, _), m in zip(loaded, models):
        print(f"   {meta['checkpoint']}: d_model {getattr(m, 'd_model', '—')}")
    configs = _test_split(seed)
    for p, d, _, _ in loaded:
        if d["gelt_obar"].shape[0] != configs.shape[0]:
            raise SystemExit(f"{p}: {d['gelt_obar'].shape[0]} test configs in the dump, "
                             f"{configs.shape[0]} in the split")

    state = {"labels": [], "obar": [[] for _ in loaded], "classical_dev": [],
             "action_dev": [], "anchor": None}
    if os.path.exists(partial):
        state = torch.load(partial, weights_only=False)
        print(f"   resuming from {partial}: {len(state['labels'])} element(s) done")
        if state["labels"] != labels[:len(state["labels"])]:
            raise SystemExit(f"{partial} was written for another element list")
    classical_e = state.get("classical_e")

    n_lv = len(levels)
    basis_levels = loaded[0][1]["meta"]["gevp_levels"]
    for i, g in enumerate(elements):
        if i < len(state["labels"]):
            continue
        t0 = time.time()
        obs, cl, wa = evaluate_element(g, configs, models, device, n_lv)
        if wa > ACTION_TOL:
            raise SystemExit(f"{g.label}: action changes by {wa:.2e} (rel) — the map "
                             f"is not a symmetry; nothing downstream is meaningful")
        if i == 0:
            # Anchor: the identity must reproduce the dump it is paired with.
            anchor = {}
            for (p, d, _, _), ob in zip(loaded, obs):
                anchor[_stem(p)] = _rel_dev(ob, d["gelt_obar"])
            basis = loaded[0][1]["Obar_basis"]
            for j, lv in enumerate(basis_levels):
                if lv in levels:
                    anchor[f"classical_APE{lv}"] = _rel_dev(cl[levels.index(lv)], basis[j])
            print("   anchor (max|Δ|/sd against the dump): "
                  + "  ".join(f"{k} {v:.1e}" for k, v in anchor.items()))
            bad = {k: v for k, v in anchor.items() if not v < ANCHOR_TOL}
            if bad:
                raise SystemExit(f"anchor failed {bad}: the checkpoint / ensemble / split "
                                 f"is not the one the dump was made from")
            state["anchor"] = anchor
            classical_e = cl
            state["classical_e"] = cl
        cdev = max(_rel_dev(cl[k], classical_e[k]) for k in range(n_lv))
        if not cdev < ANCHOR_TOL:
            raise SystemExit(f"{g.label}: the classical operator moves by {cdev:.2e} sd — "
                             f"the transform or the smearing does not commute with g")
        state["labels"].append(g.label)
        for k, ob in enumerate(obs):
            state["obar"][k].append(ob)
        state["classical_dev"].append(cdev)
        state["action_dev"].append(wa)
        torch.save(state, partial)
        nus = "  ".join(f"ν {_noninvariance(ob, state['obar'][k][0]):.2e}"
                        for k, ob in enumerate(obs))
        print(f"   [{i + 1:2d}/{len(elements)}] {g.label:12s} action {wa:.0e}  "
              f"classical {cdev:.1e}  {nus}  [{time.time() - t0:.0f} s]", flush=True)

    for k, (p, d, meta, arch) in enumerate(loaded):
        obar_g = torch.stack(state["obar"][k])
        summ = summarise(stems[k], elements, obar_g, labels)
        if ELEMENTS_MODE != "all":
            continue
        a1 = summ.pop("A1g")
        torch.save({"labels": labels, "obar_g": obar_g, "anchor": state["anchor"],
                    "classical_dev": state["classical_dev"],
                    "action_dev": state["action_dev"], "source_dump": p,
                    "meta": meta, **summ}, finals[k])
        a1_dump = os.path.join(OUT_DIR, f"{stems[k]}_a1_test_obars.pt")
        torch.save({"gelt_obar": a1, "Obar_basis": d["Obar_basis"],
                    "meta": {**meta, "cubic_projection": "A1g", "source_dump": p}},
                   a1_dump)
        print(f"   → {finals[k]}\n   → {a1_dump}")
    if ELEMENTS_MODE == "all":
        os.remove(partial)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  elements: {ELEMENTS_MODE}")
    if SMOKE:
        return smoke(device)
    if device.type != "cuda":
        raise SystemExit("no CUDA: this is 48 eval passes per group — run it on the "
                         "V100 (or --smoke=1 for the plumbing)")
    os.makedirs(OUT_DIR, exist_ok=True)
    dumps = _POSITIONAL or [p for p in DEFAULT_DUMPS if os.path.exists(p)]
    groups = {}
    for p in dumps:
        _, meta, arch = _load_dump(p)
        groups.setdefault(_group_key(meta, arch), []).append(p)
    for key, members in groups.items():
        run_group(members, device)


if __name__ == "__main__":
    main()
