"""Is the classical comparator a straw man? — the fair-fight audit, SU(2)

The Z₂ companion (``scripts/z2_fair_fight.py``) audits the attention-field
comparison of Table 5 against exact dual ground truth. This script audits the
claim the thesis actually rests on, on the group it is actually about:

    §6.2 / Table `tab:overlap` — the trained GELT operator carries more
    ground-state weight than the optimal combination of the classical
    Morningstar–Peardon basis: A₀ = 0.903 ± 0.047 against 0.837 ± 0.056, and
    ΔA₀ = +0.078 ± 0.022 (3.6σ) combining two independently sampled ensembles.

That classical arm is ``GEVP_LEVELS = [0, 2, 4, 6]`` cumulative APE levels of the
**1×1 spatial plaquette** — four smearing levels of a single loop shape. A real
Morningstar–Peardon basis carries many shapes. The question here is whether the
+0.078 survives a classical basis built as strongly as the theory allows.

The study is deliberately self-contained. It imports; it modifies nothing. To
delete it: remove this file and ``results/fair_fight/``.

-------------------------------------------------------------------------------
WHY THIS IS A DIFFERENT AUDIT FROM THE Z₂ ONE
-------------------------------------------------------------------------------
The Z₂ pre-flight found two defects in *that* comparator, and both are
consequences of Z₂'s **discrete** projection, so neither should transfer:

1. the projected APE ladder **froze after one step** — ``Z2.project`` is
   ``sign()``, which makes smearing a majority-vote automaton with a fixed
   point, so ``[0, 4, 8, 16]`` held two distinct operators, not four;
2. at exactly α = 1/2 the update hits ``V = 0`` on ties and ``project`` sends
   0 → +1, a value that does not transform — so the smearing was **not gauge
   covariant** (0.19% of links, essentially all of the ones it changed).

``SU.project`` is the SVD polar factor: continuous, no ties, no fixed point. The
prediction is therefore that neither defect exists here and the SU(2) ladder
builds radius normally. That is *checked*, not assumed — self-checks 2 and 3
below are exactly the Z₂ diagnostics re-run on this group, and confirming them
negative is what licenses saying the Z₂ vulnerability is Z₂-specific.

There is a second asymmetry, and it cuts the other way. In Z₂ the correlation
length runs to ξ ≈ 6 spatial spacings while the frozen basis was stuck at
radius ≈ 1, so the comparator was under-powered *by a factor of six*. Here
``beta_scan.pt`` puts ξ_s ≈ 1.0–1.1 spatial spacings: the 0⁺⁺ is about **one
spacing across**, APE levels 0–6 already give radius √(nα) ≈ 1.7 a_s, and a
larger loop would be bigger than the state and exponentially noisier (area law).
So the honest expectation is that the SU(2) basis is close to optimal already.
If it is, the §6.2 claim stands and the Z₂ result is explained rather than
contradicted.

-------------------------------------------------------------------------------
WHAT IS AND IS NOT COVERED
-------------------------------------------------------------------------------
Covered: the **spectroscopy** claim above — fully offline, no network forward
pass, no GPU training, because ``train_glueball.py`` already dumped the trained
operator's test-split time series and the classical basis beside it.

Not covered: the **attention-field** row (Table `tab:su2attn`,
``su2_attention_correlator.py``), whose classical arm reads A₀ = 0.88 against
the trained field's 0.88 — already level, so strengthening it could only move
that comparison further, and it needs a GPU pass over that script's own seed-11
ensemble. Also not covered: SU(2) has no dual, so there is **no exact ground
truth here** and no accuracy table. The verdict is about operator quality (A₀),
which is the only statistic available and the one §6.2 quotes.

-------------------------------------------------------------------------------
THE ARMS
-------------------------------------------------------------------------------
All built on the same 400 test configurations, projected by the same
``gevp_ground_vector``, fitted on the same window with the same fixed σ_Δ, and
jackknifed on the same blocks — the protocol is *imported* from
``fit_glueball_overlap.py``, so "same conventions as Table 4" is a fact about
the call graph.

  gelt        the trained operator, straight from the dump
  published   APE [0,2,4,6] × 1×1 — the paper's arm, straight from the dump
  deep        APE [0,2,4,6,8,12,16] × 1×1 — does more radius help?
  shapes      thin links × cubic-symmetrised R×T loops — extent without smearing
  full        deep ladder × shapes — the strongest classical opponent

-------------------------------------------------------------------------------
PRE-REGISTERED OUTCOMES
-------------------------------------------------------------------------------
* **ΔA₀ against `full` stays positive at ≳ 3σ combined** → the §6.2 headline
  survives its strongest opponent; the Z₂ vulnerability was a property of the
  discrete group's frozen smearing, and this is the sentence to write.
* **ΔA₀ collapses toward zero** → the published +0.078 was in part an
  under-powered comparator, on SU(2) too, and Table 4 must be re-stated.
* **`full` beats `gelt`** → the learned operator is not variationally better
  than a well-built classical basis, only better than the one it was compared
  to. Report it plainly; it would not touch the attention-as-observable results,
  which rest on different measurements.
* **The rebuilt `published` arm does not reproduce the dump's `Obar_basis`** →
  wrong configurations, i.e. a bug in this script. The slice check below is the
  gate on reading anything else.

Env knobs:
  SFF_DUMPS=<a.pt,b.pt>   which test-Ō dumps to audit (default: both ensembles)
  SFF_ARMS=published,full SFF_CHUNK=8 SFF_MAXLEVEL=16
  SFF_NOCACHE=1           dump-only mode: reproduce the published comparison
                          without the ensemble (no new arms)
  SFF_REPLOT=<dump.pt>    re-report and re-plot offline

Run:  python scripts/su2_fair_fight.py
"""

import math
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# train_glueball parses argv at import; hide ours so a stray argument cannot
# reconfigure the lattice under us.
_ARGV = sys.argv
sys.argv = sys.argv[:1]
import train_glueball as tg  # noqa: E402
# The estimator: fixed σ_Δ from the full-sample jackknife, v₀ recomputed inside
# every replica, delete-block jackknife of the packed differences. Imported so
# the strengthened arms are measured identically to the number they challenge.
import fit_glueball_overlap as fgo  # noqa: E402

sys.argv = _ARGV

from gelt.glueball import (  # noqa: E402
    ape_smear,
    connected_correlator,
    connected_correlator_matrix,
    fit_cosh_correlator,
    gevp_ground_vector,
    glueball_operator,
    zero_momentum,
)
from gelt.lattice import link_gauge_transformation  # noqa: E402
from gelt.sampler import staple_sum  # noqa: E402

os.makedirs("results/fair_fight", exist_ok=True)

g = tg.gaugegroup
# SU(2) needs complex linalg.det / svd, which MPS does not implement (the same
# restriction su2_attention_correlator.py records). cuda or cpu only.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Tunables ──────────────────────────────────────────────────────────────────
def _cache_for(dump):
    """The ensemble a test-Ō dump was measured on, from its filename tag."""
    seed = "_seed1" if "_ens1" in os.path.basename(dump) else ""
    return (f"datasets/glueball_configs_L{tg.L}_Lt{tg.LT}_b{tg.BETA}_xi{tg.XI}"
            f"_N{tg.N_CONFIGS}{seed}.pt")


def _tag(dump):
    """Short ensemble name: the Run-5 anchor, or a replication seed."""
    b = os.path.basename(dump)
    return f"ens{b.split('_ens')[1].split('_')[0]}" if "_ens" in b else "run5"


# Both independently sampled ensembles, so the combined 3.6σ of the paper can be
# re-derived rather than argued about. Each dump pairs with its own cache.
_DEFAULT_DUMPS = [
    ("results/glueball/best_glueball_gelt_sm0-2-4-6_test_obars.pt",
     "datasets/glueball_configs_L12_Lt24_b2.4_xi3.0_N2000.pt"),
    ("results/glueball/best_glueball_gelt_sm0-2-4-6_ens1_test_obars.pt",
     "datasets/glueball_configs_L12_Lt24_b2.4_xi3.0_N2000_seed1.pt"),
]


def _resolve(path):
    """Where the dump actually is.

    ``results/`` is gitignored, and on the V100 it is root-owned (the jobs run in
    a container), so a dump cannot be pulled or copied into it as the user. The
    tracked copies therefore live in ``dumps/`` — look there before giving up,
    so neither an env var nor a writable ``results/`` is needed to find them.
    """
    if os.path.exists(path):
        return path
    alt = os.path.join("dumps", os.path.basename(path))
    return alt if os.path.exists(alt) else path


# Paths may also be given as positional arguments — env vars do not survive
# every container wrapper, and a path that must be typed is one that can be seen.
_cli = [a for a in _ARGV[1:] if a.endswith(".pt")]
if _cli:
    DUMPS = [(_resolve(a), _cache_for(a)) for a in _cli]
elif "SFF_DUMPS" in os.environ:
    DUMPS = [(_resolve(p.strip()), _cache_for(p.strip()))
             for p in os.environ["SFF_DUMPS"].split(",")]
else:
    DUMPS = [(_resolve(d), c) for d, c in _DEFAULT_DUMPS]

CHUNK = int(os.environ.get("SFF_CHUNK", 8))
MAX_LEVEL = int(os.environ.get("SFF_MAXLEVEL", 16))
NOCACHE = os.environ.get("SFF_NOCACHE", "0") == "1"
REPLOT = os.environ.get("SFF_REPLOT", "")
KEEP_OBARS = os.environ.get("SFF_KEEP_OBARS", "1") == "1"
# Eigenvalue floor for the GEVP whitening, relative to the largest eigenvalue of
# C(t0) — the same value and for the same reason as
# `z2_attention_correlator.py`'s GEVP_EPS. `fit_glueball_overlap.project_ground`
# takes the library default 1e-12, which is twelve orders of magnitude, i.e. no
# regularisation: a small signal divided by a floored *noise* eigenvalue yields a
# huge generalized eigenvalue in a direction that is almost pure noise. Harmless
# for the published 4-operator arm, fatal for a 21-operator one at 400 configs.
GEVP_EPS = float(os.environ.get("SFF_GEVP_EPS", 1e-4))
GEVP_TD_ = int(os.environ.get("SFF_GEVP_TD", 2))  # only for the Rayleigh gate

OUT_PT = "results/fair_fight/su2_fair_fight.pt"
OUT_PNG = "results/fair_fight/su2_fair_fight.png"

# Cubic-symmetrised spatial loop shapes. `glueball_operator` already sums over
# the three spatial planes, which is the rotational scalar for R = T; a rectangle
# with R ≠ T additionally maps to T×R under a lattice rotation and has to be
# averaged with it. ξ_s ≈ 1 here, so the ladder does not need to go far — but it
# has to go far enough to show that it does not help, which is the point.
SHAPES_EXT = ((1, 1), (1, 2), (2, 2), (2, 3), (3, 3))
SHAPES_FULL = ((1, 1), (1, 2), (2, 2))

ARM_SPEC = {
    # name        levels                     shapes         from the dump?
    "published": ([0, 2, 4, 6],              ((1, 1),),     True),
    "deep":      ([0, 2, 4, 6, 8, 12, 16],   ((1, 1),),     False),
    "shapes":    ([0],                       SHAPES_EXT,    False),
    # `shapes` carries no smearing at all, so it tests thin unsmeared loops
    # rather than loop *geometry*: it was the arm `operator_decomposition.md` §6
    # predicted would be the only one able to move A₀, and it came last. This
    # one gives the same shapes the published arm's own radius, so geometry and
    # radius can be told apart — `deep` is radius alone, `shapes_sm` geometry
    # alone at matched radius, `full` both.
    "shapes_sm": ([0, 2, 4, 6],              SHAPES_EXT,    False),
    "full":      ([0, 2, 4, 6, 8, 12, 16],   SHAPES_FULL,   False),
}
# Named before the run so the choice cannot be made after seeing the A₀ column.
# SFF_STRONGEST re-points it; the verdict below additionally names whichever arm
# came out strongest in fact, since that is the opponent the claim has to beat.
STRONGEST = os.environ.get("SFF_STRONGEST", "full")
ARMS = os.environ["SFF_ARMS"].split(",") if "SFF_ARMS" in os.environ else list(ARM_SPEC)

# Okabe–Ito, ordered so no adjacent pair falls in the 6–8 ΔE band under
# deuteranopia; every series also carries a distinct marker, so identity is
# never colour-alone.
COLOR = {"gelt": "#0072B2", "published": "#D55E00", "full": "#009E73",
         "deep": "#E69F00", "shapes": "#CC79A7"}
MARKER = {"gelt": "o", "published": "s", "full": "^", "deep": "D", "shapes": "v"}


# ── Smearing and loop shapes ──────────────────────────────────────────────────
def smear_steps(U, alpha, n_steps):
    """``n_steps`` of spatial APE smearing on a whole batch at once.

    ``gelt.glueball.ape_smear`` loops over configurations in Python and calls a
    batched SVD projection per config per direction; at 400 configs × 16 levels
    that is ~19k dispatches. Spatial-only smearing rolls **only** the spatial
    axes (``staple_sum`` is called with ``nu_dirs`` = the spatial directions), so
    axis 0 — time — is never touched and the batch can be folded into it. The
    configurations then never see each other and the result is the library's, to
    SVD reproducibility; :func:`selftest` measures that residual rather than
    assuming it.
    """
    B, D, Lt = U.shape[0], U.shape[1], U.shape[2]
    dirs = list(range(1, D))  # time is axis 0
    n_staples = 2 * (len(dirs) - 1)
    flat = U.movedim(1, 0).reshape(D, B * Lt, *U.shape[3:])
    for _ in range(n_steps):
        new = flat.clone()
        for mu in dirs:
            staples = g.dagger(staple_sum(flat, mu, g, nu_dirs=dirs))
            V = (1 - alpha) * flat[mu] + (alpha / n_staples) * staples
            new[mu] = g.project(V)
        flat = new
    return flat.reshape(D, B, Lt, *U.shape[3:]).movedim(0, 1)


def loop_field(U, r, t):
    """Cubic-symmetrised spatial r×t Wilson loop, ``(B, *Λ)``."""
    O = glueball_operator(U, g, R=r, T=t)
    if r != t:
        O = 0.5 * (O + glueball_operator(U, g, R=t, T=r))
    return O


def build_basis(U, levels, shapes, alpha):
    """``(n_ops, B, Nt)`` zero-momentum basis + labels, incremental smearing."""
    table = build_table(U, levels, shapes, alpha)
    keys = [(lvl, s) for lvl in sorted(levels) for s in shapes]
    return torch.stack([table[k] for k in keys]), [_label(*k) for k in keys]


def _label(lvl, shape):
    return f"n{lvl}·{shape[0]}x{shape[1]}"


def build_table(U, levels, shapes, alpha):
    """Every (level, shape) operator from ONE incremental smearing pass.

    The arms overlap heavily — ``deep`` and ``full`` both climb the same ladder
    to level 16, and the SVD projection inside each step is the whole cost of
    this script. Smearing once and slicing per arm is therefore roughly a 2×
    saving over building each arm independently, and it also guarantees that two
    arms sharing a (level, shape) share the *same numbers* rather than two
    independently recomputed copies of them.
    """
    out = {}
    cur, done = U, 0
    for lvl in sorted(levels):
        if lvl > done:
            cur = smear_steps(cur, alpha, lvl - done)
            done = lvl
        for s in shapes:
            out[(lvl, s)] = zero_momentum(loop_field(cur, *s)).double().cpu()
    return out


# ── Self-checks: every one has an answer known in advance ─────────────────────
def selftest(U):
    """Nothing is read off this run until these pass.

    Checks 2 and 3 are the Z₂ diagnostics re-run on SU(2). Confirming them
    *negative* here is the evidence that the Z₂ defects were properties of the
    discrete projection rather than of APE smearing.
    """
    print("\nself-checks")
    ok = True

    # 1. the batched smear is the library's, up to SVD reproducibility. Not
    #    bit-exact by right: torch's batched SVD can differ between tensor
    #    shapes, and folding the batch changes the shape. O(1e-6) in complex64
    #    is reproducibility; O(1) would be a wrong fold.
    ref = ape_smear(U[:2], g, alpha=tg.SMEAR_ALPHA, n_steps=2)
    mine = smear_steps(U[:2], tg.SMEAR_ALPHA, 2)
    d = (ref - mine).abs().max().item()
    print(f"  batched smear vs gelt.glueball.ape_smear : {d:.2e}"
          f"   {'OK' if d < 1e-4 else 'FAIL'}")
    ok &= d < 1e-4

    # 2. does the ladder freeze, as it did in Z₂? A continuous projection has no
    #    fixed point, so the per-step change must decay smoothly and stay > 0.
    cur, moved = U[:4], []
    for _ in range(8):
        nxt = smear_steps(cur, tg.SMEAR_ALPHA, 1)
        moved.append((nxt - cur).abs().mean().item())
        cur = nxt
    print("  mean |ΔU| per smearing step: " + " ".join(f"{m:.4f}" for m in moved))
    frozen = any(m == 0.0 for m in moved)
    print(f"    → {'FROZEN — the Z₂ pathology transfers' if frozen else 'no fixed point (Z₂ froze after 1 step)'}")

    # 3. gauge covariance of the operator each arm feeds to the correlator.
    #    Z₂'s `published` arm reads 1.18 here; SU(2) has no tie case, so every
    #    arm must be zero to float precision.
    torch.manual_seed(11)
    sub = U[:4]
    # ``GaugeGroup.random`` has no device argument — it always builds on CPU.
    omega = g.random((len(sub),) + tuple(sub.shape[2:2 + tg.D]),
                     dtype=sub.dtype).to(sub.device)
    Ug = torch.stack([link_gauge_transformation(sub[b], omega[b], g)
                      for b in range(len(sub))])
    print("  gauge covariance of Ō(t)   (max|ΔŌ| / std Ō — 0 for a real operator)")
    cov = {}
    for name in ARMS:
        levels, shapes, _from_dump = ARM_SPEC[name]
        lv = [l for l in levels if l <= MAX_LEVEL]
        a, _ = build_basis(sub, lv, shapes, tg.SMEAR_ALPHA)
        b, _ = build_basis(Ug, lv, shapes, tg.SMEAR_ALPHA)
        v = ((a - b).abs().max() / a.std().clamp_min(1e-30)).item()
        cov[name] = v
        print(f"    {name:<12} {v:9.2e}   {'OK' if v < 1e-4 else '** NOT GAUGE INVARIANT **'}")
    return ok, cov, moved


# ── Measurement ───────────────────────────────────────────────────────────────
def test_configs(cache):
    """The contiguous test split, exactly as train_glueball.py defines it.

    The split is contiguous and chain-ordered (not shuffled), so it reproduces
    from the fractions alone with no seed — which is what makes this audit
    possible offline.
    """
    if NOCACHE or not os.path.exists(cache):
        return None
    blob = torch.load(cache, map_location="cpu", weights_only=False)
    configs = blob["configs"] if isinstance(blob, dict) else blob
    N = configs.shape[0]
    n_train = int(round(tg.TRAIN_FRACTION * N))
    n_val = int(round(tg.VAL_FRACTION * N))
    return configs[n_train + n_val:]


def verify_slice(rebuilt, dumped):
    """Did we rebuild the SAME configurations the dump was measured on?

    The dump carries the classical basis it used. Rebuilding levels [0,2,4,6]
    from the cache must reproduce it elementwise; anything else means the split
    or the cache is wrong and every strengthened arm would be measuring a
    different ensemble than the number it is compared against. This is a far
    sharper gate than comparing fitted values, which can agree by luck.
    """
    scale = dumped.abs().mean().clamp_min(1e-30)
    rel = ((rebuilt - dumped).abs().max() / scale).item()
    ok = rel < 1e-3
    print(f"  slice check: max|rebuilt − dumped| / scale = {rel:.2e}   "
          + ("OK — same configurations" if ok
             else "** MISMATCH — wrong configs; stop here **"))
    return ok


def measure(dump_path, cache_path, cov_done):
    print(f"\n── {os.path.basename(dump_path)} " + "─" * 30)
    blob = torch.load(dump_path, map_location="cpu", weights_only=False)
    gelt = blob["gelt_obar"].double()
    dumped_basis = blob["Obar_basis"].double()
    meta = blob.get("meta", {})
    t0 = int(meta.get("gevp_t0", 1))
    td = t0 + 1 if fgo.GEVP_TD is None else fgo.GEVP_TD
    jb = int(meta.get("jack_block", 10))
    B, Nt = gelt.shape
    dmin, dmax = fgo.FIT_WINDOW
    print(f"  {B} test configs × Nt = {Nt} | window Δ ∈ [{dmin}, {dmax}] | "
          f"GEVP (t0, td) = ({t0}, {td}) | block {jb}")

    bases = {"published": dumped_basis}
    labels = {"published": [f"n{n}·1x1" for n in meta.get("gevp_levels", [])]}

    configs = test_configs(cache_path)
    if configs is None:
        print(f"  ensemble cache absent ({cache_path}) — dump-only mode: the "
              "published comparison is reproduced, no strengthened arms")
    else:
        print(f"  ensemble {tuple(configs.shape)} → test split {configs.shape[0]}")
        if cov_done is None:
            cov_done = selftest(configs[:8].to(device))[1]
        # One smearing pass covering the union of every arm's requirements; the
        # arms are then slices of it.
        # `published` is always in the union whatever SFF_ARMS/SFF_MAXLEVEL say:
        # rebuilding it is the slice check, and the slice check is the gate.
        levels = sorted({l for n in ARMS for l in ARM_SPEC[n][0] if l <= MAX_LEVEL}
                        | set(ARM_SPEC["published"][0]))
        shapes = sorted({s for n in ARMS for s in ARM_SPEC[n][1]}
                        | set(ARM_SPEC["published"][1]))
        t = time.time()
        table = _chunked_table(configs, levels, shapes)
        print(f"  smearing table: {len(levels)} levels × {len(shapes)} shapes "
              f"= {len(table)} operators  ({time.time() - t:.0f}s)")

        # The gate: rebuild the published arm and match the dump elementwise.
        reb, _ = _select(table, ARM_SPEC["published"][0], ((1, 1),))
        if not verify_slice(reb, dumped_basis):
            return None
        for name in ARMS:
            if name == "published":
                continue
            lv = [l for l in ARM_SPEC[name][0] if l <= MAX_LEVEL]
            bases[name], labels[name] = _select(table, lv, ARM_SPEC[name][1])
            print(f"  {name:<10} {len(labels[name]):>2} operators")

    names = [n for n in ARM_SPEC if n in bases]

    # σ_Δ fixed from the full sample, per operator, exactly as fit_glueball_
    # overlap.py does it: every replica then minimises the same χ² surface.
    proj_full, proj_info = {}, {}
    for n in names:
        proj_full[n], proj_info[n] = _project(bases[n], t0, td)
    print(f"\n  {'arm':<12} {'n_ops':>6} {'cond C(t0)':>12}   variational gate")
    for n in names:
        i = proj_info[n]
        note = (f"FELL BACK to member {i['best']} — the GEVP picked a near-null "
                "direction" if i["fell_back"] else "ok")
        print(f"  {n:<12} {i['n_ops']:>6} {i['cond']:>12.2e}   {note}")
    sig = {"gelt": fgo.blocked_jackknife(
        lambda m: connected_correlator(gelt[m]), B, jb)[1]}
    for n in names:
        sig[n] = fgo.blocked_jackknife(
            lambda m, s=proj_full[n]: connected_correlator(s[m]), B, jb)[1]

    def fit_one(C, s):
        m, A, chi2 = fit_cosh_correlator(C, dmin, dmax, sigma=s, m_range=fgo.M_RANGE)
        return m, A * (1.0 + math.exp(-m * Nt)) / C[0].item(), chi2

    def stats(mask):
        v = {"gelt": fit_one(connected_correlator(gelt[mask]), sig["gelt"])[:2]}
        for n in names:
            proj, _ = _project(bases[n][:, mask], t0, td)
            v[n] = fit_one(connected_correlator(proj), sig[n])[:2]
        row = []
        for n in ["gelt"] + names:
            row += list(v[n])
        for n in names:                       # correlated GELT − arm differences
            row += [v["gelt"][0] - v[n][0], v["gelt"][1] - v[n][1]]
        return torch.tensor(row, dtype=torch.float64)

    mean, err = fgo.blocked_jackknife(stats, B, jb)
    order = ["gelt"] + names
    out = {"dump": dump_path, "n_cfg": B, "labels": labels, "arms": {}, "delta": {}}
    for i, n in enumerate(order):
        C = connected_correlator(gelt if n == "gelt" else proj_full[n])
        out["arms"][n] = {"m": mean[2 * i].item(), "m_err": err[2 * i].item(),
                          "A0": mean[2 * i + 1].item(), "A0_err": err[2 * i + 1].item(),
                          "profile": (C / C[0]).tolist()[:12]}
    base = 2 * len(order)
    for k, n in enumerate(names):
        out["delta"][n] = {"dm": mean[base + 2 * k].item(),
                           "dm_err": err[base + 2 * k].item(),
                           "dA0": mean[base + 2 * k + 1].item(),
                           "dA0_err": err[base + 2 * k + 1].item()}

    # The trained series and every classical arm's, on the same configurations.
    # Re-deriving the strengthened arms needs the ensemble and a GPU pass, and
    # `operator_decomposition.py` needs precisely this pairing to run against
    # `full` rather than the published basis (audit 2026-09-06 §4 item 3).
    if KEEP_OBARS:
        ob_path = ("results/fair_fight/su2_fair_fight_obars_"
                   f"{_tag(dump_path)}.pt")
        torch.save({"dump": dump_path, "n_cfg": B, "labels": labels,
                    "t0": t0, "td": td,
                    "gelt": gelt.to(torch.float32),
                    "bases": {n: bases[n].to(torch.float32) for n in names}},
                   ob_path)
        print(f"  kept Ō series → {ob_path}")

    print(f"\n  {'operator':<12} {'m·a_t':>18} {'A₀':>18}   ΔA₀ (GELT − arm)")
    for n in order:
        a = out["arms"][n]
        d = out["delta"].get(n)
        dd = (f"   {d['dA0']:+.4f} ± {d['dA0_err']:.4f} "
              f"({abs(d['dA0']) / max(d['dA0_err'], 1e-12):.1f}σ)") if d else ""
        print(f"  {n:<12} {a['m']:>8.4f} ± {a['m_err']:.4f} "
              f"{a['A0']:>8.4f} ± {a['A0_err']:.4f}{dd}")
    return out, cov_done


def _rayleigh(series):
    """C(td)/C(0) of a scalar series — what the variational projection maximises.

    Also what exposes a projection that has instead maximised noise: it is scale
    invariant, so a near-cancelling combination with an almost-zero C(0) shows up
    here and nowhere in the mass.
    """
    C = connected_correlator(series)
    return float(C[GEVP_TD_] / C[0]) if C[0] > 0 else float("-inf")


def _project(basis, t0, td, eps=GEVP_EPS):
    """v₀-projected operator, with the two guards `project_ground` does not have.

    1. a real eigenvalue floor (see GEVP_EPS), and
    2. the variational gate from `z2_attention_correlator._gevp_is_sane`: v₀
       maximises the Rayleigh quotient over the *span* of the basis, so the
       projection can never be a worse interpolator than the best single member.
       When it is, the whitening has selected a near-null direction of an
       ill-conditioned C(t0) and the honest answer is that member.

    Returns ``(series, info)``; ``info`` carries the C(t0) condition number and
    whether the gate fired, both of which belong in the report.
    """
    if basis.shape[0] == 1:
        return basis[0], {"cond": 1.0, "fell_back": False, "n_ops": 1}
    C = connected_correlator_matrix(basis)
    Ct0 = 0.5 * (C[t0] + C[t0].transpose(-1, -2))
    ev = torch.linalg.eigvalsh(Ct0)
    cond = float(ev[-1] / ev[0]) if ev[0] > 0 else float("inf")
    v0 = gevp_ground_vector(C, t0=t0, td=td, eps=eps)
    proj = torch.einsum("i,ibt->bt", v0, basis)
    singles = [_rayleigh(basis[i]) for i in range(basis.shape[0])]
    best = int(np.argmax(singles))
    if _rayleigh(proj) < singles[best] - 1e-9:
        return basis[best], {"cond": cond, "fell_back": True,
                             "n_ops": basis.shape[0], "best": best}
    return proj, {"cond": cond, "fell_back": False, "n_ops": basis.shape[0]}


def _chunked_table(configs, levels, shapes):
    """:func:`build_table` over the test split in memory-sized chunks.

    Chunking is exact: smearing is per-configuration (the fold puts the batch on
    the never-rolled time axis) and the zero-momentum sum is per-configuration
    too, so no statistic crosses a chunk boundary.
    """
    parts = []
    for i in tqdm(range(0, len(configs), CHUNK), leave=False,
                  total=math.ceil(len(configs) / CHUNK), desc="    smearing"):
        b = configs[i:i + CHUNK].to(device)
        parts.append(build_table(b, levels, shapes, tg.SMEAR_ALPHA))
    return {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}


def _select(table, levels, shapes):
    """One arm's ``(n_ops, B, Nt)`` basis + labels, sliced out of the table."""
    keys = [(lvl, s) for lvl in sorted(levels) for s in shapes]
    return torch.stack([table[k] for k in keys]), [_label(*k) for k in keys]


# ── Reporting ─────────────────────────────────────────────────────────────────
def report(rows):
    print("\n" + "=" * 78)
    print("VERDICT — does §6.2's ΔA₀ survive a stronger classical basis?")
    print("=" * 78)
    names = [n for n in ARM_SPEC if any(n in r["arms"] for r in rows)]

    print(f"\n  A₀ per ensemble")
    print("  " + f"{'arm':<12}" + "".join(f"{_tag(r['dump']):>26}" for r in rows))
    for n in ["gelt"] + names:
        cells = []
        for r in rows:
            a = r["arms"].get(n)
            cells.append(f"{a['A0']:>17.3f} ± {a['A0_err']:.3f}" if a else f"{'—':>26}")
        print(f"  {n:<12}" + "".join(cells))

    print(f"\n  ΔA₀ = GELT − arm, correlated (same configs), and combined")
    for n in names:
        ds = [r["delta"][n] for r in rows if n in r.get("delta", {})]
        if not ds:
            continue
        per = "  ".join(f"{d['dA0']:+.3f}±{d['dA0_err']:.3f}" for d in ds)
        w = [1.0 / d["dA0_err"] ** 2 for d in ds if d["dA0_err"] > 0]
        if w:
            num = sum(d["dA0"] / d["dA0_err"] ** 2 for d in ds if d["dA0_err"] > 0)
            comb, cerr = num / sum(w), math.sqrt(1 / sum(w))
            print(f"  {n:<12} {per}   →  combined {comb:+.3f} ± {cerr:.3f} "
                  f"({abs(comb) / cerr:.1f}σ)")

    # Which arm is the real opponent is an A₀ question (SU(2) has no exact truth
    # column): the highest-A₀ classical arm is the one the claim must beat. If
    # that is not the pre-registered `full`, the verdict is taken against it and
    # both are named.
    def r0(key):
        return rows[0].get(key, {}) if rows else {}

    def _mean_A0(n):
        v = [r["arms"][n]["A0"] for r in rows if n in r.get("arms", {})]
        return sum(v) / len(v) if v else float("-inf")

    # The superset gate. `deep` and `full` contain every operator `published`
    # has, so the GEVP over them optimises over a strictly larger span and their
    # A₀ cannot be lower than `published`'s — that is linear algebra, not
    # physics. When it is lower, the estimator failed on the bigger basis and no
    # verdict may be read from this run in either direction.
    pub_lv, pub_sh, _ = ARM_SPEC["published"]
    broken = []
    for n in names:
        if n == "published" or "published" not in r0("arms"):
            continue
        lv, sh, _ = ARM_SPEC[n]
        if not (set(pub_lv) <= set(lv) and set(pub_sh) <= set(sh)):
            continue                       # not a superset — may legitimately lose
        a_n, a_p = _mean_A0(n), _mean_A0("published")
        e = max(max((r["arms"][n]["A0_err"] for r in rows if n in r["arms"]),
                    default=0.0), 1e-9)
        if a_n < a_p - 2 * e:
            broken.append((n, a_n, a_p))
    if broken:
        print("\n" + "=" * 78)
        print("  NO VERDICT — the strengthened arms did not build")
        print("=" * 78)
        for n, a_n, a_p in broken:
            print(f"  `{n}` is a strict superset of `published`, so its GEVP optimises")
            print(f"  over a larger span, yet A₀ = {a_n:.3f} against {a_p:.3f}. A superset")
            print("  cannot interpolate worse: this is the estimator failing on an")
            print("  ill-conditioned C(t0), not a weaker operator.")
        print("\n  Nothing here says whether §6.2 survives — the stronger opponent was")
        print("  never built. Check the condition numbers and the variational gate")
        print("  above; raise SFF_GEVP_EPS, or cut the basis size.")
        return

    # A run with no strengthened arm (SFF_NOCACHE, or every one skipped) has no
    # verdict to give: falling back to `published` would print "survives its
    # strongest opponent" about the very comparison under audit.
    challengers = [n for n in names if n != "published"]
    if not challengers:
        print("\n  → No strengthened arm ran (dump-only mode), so there is no verdict:")
        print("    the numbers above are the published comparison reproduced, nothing")
        print("    more. Run with the ensemble cache present to build the new arms.")
        return
    best = max(challengers, key=_mean_A0)
    against = best if _mean_A0(best) > _mean_A0(STRONGEST) else STRONGEST
    if against != STRONGEST:
        print(f"\n  `{against}` (A₀ = {_mean_A0(against):.3f}) came out above the "
              f"pre-registered `{STRONGEST}` (A₀ = {_mean_A0(STRONGEST):.3f});"
              f"\n  the verdict is taken against it.")
    STRONGEST_EFF = against
    strong = [r["delta"][STRONGEST_EFF] for r in rows if STRONGEST_EFF in r.get("delta", {})]
    if strong:
        w = sum(1.0 / d["dA0_err"] ** 2 for d in strong if d["dA0_err"] > 0)
        c = sum(d["dA0"] / d["dA0_err"] ** 2 for d in strong if d["dA0_err"] > 0) / w
        s = abs(c) * math.sqrt(w)
        print(f"\n  paper quotes ΔA₀ = +0.078 ± 0.022 (3.6σ) against `published`.")
        print(f"  against `{STRONGEST_EFF}` it is {c:+.3f} ± {math.sqrt(1 / w):.3f} ({s:.1f}σ).")
        if c > 0 and s >= 3.0:
            print("\n  → The claim survives its strongest opponent. The Z₂ vulnerability")
            print("    was a property of the discrete group's frozen smearing, and does")
            print("    not transfer to SU(2).")
        elif c > 0:
            print("\n  → The claim survives in sign but loses significance. Quote the")
            print("    number against the strengthened basis, not the published one.")
        else:
            print("\n  → A well-built classical basis matches or beats the learned")
            print("    operator. §6.2 must be re-stated: the +0.078 was measured")
            print("    against an under-powered comparator. This does NOT touch the")
            print("    attention-as-observable results, which rest on other measurements.")


def plot(rows, cov, moved):
    names = [n for n in ARM_SPEC if any(n in r["arms"] for r in rows)]
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    for a in ax.ravel():
        a.grid(alpha=0.25, lw=0.6)
        a.set_axisbelow(True)

    # (0,0) A₀ per arm, both ensembles, against the learned operator.
    a = ax[0, 0]
    w = 0.8 / max(len(rows), 1)
    for j, r in enumerate(rows):
        vals = [r["arms"][n]["A0"] if n in r["arms"] else np.nan for n in names]
        errs = [r["arms"][n]["A0_err"] if n in r["arms"] else np.nan for n in names]
        a.bar(np.arange(len(names)) + j * w - 0.4, vals, yerr=errs, width=w * 0.9,
              capsize=3, label=f"ensemble {j}",
              color=[COLOR.get(n, "#888") for n in names], alpha=0.85 if j == 0 else 0.55)
        gl = r["arms"]["gelt"]
        a.axhline(gl["A0"], color=COLOR["gelt"], ls="--" if j else "-", lw=1.5,
                  label=f"GELT (ens {j})")
    a.set_xticks(range(len(names)), names)
    a.set_ylabel(r"$A_0$ — ground-state overlap")
    a.set_title("Classical arms against the learned operator")
    a.legend(fontsize=8)

    # (0,1) the difference that is the claim.
    a = ax[0, 1]
    for j, r in enumerate(rows):
        d = [r["delta"][n]["dA0"] if n in r.get("delta", {}) else np.nan for n in names]
        e = [r["delta"][n]["dA0_err"] if n in r.get("delta", {}) else np.nan for n in names]
        a.errorbar(np.arange(len(names)) + 0.04 * j, d, yerr=e, fmt="o", capsize=4,
                   ms=8, label=f"ensemble {j}")
    a.axhline(0, color="k", lw=1)
    a.axhline(0.078, color="gray", ls=":", lw=1.5, label="published +0.078")
    a.set_xticks(range(len(names)), names)
    a.set_ylabel(r"$\Delta A_0$ = GELT $-$ arm")
    a.set_title("Correlated difference, same configurations")
    a.legend(fontsize=8)

    # (1,0) the correlator each arm actually has.
    a = ax[1, 0]
    r = rows[0]
    for n in ["gelt"] + names:
        p = r["arms"].get(n, {}).get("profile")
        if p:
            a.plot(range(len(p)), p, MARKER.get(n, "o") + "-", ms=5, lw=1,
                   color=COLOR.get(n), label=n, alpha=0.9)
    a.set_yscale("log")
    a.set_xlabel(r"$\Delta t$")
    a.set_ylabel(r"$C(\Delta)/C(0)$")
    a.set_title(f"Signal per arm ({os.path.basename(r['dump'])[-20:]})")
    a.legend(fontsize=8)

    # (1,1) the Z₂ diagnostic, re-run here: no fixed point.
    a = ax[1, 1]
    if moved:
        a.plot(range(1, len(moved) + 1), moved, "o-", color=COLOR["full"], ms=7,
               label=r"SU(2): mean $|\Delta U|$ per step")
        a.axhline(0, color="k", lw=1)
        a.legend(fontsize=8)
    else:
        a.text(0.5, 0.7, "no ensemble cache —\nfreeze test not run", ha="center",
               fontsize=10, color="#555", transform=a.transAxes)
    a.annotate("Z₂ froze after 1 step\n(mean |ΔU| → exactly 0)", xy=(0.45, 0.35),
               xycoords="axes fraction", fontsize=9, color="#D55E00")
    a.set_xlabel("smearing step")
    a.set_ylabel(r"mean $|\Delta U|$")
    a.set_title("Does the APE ladder freeze? (it did in $\\mathbb{Z}_2$)")

    fig.suptitle("Fair fight, SU(2): the §6.2 classical basis rebuilt as strongly "
                 "as the theory allows", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nsaved {OUT_PNG}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    if REPLOT:
        d = torch.load(REPLOT, map_location="cpu", weights_only=False)
        report(d["rows"])
        plot(d["rows"], d.get("covariance"), d.get("moved"))
        return

    print(f"device: {device} | SU(2) {tg.L}³×{tg.LT}, β = {tg.BETA}, ξ = {tg.XI} | "
          f"arms = {','.join(ARMS)} | strongest = {STRONGEST}")
    print(f"auditing: §6.2 ΔA₀ = +0.078 ± 0.022 (3.6σ) vs the [0,2,4,6] × 1×1 basis")

    rows, cov, moved = [], None, None
    for dump, cache in DUMPS:
        if not os.path.exists(dump):
            print(f"  missing dump {dump} — skipped")
            continue
        if cov is None and not NOCACHE and os.path.exists(cache):
            cfg = test_configs(cache)
            ok, cov, moved = selftest(cfg[:8].to(device))
            if not ok:
                raise SystemExit("self-checks failed — the batched smear is wrong")
            del cfg
        got = measure(dump, cache, cov)
        if got is None:
            raise SystemExit("slice check failed — wrong configurations")
        row, cov = got
        rows.append(row)
        torch.save({"rows": rows, "arm_spec": {k: list(v) for k, v in ARM_SPEC.items()},
                    "strongest": STRONGEST, "covariance": cov, "moved": moved,
                    "meta": {"L": tg.L, "Lt": tg.LT, "beta": tg.BETA, "xi": tg.XI,
                             "fit_window": fgo.FIT_WINDOW,
                             "smear_alpha": tg.SMEAR_ALPHA}}, OUT_PT)
        print(f"  saved → {OUT_PT} ({len(rows)} ensembles)")

    if not rows:
        raise SystemExit("no dump produced a measurement")
    report(rows)
    plot(rows, cov, moved)


if __name__ == "__main__":
    main()
