"""Is the classical comparator a straw man? — the fair-fight audit (3D Z₂)

`notes/attention_as_operator.md` §6.1 and the paper's Table 5 / Table 6 compare
the attention field against "the optimal combination of a four-level smeared
Morningstar–Peardon basis", and `notes/dual_ground_truth.md` §7.7 turns that into
an accuracy statement against exact truth: classical −8.9%, attention trained
+3.8%, attention random −8.0%.

Every one of those numbers is a comparison *against one particular classical
operator*. This script asks the question a referee asks first: **was that
operator as good as a practitioner would have made it?** It rebuilds the
classical arm as strongly as the theory allows, on the same configurations,
through the same estimator, and re-runs the accuracy table.

The study is deliberately self-contained. It imports; it modifies nothing. To
delete it: remove this file and `results/fair_fight/`.

-------------------------------------------------------------------------------
WHAT THE PRE-FLIGHT ALREADY FOUND (2026-08-17, before this script existed)
-------------------------------------------------------------------------------
Two facts about `ape_smear` on Z₂ at ``SMEAR_ALPHA = 0.5``, measured on the
production β = 0.7585 ensemble. Both are re-measured here rather than trusted,
because this script exists precisely to stop taking the comparator on faith.

1. **The ladder freezes after ONE step.** Fraction of links changed per step:
   0.0017, 0.0000, 0.0000, 0.0000, … So the published basis ``[0, 4, 8, 16]``
   holds **two distinct operators, one of them repeated three times**, and the
   network's ``INPUT_SMEAR_LEVELS = (0, 4, 8, 16)`` feeds four channels of which
   three are byte-identical.

   This is structural, not a tuning accident. With ``n_staples = 2`` the update
   is ``V = (1−α)U + (α/2)(s₁+s₂)`` with ``s_i = ±1``, so

       s₁+s₂ = 0   →  V = (1−α)U          → sign unchanged, always;
       s₁+s₂ = ±2  →  V = σ(2α−1)         → the staples win iff α > 1/2.

   For α < 1/2 smearing is the **identity**; for α > 1/2 it is a majority-vote
   cellular automaton, which reaches a fixed point in a couple of sweeps. There
   is no α at which projected Z₂ APE smearing has a tunable radius. It also
   explains a recorded puzzle: §6.2's "the fallback fires at every β — the
   four-level variational basis never beats its own best member". A basis of
   rank two has nothing for a GEVP to do.

2. **At exactly α = 1/2 the smearing is not gauge covariant.** That is the
   boundary case above: ``V = 0`` whenever both staples disagree with the link,
   and ``Z2.project`` sends 0 → +1 by its ``>= 0`` rule — a value that does not
   transform. Measured on production configs: 0.185% of links change per step
   and 0.186% are non-covariant, i.e. **the smearing consists almost entirely of
   gauge-dependent tie-breaks**. ⟨O⟩ is protected by averaging (drift 1.1e-4
   relative), but the timeslice operator that actually enters the correlator
   moves by ``max|ΔŌ|/std(Ō) ≈ 1.75`` — larger than its own fluctuation.
   `tests/test_glueball.py` misses this because its covariance cases use
   α = 0.7 and α = 0.6, never 0.5.

   Direction of the consequence: gauge noise added to an operator lowers its
   ground-state overlap and biases ξ **low** — which is the direction of the
   published classical deficit. That is the reason this audit is worth running
   rather than assuming.

-------------------------------------------------------------------------------
THE ARMS
-------------------------------------------------------------------------------
All measured on the same configurations, through the same estimator, and quoted
under both estimators (multi-operator GEVP and best single member) exactly as
§9.5.1 requires.

  published   projected APE α=0.5, levels [0,4,8,16], 1×1        the paper's arm,
                                                                 rebuilt here
  covariant   projected APE α=0.7, levels [0,1,2,4]              same idea, no
                                                                 tie-breaks
  fat         *un*projected linear smearing, levels [0,2,…,32]   a real radius
                                                                 ladder
  shapes      thin links, D₄-symmetrised R×T loops               extent without
                                                                 smearing
  full        fat ladder × {1×1, 1×2, 2×2}                       the strongest
                                                                 classical arm

Unprojected smearing is legitimate here and is the only scheme with a tunable
radius: it is linear in the links, hence exactly gauge covariant, and a Z₂ loop
built from real-valued links is still gauge invariant because ω = ±1 gives
ω² = 1 whatever the link magnitude (the daggers cancel the gauge factors without
needing to be inverses). It is the Z₂ analogue of Gaussian/fat-link smearing,
which is standard in spectroscopy. No normalisation is applied — the field
shrinks geometrically with depth, which float64 absorbs, and every statistic
quoted (m, A₀, C(Δ)/C(0)) is invariant under rescaling an operator anyway.

-------------------------------------------------------------------------------
PRE-REGISTERED OUTCOMES
-------------------------------------------------------------------------------
* **The strengthened classical arm stays low against the dual truth** → the
  published comparison survives its strongest opponent and the headline is
  stronger than it was, because the obvious objection has been answered rather
  than avoided.
* **The strengthened arm closes the gap** → the published −8.9% was in part an
  under-powered comparator. The honest result becomes "the network reaches, from
  a rank-two input ladder, what a classical basis needs a genuine radius ladder
  to reach", and Table 6 must be re-stated. Report the level at which it closes.
* **The strengthened arm overshoots the truth too** → both operators share a
  systematic and the fit window, not the operator, is the suspect.
* **The `published` arm does not reproduce the published numbers** → a bug in
  this script, not a result. The regression check below is the gate on reading
  anything else.

Env knobs (all optional):
  FF_BETAS=0.745,0.752,…   FF_N_EVAL=1200   FF_CHUNK=8   FF_ARMS=published,full
  FF_NETS=0                 skip the attention arms (classical audit only)
  FF_MAXLEVEL=32            deepest linear-smearing level
  FF_SMOKE=1                seconds-long plumbing run on random links
  FF_REPLOT=<dump.pt>       re-report and re-plot offline, no GPU

Run:  python scripts/z2_fair_fight.py
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

# The imported modules read env and argv at import time. Pin the knobs that must
# line up with the published R = 6 run *before* importing, and hide our own argv
# so a stray argument cannot reconfigure them. This is the only way to configure
# them without editing them, which this study is not allowed to do.
os.environ.setdefault("ZAC_R", "6")        # the retrained R = 6 operator set
os.environ.setdefault("ZAC_N_USE", "800")  # ⇒ EVAL_START = 800, the unseen slice
os.environ.setdefault("ZAC_N_EVAL", os.environ.get("FF_N_EVAL", "1200"))
os.environ.setdefault("ZAC_CROSS", "0")
if os.environ.get("FF_SMOKE", "0") == "1":
    os.environ["ZAC_SMOKE"] = "1"

_ARGV = sys.argv
sys.argv = sys.argv[:1]
import train_z2_glueball as tz  # noqa: E402
# The estimator layer, imported rather than copied: GEVP t0=1/td=2, the window
# fixed once on the full sample, the cosh fit, Morningstar–Peardon A₀, the
# blocked jackknife at block 20 and the correlated difference are then identical
# to Table 5 *by construction* rather than by claim.
import z2_attention_correlator as zac  # noqa: E402

sys.argv = _ARGV

from gelt.glueball import (  # noqa: E402
    ape_smear,
    glueball_operator,
    zero_momentum,
)
from gelt.lattice import link_gauge_transformation  # noqa: E402
from gelt.sampler import staple_sum  # noqa: E402

os.makedirs("results/fair_fight", exist_ok=True)

g = tz.gaugegroup
device = tz.device
SMOKE = os.environ.get("FF_SMOKE", "0") == "1"


# ── Tunables ──────────────────────────────────────────────────────────────────
BETAS = [float(b) for b in os.environ["FF_BETAS"].split(",")] if "FF_BETAS" in os.environ \
    else list(zac.BETAS)
CHUNK = int(os.environ.get("FF_CHUNK", 8))
RUN_NETS = os.environ.get("FF_NETS", "1") == "1"
MAX_LEVEL = int(os.environ.get("FF_MAXLEVEL", 32))
REPLOT = os.environ.get("FF_REPLOT", "")
KEEP_OBARS = os.environ.get("FF_KEEP_OBARS", "1") == "1"

# Exact ground truth from the dual Ising model, large volume (96×48²), τ(M)-gated
# rerun. Source: notes/dual_ground_truth.md §7.7 and paper Table `tab:dual`.
# NOT read from results/dual/dual_ground_truth.pt — that dump is currently a
# DGT_SMOKE run (volumes = (24,12,12), empty results list) and would silently
# supply meaningless numbers. `_load_truth` refuses it explicitly.
DUAL_XI = {
    0.7450: (2.114, 0.045),
    0.7520: (2.962, 0.050),
    0.7560: (4.251, 0.062),
    0.7585: (6.021, 0.078),
}

# The published dump, used only as a regression check on the `published` arm.
PUBLISHED = "results/attention/z2_attention_correlator_diag_R6.pt"

OUT_PT = "results/fair_fight/z2_fair_fight.pt"
OUT_PNG = "results/fair_fight/z2_fair_fight.png"

# D₄-symmetrised spatial loop shapes. The 3D lattice has a single spatial plane,
# so an R×T loop with R ≠ T is *not* a rotational scalar on its own — it maps to
# T×R under a 90° rotation and has to be averaged with it. (R, R) is already
# symmetric and must not be double counted.
#
# The ladder has to *reach* the correlation length or the arm is under-powered
# for exactly the reason this study exists: ξ runs from 2.1 to 6.0 across the
# scan, so the extent ladder runs to 6×6. Large thin loops are noisy (area law),
# which is what `_prune`'s signal ranking and `_jack_best_single` are for.
SHAPES_EXT = ((1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3), (4, 4), (6, 6))
SHAPES_FULL = ((1, 1), (2, 2), (3, 3), (4, 4))

ARM_SPEC = {
    # name        levels                 shapes        alpha  project
    "published": ([0, 4, 8, 16],         ((1, 1),),    0.5,   True),
    "covariant": ([0, 1, 2, 4],          ((1, 1),),    0.7,   True),
    "fat":       ([0, 2, 4, 8, 16, 32],  ((1, 1),),    0.5,   False),
    "shapes":    ([0],                   SHAPES_EXT,   0.5,   False),
    "full":      ([0, 2, 4, 8, 16, 32],  SHAPES_FULL,  0.5,   False),
}
# The arm the verdict is taken against: the strongest classical opponent, named
# here *before* the run so the choice cannot be made after seeing the truth
# column. FF_STRONGEST re-points it — which is selection on the outcome and only
# defensible in one direction: handing the classical side an arm that turned out
# better than the pre-registered one makes the opponent stronger, never weaker.
STRONGEST = os.environ.get("FF_STRONGEST", "full")
ARMS = os.environ["FF_ARMS"].split(",") if "FF_ARMS" in os.environ else list(ARM_SPEC)

if SMOKE:
    BETAS = BETAS[:2]
    CHUNK, MAX_LEVEL = 2, 4
    ARM_SPEC = {
        "published": ([0, 2], ((1, 1),), 0.5, True),
        "covariant": ([0, 1], ((1, 1),), 0.7, True),
        "fat":       ([0, 2], ((1, 1),), 0.5, False),
        "shapes":    ([0], ((1, 1), (1, 2)), 0.5, False),
        "full":      ([0, 2], ((1, 1), (1, 2)), 0.5, False),
    }
    ARMS = list(ARM_SPEC)
    OUT_PT = "results/fair_fight/z2_fair_fight_smoke.pt"
    OUT_PNG = "results/fair_fight/z2_fair_fight_smoke.png"

# Okabe–Ito, reordered so no adjacent pair falls in the 6–8 ΔE band under
# deuteranopia; every series also carries a distinct marker, so identity is
# never colour-alone.
COLOR = {
    "attention trained": "#0072B2",
    "published":         "#D55E00",
    "full":              "#009E73",
    "fat":               "#E69F00",
    "attention random":  "#CC79A7",
    "covariant":         "#56B4E9",
    "shapes":            "#7f7f7f",
}
MARKER = {
    "attention trained": "o", "published": "s", "full": "^", "fat": "D",
    "attention random": "v", "covariant": "X", "shapes": "P",
}


# ── Smearing ──────────────────────────────────────────────────────────────────
def smear_steps(U, alpha, n_steps, project=True):
    """``n_steps`` of APE smearing on a whole batch at once.

    ``gelt.glueball.ape_smear`` loops over configurations in Python — ~128 tiny
    kernel launches per batch to reach level 16, as `train_z2_glueball.prepare`
    already notes. Spatial-only smearing rolls **only** the spatial axes
    (``staple_sum`` is called with ``nu_dirs`` = the spatial directions), so
    axis 0 — time — is never touched and the batch can be folded into it. The
    configurations then simply never see each other and the result is bit-exact
    against the library, which :func:`selftest_smearing` asserts at 0.0e+00
    before anything is measured.

    ``project=False`` drops the projection back onto Z₂. That is not a shortcut:
    it is the only smearing scheme on this group with a tunable radius (see the
    module docstring), it is linear and therefore exactly gauge covariant, and
    the resulting real-valued links still give gauge-invariant loops.
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
            new[mu] = g.project(V) if project else V
        flat = new
    return flat.reshape(D, B, Lt, *U.shape[3:]).movedim(0, 1)


def loop_field(U, r, t):
    """D₄-symmetrised spatial r×t Wilson loop, ``(B, *Λ)``.

    ``glueball_operator`` sums over spatial planes, which is the rotational
    scalar in D ≥ 4; in D = 3 there is exactly one spatial plane, so a rectangle
    with r ≠ t needs its own transpose added by hand to be a D₄ scalar.
    """
    O = glueball_operator(U, g, R=r, T=t)
    if r != t:
        O = 0.5 * (O + glueball_operator(U, g, R=t, T=r))
    return O


def build_basis(U, levels, shapes, alpha, project):
    """``(n_ops, B, Nt)`` zero-momentum basis + labels, one row per (level, shape).

    Smearing is incremental, as in ``smearing_operator_basis``: level 8 reuses
    level 4's links rather than re-smearing from scratch.
    """
    rows, labels = [], []
    cur, done = U, 0
    for lvl in sorted(levels):
        if lvl > done:
            cur = smear_steps(cur, alpha, lvl - done, project=project)
            done = lvl
        for (r, t) in shapes:
            rows.append(zero_momentum(loop_field(cur, r, t)).double().cpu())
            labels.append(f"n{lvl}·{r}x{t}")
    return torch.stack(rows), labels


# ── Self-checks: every one has an answer known in advance ─────────────────────
def selftest_smearing(U):
    """Nothing is read off this run until these pass.

    The failure mode of the two earlier attention studies was a statistic that
    could not have come out any other way, so each check below is one whose
    answer is fixed in advance: bit-exactness against the library, an exact zero
    for a covariant scheme, and a freeze point that is either there or is not.
    """
    print("\nself-checks")
    ok = True

    # 1. the batched smear is the library's, exactly.
    ref = ape_smear(U[:2], g, alpha=0.5, n_steps=2)
    mine = smear_steps(U[:2], 0.5, 2, project=True)
    d = (ref - mine).abs().max().item()
    print(f"  batched smear vs gelt.glueball.ape_smear : {d:.1e}"
          f"   {'OK' if d == 0.0 else 'FAIL'}")
    ok &= d == 0.0

    # 2. where does the projected ladder stop moving? (the rank-2 claim)
    for alpha in (0.5, 0.7):
        cur, flips = U[:4], []
        for _ in range(6):
            nxt = smear_steps(cur, alpha, 1, project=True)
            flips.append((nxt != cur).double().mean().item())
            cur = nxt
        frozen = next((i + 1 for i, f in enumerate(flips) if f == 0.0), None)
        print(f"  projected α={alpha}: links flipped per step "
              + " ".join(f"{f:.4f}" for f in flips)
              + (f"   → frozen after {frozen} step(s)" if frozen else "   → still moving"))

    # 3. gauge covariance of the operator each scheme feeds to the correlator.
    #    A covariant scheme gives exactly 0; the α=0.5 tie-break does not.
    torch.manual_seed(11)
    sub = U[:4]
    # ``GaugeGroup.random`` has no device argument — it always builds on CPU.
    omega = g.random((len(sub),) + tuple(sub.shape[2:5]),
                     dtype=sub.dtype).to(sub.device)
    Ug = torch.stack([link_gauge_transformation(sub[b], omega[b], g)
                      for b in range(len(sub))])
    print("  gauge covariance of Ō(t)   (max|ΔŌ| / std Ō — 0 for a real operator)")
    cov = {}
    for name in ARMS:
        levels, shapes, alpha, project = ARM_SPEC[name]
        lv = [l for l in levels if l <= MAX_LEVEL]
        a, _ = build_basis(sub, lv, shapes, alpha, project)
        b, _ = build_basis(Ug, lv, shapes, alpha, project)
        scale = a.std().clamp_min(1e-30)
        v = ((a - b).abs().max() / scale).item()
        cov[name] = v
        flag = "OK" if v < 1e-6 else "** NOT GAUGE INVARIANT **"
        print(f"    {name:<12} {v:9.2e}   {flag}")
    return ok, cov


# ── Measurement ───────────────────────────────────────────────────────────────
def _load_truth():
    """Exact ξ per β, with an explicit refusal of a smoke dump.

    The saved ``results/dual/dual_ground_truth.pt`` is currently a ``DGT_SMOKE``
    run — 12³ boxes at ξ ≈ 5, an empty results list — so reading it would supply
    numbers that mean nothing while looking authoritative. If a production dump
    ever replaces it, this is where to wire it in; until then the table from the
    paper is the reference and the provenance is a comment, not a file.
    """
    p = "results/dual/dual_ground_truth.pt"
    if os.path.exists(p):
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
            vols = d.get("volumes", {})
            if any(min(v) < 24 for v in vols.values()) or not d.get("results", {}).get("large"):
                print(f"  note: {p} is a smoke/partial run {vols} — using the "
                      "published table instead")
        except Exception as exc:
            print(f"  note: could not read {p} ({exc}) — using the published table")
    return DUAL_XI


def _dev(row, beta):
    """(percent deviation, significance) of an arm against the dual truth."""
    if beta not in DUAL_XI:
        return float("nan"), float("nan")
    truth, terr = DUAL_XI[beta]
    xi, err = zac._xi(row)
    if not np.isfinite(xi):
        return float("nan"), float("nan")
    d = xi - truth
    # No error on the arm means no significance to quote. Substituting zero
    # would turn an unresolved jackknife into a spuriously sharp deviation.
    sig = (abs(d) / math.sqrt(err ** 2 + terr ** 2)) if np.isfinite(err) else float("nan")
    return 100.0 * d / truth, sig


def attention_bases(beta, configs):
    """The trained and random attention channel bases on these configurations.

    Reuses ``zac.attention_fields`` and ``tz.config_inputs`` unchanged, so these
    rows are the same objects Table 5 quotes — this arm doubles as a second
    regression check.
    """
    ck = tz.checkpoint_path(beta)
    if not SMOKE and not os.path.exists(ck):
        print(f"  no checkpoint {ck} — attention arms skipped at this β")
        return {}
    nets = {
        "attention trained": (zac.build_model(seed=int(beta * 1e4)) if SMOKE
                              else zac.build_model(ckpt=ck)),
        "attention random": zac.build_model(seed=zac.RANDOM_SEED),
    }
    offsets = nets["attention random"].gemhsa_models[0].offsets
    dist = torch.tensor([sum(abs(c) for c in o) for o in offsets],
                        dtype=tz.MODEL_DTYPE, device=device)
    labels = zac.channel_labels()

    acc = {k: [] for k in nets}
    n = len(configs)
    for i in tqdm(range(0, n, CHUNK), desc="    attention", leave=False,
                  total=math.ceil(n / CHUNK)):
        batch = configs[i:i + CHUNK]
        W, T = tz.config_inputs(batch)
        for name, model in nets.items():
            _out, chan, _mom = zac.attention_fields(model, W, T, len(batch), dist)
            acc[name].append(chan.double().cpu())
        del W, T
    return {k: (torch.cat(v, dim=1), labels) for k, v in acc.items()}


def measure(beta, truth):
    print(f"\n── β = {beta}   (exact ξ = "
          + (f"{truth[beta][0]:.3f} ± {truth[beta][1]:.3f})" if beta in truth else "unknown)")
          + " " + "─" * 30)
    configs = zac.load_configs(beta)
    if configs is None:
        return None
    print(f"  {len(configs)} unseen configs from index {zac.EVAL_START}")
    Nt = tz.LT
    row = {"beta": beta, "n_cfg": len(configs), "arms": {}, "labels": {}}

    bases = {}
    for name in ARMS:
        levels, shapes, alpha, project = ARM_SPEC[name]
        lv = [l for l in levels if l <= MAX_LEVEL]
        t0 = time.time()
        parts, labels = [], None
        for i in tqdm(range(0, len(configs), CHUNK), desc=f"    {name:<10}",
                      leave=False, total=math.ceil(len(configs) / CHUNK)):
            b = configs[i:i + CHUNK].to(device)
            # The projected arm is reproduced in the configs' own dtype, exactly
            # as the published path did it; the linear arm runs in float64
            # because its field shrinks geometrically with depth.
            p, labels = build_basis(b if project else b.double(),
                                    lv, shapes, alpha, project)
            parts.append(p)
        bases[name] = torch.cat(parts, dim=1)
        row["labels"][name] = labels
        print(f"  {name:<10} {len(labels):>2} operators  ({time.time() - t0:.0f}s)")

    if RUN_NETS:
        for name, (chan, labels) in attention_bases(beta, configs).items():
            bases[name] = chan
            row["labels"][name] = labels

    # The per-config Ō series, kept beside the summary dump. Everything in this
    # script that is *not* recomputable offline — every correlated difference,
    # any arm pairing thought of after the run — needs these and only these.
    # ~30 MB per β against a measurement that costs a GPU hour.
    if KEEP_OBARS:
        ob_path = f"results/fair_fight/z2_fair_fight_obars_b{beta}.pt"
        torch.save({"beta": beta, "n_cfg": len(configs), "Nt": Nt,
                    "labels": row["labels"],
                    "bases": {k: v.to(torch.float32) for k, v in bases.items()}},
                   ob_path)
        print(f"  kept Ō series → {ob_path}")

    # Every arm under both estimators, as §9.5.1 requires: a line is internally
    # consistent (its ξ, its A₀ and its ΔA₀ describe the same operator) and the
    # gap between the two lines is what the variational step bought that arm.
    for name, ob in bases.items():
        res = zac._jack(ob, Nt)
        single = zac._jack_best_single(ob, Nt, row["labels"][name])
        row["arms"][name] = {"gevp": res, "single": single}
        xi, xe = zac._xi(res)
        xi1, xe1 = zac._xi(single)
        d, s = _dev(res, beta)
        d1, s1 = _dev(single, beta)
        print(f"  {name:<18} GEVP   ξ = {xi:6.2f} ± {xe:4.2f}  A₀ = {res['A0']:6.3f}"
              f"   {d:+6.1f}% ({s:.1f}σ)"
              + ("   [fell back to one channel]" if res.get("gevp_fell_back") else ""))
        print(f"  {'':<18} single ξ = {xi1:6.2f} ± {xe1:4.2f}  A₀ = {single['A0']:6.3f}"
              f"   {d1:+6.1f}% ({s1:.1f}σ)   [{single.get('channel')}]")
        zac._diagnose(name, res)

    # The load-bearing statistic: trained attention minus the STRONGEST
    # classical arm, as a blocked jackknife of the difference on shared
    # configurations, on the bases each arm's own quoted number settled on.
    # It is computed against *every* classical arm, not just the pre-registered
    # one: which arm turns out strongest is not knowable before the run, and
    # recomputing this later is impossible offline (it needs the per-config Ō
    # series, which the summary dump does not carry).
    row["delta_vs"] = {}
    if "attention trained" in bases:
        a_res = row["arms"]["attention trained"]["gevp"]
        a = zac._resolved_series(bases["attention trained"], a_res)
        for name in [n for n in ARM_SPEC if n in bases]:
            b_res = row["arms"][name]["gevp"]
            b = zac._resolved_series(bases[name], b_res)
            if a is None or b is None:
                continue
            dd = zac._corr_delta(a, b, Nt, wa=a_res.get("window"), wb=b_res.get("window"))
            zac._delta_consistency(dd, a_res, b_res, tag=f"(trained − {name})")
            row["delta_vs"][name] = dd
            if name == STRONGEST:
                row["delta_vs_strongest"] = dd
            if dd:
                n_sig = abs(dd["dA0"]) / dd["dA0_err"] if dd["dA0_err"] else float("nan")
                print(f"\n  correlated ΔA₀ (trained − {name}) = "
                      f"{dd['dA0']:+.3f} ± {dd['dA0_err']:.3f}  ({n_sig:.1f}σ)")

    # The null, on the strongest arm: scrambling configurations must leave
    # nothing fittable. If it resolves a mass, the pipeline manufactures one.
    if STRONGEST in bases:
        gen = torch.Generator().manual_seed(zac.RANDOM_SEED)
        null = zac._jack(zac._scramble_configs(bases[STRONGEST], gen), Nt)
        row["null"] = null
        print(f"  config-scramble null on {STRONGEST}: A₀ = {null['A0']:.4f}"
              f"  ({null.get('why') or 'resolved — investigate'})")
    return row


# ── Reporting ─────────────────────────────────────────────────────────────────
def regression_check(rows):
    """Does the rebuilt `published` arm reproduce the published numbers?

    Same configurations, same levels, same estimator — so it should. This is the
    gate: if it does not, every other number here is measuring a bug in this
    script rather than a property of the comparator.

    Only cells measured on the *same number* of configurations are compared. A
    short run (``FF_N_EVAL=60``) is a different sample and disagrees for an
    entirely uninteresting reason; flagging that as a mismatch would train the
    reader to ignore the one check that matters.
    """
    if not os.path.exists(PUBLISHED):
        print(f"\nregression check skipped — {PUBLISHED} absent")
        return
    pub = torch.load(PUBLISHED, map_location="cpu", weights_only=False)
    ref = {round(r["beta"], 4): r for r in pub["rows"]}
    print("\nregression check — the rebuilt `published` arm against the saved dump")
    print(f"  {'β':>7} {'ξ here':>9} {'ξ published':>13} {'Δ':>8}   {'A₀ here':>9} "
          f"{'A₀ published':>13}")
    worst, n_cmp, n_skip = 0.0, 0, 0
    for r in rows:
        p = ref.get(round(r["beta"], 4))
        arm = r["arms"].get("published")
        if p is None or arm is None:
            continue
        xi_new, _ = zac._xi(arm["gevp"])
        xi_old, _ = zac._xi(p["classical"])
        rel = abs(xi_new - xi_old) / xi_old if xi_old else float("nan")
        same_sample = r["n_cfg"] == p["n_cfg"]
        if np.isfinite(rel) and same_sample:
            worst = max(worst, rel)
            n_cmp += 1
        elif not same_sample:
            n_skip += 1
        note = "" if same_sample else f"   (n_cfg {r['n_cfg']} vs {p['n_cfg']} — not comparable)"
        print(f"  {r['beta']:>7.4f} {xi_new:>9.3f} {xi_old:>13.3f} {rel * 100:>7.2f}% "
              f"  {arm['gevp']['A0']:>9.3f} {p['classical']['A0']:>13.3f}{note}")
    # A NaN is not a pass. Counting the comparisons that actually happened is
    # what stops an all-unresolved run from printing a reassuring "OK".
    if n_cmp == 0:
        print(f"  no comparable cells — the regression check did NOT run"
              + (f" ({n_skip} β differ in statistics)" if n_skip else ""))
    else:
        verdict = ("OK" if worst < 0.01
                   else "** MISMATCH — debug before reading anything below **")
        print(f"  worst relative ξ difference over {n_cmp} β: "
              f"{worst * 100:.2f}%   {verdict}")


def report(rows, truth):
    print("\n" + "=" * 78)
    print("ACCURACY AGAINST EXACT GROUND TRUTH (dual Ising, large volume)")
    print("=" * 78)
    names = [n for n in list(ARM_SPEC) + ["attention trained", "attention random"]
             if any(n in r["arms"] for r in rows)]
    for est in ("gevp", "single"):
        print(f"\n  estimator: {est}")
        head = "  " + f"{'arm':<20}" + "".join(f"{r['beta']:>18.4f}" for r in rows) + f"{'mean':>10}"
        print(head)
        for n in names:
            cells, devs = [], []
            for r in rows:
                a = r["arms"].get(n)
                if a is None:
                    cells.append(f"{'—':>18}")
                    continue
                xi, _ = zac._xi(a[est])
                d, s = _dev(a[est], r["beta"])
                if np.isfinite(d):
                    devs.append(d)
                cells.append(f"{xi:>8.2f} {d:>+8.1f}%" if np.isfinite(xi)
                             else f"{'unresolved':>18}")
            mean = f"{np.mean(devs):>+9.1f}%" if devs else f"{'—':>10}"
            print(f"  {n:<20}" + "".join(cells) + mean)
        print(f"  {'exact ξ':<20}"
              + "".join(f"{truth.get(r['beta'], (float('nan'),))[0]:>18.3f}" for r in rows))

    print("\n  ground-state overlap A₀ (GEVP estimator)")
    print("  " + f"{'arm':<20}" + "".join(f"{r['beta']:>10.4f}" for r in rows))
    for n in names:
        cells = []
        for r in rows:
            a = r["arms"].get(n)
            v = a["gevp"]["A0"] if a else float("nan")
            cells.append(f"{v:>10.3f}" if np.isfinite(v) else f"{'—':>10}")
        print(f"  {n:<20}" + "".join(cells))

    # ── the verdict ───────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    def mean_dev(name, est="gevp"):
        d = [_dev(r["arms"][name][est], r["beta"])[0] for r in rows
             if name in r["arms"] and np.isfinite(_dev(r["arms"][name][est], r["beta"])[0])]
        return np.mean(d) if d else float("nan")

    pub_d, str_d = mean_dev("published"), mean_dev(STRONGEST)
    tr_d = mean_dev("attention trained")
    # Which classical arm came out best is a fact about the run, and it need not
    # be the one named in advance. Reporting only the pre-registered arm would
    # let the comparator lose on a technicality; reporting only the winner would
    # be selection on the outcome. Both are printed.
    cand = [(n, mean_dev(n)) for n in ARM_SPEC if n in names]
    best = min((c for c in cand if np.isfinite(c[1])), key=lambda c: abs(c[1]),
               default=(None, float("nan")))
    print(f"  published classical arm      {pub_d:+6.1f}%   (paper quotes −8.9%)")
    print(f"  strongest classical arm      {str_d:+6.1f}%   ({STRONGEST}, pre-registered)")
    if best[0] and best[0] != STRONGEST:
        print(f"  best classical arm           {best[1]:+6.1f}%   ({best[0]}, chosen after the fact)")
    print(f"  attention, trained           {tr_d:+6.1f}%   (paper quotes +3.8%)")
    # The verdict is taken against whichever classical arm is *more* accurate:
    # the fair fight exists to give the classical side its best shot.
    if np.isfinite(best[1]) and abs(best[1]) < abs(str_d):
        str_d = best[1]
    if np.isfinite(str_d) and np.isfinite(tr_d):
        if abs(str_d) > abs(tr_d):
            print("\n  → The strengthened classical basis is still further from the exact")
            print("    answer than the trained attention field. The published comparison")
            print("    survives its strongest opponent.")
        else:
            print("\n  → The strengthened classical basis is at least as accurate as the")
            print("    trained attention field. The published −8.9% was in part an")
            print("    under-powered comparator, and Table 6 must be re-stated:")
            print("    the claim becomes 'from a rank-two input ladder the network")
            print("    reaches what a classical basis needs a radius ladder to reach'.")
    # Combined over β for every classical arm, so the shrinkage of the published
    # ΔA₀ as the opponent is strengthened is readable as a ladder rather than as
    # a single number chosen for us.
    print()
    for n in [a for a in ARM_SPEC if a in names]:
        ds = [r["delta_vs"][n] for r in rows
              if r.get("delta_vs", {}).get(n)] if any("delta_vs" in r for r in rows) \
            else ([r["delta_vs_strongest"] for r in rows
                   if n == STRONGEST and r.get("delta_vs_strongest")])
        num = sum(d["dA0"] / d["dA0_err"] ** 2 for d in ds if d["dA0_err"])
        den = sum(1.0 / d["dA0_err"] ** 2 for d in ds if d["dA0_err"])
        if den:
            mark = "  ←" if n in (STRONGEST, best[0]) else ""
            print(f"  combined correlated ΔA₀ (trained − {n:<10}) = "
                  f"{num / den:+.3f} ± {math.sqrt(1 / den):.3f} "
                  f"({abs(num / den) * math.sqrt(den):.1f}σ){mark}")


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(rows, truth, cov=None):
    names = [n for n in list(ARM_SPEC) + ["attention trained", "attention random"]
             if any(n in r["arms"] for r in rows)]
    betas = [r["beta"] for r in rows]
    xt = [truth.get(b, (np.nan, np.nan))[0] for b in betas]
    xte = [truth.get(b, (np.nan, np.nan))[1] for b in betas]

    def series(n, what="xi"):
        v, e = [], []
        for r in rows:
            a = r["arms"].get(n)
            if a is None:
                v.append(np.nan)
                e.append(np.nan)
            elif what == "xi":
                x, s = zac._xi(a["gevp"])
                v.append(x)
                e.append(s)
            else:
                v.append(a["gevp"]["A0"])
                e.append(a["gevp"]["A0_err"])
        return np.array(v, float), np.array(e, float)

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    for a in ax.ravel():
        a.grid(alpha=0.25, lw=0.6)
        a.set_axisbelow(True)

    # (0,0) ξ against β, every arm, with the exact answer as a band.
    a = ax[0, 0]
    a.errorbar(betas, xt, yerr=xte, fmt="-", color="k", lw=2, capsize=4,
               label="exact (dual Ising)", zorder=5)
    a.fill_between(betas, np.array(xt) - np.array(xte), np.array(xt) + np.array(xte),
                   color="k", alpha=0.12, zorder=0)
    for n in names:
        v, e = series(n)
        a.errorbar(betas, v, yerr=e, fmt=MARKER.get(n, "o"), ls="--", lw=1,
                   ms=7, capsize=3, color=COLOR.get(n), label=n, alpha=0.9)
    a.set_xlabel(r"$\beta$")
    a.set_ylabel(r"$\xi$")
    a.set_title("Every operator against the exact correlation length")
    a.legend(fontsize=8)

    # (0,1) operator quality against the correlation length — the axis that
    # matters, since ξ/a → ∞ is the continuum limit.
    a = ax[0, 1]
    for n in names:
        v, e = series(n, "A0")
        a.errorbar(xt, v, yerr=e, xerr=xte, fmt=MARKER.get(n, "o"), ls="--", lw=1,
                   ms=7, capsize=3, color=COLOR.get(n), label=n, alpha=0.9)
    a.set_xlabel(r"exact $\xi$  [lattice spacings]")
    a.set_ylabel(r"$A_0$ — ground-state overlap")
    a.set_title("Operator quality degrades toward the continuum limit")
    a.legend(fontsize=8)

    # (1,0) fractional deviation from truth — the accuracy table, drawn.
    a = ax[1, 0]
    w = 0.8 / max(len(names), 1)
    for k, n in enumerate(names):
        d = [_dev(r["arms"][n]["gevp"], r["beta"])[0] if n in r["arms"] else np.nan
             for r in rows]
        a.bar(np.arange(len(rows)) + k * w - 0.4, d, width=w * 0.9,
              color=COLOR.get(n), label=n, alpha=0.9)
    a.axhline(0, color="k", lw=1)
    a.set_xticks(range(len(rows)), [f"{b}" for b in betas])
    a.set_xlabel(r"$\beta$")
    a.set_ylabel(r"$(\xi - \xi_{\rm exact})\,/\,\xi_{\rm exact}$  [%]")
    a.set_title("Deviation from exact ground truth")
    a.legend(fontsize=8)

    # (1,1) the correlator each arm actually has, at the largest β.
    a = ax[1, 1]
    last = rows[-1]
    for n in names:
        arm = last["arms"].get(n)
        prof = arm["gevp"].get("profile") if arm else None
        if prof:
            a.plot(range(len(prof)), prof, MARKER.get(n, "o") + "-", ms=5, lw=1,
                   color=COLOR.get(n), label=n, alpha=0.9)
    a.axhline(0, color="k", lw=0.8)
    a.set_yscale("symlog", linthresh=1e-3)
    a.set_xlabel(r"$\Delta t$")
    a.set_ylabel(r"$C(\Delta)/C(0)$")
    a.set_title(f"Signal at β = {last['beta']}: which arms are alive")
    a.legend(fontsize=8)

    fig.suptitle("Fair fight: the classical comparator, rebuilt as strongly as "
                 "the theory allows", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nsaved {OUT_PNG}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    truth = _load_truth()

    if REPLOT:
        # Re-plotting a smoke dump must not overwrite the figure of the real run
        # sitting beside it: the output names follow the dump being read.
        global OUT_PNG
        if "smoke" in os.path.basename(REPLOT) and "smoke" not in OUT_PNG:
            OUT_PNG = OUT_PNG.replace(".png", "_smoke.png")
            print(f"  smoke dump → writing {OUT_PNG}")
        d = torch.load(REPLOT, map_location="cpu", weights_only=False)
        rows = [r for r in d["rows"] if r["beta"] in BETAS] or d["rows"]
        regression_check(rows)
        report(rows, truth)
        plot(rows, truth, d.get("covariance"))
        return

    print(f"device: {device} | Z₂ 3D {tz.L}²×{tz.LT} | R = {tz.R} | "
          f"N_EVAL = {zac.N_EVAL} from index {zac.EVAL_START} | "
          f"arms = {','.join(ARMS)}" + ("  [SMOKE]" if SMOKE else ""))
    print(f"strongest classical arm: {STRONGEST}   "
          f"attention arms: {'on' if RUN_NETS else 'off'}")

    probe = zac.load_configs(BETAS[-1])
    if probe is None:
        raise SystemExit("no ensemble to run the self-checks on")
    ok, cov = selftest_smearing(probe[:8].to(device))
    if not ok:
        raise SystemExit("self-checks failed — the batched smear is not the library's")
    del probe

    rows = []
    for beta in BETAS:
        r = measure(beta, truth)
        if r is None:
            continue
        rows.append(r)
        torch.save({"rows": rows, "betas": BETAS, "arms": ARMS,
                    "arm_spec": {k: list(v) for k, v in ARM_SPEC.items()},
                    "strongest": STRONGEST, "covariance": cov, "truth": DUAL_XI,
                    "meta": {"L": tz.L, "Lt": tz.LT, "R": tz.R,
                             "n_eval": zac.N_EVAL, "eval_start": zac.EVAL_START,
                             "jack_block": zac.JACK_BLOCK,
                             "fit_window": zac.FIT_WINDOW}}, OUT_PT)
        print(f"  saved partial → {OUT_PT} ({len(rows)} ensembles)")

    if not rows:
        raise SystemExit("no ensemble produced a measurement")
    regression_check(rows)
    report(rows, truth)
    plot(rows, truth, cov)


if __name__ == "__main__":
    main()
