"""The attention field has quantum numbers: a D₄ irrep decomposition.

``notes/attention_as_operator.md`` established that the attention map of a
gauge-equivariant network is a local gauge-invariant **scalar** lattice
operator — it has a mass (ξ_A tracks the exact dual ξ at Pearson 0.9946) and
training makes it a better one (ΔA₀ = +0.11 … +0.27). The word *scalar* in that
sentence was assumed, never measured, and it is false.

Rotate a configuration by 90°, run the network, rotate the output field back.
For a genuine scalar operator — every classical smeared-loop operator, by
construction — the difference is identically zero. On the β = 0.7585 production
ensemble, as a fraction of the field's own fluctuation:

    trained (best_z2_glueball_b0.7585_R6)   0.671
    random init, seed 20260810              0.002
    random init, seed 7                     0.006

Two to three orders of magnitude, and it is *training* that produces it: the
random-init network is rotationally symmetric to float32 noise, because the L₁
ball is D₄-closed and near-uniform attention inherits the ball's symmetry.

That matters because the ensemble is D₄-invariant, so by Schur the channels do
not interfere and the correlator decomposes exactly:

    C_Ō(Δt) = Σ_ρ C_ρ(Δt),     ρ ∈ {A₁, A₂, B₁, B₂, E}

with A₁ the scalar (0⁺⁺, the target), B₁/B₂ spin-2 (2⁺⁺) and E spin-1. Two
consequences, both measured here:

  * every published A₀ and ξ_A came from the *unprojected* field Ō = Σ_x f(x),
    whose correlator carries the spin-2 tower on top of the 0⁺⁺. Projecting
    onto A₁ removes it exactly, and A₀ should rise;
  * ``LOSS_DELTAS = (1, 2)`` is a short-distance ratio, so contamination by a
    state 2–3× heavier costs the loss almost nothing — the loss cannot see the
    defect it creates. And the wrong-channel content is a second operator: after
    projection the A₁ ground state contributes **exactly zero** to the B channel
    by symmetry, which is the leakage that normally makes higher spin
    unmeasurable.

Construction. For a reduction weight w, let O_w[U](x) = Σ_Δ w(Δ) α_{x→x+Δ}[U].
For g ∈ D₄ define the pulled-back field f_g(x) = O_w[gU](gx) — run on the
rotated configuration, rotate the answer back. Then P_ρ O_w = (1/8) Σ_g χ_ρ(g)
f_g for the four 1D irreps, and the E part is the remainder. The reduction
weight is never rotated (the projector acts on the operator as a whole), and the
network is not required to be equivariant — that is the point.

Cost is 8 forward passes, not 8 configurations' worth of work: the expensive
half per configuration is APE smearing plus the R=6 transport, and both
transform exactly under D₄ without being rebuilt (``_push_W`` / ``_push_T``).
Those laws are asserted against ``config_inputs`` on genuinely rotated links,
for all eight group elements, before anything is measured — that assertion is
also the classical positive control, since it says the projector applied to a
scalar field returns pure A₁.

Design record, self-checks and pre-registered outcomes:
``notes/rotational_symmetry.md``. Read it before touching this file.

Run (on the box holding the checkpoints and ensembles):

    ZRI_N_EVAL=1200 python -u scripts/z2_rotation_irreps.py

Environment overrides:
    ZRI_N_EVAL=1200  configurations per ensemble, from index 400 (unseen by
                     every checkpoint — training used configs[:400]).
    ZRI_CHUNK=2      configurations per forward batch.
    ZRI_R=6          transport radius; must match the checkpoints being read.
    ZRI_GROUP=d4     ``d4`` (8 elements, full irrep table) or ``c4`` (4
                     elements, half the cost, A/B only — no A₁/A₂ or B₁/B₂
                     split, which needs the reflection).
    ZRI_SMOKE=1      tiny random lattice, no ensembles/checkpoints: runs the
                     transformation-law assertions and a plumbing pass in
                     seconds. The physics is meaningless; the asserts are not.
    ZRI_DUMP_OBAR=1  also store the zero-momentum series of the three analysed
                     channels (~4 MB per β per arm), so a fit window can be
                     revisited without re-running the forward passes.
    ZRI_REPLOT=<pt>  re-report and re-plot a saved dump offline (no GPU).

Writes ``results/rotation/z2_rotation_irreps[_R<r>].{pt,png}``; partial results
are saved after every ensemble so an interrupted run keeps what it measured.
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
# train_z2_glueball reads argv[1] as β at import time; hide our own argv from it
# so importing the module here cannot pick up a stray argument.
_ARGV = sys.argv
sys.argv = sys.argv[:1]
import train_z2_glueball as tz  # noqa: E402

sys.argv = _ARGV

from gelt.blocks_rope import GELT  # noqa: E402
from gelt.glueball import connected_correlator, fit_cosh_correlator  # noqa: E402
from gelt.lattice import l1_ball_offsets, random_links  # noqa: E402

os.makedirs("results/rotation", exist_ok=True)


# ── Tunables ──────────────────────────────────────────────────────────────────
SMOKE = os.environ.get("ZRI_SMOKE", "0") == "1"
if "ZRI_R" in os.environ:
    tz.R = int(os.environ["ZRI_R"])
else:
    tz.R = 6  # the retrained operators quoted in attention_as_operator.md §6.1

# β = 0.7600 is dropped for the same reason as in z2_attention_correlator.py:
# its true ξ ≈ 10 against L = 24, so that ensemble measures the box and not a
# mass gap (notes/attention_as_operator.md §8).
BETAS = [0.7450, 0.7520, 0.7560, 0.7585]
N_EVAL = int(os.environ.get("ZRI_N_EVAL", 1200))
EVAL_START = 400  # training used configs[:400]; everything above is unseen
CHUNK = int(os.environ.get("ZRI_CHUNK", 2))
GROUP = os.environ.get("ZRI_GROUP", "d4")
REPLOT = os.environ.get("ZRI_REPLOT", "")
# The zero-momentum series of the three analysed channels, kept in the dump so a
# window or a fit can be revisited offline. Off by default: it is ~4 MB per
# (β, arm), which is two orders of magnitude above everything else in results/.
DUMP_OBAR = os.environ.get("ZRI_DUMP_OBAR", "0") == "1"
# The rope_freq = 0 control arm. Costs one more set of forward passes per chunk
# and is what makes the anisotropy attributable to architecture vs training;
# ZRI_NOROPE=0 drops it if only the trained/random contrast is wanted.
NOROPE_ARM = os.environ.get("ZRI_NOROPE", "1") == "1"
# The ablation table that attributes the anisotropy to architecture vs training.
# Cheap (a few dozen configs) and it is the result that reframes the study.
ATTRIB = os.environ.get("ZRI_ATTRIB", "1") == "1"

# Statistics are copied from z2_attention_correlator.py §4 unchanged, so a
# projected A₀ is directly comparable to a published unprojected one.
FIT_WINDOW = (2, 8)
# The non-scalar channels interpolate a state 2–3× heavier, which is dead by
# Δ ≈ 8. A shorter window is not a free parameter tuned for a nicer answer: the
# m_eff profile is reported alongside, so a missing plateau is visible.
B_FIT_WINDOW = (1, 6)
MIN_FIT_POINTS = 4
JACK_BLOCK = 20
A0_MAX = 1.5
RANDOM_SEED = 20260810  # the same random arm as the published study

# The three existing reductions all use rotationally invariant weights; q1/q2
# are the natural spin-2 interpolators, added so the B channel has its best
# chance of a resolvable mass. The headline (an equivariance defect) shows up
# in the projection of an isotropic reduction just as well.
REDUCTIONS = ("self", "ell", "ent", "q1", "q2")

_TAG = f"_R{tz.R}"
OUT_PT = f"results/rotation/z2_rotation_irreps{_TAG}.pt"
OUT_PNG = f"results/rotation/z2_rotation_irreps{_TAG}.png"

device = tz.device

if SMOKE:
    tz.L, tz.LT, tz.R = 8, 12, 3
    tz.INPUT_SMEAR_LEVELS = (0, 2)
    tz.IN_CHANNELS = len(tz.INPUT_SMEAR_LEVELS)
    N_EVAL, CHUNK, EVAL_START = 8, 2, 0
    BETAS = BETAS[:2]
    FIT_WINDOW, B_FIT_WINDOW, JACK_BLOCK, MIN_FIT_POINTS = (1, 4), (1, 3), 2, 3
    OUT_PT = "results/rotation/z2_rotation_irreps_smoke.pt"
    OUT_PNG = "results/rotation/z2_rotation_irreps_smoke.png"


# ── The group, and how W, T and a scalar field transform under it ─────────────
#
# Elements are (k, s): reflect s times, then rotate k times. Two anchorings have
# to be kept apart, and conflating them is the one bug this file is exposed to:
#
#   * W and every field the network *outputs* are indexed by the base site of a
#     plaquette, which rotates about a cell centre — plain rot90 / flip;
#   * T_Δ(x) is anchored at a lattice **site**, which rotates about a site —
#     the same map plus a roll of one.
#
# Both were found numerically and both are asserted in :func:`check_transforms`
# against rebuilding config_inputs from genuinely rotated links.

ELEMENTS = [(k, s) for s in ((0, 1) if GROUP == "d4" else (0,)) for k in range(4)]
IRREPS_1D = ("A1", "A2", "B1", "B2") if GROUP == "d4" else ("A", "B")
IRREPS = IRREPS_1D + ("E",)
_CHI_R = {"A1": 1, "A2": 1, "B1": -1, "B2": -1, "A": 1, "B": -1}
_CHI_S = {"A1": 1, "A2": -1, "B1": 1, "B2": -1, "A": 1, "B": 1}


def character(rho, k, s):
    return (_CHI_R[rho] ** k) * (_CHI_S[rho] ** s)


def _offsets(with_self):
    o = l1_ball_offsets(2, tz.R)
    return ([(0, 0)] + o) if with_self else o


def _perm(with_self, fn):
    """Index permutation induced on the offset list by a map on Δ."""
    offs = _offsets(with_self)
    idx = {o: i for i, o in enumerate(offs)}
    return torch.tensor([idx[fn(*o)] for o in offs], dtype=torch.long)


_ROT = lambda a, b: (-b, a)  # noqa: E731  (i, j) -> (-j, i)
_REF = lambda a, b: (a, -b)  # noqa: E731  (i, j) -> (i, -j)


def _dims(x, n_trailing):
    """The two spatial axes of ``x``, given how many axes follow them."""
    return (x.dim() - n_trailing - 2, x.dim() - n_trailing - 1)


def _push_cell(f, k, s, n_trailing):
    d = _dims(f, n_trailing)
    if s:
        f = torch.flip(f, [d[1]])
    if k:
        f = torch.rot90(f, k, d)
    return f


def _pull_cell(f, k, s, n_trailing):
    """Inverse of :func:`_push_cell` — rot⁻ᵏ first, then the reflection back."""
    d = _dims(f, n_trailing)
    if k:
        f = torch.rot90(f, -k, d)
    if s:
        f = torch.flip(f, [d[1]])
    return f


def _push_T(T, k, s):
    """Transport under D₄: site-anchored spatial map plus the offset relabel.

    ``T'_{gΔ}(gx) = T_Δ(x)`` — the shortest paths from x to x+Δ map bijectively
    onto those from gx to gx+gΔ, so the average over them is covariant.
    """
    d = _dims(T, 2)
    if s:
        out = torch.empty_like(T)
        out[:, _P_REF_T] = torch.flip(T, [d[1]]).roll(1, d[1])
        T = out
    for _ in range(k):
        out = torch.empty_like(T)
        out[:, _P_ROT_T] = torch.rot90(T, 1, d).roll(1, d[0])
        T = out
    return T


def _push_kernel(phi, k, s):
    """Push a function on the (self-inclusive) offset list forward by g."""
    if s:
        out = torch.zeros_like(phi)
        out[..., _P_REF_K] = phi
        phi = out
    for _ in range(k):
        out = torch.zeros_like(phi)
        out[..., _P_ROT_K] = phi
        phi = out
    return phi


# Link-level action, used only by the self-check (the production path never
# rotates a configuration).
def _link_map(U, Ti, Tj, swap):
    """One D₄ generator acting on the spatial links of ``(b, 3, Lt, L, L, 1, 1)``.

    Z₂ links are real and self-inverse, so the *reversed* links that a rotation
    or a reflection produces need no dagger. For any other group they would —
    this helper is Z₂-only, and it is used only by the self-check.
    """
    Ux, Uy = U[:, 1], U[:, 2]
    nUx, nUy = torch.zeros_like(Ux), torch.zeros_like(Uy)
    if swap:  # 90° rotation: x-links become y-links, y-links become reversed x-links
        nUy[:, :, Ti, Tj] = Ux[:, :, _II, _JJ]
        nUx[:, :, (Ti - 1) % tz.L, Tj] = Uy[:, :, _II, _JJ]
    else:  # reflection: directions are preserved, the second one reverses
        nUx[:, :, Ti, Tj] = Ux[:, :, _II, _JJ]
        nUy[:, :, Ti, (Tj - 1) % tz.L] = Uy[:, :, _II, _JJ]
    return torch.stack([U[:, 0], nUx, nUy], dim=1)


def _push_links(U, k, s):
    if s:
        U = _link_map(U, _II, (-_JJ) % tz.L, False)
    for _ in range(k):
        U = _link_map(U, (-_JJ) % tz.L, _II, True)
    return U


def _rebuild_tables():
    """(Re)build every geometry table after L or R changes (SMOKE does both)."""
    global _II, _JJ, _P_ROT_T, _P_REF_T, _P_ROT_K, _P_REF_K, _DIST, _WEIGHTS
    _II, _JJ = torch.meshgrid(torch.arange(tz.L), torch.arange(tz.L), indexing="ij")
    _P_ROT_T, _P_REF_T = _perm(False, _ROT), _perm(False, _REF)
    _P_ROT_K, _P_REF_K = _perm(True, _ROT), _perm(True, _REF)
    offs = torch.tensor(_offsets(True), dtype=torch.float32)
    _DIST = offs.abs().sum(dim=1)
    n = _DIST.clamp_min(1.0) ** 2
    _WEIGHTS = {
        "ell": _DIST.to(device),
        # The spin-2 pair, normalised to O(1) and zero on the self-offset.
        "q1": ((offs[:, 0] ** 2 - offs[:, 1] ** 2) / n).to(device),
        "q2": ((2 * offs[:, 0] * offs[:, 1]) / n).to(device),
    }


_rebuild_tables()


def check_transforms(verbose=True):
    """Assert the push laws for all |G| elements against an honest rebuild.

    ``config_inputs(gU)`` recomputes APE smearing, the plaquettes and the whole
    L1-ball transport from genuinely rotated links; ``_push_W`` / ``_push_T``
    claim to produce the same thing by relabelling. If that claim is wrong every
    number downstream is wrong, and it fails silently — hence an assert rather
    than a printed diagnostic.

    This doubles as the **classical positive control**: W is a scalar field by
    construction, so a projector fed W must return pure A₁. Equality here is
    exactly that statement at the level of the inputs.
    """
    torch.manual_seed(1234)
    U = torch.stack([
        random_links(tz.L, 3, tz.gaugegroup, dtype=tz.MODEL_DTYPE, Lt=min(tz.LT, 4))
        for _ in range(1)
    ])
    lt_save, tz.LT = tz.LT, min(tz.LT, 4)
    W, T = tz.config_inputs(U)
    worst = 0.0
    for k, s in ELEMENTS:
        Wg, Tg = tz.config_inputs(_push_links(U, k, s))
        eW = (_push_cell(W, k, s, 2) - Wg).abs().max().item()
        eT = (_push_T(T, k, s) - Tg).abs().max().item()
        # The pull must invert the push exactly, or the projector mixes frames.
        eP = (_pull_cell(_push_cell(W, k, s, 2), k, s, 2) - W).abs().max().item()
        worst = max(worst, eW, eT, eP)
        assert eW == 0.0 and eT == 0.0 and eP == 0.0, (
            f"transformation law broken at g=(k={k}, s={s}): "
            f"|ΔW|={eW:.3e} |ΔT|={eT:.3e} |pull∘push−id|={eP:.3e}"
        )
    tz.LT = lt_save
    if verbose:
        print(f"  transform self-check: {len(ELEMENTS)} elements, worst error {worst:.1e}  ✓")
    return worst


# ── Model and fields ──────────────────────────────────────────────────────────
def build_model(ckpt=None, seed=None, no_rope=False):
    """The Phase-B architecture; geometry must match the checkpoint exactly.

    ``no_rope`` zeroes every ``rope_freq``, which makes the RoPE rotation the
    identity at every offset. That matters because RoPE is the *architectural*
    source of rotational-symmetry breaking: ``pair_axis = [p % D]`` ties each
    channel pair to a fixed lattice axis and gives the two axes different
    frequencies (``logspace(0, -1, n_pairs)``), so the score picks up a
    Δ-dependent phase that a 90° rotation does not preserve. Without it the
    score ``Re Tr[Q†K̃]`` is exactly D₄-covariant — T and W both are, and the
    projections act only on channel and colour indices — so this arm is the
    genuinely equivariant reference the random arm turned out not to be.
    """
    if seed is not None:
        torch.manual_seed(seed)
    model = GELT(
        gaugegroup=tz.gaugegroup, L=tz.L, D=2, R=tz.R, nhead=tz.NHEAD,
        gemhsa_layers=tz.GEMHSA_LAYERS, d_qkv=tz.D_QKV, gate=tz.GATE,
        dtype=tz.MODEL_DTYPE, mlp_hidden=tz.MLP_HIDDEN, mlp_out=1,
        reduction="none", init_scale=tz.INIT_SCALE, qk_init_scale=tz.QK_INIT_SCALE,
        mlp_zero_init=False, d_model=tz.D_MODEL, grad_checkpoint=False,
        in_channels=tz.IN_CHANNELS,
    ).to(device)
    if ckpt is not None:
        state = torch.load(ckpt, map_location=device, weights_only=True)
        # _nbr_idx is an L-dependent geometry buffer, rebuilt by the constructor;
        # every *trainable* parameter is L-independent, which is what makes the
        # operator transferable across volumes at all.
        model.load_state_dict({k: v for k, v in state.items() if "_nbr_idx" not in k},
                              strict=False)
    if no_rope:
        with torch.no_grad():
            for layer in model.gemhsa_models:
                layer.rope_freq.zero_()
    model.eval()
    return model


def channel_labels():
    """Ordering must match :func:`extract_fields`: out, controls, then
    layer → reduction → head."""
    return (["out", "class:thin", "class:smeared"]
            + [f"L{l + 1}h{h}:{r}"
               for l in range(tz.GEMHSA_LAYERS)
               for r in REDUCTIONS
               for h in range(tz.NHEAD)])


@torch.no_grad()
def extract_fields(model, W, T):
    """All per-site scalar fields of one (already pushed) input batch.

    Returns ``(n_ch, B, L, L)``. Channel 0 is the network's own variational
    operator; channels 1–2 are the thin and most-smeared plaquette fields, which
    are scalars by construction and therefore the run's exact-zero control row.
    """
    site = model(W, T)  # (B, L, L)
    fields = [site, W[:, 0, ..., 0, 0], W[:, -1, ..., 0, 0]]
    for layer in model.gemhsa_models:
        a = layer._last_alpha  # (B, H, n_off, L, L)
        for name in REDUCTIONS:
            if name == "self":
                f = a[:, :, 0]
            elif name == "ent":
                f = -(a.clamp_min(1e-12).log() * a).sum(dim=2)
            else:
                f = (a * _WEIGHTS[name].view(1, 1, -1, 1, 1)).sum(dim=2)
            for h in range(f.shape[1]):
                fields.append(f[:, h])
    return torch.stack(fields)


@torch.no_grad()
def kernel_moments(model):
    """Σ over sites of α, per (layer, head, offset) — the mean attention kernel.

    Free (the forward pass already happened) and it answers a question the
    projected *field* cannot: whether the anisotropy sits in the weights (a
    fixed learned filter, a spurion that projection removes) or in the
    configuration-dependent fluctuations (the operator choosing an axis per
    region, which would be a far stronger finding).
    """
    return torch.stack([
        layer._last_alpha.double().sum(dim=(0, 3, 4)) for layer in model.gemhsa_models
    ])  # (n_layers, H, n_off)


@torch.no_grad()
def project_chunk(model, W, T, n_cfg):
    """Irrep-projected zero-momentum series and site moments for one batch.

    Returns ``(obar, mom, raw_kernel)`` with ``obar[ρ]`` of shape
    ``(n_ch, n_cfg, Lt)`` and ``mom[ρ]`` of shape ``(n_ch, 2)`` holding Σf and
    Σf² over sites — the site-level variance split, which needs no fit and whose
    null is an exact zero.
    """
    acc = {rho: None for rho in IRREPS_1D}
    f_id = None
    kern = None
    for k, s in ELEMENTS:
        f = extract_fields(model, _push_cell(W, k, s, 2), _push_T(T, k, s))
        f = _pull_cell(f, k, s, 0)
        for rho in IRREPS_1D:
            c = character(rho, k, s)
            term = f if c > 0 else -f
            acc[rho] = term.clone() if acc[rho] is None else acc[rho] + term
        if (k, s) == (0, 0):
            f_id = f
            kern = kernel_moments(model)
    proj = {rho: acc[rho] / len(ELEMENTS) for rho in IRREPS_1D}
    proj["E"] = f_id - sum(proj.values())
    proj["raw"] = f_id

    obar, mom = {}, {}
    for rho, f in proj.items():
        obar[rho] = f.reshape(f.shape[0], n_cfg, tz.LT, tz.L, tz.L).sum(dim=(-2, -1))
        mom[rho] = torch.stack([f.double().sum(dim=(1, 2, 3)),
                                f.double().pow(2).sum(dim=(1, 2, 3))], dim=1)
    return obar, mom, kern


# ── Statistics (single series; the irrep projection replaces the GEVP) ────────
def _window(C, bounds, min_pts=MIN_FIT_POINTS):
    """Largest ``[lo, d] ⊆ bounds`` over which the correlator stays positive.

    Chosen once on the full sample and reused inside every jackknife replica —
    an estimator that changes between replicas does not estimate anything.
    """
    lo, hi = bounds
    if not torch.isfinite(C[0]) or C[0] <= 0:
        return None
    d = lo - 1
    for k in range(lo, min(hi, len(C) - 1) + 1):
        if not torch.isfinite(C[k]) or C[k] <= 0:
            break
        d = k
    return (lo, d) if d - lo + 1 >= min_pts else None


def _fit(series, nt, window):
    C = connected_correlator(series)
    if not torch.isfinite(C[: window[1] + 1]).all() or C[0] <= 0:
        return float("nan"), float("nan")
    m, A, _ = fit_cosh_correlator(C, *window)
    if not np.isfinite(m) or m <= 0:
        return float("nan"), float("nan")
    A0 = A * (1.0 + math.exp(-m * nt)) / C[0].item()
    # An overlap fraction outside [0, A0_MAX] is a fit that ran to its grid edge,
    # not a number; the published study lost one cross-β cell to exactly this.
    if not (0.0 <= A0 <= A0_MAX):
        return float("nan"), float("nan")
    return float(m), float(A0)


def _jack_err(vals):
    good = np.asarray(vals, dtype=float)
    good = good[np.isfinite(good)]
    if len(good) < 2:
        return float("nan")
    n = len(good)
    return float(np.sqrt((n - 1) / n * ((good - good.mean()) ** 2).sum()))


def _blocks(B, block=JACK_BLOCK):
    n = max(2, B // block)
    for i in range(n):
        keep = torch.ones(B, dtype=torch.bool)
        keep[i * block: (i + 1) * block] = False
        yield keep


def measure(series, nt, bounds=FIT_WINDOW, block=JACK_BLOCK):
    """Blocked-jackknife (m, A₀) of one zero-momentum series."""
    series = series.double()
    C = connected_correlator(series)
    win = _window(C, bounds)
    prof = [float(x) for x in (C / C[0])[:12]] if C[0] > 0 else None
    meff = ([float(torch.log(C[d] / C[d + 1])) if C[d] > 0 and C[d + 1] > 0 else float("nan")
             for d in range(min(8, len(C) - 1))] if C[0] > 0 else None)
    out = {"window": win, "profile": prof, "m_eff": meff,
           "signal": float(C[2] / C[0]) if C[0] > 0 and len(C) > 2 else float("nan")}
    if win is None:
        out.update({"m": float("nan"), "m_err": float("nan"),
                    "A0": float("nan"), "A0_err": float("nan")})
        return out
    m, a0 = _fit(series, nt, win)
    ms, a0s = zip(*(_fit(series[keep], nt, win) for keep in _blocks(len(series), block)))
    out.update({"m": m, "m_err": _jack_err(ms), "A0": a0, "A0_err": _jack_err(a0s)})
    return out


def corr_delta(a, b, nt, wa=FIT_WINDOW, wb=FIT_WINDOW, block=JACK_BLOCK):
    """Blocked jackknife of (A₀_a − A₀_b, ξ_a − ξ_b) on SHARED configurations.

    The projected and unprojected fields are two reductions of the *same*
    forward pass on the same configurations, so quoting the difference as
    √(σ² + σ²) throws away a cancellation that is nearly total here — much more
    so than for the trained/random comparison it was introduced for.
    """
    a, b = a.double(), b.double()
    Wa = _window(connected_correlator(a), wa)
    Wb = _window(connected_correlator(b), wb)
    if Wa is None or Wb is None:
        return None

    def stat(mask):
        ma, aa = _fit(a[mask], nt, Wa)
        mb, ab = _fit(b[mask], nt, Wb)
        ok = all(np.isfinite(v) for v in (ma, mb)) and ma > 0 and mb > 0
        return aa - ab, (1.0 / ma - 1.0 / mb) if ok else float("nan")

    dA, dxi = stat(torch.ones(len(a), dtype=torch.bool))
    sA, sx = zip(*(stat(keep) for keep in _blocks(len(a), block)))
    return {"dA0": dA, "dA0_err": _jack_err(sA), "dxi": dxi, "dxi_err": _jack_err(sx)}


def variance_split(mom, n_site):
    """Fraction of each field's site-level variance carried by each irrep.

    The null is an **exact zero** for every non-A₁ channel — that is what makes
    this statistic different from ℓ_att, which was bounded by R and centred by
    the ball geometry. Returned alongside Σ_ρ Var / Var(raw), the Schur check:
    the cross-channel terms vanish only if the ensemble really is D₄-symmetric.
    """
    var = {}
    for rho, m in mom.items():
        mean = m[:, 0] / n_site
        var[rho] = (m[:, 1] / n_site - mean ** 2).clamp_min(0.0)
    tot = sum(var[r] for r in IRREPS)
    return ({r: (var[r] / tot.clamp_min(1e-300)) for r in IRREPS},
            (tot / var["raw"].clamp_min(1e-300)),
            var)


@torch.no_grad()
def attribution(beta, labels, n_cfg=24):
    """Where the rotational-symmetry breaking comes from. Three ablations.

    The naive expectation — "the equivariant construction is symmetric, so any
    anisotropy is learned" — is **false**, and this diagnostic is what shows it.
    Two architectural sources sit under the learned one:

    1. **The anchoring mismatch.** ``W`` is the plaquette *based at*
       site x, i.e. the loop x → x+μ̂ → x+μ̂+ν̂ → x+ν̂ → x. Under a 90° lattice
       rotation that loop maps to the one whose base corner is ``Rx − μ̂`` — the
       plaquette field rotates about a **cell centre**. ``T_Δ(x)`` is a
       transporter between lattice **sites** and rotates about a site. The block
       pairs ``W[x]`` with ``T[·, x]`` at equal array index, and the two indices
       do not move together, so the composite is not D₄-covariant even though
       each input separately is. Feeding a trivial transport removes exactly
       this term, and the block becomes equivariant to float32 noise.
    2. **RoPE.** ``pair_axis = [p % D]`` ties each channel pair to a fixed
       lattice axis and ``logspace(0, -1, n_pairs)`` gives the axes different
       frequencies, so the score picks up a Δ-dependent phase that a rotation
       does not preserve. Zeroing ``rope_freq`` removes this term.

    Neither is visible in the *output* field at initialisation, because
    ``init_scale`` makes the attention path a small perturbation on the
    (exactly scalar) residual W — which is also why the random arm's operator
    agrees with the classical one in attention_as_operator.md §6.1.2.

    Reported as the non-A₁ variance fraction, the same statistic as the main
    tables, averaged over the isotropic reductions (self/ell/ent). The bottom
    row — no RoPE *and* trivial T — must be **exactly 0**: that is the exact-zero
    null the whole study rests on, and it is the strongest validation of the
    projector, stronger than any of the checks in :func:`check_transforms`.

    Which of the two architectural terms dominates is left to the run. On the
    smoke configuration (R=3, random links) RoPE is the larger of the two by
    ~5×, but that geometry is not the production one.
    """
    configs = load_configs(beta)
    if configs is None:
        return None
    configs = configs[:n_cfg]
    iso = [i for i, l in enumerate(labels)
           if l.startswith("L") and l.split(":")[1] in ("self", "ell", "ent")]
    i_out = labels.index("out")
    a1 = IRREPS_1D[0]

    arms = {"random": build_model(seed=RANDOM_SEED),
            "no RoPE": build_model(seed=RANDOM_SEED, no_rope=True)}
    ck = tz.checkpoint_path(beta, tz.R)
    if os.path.exists(ck):
        arms["trained"] = build_model(ckpt=ck)
        arms["trained, no RoPE"] = build_model(ckpt=ck, no_rope=True)

    rows = []
    for name, model in arms.items():
        for tname, trivial in (("real T", False), ("trivial T", True)):
            mom = {rho: torch.zeros(len(labels), 2, dtype=torch.float64)
                   for rho in IRREPS + ("raw",)}
            n_site = 0
            for i in range(0, len(configs), CHUNK):
                batch = configs[i: i + CHUNK]
                W, T = tz.config_inputs(batch)
                if trivial:
                    T = torch.ones_like(T)
                _, m, _ = project_chunk(model, W, T, len(batch))
                for rho in m:
                    mom[rho] += m[rho].cpu()
                n_site += len(batch) * tz.LT * tz.L * tz.L
                del W, T
            frac, _, _ = variance_split(mom, n_site)
            non = 1.0 - frac[a1]
            rows.append((name, tname, float(non[iso].mean()), float(non[i_out])))

    print("\n" + "=" * 78)
    print(f"ATTRIBUTION — where the symmetry breaking comes from (β = {beta}, "
          f"{len(configs)} configs)")
    print("=" * 78)
    print("  non-A₁ variance fraction; 0 = an exactly rotation-equivariant operator")
    print(f"  {'arm':<18s}{'transport':<12s}{'attention (iso)':>17s}{'output field':>15s}")
    for name, tname, a, o in rows:
        print(f"  {name:<18s}{tname:<12s}{a:17.4f}{o:15.4f}")
    print("\n  trivial T isolates the anchoring mismatch; no-RoPE isolates the")
    print("  positional encoding; what survives both is genuinely learned.")
    return rows


def kernel_split(kern):
    """The same decomposition applied to the mean attention kernel ᾱ(Δ).

    The uniform component of a softmax row is pure A₁ and carries no shape
    information, so it is removed first; what is left is the anisotropy of the
    learned *filter*, as opposed to the anisotropy of its fluctuations.
    """
    phi = kern - kern.mean(dim=-1, keepdim=True)  # (n_layers, H, n_off)
    acc = {rho: sum(character(rho, k, s) * _push_kernel(phi, k, s) for k, s in ELEMENTS)
                 / len(ELEMENTS) for rho in IRREPS_1D}
    acc["E"] = phi - sum(acc.values())
    p = phi.pow(2).sum(dim=-1).clamp_min(1e-300)
    return {rho: (acc[rho].pow(2).sum(dim=-1) / p) for rho in IRREPS}


# ── Per-ensemble driver ───────────────────────────────────────────────────────
def load_configs(beta):
    if SMOKE:
        torch.manual_seed(int(beta * 1e4))
        return torch.stack([
            random_links(tz.L, 3, tz.gaugegroup, dtype=tz.MODEL_DTYPE, Lt=tz.LT)
            for _ in range(N_EVAL)
        ])
    tz.BETA = beta
    tz.CACHE = f"datasets/z2_configs_L{tz.L}_Lt{tz.LT}_b{beta}_N{tz.N_CONFIGS}.pt"
    if not os.path.exists(tz.CACHE):
        print(f"  MISSING ensemble {tz.CACHE} — skipping β={beta}")
        return None
    sl = tz.ensemble()[EVAL_START: EVAL_START + N_EVAL]
    if len(sl) < 2 * JACK_BLOCK:
        print(f"  only {len(sl)} unseen configs at β={beta} — skipping")
        return None
    return sl.to(tz.MODEL_DTYPE)


def measure_ensemble(beta, nets, labels):
    configs = load_configs(beta)
    if configs is None:
        return None
    n_cfg = len(configs)
    print(f"  {n_cfg} unseen configs (index {EVAL_START}…{EVAL_START + n_cfg}), "
          f"{len(ELEMENTS)} group elements × {len(nets)} networks")

    acc = {name: {rho: [] for rho in IRREPS + ("raw",)} for name in nets}
    mom = {name: {rho: torch.zeros(len(labels), 2, dtype=torch.float64)
                  for rho in IRREPS + ("raw",)} for name in nets}
    kern = {name: None for name in nets}
    n_site = 0

    t0 = time.time()
    n_chunks = math.ceil(n_cfg / CHUNK)
    for ci, i in enumerate(tqdm(range(0, n_cfg, CHUNK), desc=f"β={beta}", total=n_chunks)):
        batch = configs[i: i + CHUNK]
        b = len(batch)
        # The expensive half — APE smearing and the R-ball transport — depends on
        # the configuration alone and transforms exactly under D₄, so it is paid
        # once for all |G| elements and all networks.
        W, T = tz.config_inputs(batch)
        for name, model in nets.items():
            obar, m, kk = project_chunk(model, W, T, b)
            for rho in obar:
                acc[name][rho].append(obar[rho].double().cpu())
                mom[name][rho] += m[rho].cpu()
            kern[name] = kk if kern[name] is None else kern[name] + kk
        n_site += b * tz.LT * tz.L * tz.L
        del W, T
        if ci == 0:
            dt = time.time() - t0
            print(f"    first chunk {dt:.1f}s → ETA {dt * n_chunks / 60:.1f} min "
                  f"for this ensemble")

    row = {"beta": beta, "n_cfg": n_cfg, "nets": {}}
    for name in nets:
        obar = {rho: torch.cat(acc[name][rho], dim=1) for rho in acc[name]}
        frac, schur, _ = variance_split(mom[name], n_site)
        row["nets"][name] = {
            "obar": obar,  # (n_ch, n_cfg, Lt) per irrep — everything else derives
            "frac": {r: frac[r].tolist() for r in frac},
            "schur": schur.tolist(),
            "kernel": {r: v.tolist() for r, v in kernel_split(kern[name] / n_cfg).items()},
        }
    return row


def analyse(row, labels):
    """Fits and correlated differences. Separated from measurement so a saved
    dump can be re-analysed offline without touching a GPU."""
    nt = tz.LT
    i_out = labels.index("out")
    att = [i for i, l in enumerate(labels) if l.startswith("L")]
    b_irreps = [r for r in IRREPS_1D if r.startswith("B")]
    a1 = IRREPS_1D[0]

    # The channel is selected ONCE, on the trained arm, and reused for the
    # random arm. Re-selecting best-of-N per arm is a bias that measures 7.4× on
    # pure noise (attention_as_operator.md §6.2 defect 5).
    names = list(row["nets"])
    lead = "trained" if "trained" in names else names[0]

    def sig(o, ch):
        C = connected_correlator(o[ch].double())
        return float(C[2] / C[0]) if C[0] > 0 else float("-inf")

    o_lead = row["nets"][lead]["obar"]
    ch_a1 = max(att, key=lambda c: sig(o_lead[a1], c))
    ch_b = max(((c, r) for c in att for r in b_irreps),
               key=lambda cr: sig(o_lead[cr[1]], cr[0]))

    for name in names:
        o = row["nets"][name]
        res = {"ch_a1": labels[ch_a1], "ch_b": (labels[ch_b[0]], ch_b[1])}
        for tag, ch in (("out", i_out), ("att", ch_a1)):
            res[f"{tag}_raw"] = measure(o["obar"]["raw"][ch], nt)
            res[f"{tag}_{a1}"] = measure(o["obar"][a1][ch], nt)
            res[f"{tag}_delta"] = corr_delta(o["obar"][a1][ch], o["obar"]["raw"][ch], nt)
        for rho in b_irreps + ["E"]:
            res[f"out_{rho}"] = measure(o["obar"][rho][i_out], nt, B_FIT_WINDOW)
        res["att_B"] = measure(o["obar"][ch_b[1]][ch_b[0]], nt, B_FIT_WINDOW)
        # ‖f − P_A₁ f‖ / ‖f − ⟨f⟩‖ on the zero-momentum series: the §1 scalar,
        # in the same units the paper would quote.
        raw, pa1 = o["obar"]["raw"][i_out].double(), o["obar"][a1][i_out].double()
        res["breaking_out"] = float((raw - pa1).std() / raw.std().clamp_min(1e-300))
        o["fits"] = res
    return row


def slim(row, labels):
    """A saveable copy: fits and fractions always, raw series only on request.

    ``obar`` is (n_ch, n_cfg, Lt) per irrep — hundreds of MB per ensemble, which
    is not what ``results/`` is for. The fits, profiles, m_eff and variance
    fractions are everything the report and the figure need.
    """
    keep = None
    if DUMP_OBAR:
        f = row["nets"][next(iter(row["nets"]))]["fits"]
        idx = sorted({labels.index("out"), labels.index(f["ch_a1"]),
                      labels.index(f["ch_b"][0])})
        keep = idx
    out = {"beta": row["beta"], "n_cfg": row["n_cfg"], "nets": {}}
    for name, n in row["nets"].items():
        e = {k: v for k, v in n.items() if k != "obar"}
        if keep is not None:
            e["obar"] = {rho: n["obar"][rho][keep].float() for rho in n["obar"]}
            e["obar_channels"] = [labels[i] for i in keep]
        out["nets"][name] = e
    return out


# ── Reporting ─────────────────────────────────────────────────────────────────
def _fmt(v, e=None, nd=3):
    if v is None or not np.isfinite(v):
        return "  —  "
    return f"{v:.{nd}f}" + (f" ± {e:.{nd}f}" if e is not None and np.isfinite(e) else "")


def report(rows, labels):
    a1 = IRREPS_1D[0]
    i_out = labels.index("out")
    i_ctl = labels.index("class:smeared")

    print("\n" + "=" * 78)
    print("CONTROL — the classical smeared plaquette field, same projector")
    print("=" * 78)
    print("  A scalar by construction: every non-A₁ fraction must be 0 to machine")
    print("  precision. Anything else invalidates the run.")
    for r in rows:
        for name, n in r["nets"].items():
            non = max(n["frac"][rho][i_ctl] for rho in IRREPS if rho != a1)
            print(f"  β={r['beta']}  {name:<12s}  max non-{a1} fraction = {non:.2e}")

    print("\n" + "=" * 78)
    print("IRREP VARIANCE SPLIT of the network's own operator field")
    print("=" * 78)
    hdr = "  β      arm           " + "".join(f"{r:>9s}" for r in IRREPS) + "    Schur   break"
    print(hdr)
    for r in rows:
        for name, n in r["nets"].items():
            fr = "".join(f"{n['frac'][rho][i_out]:9.4f}" for rho in IRREPS)
            print(f"  {r['beta']:<6.4f} {name:<13s}{fr}  {n['schur'][i_out]:7.4f}"
                  f"  {n['fits']['breaking_out']:6.3f}")

    print("\n  Same split for the mean attention kernel ᾱ(Δ) (uniform part removed),")
    print("  averaged over layers and heads — anisotropy in the weights, not the field.")
    print(f"  NOTE: unlike the field split, the null here is NOT zero — ᾱ has a radial")
    print(f"  profile (pure {a1}) and a finite-sample noise floor of d_ρ²/|G|. The random")
    print("  arm is the null. A trained/random gap in the non-scalar rows means the")
    print("  anisotropy is a learned fixed filter; no gap means it lives in the")
    print("  configuration-dependent fluctuations instead.")
    for r in rows:
        for name, n in r["nets"].items():
            k = {rho: float(np.mean(n["kernel"][rho])) for rho in IRREPS}
            print(f"  {r['beta']:<6.4f} {name:<13s}"
                  + "".join(f"{k[rho]:9.4f}" for rho in IRREPS))

    print("\n" + "=" * 78)
    print(f"DOES THE {a1} PROJECTION IMPROVE THE OPERATOR?")
    print("=" * 78)
    print("  ΔA₀ and Δξ are blocked jackknives of the difference on shared configs.")
    for tag, what in (("out", "network operator"), ("att", "attention channel")):
        print(f"\n  --- {what} ---")
        print("  β      arm            A₀ raw          A₀ " + a1
              + "         ΔA₀ (corr)        ξ raw    ξ " + a1)
        for r in rows:
            for name, n in r["nets"].items():
                f = n["fits"]
                raw, pj, d = f[f"{tag}_raw"], f[f"{tag}_{a1}"], f[f"{tag}_delta"]
                xr = 1 / raw["m"] if np.isfinite(raw["m"]) and raw["m"] > 0 else float("nan")
                xp = 1 / pj["m"] if np.isfinite(pj["m"]) and pj["m"] > 0 else float("nan")
                sig = ""
                if d and np.isfinite(d["dA0"]) and np.isfinite(d["dA0_err"]) and d["dA0_err"] > 0:
                    sig = f"  ({abs(d['dA0'] / d['dA0_err']):.1f}σ)"
                print(f"  {r['beta']:<6.4f} {name:<13s} "
                      f"{_fmt(raw['A0'], raw['A0_err'])}  {_fmt(pj['A0'], pj['A0_err'])}  "
                      f"{_fmt(d['dA0'], d['dA0_err']) if d else '  —  '}{sig:<9s}"
                      f"  {_fmt(xr, None, 2)}  {_fmt(xp, None, 2)}")

    print("\n" + "=" * 78)
    print("THE NON-SCALAR CHANNELS — is there a spin-2 state in there?")
    print("=" * 78)
    print(f"  Window {B_FIT_WINDOW} (a heavier state dies before Δ=8). m_eff is printed")
    print("  so a missing plateau is visible rather than hidden in a fitted number.")
    b_keys = [f"out_{r}" for r in IRREPS_1D if r.startswith("B")] + ["out_E", "att_B"]
    for r in rows:
        for name, n in r["nets"].items():
            f = n["fits"]
            base = f[f"out_{a1}"]["m"]
            for key in b_keys:
                res = f.get(key)
                if res is None or not np.isfinite(res["m"]):
                    continue
                ratio = res["m"] / base if np.isfinite(base) and base > 0 else float("nan")
                me = " ".join(f"{v:5.2f}" for v in (res["m_eff"] or [])[:6])
                print(f"  {r['beta']:<6.4f} {name:<12s} {key:<10s} "
                      f"m={_fmt(res['m'], res['m_err'])}  m/m_{a1}={_fmt(ratio, None, 2)}"
                      f"  A₀={_fmt(res['A0'], res['A0_err'])}  m_eff: {me}")

    print("\n" + "=" * 78)
    print(f"{a1} FRACTION BY REDUCTION — self-check 4, in both directions")
    print("=" * 78)
    print(f"  For an equivariant network the isotropic reductions (self/ell/ent) are")
    print(f"  pure {a1} and the spin-2 reductions (q1/q2) have **no** {a1} part at all:")
    print(f"  expected 1.000 / 1.000 / 1.000 / 0.000 / 0.000 on the random arm.")
    for r in rows:
        for name, n in r["nets"].items():
            by = {}
            for c, lab in enumerate(labels):
                if lab.startswith("L"):
                    by.setdefault(lab.split(":")[1], []).append(n["frac"][a1][c])
            print(f"  {r['beta']:<6.4f} {name:<12s} "
                  + "  ".join(f"{k}:{np.mean(v):.3f}" for k, v in by.items()))

    print("\n" + "=" * 78)
    print("PER-(LAYER, HEAD, REDUCTION) non-scalar fraction — where the anisotropy lives")
    print("=" * 78)
    for r in rows[-1:]:
        print(f"  β = {r['beta']}")
        for name, n in r["nets"].items():
            worst = sorted(((sum(n["frac"][rho][c] for rho in IRREPS if rho != a1), labels[c])
                            for c in range(len(labels)) if labels[c].startswith("L")),
                           reverse=True)[:8]
            print(f"    {name:<12s} " + "  ".join(f"{l}:{v:.3f}" for v, l in worst))


def plot(rows, labels):
    a1 = IRREPS_1D[0]
    i_out = labels.index("out")
    betas = [r["beta"] for r in rows]
    names = [n for n in ("norope", "random", "trained")
             if all(n in r["nets"] for r in rows)]
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    # (a) stacked irrep split of the network operator, arm by arm
    a0 = ax[0, 0]
    x = np.arange(len(rows) * len(names))
    bottom = np.zeros(len(x))
    cols = {"A1": "#4C72B0", "A2": "#DD8452", "B1": "#C44E52", "B2": "#8172B3",
            "E": "#937860", "A": "#4C72B0", "B": "#C44E52"}
    for rho in IRREPS:
        v = np.array([r["nets"][n]["frac"][rho][i_out] for r in rows for n in names])
        a0.bar(x, v, bottom=bottom, label=rho, color=cols.get(rho, None))
        bottom += v
    a0.set_xticks(x)
    a0.set_xticklabels([f"{r['beta']}\n{n[:9]}" for r in rows for n in names], fontsize=7)
    a0.set_ylabel("fraction of site-level variance")
    a0.set_title("Irrep content of the learned operator\n(classical operator = 1.0 in "
                 + a1 + " by construction)")
    a0.legend(fontsize=8, ncol=len(IRREPS))

    # (b) A₀, raw vs projected
    a1p = ax[0, 1]
    for name, mk in zip(names, ("o", "s", "^")):
        for key, ls, lab in (("out_raw", "--", "raw"), (f"out_{a1}", "-", a1)):
            y = [r["nets"][name]["fits"][key]["A0"] for r in rows]
            e = [r["nets"][name]["fits"][key]["A0_err"] for r in rows]
            a1p.errorbar(betas, y, yerr=e, marker=mk, ls=ls, capsize=3,
                         label=f"{name} {lab}")
    a1p.set_xlabel("β")
    a1p.set_ylabel("A₀")
    a1p.set_title("Ground-state overlap before and after projection")
    a1p.legend(fontsize=7)
    a1p.grid(alpha=0.3)

    # (c) correlated ΔA₀ with its own error — the load-bearing panel
    a2 = ax[1, 0]
    w = 0.35
    for i, name in enumerate(names):
        d = [r["nets"][name]["fits"]["out_delta"] for r in rows]
        y = [x_["dA0"] if x_ else np.nan for x_ in d]
        e = [x_["dA0_err"] if x_ else np.nan for x_ in d]
        a2.bar(np.arange(len(rows)) + i * w, y, w, yerr=e, capsize=3, label=name)
    a2.axhline(0, color="k", lw=1)
    a2.set_xticks(np.arange(len(rows)) + w / 2)
    a2.set_xticklabels([str(b) for b in betas])
    a2.set_xlabel("β")
    a2.set_ylabel(f"A₀({a1}) − A₀(raw), correlated")
    a2.set_title("What the projection buys")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3, axis="y")

    # (d) non-scalar fraction per attention channel
    a3 = ax[1, 1]
    att = [c for c in range(len(labels)) if labels[c].startswith("L")]
    for name, mk in zip(names, ("o", "s", "^")):
        v = [sum(rows[-1]["nets"][name]["frac"][rho][c] for rho in IRREPS if rho != a1)
             for c in att]
        a3.plot(range(len(att)), v, mk, ls="none", label=name, alpha=0.8)
    a3.set_xticks(range(len(att)))
    a3.set_xticklabels([labels[c] for c in att], rotation=90, fontsize=5)
    a3.set_ylabel(f"1 − v_{a1}")
    a3.set_title(f"Non-scalar content per attention channel (β = {betas[-1]})")
    a3.legend(fontsize=8)
    a3.grid(alpha=0.3, axis="y")

    fig.suptitle(f"D₄ irrep decomposition of the attention field — Z₂ 3D, "
                 f"L={tz.L} Lt={tz.LT} R={tz.R}, {rows[0]['n_cfg']} unseen configs")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"\nwrote {OUT_PNG}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    if REPLOT:
        d = torch.load(REPLOT, map_location="cpu", weights_only=False)
        rows, labels = d["rows"], d["labels"]
        # Reports the fits stored at measurement time. Changing a fit window
        # means re-running: the dump keeps the zero-momentum series only under
        # ZRI_DUMP_OBAR=1, and then only for the three analysed channels, keyed
        # by label in ``obar_channels``.
        report(rows, labels)
        plot(rows, labels)
        return

    print(f"D₄ irrep decomposition | group={GROUP} ({len(ELEMENTS)} elements) | "
          f"R={tz.R} | N_EVAL={N_EVAL} | device={device}")
    check_transforms()

    labels = channel_labels()
    if ATTRIB:
        attribution(BETAS[-1], labels)
    rows = []
    for beta in BETAS:
        print(f"\nβ = {beta}")
        # Arms are keyed by role, not by checkpoint name, so every β has the
        # same keys and the cross-β tables and panels line up. Three arms
        # separate the two sources of anisotropy: `norope` is the exactly
        # equivariant reference, `random` adds RoPE and no learning, `trained`
        # adds learning on top of RoPE.
        nets = {"random": build_model(seed=RANDOM_SEED)}
        if NOROPE_ARM:
            nets["norope"] = build_model(seed=RANDOM_SEED, no_rope=True)
        ck = tz.checkpoint_path(beta, tz.R)
        if SMOKE:
            nets["trained"] = build_model(seed=1)  # no checkpoints in smoke mode
        elif os.path.exists(ck):
            nets["trained"] = build_model(ckpt=ck)
        else:
            print(f"  MISSING checkpoint {ck} — random arm only")
        row = measure_ensemble(beta, nets, labels)
        if row is None:
            continue
        analyse(row, labels)
        rows.append(slim(row, labels))
        torch.save({"rows": rows, "labels": labels, "irreps": IRREPS, "group": GROUP,
                    "meta": {"L": tz.L, "Lt": tz.LT, "R": tz.R, "N_EVAL": N_EVAL,
                             "eval_start": EVAL_START, "fit_window": FIT_WINDOW,
                             "b_fit_window": B_FIT_WINDOW, "jack_block": JACK_BLOCK,
                             "reductions": REDUCTIONS}}, OUT_PT)
        print(f"  saved partial results → {OUT_PT}")

    if not rows:
        raise SystemExit("no ensemble produced results")
    report(rows, labels)
    plot(rows, labels)


if __name__ == "__main__":
    main()
