"""The attention map as a lattice operator: does it carry the mass gap?

Clause 2 of the conference abstract ("whether the attention range correlates
with physical correlation length") is closed **negative twice** in
``notes/topological_localization.md`` §6 (SU(2): ξ_s ≤ 1.1 spacings, unaskable)
and §6.1 (3D Z₂: ξ reaches 5.3, but ℓ_att sat on its uniform value at both R=6
and R=12). This script tests the diagnosis in ``notes/attention_as_operator.md``:
both failures measured the wrong object.

``ℓ_att = Σ_Δ |Δ|₁ ᾱ(Δ)`` is the first radial moment of the attention averaged
over the lattice — a property of the learned *kernel*, bounded above by R and
centred by the ball geometry (ℓ_uniform = 4.28 at R=6, 8.31 at R=12; every
measurement landed there). Averaging over sites is precisely the operation that
discards what attention has and convolution does not.

Keep the variation instead. The score ``Re Tr[Q†K̃]`` is gauge **invariant**, so
any reduction of the attention row,

    A(x) = f(α_{x→·})     ∈ {α_{x→x},  Σ_Δ|Δ|₁α,  −Σ_Δ α log α},

is a gauge-invariant local scalar field built from the links — an ordinary
lattice operator. The Z₂ operator is per-timeslice, so ``A(t,·)`` depends on
slice t alone and its zero-momentum correlator has an exact transfer-matrix
decomposition:

    C_A(Δt) = ⟨δĀ(t+Δt) δĀ(t)⟩ = Σ_n |⟨0|Â|n⟩|² e^{−m_n Δt}.

So **the attention field has a mass**, and ξ_A = 1/m_A should equal the ξ the
classical scan measures — with no ceiling at R and no geometric offset, because
C_A correlates a fluctuation (mean zero by construction) rather than moments of
a kernel. For a fixed convolutional kernel the fluctuation vanishes identically
and this measurement does not exist: the *existence* of a signal is itself the
content-dependence result.

Three arms, because "any local gauge-invariant scalar decays at the gap" is a
theorem and would be satisfied by an untrained network too:

  * trained network on its **own** β (diagonal)   — the headline ξ_A vs ξ;
  * trained network on **other** β (off-diagonal) — ξ_A must follow the
    *evaluation* ensemble, not the training β. This also removes the confound
    that closed §6.1 (operator quality degrading along the plotted axis: it is
    now a property of the row, while the physics is a property of the column);
  * **random-init** network                       — what equivariance alone
    gives, the baseline against which the trained ground-state overlap A₀ is
    the "training taught the routing physics" claim.

Everything the script needs already exists: five Z₂ ensembles (24²×48, N=2000)
with classical ξ = 2.05 → 5.28 in ``datasets/z2_beta_scan.pt``, and five R=12
checkpoints ``best_z2_glueball_b<β>.pth``. Training used ``configs[:400]``, so
the evaluation slice ``configs[400:]`` is unseen by every checkpoint, diagonal
included. No sampling, no training — the per-configuration cost (APE smearing
plus the R=12 transport) is paid once and shared across all six networks.

Statistical conventions are copied from ``scripts/z2_beta_scan.py`` so the two
masses are directly comparable: GEVP projection (t0=1, td=2), cosh fit over
Δ ∈ [2,8], blocked jackknife over configurations with block 20. The classical
smeared-plaquette basis is re-measured **on the same configurations**, so
ξ_A − ξ_class is a correlated difference rather than two independent numbers.

Run (on the box holding the checkpoints and ensembles), in this order:

    # 1. the headline, where precision matters: diagonal + random arm only,
    #    so the configuration budget buys error bars instead of matrix cells
    ZAC_CROSS=0 ZAC_N_EVAL=1600 python -u scripts/z2_attention_correlator.py

    # 2. the control matrix, where only the row/column contrast matters
    ZAC_N_EVAL=400 python -u scripts/z2_attention_correlator.py

The script prints a wall-clock ETA after its first chunk, so an overlong
configuration budget shows up in the first ten seconds rather than at hour six.

Environment overrides:
    ZAC_N_EVAL=400   configurations per ensemble, taken from index 400 onward.
                     Cost is linear in it; so is the shrinkage of every error
                     bar. The classical arm is re-measured on the same slice,
                     so this also sets the precision of the reference ξ.
    ZAC_CHUNK=1      configurations per forward batch. Raise to 2–4 for
                     throughput if the box has the memory (peak transient is
                     ≈2.5 GB per configuration in the batch at R=12).
    ZAC_CROSS=1      0 ⇒ only the matched network + the random control per
                     ensemble. 5× cheaper; drops the off-diagonal arm.
    ZAC_SMOKE=1      tiny random lattice, no caches/checkpoints — runs the
                     synthetic fit self-check plus a plumbing pass in seconds.

Writes ``datasets/z2_attention_correlator[_diag].pt`` (partial results are saved
after every ensemble, so an interrupted run keeps what it measured) and the
matching ``.png``. The two modes write different names and never clobber each
other.
"""

import math
import os
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# train_z2_glueball reads argv[1] as β at import time; hide our own argv/env
# parsing from it so importing the module here cannot pick up a stray argument.
_ARGV = sys.argv
sys.argv = sys.argv[:1]
import train_z2_glueball as tz  # noqa: E402  (path fix must precede the import)

sys.argv = _ARGV

from gelt.blocks_rope import GELT  # noqa: E402
from gelt.glueball import (  # noqa: E402
    connected_correlator,
    connected_correlator_matrix,
    fit_cosh_correlator,
    gevp_ground_vector,
    smearing_operator_basis,
)
from gelt.lattice import random_links  # noqa: E402


# ── Tunables ──────────────────────────────────────────────────────────────────
BETA_C = 0.7614
BETAS = [0.7450, 0.7520, 0.7560, 0.7585, 0.7600]

# Training used configs[:N_USE]; everything from there on is unseen by every
# checkpoint, so the SAME evaluation slice is clean for all 25 matrix cells.
EVAL_START = tz.N_USE
N_EVAL = int(os.environ.get("ZAC_N_EVAL", 400))
# R (and with it N_USE, and with it the unseen slice) must match the checkpoints
# being read. ZAC_R=6 ZAC_N_USE=800 pairs with the retrained operators; the
# defaults still read the original R=12 set.
if "ZAC_R" in os.environ:
    tz.R = int(os.environ["ZAC_R"])
if "ZAC_N_USE" in os.environ:
    tz.N_USE = int(os.environ["ZAC_N_USE"])
    EVAL_START = tz.N_USE
CHUNK = int(os.environ.get("ZAC_CHUNK", 1))
CROSS = os.environ.get("ZAC_CROSS", "1") == "1"
SMOKE = os.environ.get("ZAC_SMOKE", "0") == "1"

# Matched to z2_beta_scan.py — the classical ξ these numbers are compared to
# was produced with exactly these conventions.
GEVP_T0, GEVP_TD = 1, 2
FIT_WINDOW = (2, 8)
JACK_BLOCK = 20
SMEAR_LEVELS = [0, 4, 8, 16]

# The three natural scalars of an attention row. "ell" is deliberately the
# statistic that failed as a mean — read here as a *field*, which is the whole
# point of the exercise.
REDUCTIONS = ("self", "ell", "ent")
RANDOM_SEED = 20260810

# Channels whose site-to-site fluctuation is below this fraction of their mean
# carry no signal and would only make C(t0) singular in the GEVP.
MIN_REL_FLUCT = 1e-6

_TAG = ("" if CROSS else "_diag") + tz.artifact_tag()
OUT_PT = f"datasets/z2_attention_correlator{_TAG}.pt"
OUT_PNG = f"z2_attention_correlator{_TAG}.png"

device = tz.device


# ── Smoke mode: shrink the lattice so the plumbing runs in seconds ────────────
if SMOKE:
    tz.L, tz.LT, tz.R = 8, 12, 3
    tz.INPUT_SMEAR_LEVELS = (0, 2)
    tz.IN_CHANNELS = len(tz.INPUT_SMEAR_LEVELS)
    N_EVAL, CHUNK = 8, 2
    BETAS = BETAS[:2]
    SMEAR_LEVELS = [0, 2]
    FIT_WINDOW, JACK_BLOCK = (1, 4), 2
    OUT_PT, OUT_PNG = "datasets/z2_attention_correlator_smoke.pt", "z2_attention_correlator_smoke.png"


def build_model(ckpt=None, seed=None):
    """The Phase-B architecture, optionally loaded from a checkpoint.

    Mirrors ``scripts/z2_attention_readout.py``'s builder — every geometric
    hyperparameter has to match the checkpoint or the offsets (and hence the
    attention rows) would not line up.
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
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model


def channel_labels():
    """Ordering must match :func:`attention_fields`: layer → reduction → head."""
    return [
        f"L{l + 1}h{h}:{r}"
        for l in range(tz.GEMHSA_LAYERS)
        for r in REDUCTIONS
        for h in range(tz.NHEAD)
    ]


@torch.no_grad()
def attention_fields(model, W, T, n_cfg, dist):
    """Zero-momentum attention operators Ā(t) for one configuration batch.

    Returns ``(out, chans, moments)``:

    ``out``     ``(n_cfg, Nt)`` — the network's own operator Ō(t). Kept as a
                reference channel and as a check that the checkpoint loaded
                correctly (its mass must reproduce the recorded ``m_net``).
    ``chans``   ``(n_ch, n_cfg, Nt)`` — one zero-momentum series per
                (layer, reduction, head).
    ``moments`` ``(n_ch, 2)`` — running Σ A and Σ A² over *sites* (not the
                zero-momentum sum), so the site-level fluctuation-to-mean ratio
                can be quoted. That ratio is the effect size of the whole
                claim: a convolution's is identically zero.
    """
    site = model(W, T)  # (n_cfg·Lt, L, L)
    out = site.reshape(n_cfg, tz.LT, *site.shape[1:]).sum(dim=(-2, -1))

    chans, moments = [], []
    for layer in model.gemhsa_models:
        a = layer._last_alpha  # (n_cfg·Lt, H, n_off, L, L)
        red = {
            "self": a[:, :, 0],
            "ell": (a * dist.view(1, 1, -1, 1, 1)).sum(dim=2),
            "ent": -(a.clamp_min(1e-12).log() * a).sum(dim=2),
        }
        for name in REDUCTIONS:
            f = red[name]  # (n_cfg·Lt, H, L, L)
            moments.append(
                torch.stack(
                    [f.double().sum(dim=(0, 2, 3)), f.double().pow(2).sum(dim=(0, 2, 3))],
                    dim=1,
                )  # (H, 2)
            )
            f = f.reshape(n_cfg, tz.LT, *f.shape[1:])  # (n_cfg, Lt, H, L, L)
            chans.append(f.sum(dim=(-2, -1)).permute(2, 0, 1))  # (H, n_cfg, Lt)
    return out, torch.cat(chans, dim=0), torch.cat(moments, dim=0)


def _standardize(obar):
    """Per-channel unit variance before the GEVP.

    The GEVP is invariant under any invertible linear recombination of the
    basis, so this changes nothing mathematically — but the raw channels differ
    by orders of magnitude (entropy ≈ 5 against a self-weight ≈ 0.01) and the
    eigenvalue floor in the whitening would otherwise decide the answer.
    """
    s = obar.reshape(obar.shape[0], -1).std(dim=1).clamp_min(1e-30)
    return obar / s.view(-1, 1, 1)


def _fit(obar, nt):
    """(m, A₀) from a cosh fit to the GEVP-projected correlator.

    ``obar`` is ``(n_ch, B, Nt)``. Single-channel input skips the GEVP. Returns
    NaN rather than raising when the correlator is not positive across the fit
    window — an unresolved cell must be visible as unresolved, not as a number.
    """
    if obar.shape[0] == 1:
        proj = obar[0]
    else:
        try:
            C = connected_correlator_matrix(obar)
            v0 = gevp_ground_vector(C, t0=GEVP_T0, td=GEVP_TD)
            proj = torch.einsum("i,ibt->bt", v0, obar)
        except Exception:
            return float("nan"), float("nan")
    Cp = connected_correlator(proj)
    lo, hi = FIT_WINDOW
    win = Cp[lo : hi + 1]
    if not torch.isfinite(Cp[: hi + 1]).all() or (win <= 0).any() or Cp[0] <= 0:
        return float("nan"), float("nan")
    m, A, _chi2 = fit_cosh_correlator(Cp, lo, hi)
    if not np.isfinite(m) or m <= 0:
        return float("nan"), float("nan")
    # Morningstar–Peardon ground-state overlap fraction, as in
    # scripts/fit_glueball_overlap.py.
    A0 = A * (1.0 + math.exp(-m * nt)) / Cp[0].item()
    return float(m), float(A0)


def _jack(obar, nt, block=JACK_BLOCK):
    """Blocked-jackknife (m, A₀) with errors, resampling *configurations*.

    Blocks rather than single deletions because τ_int reaches 6.3 near β_c and
    neighbouring configurations are correlated; deleting one at a time would
    understate the error exactly where the interesting points are.
    """
    obar = _standardize(obar.double())
    m, a0 = _fit(obar, nt)
    B = obar.shape[1]
    n_blocks = max(2, B // block)
    ms, a0s = [], []
    for b in range(n_blocks):
        keep = torch.ones(B, dtype=torch.bool)
        keep[b * block : (b + 1) * block] = False
        mm, aa = _fit(obar[:, keep], nt)
        ms.append(mm)
        a0s.append(aa)
    ms = np.array(ms, dtype=float)
    a0s = np.array(a0s, dtype=float)

    def _err(vals):
        good = vals[np.isfinite(vals)]
        if len(good) < 2:
            return float("nan")
        n = len(good)
        return float(np.sqrt((n - 1) / n * ((good - good.mean()) ** 2).sum()))

    return {"m": m, "m_err": _err(ms), "A0": a0, "A0_err": _err(a0s),
            "n_resolved": int(np.isfinite(ms).sum()), "n_blocks": n_blocks}


def _xi(row):
    """ξ = 1/m with the error propagated, NaN-safe."""
    m, e = row["m"], row["m_err"]
    if not np.isfinite(m) or m <= 0:
        return float("nan"), float("nan")
    return 1.0 / m, (1.0 / m) * (e / m if np.isfinite(e) else float("nan"))


def _shuffle_time(obar, gen):
    """Null arm: destroy the temporal ordering, keep everything else.

    One independent permutation of the timeslices per configuration, shared
    across channels (so equal-time cross-channel structure survives and only
    the correlator in Δt is destroyed). C_A(Δt>0) must come out consistent with
    zero; if it does not, the pipeline is manufacturing a mass.
    """
    n_ch, B, Nt = obar.shape
    idx = torch.stack([torch.randperm(Nt, generator=gen) for _ in range(B)])
    return torch.gather(obar, 2, idx.unsqueeze(0).expand(n_ch, -1, -1))


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
    configs = tz.ensemble()
    sl = configs[EVAL_START : EVAL_START + N_EVAL]
    if len(sl) < 2 * JACK_BLOCK:
        print(f"  only {len(sl)} unseen configs at β={beta} — skipping")
        return None
    return sl.to(tz.MODEL_DTYPE)


def measure_ensemble(beta, nets, labels, dist):
    """Every network's attention operators + the classical basis, one ensemble."""
    configs = load_configs(beta)
    if configs is None:
        return None
    n_cfg = len(configs)
    print(f"  {n_cfg} unseen configs (index {EVAL_START}…{EVAL_START + n_cfg})")

    acc = {name: {"out": [], "chan": []} for name in nets}
    mom = {name: torch.zeros(len(labels), 2, dtype=torch.float64) for name in nets}
    n_site_samples = 0
    classical = []

    t0 = time.time()
    n_chunks = math.ceil(n_cfg / CHUNK)
    for ci, i in enumerate(tqdm(range(0, n_cfg, CHUNK), desc=f"β={beta}", total=n_chunks)):
        batch = configs[i : i + CHUNK]
        b = len(batch)
        # The expensive half — APE smearing + the R=12 transport — depends on
        # the configuration alone, so it is paid once for all six networks.
        W, T = tz.config_inputs(batch)
        classical.append(
            smearing_operator_basis(batch.to(device), tz.gaugegroup, SMEAR_LEVELS,
                                    alpha=tz.SMEAR_ALPHA).double().cpu()
        )
        for name, model in nets.items():
            out, chan, moments = attention_fields(model, W, T, b, dist)
            acc[name]["out"].append(out.double().cpu())
            acc[name]["chan"].append(chan.double().cpu())
            mom[name] += moments.cpu()
        n_site_samples += b * tz.LT * tz.L * tz.L
        del W, T
        if ci == 0:
            dt = time.time() - t0
            print(f"    first chunk {dt:.1f}s → ETA {dt * n_chunks / 60:.1f} min "
                  f"for this ensemble ({len(nets)} networks)")

    Nt = tz.LT
    res = {"beta": beta, "n_cfg": n_cfg, "nets": {}}

    cl = torch.cat(classical, dim=1)  # (n_levels, B, Nt)
    res["classical"] = _jack(cl, Nt)
    xi_c, xi_ce = _xi(res["classical"])
    print(f"  classical smeared basis:  m = {res['classical']['m']:.4f} "
          f"± {res['classical']['m_err']:.4f}   ξ = {xi_c:.2f} ± {xi_ce:.2f}")

    gen = torch.Generator().manual_seed(RANDOM_SEED)
    for name in nets:
        chan = torch.cat(acc[name]["chan"], dim=1)  # (n_ch, B, Nt)
        out = torch.cat(acc[name]["out"], dim=0).unsqueeze(0)  # (1, B, Nt)

        # Site-level fluctuation-to-mean ratio per channel: the effect size.
        s1, s2 = mom[name][:, 0] / n_site_samples, mom[name][:, 1] / n_site_samples
        rel = (s2 - s1**2).clamp_min(0).sqrt() / s1.abs().clamp_min(1e-30)
        keep = rel > MIN_REL_FLUCT
        if keep.sum() == 0:
            print(f"  {name}: every attention channel is constant — no signal to fit")
            res["nets"][name] = {"rel_fluct": rel, "dead": True}
            continue

        entry = {
            "rel_fluct": rel,
            "kept": keep,
            "attention": _jack(chan[keep], Nt),
            "output": _jack(out, Nt),
            "shuffled": _jack(_shuffle_time(_standardize(chan[keep].double()), gen), Nt),
            "per_head": {},
        }
        # Per-(layer, head): the three reductions of one head as a mini-basis.
        for l in range(tz.GEMHSA_LAYERS):
            for h in range(tz.NHEAD):
                idx = [labels.index(f"L{l + 1}h{h}:{r}") for r in REDUCTIONS]
                idx = [j for j in idx if keep[j]]
                if idx:
                    entry["per_head"][f"L{l + 1}h{h}"] = _jack(chan[idx], Nt)
        res["nets"][name] = entry

        xi_a, xi_ae = _xi(entry["attention"])
        xi_o, _ = _xi(entry["output"])
        m_sh = entry["shuffled"]["m"]
        print(f"  {name:>14}: ξ_att = {xi_a:6.2f} ± {xi_ae:5.2f}   "
              f"A₀ = {entry['attention']['A0']:.3f} ± {entry['attention']['A0_err']:.3f}   "
              f"ξ_out = {xi_o:6.2f}   shuffled m = {m_sh:.3f}   "
              f"δA/A = {rel[keep].mean():.2e}")
    return res


def net_names(rows):
    """Ordered union of network names across ensembles.

    In ``ZAC_CROSS=0`` mode each ensemble carries a *different* matched network,
    so no single row's key set is the column list.
    """
    names = [f"train@{b}" for b in BETAS if any(f"train@{b}" in r["nets"] for r in rows)]
    if any("random" in r["nets"] for r in rows):
        names.append("random")
    return names


def _xi_cell(rows, j, name):
    e = rows[j]["nets"].get(name)
    if e is None or e.get("dead"):
        return float("nan")
    return _xi(e["attention"])[0]


def report(rows):
    print("\n" + "=" * 88)
    print("ξ from the attention field vs ξ from the classical smeared basis "
          "(same configurations)")
    print("=" * 88)
    names = net_names(rows)
    hdr = f"{'β':>8} {'ξ_class':>16} " + " ".join(f"{n:>14}" for n in names)
    print(hdr)
    print("-" * len(hdr))
    for j, r in enumerate(rows):
        xi_c, xi_ce = _xi(r["classical"])
        line = f"{r['beta']:>8} {xi_c:>8.2f} ±{xi_ce:>6.2f} "
        for name in names:
            if name not in r["nets"]:
                line += f"{'—':>14} "
            elif r["nets"][name].get("dead"):
                line += f"{'dead':>14} "
            else:
                line += f"{_xi_cell(rows, j, name):>14.2f} "
        print(line)

    # The control that closed §6.1's confound: ξ_A must depend on the ensemble
    # (column), not on which β the network was trained at (row).
    if CROSS:
        names = [n for n in names if n.startswith("train@")]
        if len(names) > 1:
            col = np.array([[_xi_cell(rows, j, n) for j in range(len(rows))]
                            for n in names])
            if not np.isfinite(col).any():
                return
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                spread_row = np.nanmean(np.nanstd(col, axis=0))   # across training β
                spread_col = np.nanmean(np.nanstd(col, axis=1))   # across ensembles
            print("-" * len(hdr))
            print(f"cross-evaluation: spread of ξ_att across ENSEMBLES "
                  f"(the physics)  = {spread_col:.2f}")
            print(f"                  spread of ξ_att across TRAINING β "
                  f"(memorisation) = {spread_row:.2f}")
            print("  content-dependence requires the first to dominate the second.")


def fit_exponent(betas, xis, errs):
    """Effective ν from ξ ~ (β_c − β)^(−ν), weighted log-log line.

    Quotable ONLY as a consistency check between the attention and the classical
    scan on the same points: ξ ≤ 5.3 at L = 24 is not the asymptotic regime and
    this returns ≈ 0.39, not the 3D Ising 0.63.
    """
    b, x, e = np.asarray(betas), np.asarray(xis), np.asarray(errs)
    ok = np.isfinite(x) & (x > 0) & np.isfinite(e) & (e > 0)
    if ok.sum() < 3:
        return float("nan")
    t = np.log(BETA_C - b[ok])
    y = np.log(x[ok])
    w = (x[ok] / e[ok]) ** 2  # weight of log ξ
    p = np.polyfit(t, y, 1, w=np.sqrt(w))
    return -p[0]


def _finite_max(*arrays, default=1.0):
    """Largest finite entry across the arguments; NaN-only input gives default.

    A partially unresolved scan is the normal outcome, not an error — the
    figure has to survive it and show which cells are missing.
    """
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    return float(vals.max()) if len(vals) else default


def plot(rows):
    names = net_names(rows)
    betas = [r["beta"] for r in rows]
    xi_c = [_xi(r["classical"])[0] for r in rows]
    xi_ce = [_xi(r["classical"])[1] for r in rows]

    def arm(name, key="attention", what="xi"):
        vals, errs = [], []
        for r in rows:
            e = r["nets"].get(name)
            if e is None or e.get("dead"):
                vals.append(np.nan)
                errs.append(np.nan)
            elif what == "xi":
                v, s = _xi(e[key])
                vals.append(v)
                errs.append(s)
            else:
                vals.append(e[key]["A0"])
                errs.append(e[key]["A0_err"])
        return np.array(vals), np.array(errs)

    matched = [f"train@{b}" for b in betas]
    xi_m, xi_me = [], []
    for r, n in zip(rows, matched):
        e = r["nets"].get(n)
        v, s = (np.nan, np.nan) if e is None or e.get("dead") else _xi(e["attention"])
        xi_m.append(v)
        xi_me.append(s)
    xi_m, xi_me = np.array(xi_m), np.array(xi_me)
    xi_r, xi_re = arm("random")

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # (0,0) the headline: ξ from the attention field against ξ from the classical
    # basis, measured on the same configurations.
    a = ax[0, 0]
    lim = [0, _finite_max(xi_c, xi_m, xi_r) * 1.25]
    a.plot(lim, lim, ls=":", color="gray", label="y = x")
    a.errorbar(xi_c, xi_m, xerr=xi_ce, yerr=xi_me, fmt="o", capsize=4,
               label="trained, matched β")
    a.errorbar(xi_c, xi_r, xerr=xi_ce, yerr=xi_re, fmt="s", capsize=4, alpha=0.7,
               label="random init")
    for xc, ym, b in zip(xi_c, xi_m, betas):
        if np.isfinite(ym):
            a.annotate(f"{b}", (xc, ym), fontsize=8, xytext=(4, 4),
                       textcoords="offset points")
    a.set_xlabel(r"$\xi$ — classical smeared basis")
    a.set_ylabel(r"$\xi_A$ — attention field")
    a.set_title("The attention field carries the mass gap")
    a.set_xlim(lim)
    a.set_ylim(lim)
    a.legend()

    # (0,1) cross-evaluation matrix: rows = training β, columns = ensemble β.
    a = ax[0, 1]
    rown = names
    M = np.array([[_xi_cell(rows, j, n) for j in range(len(rows))] for n in rown])
    im = a.imshow(M, cmap="viridis", aspect="auto")
    a.set_xticks(range(len(betas)), [f"{b}" for b in betas])
    a.set_yticks(range(len(rown)), rown)
    a.set_xlabel(r"evaluation ensemble $\beta$")
    a.set_ylabel(r"network trained at $\beta$")
    a.set_title(r"$\xi_A$: follows the column (physics), not the row")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                a.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                       color="w", fontsize=9)
    fig.colorbar(im, ax=a, label=r"$\xi_A$")

    # (1,0) scaling: both lengths against β_c − β, with the effective exponent.
    a = ax[1, 0]
    d = BETA_C - np.array(betas)
    nu_c = fit_exponent(betas, xi_c, xi_ce)
    nu_a = fit_exponent(betas, xi_m, xi_me)
    a.errorbar(d, xi_c, yerr=xi_ce, fmt="o-", capsize=4,
               label=rf"classical ($\nu_{{\rm eff}}$ = {nu_c:.2f})")
    a.errorbar(d, xi_m, yerr=xi_me, fmt="s-", capsize=4,
               label=rf"attention ($\nu_{{\rm eff}}$ = {nu_a:.2f})")
    if np.isfinite(np.asarray(xi_c, dtype=float)).any() or np.isfinite(xi_m).any():
        a.set_xscale("log")
        a.set_yscale("log")
    a.invert_xaxis()
    a.set_xlabel(r"$\beta_c - \beta$")
    a.set_ylabel(r"$\xi$")
    a.set_title(r"$\xi_{\rm eff}$ scaling (NOT the asymptotic $\nu$ = 0.63)")
    a.legend()

    # (1,1) how good an operator the gaze is, and the null that must be flat.
    a = ax[1, 1]
    A_m, A_me = [], []
    for r, n in zip(rows, matched):
        e = r["nets"].get(n)
        if e is None or e.get("dead"):
            A_m.append(np.nan)
            A_me.append(np.nan)
        else:
            A_m.append(e["attention"]["A0"])
            A_me.append(e["attention"]["A0_err"])
    A_r, A_re = arm("random", what="A0")
    A_c = [r["classical"]["A0"] for r in rows]
    A_ce = [r["classical"]["A0_err"] for r in rows]
    a.errorbar(betas, A_m, yerr=A_me, fmt="o-", capsize=4, label="attention (trained)")
    a.errorbar(betas, A_r, yerr=A_re, fmt="s-", capsize=4, label="attention (random init)")
    a.errorbar(betas, A_c, yerr=A_ce, fmt="^-", capsize=4, label="classical basis")
    a.axvline(BETA_C, ls=":", color="red")
    a.set_xlabel(r"$\beta$")
    a.set_ylabel(r"$A_0$ (ground-state overlap)")
    a.set_title("Does training teach the routing the physics?")
    a.legend()

    fig.suptitle(
        "Attention as a lattice operator — 3D Z₂, "
        f"{rows[0]['n_cfg']} unseen configs/ensemble, R = {tz.R}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"\nwrote {OUT_PNG}")


def operator_quality(beta):
    """m_net/m_class for the checkpoint at this β, from the training rows.

    Printed up front so an undertrained set of operators is visible *before* the
    run rather than in the notes afterwards. The cross-evaluation arm survives
    unequal quality by design (it is a property of the row, the physics of the
    column) — but the diagonal does not, and A₀ least of all.
    """
    if not os.path.exists(tz.ROWS):
        return None
    rows = torch.load(tz.ROWS, weights_only=False)
    r = rows.get(beta)
    if r is None:
        return None
    ratio = r.get("ratio")
    if ratio is None and r.get("m_net") is not None:
        m_cl, _ = tz.classical_mass(beta)
        ratio = float(r["m_net"]) / m_cl if m_cl else None
    return ratio


def selftest():
    """Does the analysis half recover a mass it was given?

    Smoke mode feeds Haar-random links, which have no correlator at all, so the
    plumbing pass exercises shapes but never the fit. This synthesises a
    two-channel periodic correlator with a known ground state plus an excited
    contaminant and checks that the GEVP + cosh fit + blocked jackknife return
    it — the same arithmetic tests/test_glueball.py covers for the classical
    path, re-run through *this* script's wrapper.
    """
    m0, m1, nt, B = 0.25, 0.9, tz.LT, 400
    g = torch.Generator().manual_seed(7)
    t = torch.arange(nt, dtype=torch.float64)
    sep = torch.remainder(t.view(-1, 1) - t.view(1, -1), nt)
    sep = torch.minimum(sep, nt - sep)  # periodic time separation

    def gaussian_field(m):
        """B time series whose covariance IS the periodic single-state form."""
        cov = torch.exp(-m * sep) + torch.exp(-m * (nt - sep))
        evals, evecs = torch.linalg.eigh(cov)
        half = evecs @ torch.diag(evals.clamp_min(0).sqrt()) @ evecs.T
        return torch.randn(B, nt, generator=g, dtype=torch.float64) @ half

    x0, x1 = gaussian_field(m0), gaussian_field(m1)
    # Two operators with different admixtures, so the GEVP has something to do:
    # C_ij(Δ) = w0_i w0_j z_{m0}(Δ) + w1_i w1_j z_{m1}(Δ), exactly two states.
    obar = torch.stack([w0 * x0 + w1 * x1 for w0, w1 in ((1.0, 0.6), (1.0, -0.4))])
    res = _jack(obar, nt, block=20)
    print(f"selftest: injected m = {m0}  →  fitted {res['m']:.4f} ± {res['m_err']:.4f}"
          f"   A₀ = {res['A0']:.3f}")
    return res


def main():
    if SMOKE:
        selftest()
    print(f"device: {device} | Z₂ 3D {tz.L}²×{tz.LT} | R = {tz.R} | "
          f"N_EVAL = {N_EVAL} from index {EVAL_START} | cross = {CROSS}"
          + ("  [SMOKE]" if SMOKE else ""))

    nets_all = {}
    quality = {}
    for b in BETAS:
        ck = tz.checkpoint_path(b)
        if SMOKE:
            nets_all[f"train@{b}"] = build_model(seed=int(b * 1e4))
        elif os.path.exists(ck):
            nets_all[f"train@{b}"] = build_model(ckpt=ck)
            quality[b] = operator_quality(b)
        else:
            print(f"  no checkpoint {ck} — that row of the matrix will be absent")
    nets_all["random"] = build_model(seed=RANDOM_SEED)
    if len(nets_all) < 2:
        raise SystemExit("no checkpoints found — nothing to measure")

    labels = channel_labels()
    offsets = nets_all["random"].gemhsa_models[0].offsets
    dist = torch.tensor([sum(abs(c) for c in o) for o in offsets],
                        dtype=tz.MODEL_DTYPE, device=device)
    print(f"{len(nets_all)} networks | {len(labels)} attention channels "
          f"| {len(offsets)} offsets")
    if quality:
        q = "  ".join(f"β={b}: {r:.2f}" if r else f"β={b}: ?" for b, r in quality.items())
        print(f"operator quality m_net/m_class — {q}")
        bad = [b for b, r in quality.items() if r and r > tz.GATE_RATIO]
        if bad:
            print(f"  WARNING: {len(bad)}/{len(quality)} checkpoints are above the "
                  f"{tz.GATE_RATIO} gate ({bad}).")
            print("  ξ_A and the row/column control are still readable; the diagonal's")
            print("  A₀ comparison against the random arm is not, until these converge.")

    rows = []
    for beta in BETAS:
        print(f"\n── ensemble β = {beta}  (β_c − β = {BETA_C - beta:.4f}) " + "─" * 24)
        nets = nets_all if CROSS else {
            k: v for k, v in nets_all.items() if k in (f"train@{beta}", "random")
        }
        r = measure_ensemble(beta, nets, labels, dist)
        if r is None:
            continue
        rows.append(r)
        os.makedirs("datasets", exist_ok=True)
        torch.save({"rows": rows, "labels": labels, "betas": BETAS, "beta_c": BETA_C,
                    "meta": {"L": tz.L, "Lt": tz.LT, "R": tz.R, "N_EVAL": N_EVAL,
                             "eval_start": EVAL_START, "fit_window": FIT_WINDOW,
                             "jack_block": JACK_BLOCK, "reductions": REDUCTIONS,
                             "smear_levels": SMEAR_LEVELS, "cross": CROSS}}, OUT_PT)
        print(f"  saved partial → {OUT_PT} ({len(rows)} ensembles)")

    if not rows:
        raise SystemExit("no ensemble produced a measurement")

    report(rows)

    # Per-head detail for the matched networks: which heads carry the gap.
    print("\nper-head ξ_A (matched network, resolved heads only)")
    for r in rows:
        e = r["nets"].get(f"train@{r['beta']}")
        if e is None or e.get("dead"):
            continue
        cells = []
        for k, v in e["per_head"].items():
            xi_h, _ = _xi(v)
            if np.isfinite(xi_h):
                cells.append(f"{k} {xi_h:5.2f}")
        print(f"  β={r['beta']}  ξ_class={_xi(r['classical'])[0]:.2f}  " + "  ".join(cells))

    plot(rows)


if __name__ == "__main__":
    main()
