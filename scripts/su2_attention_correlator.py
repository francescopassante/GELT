"""The attention map as a lattice operator, in SU(2): Table 5 on a non-abelian gauge group.

``scripts/z2_attention_correlator.py`` established the result in 3D Z₂
(``notes/attention_as_operator.md`` §6.1): the gauge-invariant score makes any
reduction ``A(x) = f(α_{x→·})`` a **local scalar lattice operator**, so its
zero-momentum connected correlator decays with the mass gap, and ξ_A = 1/m_A
tracks the classical ξ at Pearson 0.9966 while the *trained* arm carries far
more ground-state overlap than a randomly initialized network of the same
architecture. This script runs the identical measurement on **anisotropic SU(2)
in 3+1D**, i.e. it reproduces the paper's Table 5 (``tab:z2``) on the group the
thesis actually cares about.

Why this is worth doing, and why it was not obvious that it could be
-------------------------------------------------------------------
Z₂ was chosen for the *range* study because it is the one lattice family in this
repo where the spatial correlation length reaches several lattice spacings
(``notes/topological_localization.md`` §6: SU(2) sits at ξ_s ≤ 1.1). But ξ_s is
the wrong number here. The attention field is a **per-timeslice** operator and
its correlator is measured in **time**, and the anisotropic lattice
(a_t = a_s/ξ, ξ = 3) is precisely the one that made the 0⁺⁺ mass resolvable:
``results/attention/beta_scan.pt`` records m·a_t = 0.581 … 0.303 over
β ∈ [2.1, 2.7], i.e. **ξ_t = 1/(m·a_t) = 1.7 … 3.3 temporal spacings** at 3–8%
precision. That is a real correlator with a factor-1.9 dynamic range, measured
on the same lattice family the §6.2 spectroscopy result was established on. The
ceiling that killed ℓ_att does not exist for ξ_A: C_A correlates a *fluctuation*
about the mean, not a moment of a kernel, so nothing bounds it by R.

Scope: one row, at β = 2.4
-------------------------
The default run measures a **single coupling**, and produces everything one row
of Table 5 carries: ξ_class, ξ_A trained, ξ_A random, the three A₀, and the
correlated ΔA₀ — plus the per-row detail the paper prints beside the table (the
C(Δ)/C(0) profiles of every arm, the time-shuffle zero-mode check, the
config-scramble null, and the site-level δA/A).

β = 2.4 because it is the only coupling with a trained operator, which makes
this a genuine **diagonal** cell — network and ensemble at the same β, as every
row of Table 5 is. A fresh operator elsewhere costs ~15 h of V100 (47 min
sampling + ~14 h training, ``logs/overnight.log``), so a five-row table is a
three-day batch. β = 2.4 is also the anchor of the §6.2 spectroscopy result and
carries ξ_t = 3.00, near the top of the range this lattice family reaches.

What a single row cannot support is the *aggregate* claims: Pearson(ξ_A,
ξ_class), the slope, the dynamic range, and the cross-β control that shows ξ_A
following the ensemble rather than the training coupling. Those need the scan,
and :func:`report` declines to print them below three couplings rather than
emitting a NaN that reads like a failed measurement. Setting
``SAC_BETAS=2.1,2.3,2.4,2.5,2.7`` runs the scan; the trained arm is then the
β = 2.4 operator evaluated everywhere (matched in its own row, off-diagonal in
the rest), which Table 6 (``tab:cross``) licenses — it measures the ratio
(spread across evaluation ensembles)/(spread across training β) = 12.0, with a
mean diagonal advantage in A₀ of only +0.038 against a mean trained-minus-random
of +0.156, i.e. ξ_A is a property of the configuration in front of the network.

One trained arm is run: the Run-5 β = 2.4 operator
(``best_glueball_gelt_sm0-2-4-6.pth``, seed-0 ensemble). The replication
checkpoint (``…_ens1.pth``, seed-1 ensemble) is a second, equally valid trained
arm trained on an independent ensemble — running it too would transport §6.2's
A₀ replication statement to the attention field, and it is one entry in
``TRAINED_CKPTS`` away.

Every ensemble measured here is sampled at ``SAC_SEED`` (default 11), which is
neither of the seeds (0, 1, 2) any checkpoint was trained on, and at a different
N — so the cache keys cannot collide and **every configuration is unseen by
every network, at every β including the matched one**. No train/val/test slicing
is needed or used.

Statistics
----------
The estimator layer is *imported* from ``scripts/z2_attention_correlator.py``
rather than re-implemented: GEVP projection (t0 = 1, td = 2) with the
variational self-check that catches a near-null direction of C(t0), the largest
positive fit window inside Δ ∈ [2, 8] chosen once on the full sample, the cosh
fit and Morningstar–Peardon A₀, the blocked jackknife with block 20, the
config-scramble null and the time-shuffle zero-mode check, and the correlated
trained − random ΔA₀ on shared configurations. Reusing the code is the point:
"same conventions as Table 5" is then a fact about the call graph rather than a
claim in a caption. Only the *inputs* change — SU(2)'s classical basis is the
APE ladder (0, 2, 4, 6) that ``train_glueball.py``/``measure_glueball.py`` use,
not Z₂'s (0, 4, 8, 16).

Run (on the box with the V100 and the checkpoints):

    # plumbing check first (seconds, physics meaningless)
    SAC_SMOKE=1 SAC_DEVICE=cpu python -u scripts/su2_attention_correlator.py

    # the row: one coupling, ~40 min sampling + measurement
    python -u scripts/su2_attention_correlator.py

    # the full scan, if the aggregates are wanted later
    SAC_BETAS=2.1,2.3,2.4,2.5,2.7 SAC_N_EVAL=800 \
        python -u scripts/su2_attention_correlator.py

Environment overrides:
    SAC_BETAS=2.4     comma-separated couplings. One by default; the aggregate
                      statistics are suppressed below three.
    SAC_N_EVAL=1600   configurations per ensemble. Every error bar shrinks as
                      1/√N and the cost is linear; Table 5's diagonal quotes
                      1200 per ensemble at 5–7% on both ξ arms.
    SAC_CHUNK=4       configurations per forward batch (memory knob: each
                      expands to CHUNK·24 timeslices through the 3D network).
    SAC_SEED=11       sampler seed for the evaluation ensembles.
    SAC_KEEP=1        cache sampled ensembles under datasets/ (≈5.3 MB/config,
                      so ≈8.5 GB at the default N = 1600). 0 = sample, measure,
                      discard — cheap on disk, expensive to re-analyse.
    SAC_SMOKE=1       tiny lattice, random links, no checkpoints — plumbing only.
    SAC_REPLOT=<pt>   re-report and re-plot a saved dump offline (no GPU).

Writes ``results/attention/su2_attention_correlator.{pt,png}`` (partial results
are saved after every ensemble, so an interrupted run keeps what it measured)
and a paste-ready ``…_table.tex`` in the column layout of the paper's Table 5.
"""

import functools
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
# Both imported modules parse sys.argv / env at import time (train_z2_glueball
# reads argv[1] as β; z2_attention_correlator reads its own ZAC_* knobs). Hide
# our argv from them so a stray argument here cannot reconfigure either.
_ARGV = sys.argv
sys.argv = sys.argv[:1]
import train_glueball as tg  # noqa: E402  (path fix must precede the imports)
# The estimator layer. Imported, not copied: identical GEVP / window / cosh /
# jackknife / null conventions to the Z₂ run is what makes this a reproduction
# of Table 5 rather than a differently-tuned measurement of the same idea.
import z2_attention_correlator as zac  # noqa: E402

sys.argv = _ARGV

from gelt.blocks_rope import GELT  # noqa: E402
from gelt.glueball import smearing_operator_basis  # noqa: E402
from gelt.lattice import random_links  # noqa: E402
from gelt.sampler import heatbath_overrelaxation_sweep, mcmc_ensemble  # noqa: E402

for _d in ("results/attention", "datasets"):
    os.makedirs(_d, exist_ok=True)


# ── Tunables ──────────────────────────────────────────────────────────────────
# One coupling by default, and β = 2.4 specifically: it is the only β with a
# trained operator, so this row is a genuine **diagonal** cell — network and
# ensemble at the same coupling, exactly like every row of the paper's Table 5.
# Any other β would be off-diagonal and would additionally be answering a
# question (does ξ_A follow the ensemble rather than the training β?) that
# needs the full matrix to be worth asking. β = 2.4 is also the anchor of the
# whole §6.2 program — the GEVP plateau m·a_t ≈ 0.33 and the A₀ = 0.903 result
# are quoted there — and it carries ξ_t = 3.00, near the top of the range the
# lattice family reaches.
#
# The scan is still one variable away:
#     SAC_BETAS=2.1,2.3,2.4,2.5,2.7
# which is beta_scan.py's set, so its recorded m·a_t cross-checks the classical
# arm at every point. Note β = 2.7 is NOT monotonic in ξ (it reads the same mass
# as β = 2.4): unlike Z₂'s dropped β = 0.760 this is not a finite-volume
# artefact — ξ_s ≈ 1.0 against L = 12 is L/ξ_s = 12 — but coarse-a_s strong
# coupling plus a bare anisotropy whose renormalisation drifts with β. See
# notes/attention_as_operator.md §9.2 for why it is worth keeping.
BETAS = [float(b) for b in os.environ.get("SAC_BETAS", "2.4").split(",")]

# 1600 because a single-coupling run spends its whole budget on one row: the
# paper's Table 5 diagonal quotes 1200 configurations per ensemble, and every
# error bar here shrinks as 1/√N while the cost is linear. Drop it to ~800 if
# the box is short of time or disk; the fit window will start collapsing toward
# its 4-point minimum below ~400.
N_EVAL = int(os.environ.get("SAC_N_EVAL", 1600))
CHUNK = int(os.environ.get("SAC_CHUNK", 4))
# Neither 0 (Run 5) nor 1/2 (the replication phases), so no cache key can
# collide with a training ensemble and no configuration measured here was ever
# seen by a checkpoint.
ENSEMBLE_SEED = int(os.environ.get("SAC_SEED", 11))
KEEP_CONFIGS = os.environ.get("SAC_KEEP", "1") == "1"
SMOKE = os.environ.get("SAC_SMOKE", "0") == "1"
REPLOT = os.environ.get("SAC_REPLOT", "")

# Lattice / sampler: measure_glueball.py's parameters exactly, so the ensembles
# are drawn from the same distribution the checkpoints were trained on.
L, D, LT, XI = tg.L, tg.D, tg.LT, tg.XI
N_THERM, N_SKIP, N_OR = tg.N_THERM, tg.N_SKIP, tg.N_OR
gaugegroup = tg.gaugegroup

# The classical comparator: the same APE ladder train_glueball.py builds its
# GEVP anchor from. (Z₂ used (0, 4, 8, 16) — a coarser ladder for a coarser ξ.)
SMEAR_LEVELS = tg.GEVP_LEVELS

# The trained arm: the Run-5 β = 2.4 operator. The replication checkpoint
# (`tg.CHECKPOINT.replace(".pth", "_ens1.pth")`, trained on the independent
# seed-1 ensemble) is a second, equally valid trained arm — add it back as one
# more entry here if a replication of the attention measurement is wanted.
TRAINED_CKPTS = {
    "trained": tg.CHECKPOINT,
}
TRAIN_BETA = tg.BETA  # the coupling every checkpoint was trained at

# Everything below is the Z₂ run's, unchanged — see the module docstring.
REDUCTIONS = zac.REDUCTIONS
JACK_BLOCK = zac.JACK_BLOCK
PER_HEAD_BLOCK = zac.PER_HEAD_BLOCK
MIN_REL_FLUCT = zac.MIN_REL_FLUCT
RANDOM_SEED = zac.RANDOM_SEED

OUT_PT = "results/attention/su2_attention_correlator.pt"
OUT_PNG = "results/attention/su2_attention_correlator.png"
OUT_TEX = "results/attention/su2_attention_correlator_table.tex"

# beta_scan.py's recorded classical masses — an independent measurement of the
# same quantity on independently sampled ensembles, printed as a cross-check.
BETA_SCAN = "results/attention/beta_scan.pt"

# SAC_DEVICE forces the backend. MPS cannot run this path at all — the SU(2)
# projection needs complex `linalg.det`/`svd`, which Metal does not implement —
# so a laptop smoke test has to say `SAC_DEVICE=cpu`.
device = torch.device(os.environ.get("SAC_DEVICE") or (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
))


# ── Smoke mode: shrink everything so the plumbing runs in seconds ─────────────
if SMOKE:
    L, LT = 4, 8
    tg.L, tg.LT = L, LT
    tg.INPUT_SMEAR_LEVELS = (0, 2)
    SMEAR_LEVELS = [0, 2]
    BETAS = BETAS[:2]
    # ≥ 2·JACK_BLOCK configs, or the blocked jackknife deletes everything and
    # every arm returns NaN for a reason that has nothing to do with plumbing.
    N_EVAL, CHUNK = 48, 8
    TRAINED_CKPTS = {}
    OUT_PT = "results/attention/su2_attention_correlator_smoke.pt"
    OUT_PNG = "results/attention/su2_attention_correlator_smoke.png"
    OUT_TEX = "results/attention/su2_attention_correlator_smoke.tex"


def build_model(ckpt=None, seed=None):
    """The §6.2 architecture, optionally loaded from a checkpoint.

    Every geometric hyperparameter has to match ``train_glueball.py``'s or the
    offsets — and hence the attention rows — would not line up with the
    checkpoint's. Read from ``tg`` rather than restated for exactly that reason.
    """
    if seed is not None:
        torch.manual_seed(seed)
    model = GELT(
        gaugegroup=gaugegroup, L=L, D=3, R=tg.R, nhead=tg.NHEAD,
        gemhsa_layers=tg.GEMHSA_LAYERS, d_qkv=tg.D_QKV, gate=tg.GATE,
        dtype=tg.MODEL_DTYPE, mlp_hidden=tg.MLP_HIDDEN, mlp_out=1,
        reduction="none", init_scale=tg.INIT_SCALE,
        qk_init_scale=tg.QK_INIT_SCALE, mlp_zero_init=False,
        d_model=tg.D_MODEL, grad_checkpoint=False,
        in_channels=3 * len(tg.INPUT_SMEAR_LEVELS),
    ).to(device)
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model


def channel_labels():
    """Ordering must match :func:`attention_fields`: layer → reduction → head."""
    return [
        f"L{l + 1}h{h}:{r}"
        for l in range(tg.GEMHSA_LAYERS)
        for r in REDUCTIONS
        for h in range(tg.NHEAD)
    ]


@torch.no_grad()
def attention_fields(model, W, T, n_cfg, dist):
    """Zero-momentum attention operators Ā(t) for one configuration batch.

    The 3D analogue of ``z2_attention_correlator.attention_fields``: the spatial
    lattice is L³ rather than L², so the zero-momentum projection sums three
    axes instead of two. Everything else — the three reductions of the attention
    row, the (layer, reduction, head) channel ordering, the site-level moments —
    is identical.

    Returns ``(out, chans, moments)``:

    ``out``     ``(n_cfg, Nt)`` — the network's own glueball operator Ō(t),
                i.e. ``train_glueball.network_obar``. Kept as a reference
                channel and as a check that the checkpoint loaded correctly.
    ``chans``   ``(n_ch, n_cfg, Nt)`` — one zero-momentum series per
                (layer, reduction, head).
    ``moments`` ``(n_ch, 2)`` — running Σ A and Σ A² over *sites*, so the
                site-level fluctuation-to-mean ratio δA/A can be quoted. That
                ratio is the effect size of the whole claim: a fixed
                convolutional kernel's is identically zero.
    """
    site = model(W, T)  # (n_cfg·Lt, L, L, L)
    out = site.reshape(n_cfg, LT, *site.shape[1:]).sum(dim=(-3, -2, -1))

    chans, moments = [], []
    for layer in model.gemhsa_models:
        a = layer._last_alpha  # (n_cfg·Lt, H, n_off, L, L, L)
        red = {
            "self": a[:, :, 0],
            "ell": (a * dist.view(1, 1, -1, 1, 1, 1)).sum(dim=2),
            "ent": -(a.clamp_min(1e-12).log() * a).sum(dim=2),
        }
        for name in REDUCTIONS:
            f = red[name]  # (n_cfg·Lt, H, L, L, L)
            moments.append(
                torch.stack(
                    [f.double().sum(dim=(0, 2, 3, 4)),
                     f.double().pow(2).sum(dim=(0, 2, 3, 4))],
                    dim=1,
                )  # (H, 2)
            )
            f = f.reshape(n_cfg, LT, *f.shape[1:])  # (n_cfg, Lt, H, L, L, L)
            chans.append(f.sum(dim=(-3, -2, -1)).permute(2, 0, 1))  # (H, n_cfg, Lt)
    return out, torch.cat(chans, dim=0), torch.cat(moments, dim=0)


def cache_path(beta):
    """Distinct from every training cache key: different N *and* different seed."""
    return (f"datasets/glueball_configs_L{L}_Lt{LT}_b{beta}_xi{XI}"
            f"_N{N_EVAL}_seed{ENSEMBLE_SEED}.pt")


def load_configs(beta):
    """The evaluation ensemble at this β — cached, or sampled and (maybe) cached.

    Sampling is the expensive half of the run (~20 min per β at N = 800 on a
    V100, from ``logs/replication_ens1.log``'s 3.57 sweeps/s), so the cache is
    worth ~4 GB per β unless the box is short of disk (``SAC_KEEP=0``).
    """
    if SMOKE:
        torch.manual_seed(int(beta * 1e4))
        return torch.stack([
            random_links(L, D, gaugegroup, dtype=tg.MODEL_DTYPE, Lt=LT)
            for _ in range(N_EVAL)
        ])
    path = cache_path(beta)
    if os.path.exists(path):
        print(f"  loading cached {path}")
        return torch.load(path).to(tg.MODEL_DTYPE)
    print(f"  sampling N={N_EVAL} at β={beta}, seed {ENSEMBLE_SEED} "
          f"(the long pole) …")
    # Seed per (β, run seed) so each ensemble is reproducible on its own and no
    # two couplings share a random stream.
    torch.manual_seed(ENSEMBLE_SEED * 1000 + int(round(beta * 100)))
    np.random.seed(ENSEMBLE_SEED)
    sweep = functools.partial(heatbath_overrelaxation_sweep, n_or=N_OR, xi=XI)
    configs, acc = mcmc_ensemble(
        L=L, D=D, gaugegroup=gaugegroup, beta=beta, n_configs=N_EVAL,
        n_therm=N_THERM, n_skip=N_SKIP, sweep_fn=sweep, progress=True, Lt=LT,
    )
    print(f"  acceptance = {acc:.2f}")
    if KEEP_CONFIGS:
        torch.save(configs, path)
        print(f"  cached → {path}")
    return configs.to(tg.MODEL_DTYPE)


def measure_ensemble(beta, nets, labels, dist):
    """Every network's attention operators + the classical basis, one ensemble.

    Structurally identical to the Z₂ version; the differences are that the
    per-config inputs come from ``train_glueball.config_inputs`` (4D config →
    per-timeslice 3D W and T) and that the classical basis is the SU(2) APE
    ladder. Every estimator called on the results is ``zac``'s.
    """
    configs = load_configs(beta)
    n_cfg = len(configs)
    print(f"  {n_cfg} configs, none seen by any checkpoint")

    acc = {name: {"out": [], "chan": []} for name in nets}
    mom = {name: torch.zeros(len(labels), 2, dtype=torch.float64) for name in nets}
    n_site_samples = 0
    classical = []

    t0 = time.time()
    n_chunks = -(-n_cfg // CHUNK)
    for ci, i in enumerate(tqdm(range(0, n_cfg, CHUNK), desc=f"β={beta}",
                                total=n_chunks)):
        batch = configs[i : i + CHUNK]
        b = len(batch)
        # The expensive half — the APE ladder plus the 3D transport — depends on
        # the configuration alone, so it is paid once for all networks.
        W, T = tg.config_inputs(batch, device)
        classical.append(
            smearing_operator_basis(batch.to(device), gaugegroup, SMEAR_LEVELS,
                                    alpha=tg.SMEAR_ALPHA).double().cpu()
        )
        for name, model in nets.items():
            out, chan, moments = attention_fields(model, W, T, b, dist)
            acc[name]["out"].append(out.double().cpu())
            acc[name]["chan"].append(chan.double().cpu())
            mom[name] += moments.cpu()
        n_site_samples += b * LT * L**3
        del W, T
        if ci == 0:
            dt = time.time() - t0
            print(f"    first chunk {dt:.1f}s → ETA {dt * n_chunks / 60:.1f} min "
                  f"for this ensemble ({len(nets)} networks)")

    res = {"beta": beta, "n_cfg": n_cfg, "nets": {}}

    cl = torch.cat(classical, dim=1)  # (n_levels, B, Nt)
    res["classical"] = zac._jack(cl, LT)
    # The same basis read by its best single member, so every arm — classical,
    # trained, random — is quoted under both estimators and the table can be
    # read either way. The GEVP is the variational optimum but it is fitted on
    # the configurations it is scored on; the best single member is not.
    res["classical_single"] = zac._jack_best_single(
        cl, LT, [f"APE{n}" for n in SMEAR_LEVELS])
    xi_c, xi_ce = zac._xi(res["classical"])
    xi_c1, _ = zac._xi(res["classical_single"])
    print(f"  classical APE basis {SMEAR_LEVELS}:  m·a_t = "
          f"{res['classical']['m']:.4f} ± {res['classical']['m_err']:.4f}   "
          f"ξ_t = {xi_c:.2f} ± {xi_ce:.2f}   "
          f"A₀ = {res['classical']['A0']:.3f}   "
          f"[best single {res['classical_single'].get('channel')}: "
          f"ξ_t = {xi_c1:.2f}, A₀ = {res['classical_single']['A0']:.3f}]")
    zac._diagnose("classical", res["classical"])

    gen = torch.Generator().manual_seed(RANDOM_SEED)
    best_series, gevp_series = {}, {}
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

        kept_labels = [labels[j] for j in range(len(labels)) if keep[j]]
        attn_single = zac._jack_best_single(chan[keep], LT, kept_labels)
        # The null must be the SAME operator with its configs scrambled — letting
        # it re-select a best channel out of six makes it positive by
        # construction (the Z₂ run measured that inflation: A₀ 0.005 → 0.23).
        std = zac._standardize(chan[keep].double())
        i_best = attn_single.get("idx")
        null_src = std if i_best is None else std[i_best : i_best + 1]
        entry = {
            "rel_fluct": rel,
            "kept": keep,
            "attention": zac._jack(chan[keep], LT),
            "attention_single": attn_single,
            "output": zac._jack(out, LT),
            "shuffled": zac._jack(zac._shuffle_time(null_src, gen), LT),
            "scrambled": zac._jack(zac._scramble_configs(null_src, gen), LT),
            "per_head": {},
        }
        for l in range(tg.GEMHSA_LAYERS):
            for h in range(tg.NHEAD):
                idx = [labels.index(f"L{l + 1}h{h}:{r}") for r in REDUCTIONS]
                idx = [j for j in idx if keep[j]]
                if idx:
                    entry["per_head"][f"L{l + 1}h{h}"] = zac._jack(
                        chan[idx], LT, block=PER_HEAD_BLOCK
                    )
        res["nets"][name] = entry

        xi_a, xi_ae = zac._xi(entry["attention"])
        xi_s, _ = zac._xi(entry["attention_single"])
        xi_o, _ = zac._xi(entry["output"])
        r_keep = rel[keep]
        print(f"  {name:>14}: ξ_att = {xi_a:6.2f} ± {xi_ae:5.2f}   "
              f"A₀ = {entry['attention']['A0']:.3f} ± "
              f"{entry['attention']['A0_err']:.3f}   "
              f"ξ_1ch = {xi_s:6.2f} ({entry['attention_single'].get('channel')})   "
              f"ξ_out = {xi_o:6.2f}   "
              f"scrambled-null A₀ = {entry['scrambled']['A0']:.4f}")
        print(f"      δA/A over {int(keep.sum())} channels: "
              f"median {r_keep.median():.3f}  min {r_keep.min():.3f}  "
              f"max {r_keep.max():.3f}")
        zac._diagnose(f"{name} attention", entry["attention"])
        zac._diagnose(f"{name} scrambled-NULL", entry["scrambled"])
        # The time-shuffle is a zero-mode consistency check, not a null: it
        # preserves each configuration's mean, so its plateau must sit at
        # (2ξ−1)/Nt.
        if np.isfinite(xi_s):
            plateau = float(np.mean(entry["shuffled"]["profile"][1:8]))
            print(f"      zero-mode check: time-shuffle plateau {plateau:.3f} "
                  f"vs (2ξ−1)/Nt = {(2 * xi_s - 1) / LT:.3f}")

        # The series the correlated difference is taken on is the one the
        # `attention` row quotes — GEVP basis, or the single channel it fell
        # back to. On SU(2) the fallback does *not* fire uniformly (it did at
        # every Z₂ β), so this is where the two definitions part company.
        gevp_series[name] = zac._resolved_series(chan[keep], entry["attention"])
        idx = entry["attention_single"].get("idx")
        if idx is not None:
            best_series[name] = std[idx : idx + 1]
            tau = entry["attention_single"].get("tau_int", float("nan"))
            flag = "" if not np.isfinite(tau) or JACK_BLOCK >= 2 * tau else \
                "  ← block < 2τ_int, errors understated"
            print(f"      τ_int({entry['attention_single']['channel']}) = "
                  f"{tau:.2f} vs jackknife block {JACK_BLOCK}{flag}")

    # Correlated trained − random difference on shared configurations: the
    # significance statement for "training makes the routing a better operator".
    # Taken on the *resolved* bases, so ΔA₀ is exactly the difference of the two
    # A₀ the table quotes in the same row; `delta_vs_random_single` keeps the
    # conditioning-free single-channel version beside it.
    res["delta_vs_random"] = {}
    res["delta_vs_random_single"] = {}
    if "random" in gevp_series:
        e_r = res["nets"]["random"]["attention"]
        for name in gevp_series:
            if name == "random":
                continue
            e_t = res["nets"][name]["attention"]
            dd = zac._corr_delta(gevp_series[name], gevp_series["random"], LT,
                                 wa=e_t.get("window"), wb=e_r.get("window"))
            if dd:
                res["delta_vs_random"][name] = dd
                zac._delta_consistency(dd, e_t, e_r, tag=f"[{name}]")
                sig = dd["dA0"] / dd["dA0_err"] if dd["dA0_err"] else float("nan")
                print(f"  {name} − random (shared configs): "
                      f"ΔA₀ = {dd['dA0']:+.3f} ± {dd['dA0_err']:.3f} ({sig:+.1f}σ)"
                      f"   Δξ = {dd['dxi']:+.2f} ± {dd['dxi_err']:.2f}"
                      f"   [{dd['n_ops'][0]} vs {dd['n_ops'][1]} ops]")
    if "random" in best_series:
        for name in best_series:
            if name == "random":
                continue
            ds = zac._corr_delta(best_series[name], best_series["random"], LT)
            if ds:
                res["delta_vs_random_single"][name] = ds
                sig = ds["dA0"] / ds["dA0_err"] if ds["dA0_err"] else float("nan")
                print(f"  {name} − random, best single channel: "
                      f"ΔA₀ = {ds['dA0']:+.3f} ± {ds['dA0_err']:.3f} ({sig:+.1f}σ)")
    return res


# ── Reporting ─────────────────────────────────────────────────────────────────
def _arm(rows, name, key="attention", what="xi"):
    """(values, errors) of one arm across ensembles, NaN where it did not resolve."""
    vals, errs = [], []
    for r in rows:
        e = r["nets"].get(name)
        if e is None or e.get("dead"):
            vals.append(np.nan)
            errs.append(np.nan)
        elif what == "xi":
            v, s = zac._xi(e[key])
            vals.append(v)
            errs.append(s)
        else:
            vals.append(e[key]["A0"])
            errs.append(e[key]["A0_err"])
    return np.array(vals, dtype=float), np.array(errs, dtype=float)


def _pearson_slope(x, y):
    """Pearson r and the least-squares slope of y on x, over the finite pairs."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    xr, yr = x[ok], y[ok]
    r = float(np.corrcoef(xr, yr)[0, 1])
    return r, float(np.polyfit(xr, yr, 1)[0])


def _fmt(v, e, digits=2):
    """``2.24(12)`` — the paper's Table 5 format."""
    if not np.isfinite(v):
        return "—"
    if not np.isfinite(e):
        return f"{v:.{digits}f}"
    scale = 10 ** digits
    return f"{v:.{digits}f}({int(round(e * scale)):d})"


def profile_block(r, names):
    """Everything a single row of Table 5 carries beyond its five numbers.

    The paper prints exactly this next to the table for its largest-ξ ensemble:
    the normalized correlator C(Δ)/C(0) of each arm, where "the trained
    attention field is a visibly cleaner exponential" is read off directly; the
    time-shuffle plateau, which is a zero-mode *consistency check* and must sit
    at (2ξ_A − 1)/N_t rather than at zero; and the config-scramble, which is the
    actual null and must not fit at all. The site-level δA/A is added because it
    is the effect size of the whole claim — a fixed convolutional kernel's is
    identically zero, so a nonzero value is what makes the measurement exist.
    """
    print(f"\n  β = {r['beta']}  ({r['n_cfg']} configs) — correlator profiles, "
          f"C(Δ)/C(0):")
    hdr = "    " + f"{'arm':>26}" + "".join(f"{d:>9}" for d in range(1, 6))
    print(hdr)
    rowsrc = [("classical basis", r["classical"])]
    for n in names:
        e = r["nets"].get(n)
        if e and not e.get("dead"):
            rowsrc.append((n, e["attention"]))
    tr = names[0] if names and names[0] != "random" else None
    if tr and tr in r["nets"] and not r["nets"][tr].get("dead"):
        rowsrc.append(("time-shuffle (zero mode)", r["nets"][tr]["shuffled"]))
        rowsrc.append(("config-scramble (null)", r["nets"][tr]["scrambled"]))
    for label, src in rowsrc:
        prof = src.get("profile") or []
        cells = "".join(f"{prof[d]:>+9.3f}" if d < len(prof) else f"{'—':>9}"
                        for d in range(1, 6))
        print(f"    {label:>26}{cells}")

    for n in names:
        e = r["nets"].get(n)
        if not e or e.get("dead"):
            continue
        rel = e["rel_fluct"][e["kept"]]
        single = e["attention_single"]
        print(f"    {n}: δA/A median {rel.median():.3f} over "
              f"{int(e['kept'].sum())} channels | best single channel "
              f"{single.get('channel')} (ξ = {zac._xi(single)[0]:.2f}) | "
              f"network's own operator ξ = {zac._xi(e['output'])[0]:.2f} | "
              f"scramble-null A₀ = {e['scrambled']['A0']:.4f}")


def report(rows):
    betas = np.array([r["beta"] for r in rows], dtype=float)
    xi_c, xi_ce = np.array([zac._xi(r["classical"])[0] for r in rows]), \
        np.array([zac._xi(r["classical"])[1] for r in rows])
    A_c, _ = np.array([r["classical"]["A0"] for r in rows]), None
    names = [n for n in list(TRAINED_CKPTS) + ["random"]
             if any(n in r["nets"] for r in rows)]

    print("\n" + "=" * 96)
    print("Table 5, SU(2): ξ from the attention field vs the classical smeared "
          "basis (same configurations)")
    print("=" * 96)
    # Every β is printed twice, once per estimator, and each line is internally
    # consistent: ξ, A₀ and ΔA₀ on one line all come from the same operator, so
    # the A₀ column subtracts to the ΔA₀ column within a line and never across.
    hdr = (f"{'β':>6} {'est':>7} {'ξ_class (a_t)':>16} "
           + " ".join(f"{n:>16}" for n in names)
           + f" {'A₀ tr/rnd/cl':>20} {'ΔA₀ (trained−rnd)':>22}")
    print(hdr)
    print("-" * len(hdr))
    for j, r in enumerate(rows):
        for est, ckey, akey, dkey in (
            ("GEVP", "classical", "attention", "delta_vs_random"),
            ("single", "classical_single", "attention_single",
             "delta_vs_random_single"),
        ):
            cl = r.get(ckey)
            if cl is None:  # a dump written before both estimators were kept
                continue
            xv, xe = zac._xi(cl)
            line = f"{r['beta']:>6} {est:>7} {_fmt(xv, xe):>16} "
            for n in names:
                v, e = _arm([r], n, key=akey)
                line += f"{_fmt(v[0], e[0]):>16} "
            a_tr = (_arm([r], names[0], key=akey, what="A0")[0][0]
                    if names else float("nan"))
            a_rn = _arm([r], "random", key=akey, what="A0")[0][0]
            line += f"{a_tr:>6.2f}/{a_rn:.2f}/{cl['A0']:.2f} "
            dd = r.get(dkey, {}).get(names[0] if names else "")
            line += (f"{dd['dA0']:>+13.3f}({int(round(dd['dA0_err'] * 1000))})"
                     if dd else f"{'—':>18}")
            print(line)

    print("-" * len(hdr))

    # The per-ensemble detail the paper quotes inline next to Table 5: the
    # normalized correlator profiles, the site-level fluctuation ratio, and the
    # two control arms. This is the whole content of a *row*, and unlike the
    # aggregates below it is complete at a single coupling.
    for r in rows:
        profile_block(r, names)

    # Aggregates across couplings. A Pearson and a slope over one or two points
    # are not statistics, so they are not printed as if they were — with a
    # single β the row-level numbers above are the entire result, and saying so
    # is better than emitting a NaN that looks like a failed measurement.
    if len(rows) >= 3:
        print("-" * len(hdr))
        for est, ckey, akey in (("GEVP", "classical", "attention"),
                                ("single", "classical_single", "attention_single")):
            if any(r.get(ckey) is None for r in rows):
                continue
            xi_ref = np.array([zac._xi(r[ckey])[0] for r in rows])
            for n in names:
                xi_a, _ = _arm(rows, n, key=akey)
                r_p, slope = _pearson_slope(xi_ref, xi_a)
                print(f"  [{est:>6}] Pearson(ξ_A[{n}], ξ_class) = {r_p:.4f}   "
                      f"slope = {slope:.2f}")
        ok = np.isfinite(xi_c) & (xi_c > 0)
        if ok.sum() > 1:
            print(f"  classical dynamic range: "
                  f"×{xi_c[ok].max() / xi_c[ok].min():.2f} "
                  f"(ξ_t = {xi_c[ok].min():.2f} … {xi_c[ok].max():.2f} "
                  f"temporal spacings)")
    else:
        print("-" * len(hdr))
        print(f"  {len(rows)} coupling(s): Pearson(ξ_A, ξ_class), the slope and the")
        print("  dynamic range are cross-β aggregates and are not reported. The")
        print("  single-β claims are the ones above: ξ_A agrees with the classical")
        print("  ξ on the same configurations, ΔA₀ > 0 against the random arm, and")
        print("  the config-scramble null does not fit.")

    # Independent cross-check: beta_scan.py measured the same classical mass on
    # its own ensembles with the GEVP m_eff plateau rather than a cosh fit. The
    # two need not agree exactly (different estimator, different configs) but a
    # large discrepancy would mean this script's classical arm is wrong.
    if os.path.exists(BETA_SCAN):
        bs = torch.load(BETA_SCAN, map_location="cpu", weights_only=False)
        ref = dict(zip(bs["betas"], bs["m_at"]))
        pairs = [(b, ref[b], r["classical"]["m"]) for b, r in zip(betas, rows)
                 if b in ref and np.isfinite(r["classical"]["m"])]
        if pairs:
            print("\n  cross-check vs results/attention/beta_scan.pt "
                  "(GEVP m_eff plateau, independent ensembles):")
            for b, m_ref, m_here in pairs:
                print(f"    β={b}: m·a_t = {m_here:.3f} here vs {m_ref:.3f} "
                      f"there ({100 * (m_here - m_ref) / m_ref:+.0f}%)")


def write_tex(rows):
    """A paste-ready Table 5 in the paper's column layout."""
    names = [n for n in list(TRAINED_CKPTS) + ["random"]
             if any(n in r["nets"] for r in rows)]
    tr = names[0] if names and names[0] != "random" else None
    lines = [
        r"% generated by scripts/su2_attention_correlator.py — do not hand-edit",
        r"\begin{table}[t]",
        r"\caption{Correlation lengths and ground-state overlaps of the attention",
        rf" field on ${rows[0]['n_cfg']}$ configurations per ensemble, none seen by any",
        r" checkpoint, against the classical APE basis measured on the same",
        r" configurations and against a randomly initialized network of identical",
        rf" architecture. Anisotropic SU(2), $L^3\times N_t = {L}^3\times{LT}$,",
        rf" $\xi_{{\rm bare}}={XI}$; lengths in temporal lattice spacings $a_t$. The",
        rf" trained arm is the $\beta={TRAIN_BETA}$ operator of Sec.~\ref{{sec:spectroscopy}}",
        r" evaluated on every ensemble (matched only in its own row).",
        r" Each coupling is quoted under both estimators: the multi-channel",
        r" GEVP, which is the variational optimum but is fitted on the",
        r" configurations it is scored on, and the best single member of the",
        r" same basis, which is not. A line is internally consistent —",
        r" $\Delta A_0$ is a blocked jackknife of the trained $-$ random",
        r" \emph{difference} on shared configurations, taken on the two",
        r" operators that line quotes, so it is their difference.}",
        r"\label{tab:su2attn}",
        r"\begin{ruledtabular}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\begin{tabular}{llccccc}",
        r"$\beta$ & est. & $\xi_{\rm class}$ & $\xiA$ trained & $\xiA$ random"
        r" & $A_0$ tr./rnd./cl. & $\Delta A_0$ \\",
        r"\colrule",
    ]
    ESTIMATORS = (
        ("GEVP", "classical", "attention", "delta_vs_random"),
        ("single", "classical_single", "attention_single",
         "delta_vs_random_single"),
    )
    for j, r in enumerate(rows):
        for i, (est, ckey, akey, dkey) in enumerate(ESTIMATORS):
            cl = r.get(ckey)
            if cl is None:  # a dump written before both estimators were kept
                continue
            xi_c, xi_ce = zac._xi(cl)
            nan = (np.array([np.nan]), np.array([np.nan]))
            v_t, e_t = _arm([r], tr, key=akey) if tr else nan
            v_r, e_r = _arm([r], "random", key=akey)
            a_t = _arm([r], tr, key=akey, what="A0")[0][0] if tr else float("nan")
            a_r = _arm([r], "random", key=akey, what="A0")[0][0]
            dd = r.get(dkey, {}).get(tr)
            # A table generated from a dump written before the difference moved
            # onto the resolved bases would silently print a single-channel ΔA₀
            # beside a multi-channel A₀ column. Say so rather than emit it
            # quietly.
            if dd and tr:
                zac._delta_consistency(dd, r["nets"][tr][akey],
                                       r["nets"]["random"][akey],
                                       tag=f"[β={r['beta']:g}, {est}, stale dump?]")
            d = (f"${dd['dA0']:+.3f}({int(round(dd['dA0_err'] * 1000))})$"
                 if dd else "---")
            head = f"${r['beta']:g}$" if i == 0 else ""
            lines.append(
                f"{head} & {est} & ${_fmt(xi_c, xi_ce)}$ "
                f"& ${_fmt(v_t[0], e_t[0])}$ "
                f"& ${_fmt(v_r[0], e_r[0])}$ & ${a_t:.2f}/{a_r:.2f}/"
                f"{cl['A0']:.2f}$ & {d} \\\\"
            )
    lines += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table}", ""]
    # encoding is explicit: the V100 box runs with an ASCII default locale, and
    # the generated header carries an em dash, so the write died there after a
    # 1.5 h measurement had already succeeded.
    with open(OUT_TEX, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nwrote {OUT_TEX}")


def plot_single(rows):
    """Figure for a one-coupling run.

    The scan figure's two headline panels are a scatter of ξ_A against ξ_class
    and a curve over β; with one ensemble both are a single point and say
    nothing. What a single row *does* support is the three comparisons made on
    the same configurations — the correlator shapes, the two lengths, and the
    two overlaps — so the panels are those, with the controls drawn in.
    """
    r = rows[0]
    names = [n for n in list(TRAINED_CKPTS) + ["random"] if n in r["nets"]]
    tr = names[0] if names and names[0] != "random" else None
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    # (0) the correlator shapes. The claim is visible here before any fit: the
    # trained arm should decay more slowly and more cleanly than the classical
    # basis and the untrained network measured on the same configurations.
    a = ax[0]
    a.plot(r["classical"].get("profile") or [], "o-", color="k",
           label="classical APE basis")
    for n, fmt in zip(names, ["s--", "^--", "v--"]):
        e = r["nets"][n]
        if not e.get("dead") and e["attention"].get("profile"):
            a.plot(e["attention"]["profile"], fmt, alpha=0.85, label=n)
    if tr and not r["nets"][tr].get("dead"):
        a.plot(r["nets"][tr]["shuffled"].get("profile") or [], ":", color="gray",
               label="time-shuffle (zero mode)")
        a.plot(r["nets"][tr]["scrambled"].get("profile") or [], "x:",
               color="crimson", label="config-scramble (null)")
    a.axhline(0.0, color="gray", lw=0.5)
    a.set_xlabel(r"$\Delta t$")
    a.set_ylabel(r"$C(\Delta)/C(0)$")
    a.set_title(rf"SU(2) $\beta$ = {r['beta']}: the attention field is an operator")
    a.legend(fontsize=8)

    # (1) and (2): the two numbers of the row, arm by arm, with the classical
    # basis drawn as a band because it is the comparator rather than a category.
    labels = ["classical"] + names
    xi_v = [zac._xi(r["classical"])[0]] + [zac._xi(r["nets"][n]["attention"])[0]
                                           for n in names]
    xi_e = [zac._xi(r["classical"])[1]] + [zac._xi(r["nets"][n]["attention"])[1]
                                           for n in names]
    a0_v = [r["classical"]["A0"]] + [r["nets"][n]["attention"]["A0"] for n in names]
    a0_e = [r["classical"]["A0_err"]] + [r["nets"][n]["attention"]["A0_err"]
                                         for n in names]
    # The same three arms read by the best single member of each basis, drawn
    # hollow beside the filled GEVP points: the gap between the two markers of
    # one arm is what the variational step bought that arm, and it is much
    # larger for a random network's redundant channels than for a trained one's.
    cl1 = r.get("classical_single")
    if cl1 is not None:
        xi1_v = [zac._xi(cl1)[0]] + [zac._xi(r["nets"][n]["attention_single"])[0]
                                     for n in names]
        xi1_e = [zac._xi(cl1)[1]] + [zac._xi(r["nets"][n]["attention_single"])[1]
                                     for n in names]
        a01_v = [cl1["A0"]] + [r["nets"][n]["attention_single"]["A0"] for n in names]
        a01_e = [cl1["A0_err"]] + [r["nets"][n]["attention_single"]["A0_err"]
                                   for n in names]
    else:
        xi1_v = xi1_e = a01_v = a01_e = None

    y = np.arange(len(labels))
    for a, v, e, v1, e1, xlabel, title in (
        (ax[1], xi_v, xi_e, xi1_v, xi1_e,
         r"$\xi_t = 1/(m\,a_t)$  [temporal spacings]",
         "Correlation length, same configurations"),
        (ax[2], a0_v, a0_e, a01_v, a01_e,
         r"$A_0$  [ground-state overlap fraction]", "Operator quality"),
    ):
        a.errorbar(v, y, xerr=e, fmt="o", capsize=4, color="tab:blue",
                   label="GEVP basis")
        if v1 is not None:
            a.errorbar(v1, y + 0.16, xerr=e1, fmt="o", capsize=4, mfc="none",
                       color="tab:orange", label="best single member")
        a.axvline(v[0], color="k", ls=":", lw=1)
        if np.isfinite(e[0]):
            a.axvspan(v[0] - e[0], v[0] + e[0], color="k", alpha=0.08)
        a.set_yticks(y, labels)
        a.invert_yaxis()
        a.set_xlabel(xlabel)
        a.set_title(title)
        a.legend(fontsize=8)
    # The correlated trained − random difference goes in the title rather than
    # inside the axes: it is the significance statement of the panel, and the
    # only empty region of a 4-row point plot is wherever the next run's error
    # bars happen not to be.
    dd = r.get("delta_vs_random", {}).get(tr) if tr else None
    ds = r.get("delta_vs_random_single", {}).get(tr) if tr else None
    if dd and dd["dA0_err"]:
        sub = ""
        if ds and ds["dA0_err"]:
            sub = (rf", single {ds['dA0']:+.3f} ± {ds['dA0_err']:.3f} "
                   rf"({ds['dA0'] / ds['dA0_err']:+.1f}$\sigma$)")
        ax[2].set_title(
            "Operator quality\n"
            rf"$\Delta A_0$ GEVP {dd['dA0']:+.3f} ± {dd['dA0_err']:.3f} "
            rf"({dd['dA0'] / dd['dA0_err']:+.1f}$\sigma$){sub}, correlated"
        )

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"wrote {OUT_PNG}")


def plot(rows):
    if len(rows) == 1:
        return plot_single(rows)
    betas = [r["beta"] for r in rows]
    xi_c, xi_ce = np.array([zac._xi(r["classical"])[0] for r in rows]), \
        np.array([zac._xi(r["classical"])[1] for r in rows])
    names = [n for n in list(TRAINED_CKPTS) + ["random"]
             if any(n in r["nets"] for r in rows)]
    tr = names[0] if names and names[0] != "random" else None

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # (0,0) the headline: ξ from the attention field against ξ from the classical
    # basis, on the same configurations.
    a = ax[0, 0]
    all_xi = [xi_c] + [_arm(rows, n)[0] for n in names]
    lim = [0, zac._finite_max(*all_xi) * 1.25]
    a.plot(lim, lim, ls=":", color="gray", label="y = x")
    for n, fmt in zip(names, ["o", "^", "s"]):
        v, e = _arm(rows, n)
        r_p, slope = _pearson_slope(xi_c, v)
        a.errorbar(xi_c, v, xerr=xi_ce, yerr=e, fmt=fmt, capsize=4, alpha=0.85,
                   label=f"{n} (r = {r_p:.3f}, slope {slope:.2f})")
    if tr:
        for xc, ym, b in zip(xi_c, _arm(rows, tr)[0], betas):
            if np.isfinite(ym):
                a.annotate(f"{b}", (xc, ym), fontsize=8, xytext=(4, 4),
                           textcoords="offset points")
    a.set_xlabel(r"$\xi_t$ — classical APE basis")
    a.set_ylabel(r"$\xi_A$ — attention field")
    a.set_title("SU(2): the attention field carries the mass gap")
    a.set_xlim(lim)
    a.set_ylim(lim)
    a.legend(fontsize=8)

    # (0,1) both lengths against β. Not a scaling plot — SU(2) has no critical
    # point here — but it shows the arms tracking each other coupling by
    # coupling, including through β = 2.7's non-monotonic point.
    a = ax[0, 1]
    a.errorbar(betas, xi_c, yerr=xi_ce, fmt="o-", capsize=4, color="k",
               label="classical")
    for n, fmt in zip(names, ["s--", "^--", "v--"]):
        v, e = _arm(rows, n)
        a.errorbar(betas, v, yerr=e, fmt=fmt, capsize=4, alpha=0.85, label=n)
    if os.path.exists(BETA_SCAN):
        bs = torch.load(BETA_SCAN, map_location="cpu", weights_only=False)
        m = np.array(bs["m_at"], dtype=float)
        a.plot(bs["betas"], 1.0 / m, "x:", color="gray",
               label="beta_scan.py GEVP (indep. ensembles)")
    a.set_xlabel(r"$\beta$")
    a.set_ylabel(r"$\xi_t = 1/(m\,a_t)$")
    a.set_title("Both operators, coupling by coupling")
    a.legend(fontsize=8)

    # (1,0) operator quality: A₀ per arm, with the config-scramble null that
    # must sit at zero.
    a = ax[1, 0]
    a.plot(betas, [r["classical"]["A0"] for r in rows], "o-", color="k",
           label="classical")
    arm_color = {}
    for n, fmt in zip(names, ["s--", "^--", "v--"]):
        v, e = _arm(rows, n, what="A0")
        cont = a.errorbar(betas, v, yerr=e, fmt=fmt, capsize=4, alpha=0.85,
                          label=n)
        arm_color[n] = cont.lines[0].get_color()
    # The same arms read by their best single member, hollow and in the arm's
    # own colour: the gap between a filled and a hollow marker of one arm is
    # what the variational step bought it, and the random network — 24
    # redundant noisy channels — gains most.
    if all(r.get("classical_single") is not None for r in rows):
        a.plot(betas, [r["classical_single"]["A0"] for r in rows], "o:",
               color="k", mfc="none", alpha=0.6, label="best single member")
        for n, fmt in zip(names, ["s:", "^:", "v:"]):
            v, e = _arm(rows, n, key="attention_single", what="A0")
            a.errorbar(betas, v, yerr=e, fmt=fmt, capsize=3, mfc="none",
                       alpha=0.6, color=arm_color[n])
    if tr:
        null = [r["nets"][tr]["scrambled"]["A0"] if tr in r["nets"]
                and not r["nets"][tr].get("dead") else np.nan for r in rows]
        a.plot(betas, null, "x:", color="crimson", label="config-scramble null")
    a.axhline(0.0, color="gray", lw=0.5)
    a.set_xlabel(r"$\beta$")
    a.set_ylabel(r"$A_0$ — ground-state overlap fraction")
    a.set_title("How good an operator the attention is")
    a.legend(fontsize=8)

    # (1,1) the normalized correlator at the largest classical ξ — the profile
    # the paper quotes inline. A cleaner exponential is the whole claim.
    a = ax[1, 1]
    j = int(np.nanargmax(xi_c)) if np.isfinite(xi_c).any() else 0
    r = rows[j]
    prof = r["classical"].get("profile")
    if prof:
        a.plot(range(len(prof)), prof, "o-", color="k", label="classical basis")
    for n, fmt in zip(names, ["s--", "^--", "v--"]):
        e = r["nets"].get(n)
        if e and not e.get("dead") and e["attention"].get("profile"):
            p = e["attention"]["profile"]
            a.plot(range(len(p)), p, fmt, alpha=0.85, label=n)
    if tr and tr in r["nets"] and r["nets"][tr].get("shuffled", {}).get("profile"):
        p = r["nets"][tr]["shuffled"]["profile"]
        a.plot(range(len(p)), p, ":", color="gray", label="time-shuffle (zero mode)")
    a.axhline(0.0, color="gray", lw=0.5)
    a.set_xlabel(r"$\Delta t$")
    a.set_ylabel(r"$C(\Delta)/C(0)$")
    a.set_title(rf"Correlator profiles at $\beta$ = {r['beta']} (largest $\xi$)")
    a.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"wrote {OUT_PNG}")


def replot(path):
    """Re-report and re-plot a saved dump offline — no GPU, no ensembles."""
    global OUT_PNG, OUT_TEX
    d = torch.load(path, map_location="cpu", weights_only=False)
    rows = d["rows"]
    OUT_PNG = path.replace(".pt", ".png")
    OUT_TEX = path.replace(".pt", "_table.tex")
    report(rows)
    write_tex(rows)
    plot(rows)


def main():
    if REPLOT:
        for p in REPLOT.split(","):
            replot(p.strip())
        return

    print(f"device: {device} | anisotropic SU(2) {L}³×{LT}, ξ_bare = {XI} | "
          f"R = {tg.R} | N_EVAL = {N_EVAL} | seed {ENSEMBLE_SEED}"
          + ("  [SMOKE]" if SMOKE else ""))
    if not SMOKE and KEEP_CONFIGS:
        gb = N_EVAL * 4 * LT * L**3 * 4 * 8 / 1e9
        print(f"  caching ensembles: ≈{gb:.1f} GB per β, "
              f"{gb * len(BETAS):.1f} GB total (SAC_KEEP=0 to disable)")

    nets = {}
    for name, ck in TRAINED_CKPTS.items():
        if os.path.exists(ck):
            nets[name] = build_model(ckpt=ck)
            print(f"  loaded {name} ← {ck}")
        else:
            print(f"  no checkpoint {ck} — arm {name!r} absent")
    nets["random"] = build_model(seed=RANDOM_SEED)
    if len(nets) < 2 and not SMOKE:
        raise SystemExit(
            "no trained checkpoint found — the trained arm is the whole point. "
            f"Expected one of {list(TRAINED_CKPTS.values())}."
        )

    labels = channel_labels()
    offsets = nets["random"].gemhsa_models[0].offsets
    # Real, not MODEL_DTYPE: the score is Re Tr[Q†K̃] so α is a real softmax, and
    # a complex weight here would silently promote the whole reduction.
    dist = torch.tensor([sum(abs(c) for c in o) for o in offsets],
                        dtype=torch.float32, device=device)
    print(f"{len(nets)} networks | {len(labels)} attention channels "
          f"| {len(offsets)} offsets")

    rows = []
    for beta in BETAS:
        print(f"\n── ensemble β = {beta} " + "─" * 40)
        rows.append(measure_ensemble(beta, nets, labels, dist))
        torch.save({"rows": rows, "labels": labels, "betas": BETAS,
                    "meta": {"L": L, "Lt": LT, "xi": XI, "R": tg.R,
                             "N_EVAL": N_EVAL, "seed": ENSEMBLE_SEED,
                             "fit_window": zac.FIT_WINDOW,
                             "jack_block": JACK_BLOCK,
                             "reductions": REDUCTIONS,
                             "smear_levels": SMEAR_LEVELS,
                             "train_beta": TRAIN_BETA,
                             "checkpoints": TRAINED_CKPTS}}, OUT_PT)
        print(f"  saved partial → {OUT_PT} ({len(rows)} ensembles)")

    report(rows)
    write_tex(rows)
    plot(rows)


if __name__ == "__main__":
    main()
