"""GELT as a learned variational 0⁺⁺ glueball operator (deliverable §6.2).

Trains ``GELT(reduction="none")`` on a **multi-Δ Rayleigh loss**
``−mean_Δ C(Δ)/C(0)`` (Δ ∈ ``LOSS_DELTAS``). The per-timeslice transfer matrix
bounds every ratio by e^{−m₀·Δ}, so each converged ratio *is* a
glueball-mass estimate and the trained network *is* the
optimal-overlap interpolating operator (notes/glueball_spectroscopy.md §1). The
result is validated against the classical multi-level GEVP plateau
``m·a_t ≈ 0.33`` measured by ``scripts/measure_glueball.py`` on the same cached
anisotropic ensemble.

This script implements the **audit (2026-07-01)** "Revised §6.2 checklist". The
binding constraint is that GELT must be a **per-timeslice 3D** operator: the
Rayleigh/transfer-matrix argument requires Ō(t) to depend on the fields of
timeslice ``t`` only. So each config's *spatial* links at fixed time are fed as a
``D=3`` configuration (spatial-plaquette input + 3D L1-ball transport), batched
over (config × timeslice). Letting the network see temporal links/plaquettes
would give O(t) a temporal receptive field, void the variational upper bound, and
make the loss gameable toward m → 0 (audit "Critical" item 1). Concretely:

  * ``GELT(D=3, L=12, reduction="none", mlp_zero_init=False)``. ``mlp_zero_init``
    MUST be False: a zero-init readout makes the output identically 0, and since
    C(0), C(1) are both quadratic in the output the Rayleigh gradient at O ≡ 0 is
    *exactly* zero — training never leaves the saddle (audit item 3).
  * Input ``W`` = ``plaquette_tensor`` of the 3D slice (3 spatial planes), and
    transport ``T`` = ``build_transport_average`` of the 3D slice, both computed
    **on the fly per batch** (cheap at 12³ — audit item 4; no precomputed-T
    dataset, and the 4D transport memory wall is sidestepped).

Every reported mass is a **blocked jackknife on held-out configs only** (audit
item 4 / "Moderate"): a network optimizing the *empirical* C(1)/C(0) overfits the
noise, so the variational bound can be spuriously violated on the training
sample; and residual autocorrelation on the anisotropic ensemble is not fully
characterized, so we delete blocks (~10 configs) rather than single configs.

Run:
    python scripts/train_glueball.py
"""

import functools
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from gelt.blocks_rope import GELT
from gelt.glueball import (
    ape_smear,
    connected_correlator,
    effective_mass,
    smearing_operator_basis,
    connected_correlator_matrix,
    gevp_eigenvalues,
    gevp_effective_mass,
)
from gelt.lattice import SU, build_transport_average, plaquette_tensor
from gelt.sampler import heatbath_overrelaxation_sweep, mcmc_ensemble


# ── Tunables ──────────────────────────────────────────────────────────────────
# Ensemble parameters — these MUST match scripts/measure_glueball.py exactly so
# the cache key resolves to the same file and the classical GEVP anchor (measured
# there) applies to this ensemble.
L = 12  # spatial extent (the lattice is LT × L^(D-1))
D = 4  # 3+1 dimensions; time is lattice axis 0
BETA = 2.4  # SU(2) coupling; on the anisotropic lattice β_t = β·ξ, β_s = β/ξ
N_CONFIGS = 2000
N_THERM = 300
N_SKIP = 5
N_OR = 4  # overrelaxation sweeps per heat-bath sweep
XI = 3.0  # bare anisotropy a_s/a_t (>1 ⇒ finer time)
LT = 2 * L  # temporal extent (time = axis 0)
gaugegroup = SU(2)
NC = gaugegroup.nc

# Same cache key as measure_glueball.py — resolves to the identical file.
CACHE = f"datasets/glueball_configs_L{L}_Lt{LT}_b{BETA}_xi{XI}_N{N_CONFIGS}.pt"

# GELT / training hyperparameters.
R = 2  # L1-ball radius of the (3D) transport — the "smearing level" budget (§7)
GEMHSA_LAYERS = 4  # the value path is bilinear, so loop degree doubles per
#                    layer: 4 layers reach degree ≤ 16 in the plaquettes. At 3
#                    (≤ 8) the operator demonstrably lost to APE ×6, whose
#                    branched staple content is far deeper within the same
#                    radius-6 support.
NHEAD = 2  # 2 heads: still tiny (~5k params) but feeds head-specialization study
D_QKV = 6  # even (RoPE) and ≥ 2·D = 6 so every spatial axis gets a real rotation
#            (caveat 3: pair_axis = p % D leaves axes unrotated when d_qkv < 2D)
D_MODEL = 16
MLP_HIDDEN = 32
INIT_SCALE = 10.0
QK_INIT_SCALE = 1.0
GATE = "softplus"
MODEL_DTYPE = torch.complex64

LR = 3e-3
WEIGHT_DECAY = 1e-3  # AdamW decoupled decay: the unregularized run's val curve
#                      turned up at ~epoch 10 while train kept falling (overfit
#                      on ~1400 configs) — this is the cheap counter before
#                      sampling a bigger ensemble.
# This loss converges in tens of steps on a ~5k-param model — 60 epochs at
# ~175 steps/epoch (batch 8) is ~10k steps, ample. (The earlier 400/40 was tuned
# for the 160k-step regime the StepLR(150) assumed; both were an order of
# magnitude oversized.) 40 → 60 with AdamW: weight decay shifts the val minimum
# later than the unregularized run's ~epoch-10 turn-up.
EPOCHS = 60
PATIENCE = 10
BATCH_CONFIGS = 6  # configs per minibatch; each expands to BATCH_CONFIGS·LT 3D
#                    slices through the network. This is the memory knob (T and
#                    the per-layer K/V scale with it) AND the VEV-estimate knob
#                    (the batch mean ⟨Ō⟩ in the loss is noisier for small
#                    batches — the §4 ratio-estimator bias). V100 profile at
#                    4 layers WITH grad checkpointing: batch 4 = 17 GiB,
#                    batch 6 ≈ 25 GiB (fits 32 GiB), batch 8 OOMs. Without
#                    checkpointing even batch 4 OOMs (~7.4 GiB acts/config).
#                    Check with profile_glueball_step.py.
GRAD_CHECKPOINT = True  # recompute each GEMHSA layer in backward (~×layers cut
#                         in stored activations for ~one extra forward) — the
#                         stored K/V neighbourhoods, not the transport build,
#                         are the memory wall (V100 profile: transport 1.9%).
EPS = 1e-8  # floor for the Rayleigh denominator C(0): guards the
#             constant-operator collapse C(0) → 0 (§4 pitfall 3). C(0) is the
#             ONLY denominator — chained ratios C(Δ)/C(Δ−1) put the *signed*
#             batch estimate C(1) downstairs, and on a 6-config batch its noise
#             (~C(0)/√(b·Lt)) crosses zero routinely: the clamped ratio
#             C(2)/EPS ~ ±1e15 stayed finite (so the isfinite guard never
#             fired) and its 1/EPS-scaled gradient — clipped to norm 1 but pure
#             garbage in *direction* — pumped the operator scale to C(0)~5e8
#             (the epoch-14 train=2e14 blowup).
LOSS_DELTAS = (1, 2)  # Rayleigh ratios C(Δ)/C(0) entering the loss. The
#                       per-timeslice operator keeps the transfer matrix
#                       positive, so EVERY ratio is bounded by e^{−m₀·Δ}; the
#                       Δ=2 term makes training select for the reported
#                       m_eff(Δ≥1) plateau instead of only the
#                       most-contaminated Δ=0→1 ratio.
SCALE_REG = 1e-2  # coefficient of the (log C(0))² scale pin. The ratios
#                   C(Δ)/C(0) are invariant under Ō → λŌ, so the loss has a
#                   scale-flat direction and nothing stops the operator scale
#                   from drifting. With thin inputs the drift stayed benign;
#                   the smeared-input channels are smooth and site-coherent,
#                   the attention sums add constructively, and the bilinear
#                   value path squares that gain per layer — the first
#                   [0,2,4,6] run hit C(0) ~ 1e73 (Ō ~ 1e36, float32 edge) by
#                   epoch 2: non-finite grads, val=nan, and since nan never
#                   improves best_val_loss, NO checkpoint was ever saved. The
#                   (log C(0))² term is variationally neutral (ratios
#                   untouched) and restores C(0) → 1.

# Contiguous three-way split of the *chain-ordered* ensemble (NOT shuffled).
# The ensemble is MCMC-ordered, so neighbouring configs are autocorrelated;
# a random split would (a) leak train↔held correlations and (b) defeat the
# blocked jackknife, which only removes autocorrelation on chain-ordered data.
# Train fits the operator; VAL selects the checkpoint; TEST is untouched until
# the final report — so the reported mass is not biased low by model selection.
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.1  # (TEST_FRACTION = 1 − TRAIN − VAL = 0.2)
JACK_BLOCK = 10  # blocked-jackknife block size (configs) for reported masses
GEVP_LEVELS = [0, 2, 4, 6]  # classical anchor smearing basis (matches measure_)
GEVP_T0 = 1
SMEAR_ALPHA = 0.5
INPUT_SMEAR_LEVELS = (0, 2, 4, 6)  # cumulative APE levels of the plaquette
#   channels fed to GELT; (0,) is the original thin-link input. Stacking several
#   levels on the channel axis (in_channels = 3·n_levels) hands the network the
#   deep branched staple content it cannot rebuild from thin links at depth 4
#   (Run 4 in the notes: loop degree ≤ 16 vs APE×6's ~3⁶ — the overlap gap).
#   Smearing is spatial-only (ape_smear), so the per-timeslice transfer-matrix
#   bound is untouched; GELT becomes a spatially-resolved nonlinear
#   generalization of the GEVP over the same smearing ladder. The transport T
#   is built from the FIRST (least smeared) level.
CHECKPOINT = "best_glueball_gelt" + (
    ""
    if tuple(INPUT_SMEAR_LEVELS) == (0,)
    else "_sm" + "-".join(str(lv) for lv in INPUT_SMEAR_LEVELS)
) + ".pth"
# The checkpoint name encodes the input smearing levels: changing them changes
# the ChannelLift shape, so checkpoints at different levels are incompatible
# and must never RESUME into (or overwrite) each other.
RESUME = True  # warm-start from CHECKPOINT if it exists (optimizer state and the
#                cosine schedule restart fresh); set False to train from scratch
EVAL_ONLY = False  # skip training entirely: load CHECKPOINT and run only the
#                    test-split eval, the Ō-array dump, and the plots — one
#                    forward pass over val (if RESUME) + test. The cheap way to
#                    (re)produce the offline-analysis dump consumed by
#                    scripts/fit_glueball_overlap.py from an existing run.


# ── Rayleigh loss & per-batch operator ────────────────────────────────────────
def network_obar(model, U4_batch, device):
    """Zero-momentum operator Ō(t) of a config minibatch via the 3D GELT.

    ``U4_batch`` : ``(b, 4, Lt, L, L, L, nc, nc)`` full 4D SU(2) configs (time =
    lattice axis 0, i.e. tensor dim 2). Returns ``(b, Lt)`` real Ō[config, t].

    The spatial links (directions 1..3) at each fixed time are pulled out and the
    time axis is *folded into the batch*, so the network only ever sees a single
    timeslice's spatial links — a legitimate single-timeslice operator (audit
    item 1). ``W`` (spatial plaquettes) and ``T`` (3D L1-ball transport) are
    built on the fly per batch (audit item 4). ``W`` stacks one 3-channel
    plaquette block per ``INPUT_SMEAR_LEVELS`` entry (in_channels = 3·n_levels);
    ``T`` comes from the first (least smeared) level.
    """
    b, _, Lt = U4_batch.shape[0], U4_batch.shape[1], U4_batch.shape[2]
    U = U4_batch.to(device)
    # One plaquette-channel block per cumulative APE level (incremental, as in
    # smearing_operator_basis). ape_smear is spatial-only with time = axis 0, so
    # each timeslice's smeared links depend on that timeslice alone — the
    # single-timeslice operator property survives the smearing.
    Ws, U3_first, done = [], None, 0
    for lvl in sorted(INPUT_SMEAR_LEVELS):
        if lvl > done:
            U = ape_smear(U, gaugegroup, alpha=SMEAR_ALPHA, n_steps=lvl - done)
            done = lvl
        # (b, 3, Lt, L, L, L, nc, nc) → move the lattice time axis in front of the
        # spatial-lattice axes so (config, timeslice) fold contiguously into batch.
        Usp = U[:, 1:].movedim(2, 1).contiguous()  # (b, Lt, 3, L,L,L, nc,nc)
        U3 = Usp.reshape(b * Lt, 3, L, L, L, NC, NC)  # 3D slices, batch=b·Lt
        if U3_first is None:
            U3_first = U3  # least-smeared level — supplies the transport below
        Ws.append(plaquette_tensor(U3, gaugegroup))  # (b·Lt, 3, L,L,L, nc,nc) spatial planes
    W = torch.cat(Ws, dim=1)  # (b·Lt, 3·n_levels, L,L,L, nc,nc)
    T = build_transport_average(U3_first, R, gaugegroup)  # (b·Lt, n_off, L,L,L, nc,nc)
    O = model(W, T)  # (b·Lt, L, L, L) per-site invariant scalar
    Obar = O.sum(dim=(1, 2, 3))  # zero-momentum projection → (b·Lt,)
    return Obar.view(b, Lt)


def rayleigh_loss(Obar, deltas=LOSS_DELTAS):
    """−mean_Δ C(Δ)/max(C(0), ε) with batch-estimated VEV subtraction and
    t₀-averaging.

    ``Obar`` : ``(b, Lt)``. Returns ``(loss, C0, R)`` with R = C(1)/(C(0)+ε),
    the monitoring ratio (m = −log R at the optimum of the Δ=1 term). Each
    C(Δ) is averaged over every time origin (the ``roll``, periodic in time).
    """
    # float64: at large operator scale (C(0) grew to ~5e8 mid-run) the VEV
    # subtraction and the C(Δ) sums cancel catastrophically in float32; Ō is
    # (b, Lt), so the cast is free and the gradient flows through it.
    Obar = Obar.double()
    mu = Obar.mean()  # ⟨Ō⟩ — 0⁺⁺ has a NONZERO VEV; must subtract
    d = Obar - mu
    # C(Δ) for every separation the ratios touch; roll(0) is C(0), the
    # connected variance.
    need = sorted({0, 1} | set(deltas))
    C = {dt: (d.roll(-dt, dims=1) * d).mean() for dt in need}
    # Every ratio is C(Δ)/C(0): the denominator is a variance (≥ 0, and large
    # whenever the operator is nontrivial), so it cannot cross zero the way
    # the signed C(1) did under the old chained C(Δ)/C(Δ−1) form — whose
    # clamped 1/EPS ratios stayed finite but trained on garbage gradients
    # (see the EPS tunable note). Cauchy–Schwarz gives |C(Δ)| ≤ C(0), so the
    # loss is bounded in [−1, 1] by construction, and the transfer matrix
    # still bounds C(Δ)/C(0) ≤ e^{−m₀·Δ}: the ground-state operator remains
    # the joint optimum, so nothing changes variationally. clamp_min guards
    # only the constant-operator collapse C(0) → 0 (and zeroes the explosive
    # quotient-rule gradient while clamped).
    C0c = C[0].clamp_min(EPS)
    ratios = [C[dt] / C0c for dt in deltas]
    # Scale pin (see SCALE_REG): lifts the Ō → λŌ flat direction toward
    # C(0) = 1 without touching the scale-invariant Rayleigh ratios.
    loss = -sum(ratios) / len(ratios) + SCALE_REG * C0c.log().pow(2)
    Rq = C[1] / (C[0] + EPS)
    return loss, C[0], Rq


# ── Blocked jackknife on held-out configs ─────────────────────────────────────
def _blocks(B, block_size):
    idx = torch.arange(B)
    return [idx[i : i + block_size] for i in range(0, B, block_size)]


def blocked_jackknife_meff(Obar, block_size):
    """Delete-block jackknife mean/err of m_eff(Δ) for a single operator.

    ``Obar`` : ``(B, Nt)`` (config axis first). Deletes consecutive blocks of
    ``block_size`` configs rather than single configs, because residual
    autocorrelation on the anisotropic ensemble is not fully characterized (audit
    "Moderate"), so leave-one-out would underestimate the error bars.
    """
    B = Obar.shape[0]
    blocks = _blocks(B, block_size)
    samples = []
    for bl in blocks:
        mask = torch.ones(B, dtype=torch.bool)
        mask[bl] = False
        samples.append(effective_mass(connected_correlator(Obar[mask])))
    samples = torch.stack(samples)
    n = len(blocks)
    mean = samples.mean(dim=0)
    err = ((n - 1) / n * ((samples - mean) ** 2).sum(dim=0)).sqrt()
    return mean, err


def blocked_jackknife_gevp_meff(Obar_basis, block_size, t0):
    """Delete-block jackknife of the GEVP ground-state m_eff(Δ).

    ``Obar_basis`` : ``(n_ops, B, Nt)`` (config axis is axis 1). Returns the
    ground-state (column 0) mean/err, mirroring the single-operator version.
    """
    B = Obar_basis.shape[1]
    blocks = _blocks(B, block_size)
    samples = []
    for bl in blocks:
        mask = torch.ones(B, dtype=torch.bool)
        mask[bl] = False
        C = connected_correlator_matrix(Obar_basis[:, mask])
        samples.append(gevp_effective_mass(gevp_eigenvalues(C, t0=t0)))
    samples = torch.stack(samples)  # (n_blocks, Nt-1, n_ops)
    n = len(blocks)
    mean = samples.mean(dim=0)
    err = ((n - 1) / n * ((samples - mean) ** 2).sum(dim=0)).sqrt()
    return mean[:, 0], err[:, 0]  # ground state


def blocked_jackknife_meff_diff(Obar_single, Obar_basis, block_size, t0):
    """Delete-block jackknife of m_eff(single) − m_eff(GEVP ground state).

    Both operators are measured on the SAME configs, so their separate error
    bars are strongly correlated and comparing them as if independent
    overstates the uncertainty of the difference. Jackknifing the difference
    itself lets the correlated fluctuations cancel — this is the number that
    decides whether "GELT sits below the GEVP" is significant.
    """
    B = Obar_single.shape[0]
    blocks = _blocks(B, block_size)
    samples = []
    for bl in blocks:
        mask = torch.ones(B, dtype=torch.bool)
        mask[bl] = False
        m_single = effective_mass(connected_correlator(Obar_single[mask]))
        C = connected_correlator_matrix(Obar_basis[:, mask])
        m_gevp = gevp_effective_mass(gevp_eigenvalues(C, t0=t0))[:, 0]
        samples.append(m_single - m_gevp)
    samples = torch.stack(samples)
    n = len(blocks)
    mean = samples.mean(dim=0)
    err = ((n - 1) / n * ((samples - mean) ** 2).sum(dim=0)).sqrt()
    return mean, err


# ── Held-out network operator (no grad, minibatched for memory) ───────────────
@torch.no_grad()
def held_out_obar(model, configs, device):
    """Assemble the network's Ō for every held-out config → ``(N_held, Lt)``."""
    model.eval()
    obars = []
    for i in range(0, configs.shape[0], BATCH_CONFIGS):
        batch = configs[i : i + BATCH_CONFIGS]
        obars.append(network_obar(model, batch, device).cpu())
    return torch.cat(obars, dim=0)


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    # ── Ensemble: load the cached anisotropic ensemble, or regenerate it with
    # exactly measure_glueball.py's sampling parameters (SU(2) heat-bath + OR,
    # ξ=3.0, Lt=24, β=2.4). Sampling is the expensive step — do it once.
    if os.path.exists(CACHE):
        print(f"Loading cached ensemble {CACHE} …")
        configs = torch.load(CACHE)
    else:
        print(
            f"Cache absent — sampling SU(2) ensemble  L={L} Lt={LT} D={D} "
            f"β={BETA} ξ={XI}  N={N_CONFIGS} (this is the expensive step) …"
        )
        sweep = functools.partial(heatbath_overrelaxation_sweep, n_or=N_OR, xi=XI)
        configs, acc = mcmc_ensemble(
            L=L,
            D=D,
            gaugegroup=gaugegroup,
            beta=BETA,
            n_configs=N_CONFIGS,
            n_therm=N_THERM,
            n_skip=N_SKIP,
            sweep_fn=sweep,
            progress=True,
            Lt=LT,
        )
        print(f"     acceptance = {acc:.2f}")
        os.makedirs("datasets", exist_ok=True)
        torch.save(configs, CACHE)
        print(f"     cached ensemble → {CACHE}")

    configs = configs.to(MODEL_DTYPE)  # complex64 links for the complex model
    N = configs.shape[0]
    print(f"ensemble: {tuple(configs.shape)}  {configs.dtype}")

    # ── Hard, CONTIGUOUS three-way split of the chain-ordered ensemble, BEFORE
    # any training. No shuffle: the ensemble is MCMC-ordered, so a contiguous cut
    # keeps train / val / test mutually decorrelated (up to the boundary τ_int)
    # and keeps each slice chain-ordered so the blocked jackknife actually
    # removes autocorrelation. VAL selects the checkpoint; TEST is untouched
    # until the final report, so the reported mass is not biased low by model
    # selection.
    n_train = int(round(TRAIN_FRACTION * N))
    n_val = int(round(VAL_FRACTION * N))
    train_configs = configs[:n_train]
    val_configs = configs[n_train : n_train + n_val]
    test_configs = configs[n_train + n_val :]
    print(
        f"split (contiguous, chain-ordered): {train_configs.shape[0]} train / "
        f"{val_configs.shape[0]} val / {test_configs.shape[0]} test "
        f"(block size {JACK_BLOCK} → "
        f"{-(-test_configs.shape[0] // JACK_BLOCK)} jackknife blocks on test)"
    )

    # ── Model. reduction="none" → per-site invariant scalar field O(x);
    # mlp_zero_init=False is MANDATORY (audit item 3).
    model = GELT(
        gaugegroup=gaugegroup,
        L=L,
        D=3,  # per-timeslice 3D operator (audit item 1)
        R=R,
        nhead=NHEAD,
        gemhsa_layers=GEMHSA_LAYERS,
        d_qkv=D_QKV,
        gate=GATE,
        dtype=MODEL_DTYPE,
        mlp_hidden=MLP_HIDDEN,
        mlp_out=1,
        reduction="none",
        init_scale=INIT_SCALE,
        qk_init_scale=QK_INIT_SCALE,
        mlp_zero_init=False,
        d_model=D_MODEL,
        grad_checkpoint=GRAD_CHECKPOINT,
        # 3 spatial-plaquette channels per smearing level (see INPUT_SMEAR_LEVELS).
        in_channels=3 * len(INPUT_SMEAR_LEVELS),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"GELT(D=3, R={R}, layers={GEMHSA_LAYERS}, d_qkv={D_QKV}) | "
        f"input smear levels {list(INPUT_SMEAR_LEVELS)} | params {n_params:,}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Cosine anneal over the (now short) run — StepLR(150) never fired at 40
    # epochs. One smooth decay from LR to ~0 across EPOCHS.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Optional warm start: reload the best checkpoint of a previous run and
    # keep training. best_val_loss is seeded with the checkpoint's own val loss
    # so a worse first epoch cannot overwrite better weights already on disk.
    best_val_loss = float("inf")
    if RESUME and os.path.exists(CHECKPOINT):
        model.load_state_dict(
            torch.load(CHECKPOINT, map_location=device, weights_only=True)
        )
        val0 = held_out_obar(model, val_configs, device)
        best_val_loss = rayleigh_loss(val0)[0].item()
        print(
            f"Resumed weights from {CHECKPOINT} — starting val Rayleigh loss "
            f"{best_val_loss:.4f} (optimizer/schedule start fresh)."
        )

    # ── Training loop. Minibatch over configs; each config's LT timeslices go
    # through the 3D network together so the temporal correlator can be formed.
    # Model selection is on VAL only (never test) so the reported test mass is
    # not biased low by the checkpoint choice.
    train_hist, val_hist = [], []
    epoch_bar = tqdm(range(0 if EVAL_ONLY else EPOCHS))
    epochs_no_improve = 0
    # Ctrl-C breaks cleanly out of training and falls through to the eval + plot
    # below, run on the best checkpoint so far — so an interrupted run still
    # produces the mass numbers and glueball_gelt.png, not just orphaned weights.
    # The best weights are already on disk (checkpointed every improvement), and
    # train_hist / val_hist survive because we break rather than raise.
    try:
        for epoch in epoch_bar:
            model.train()
            order = torch.randperm(train_configs.shape[0])
            run_loss, run_C0, run_R, nb = 0.0, 0.0, 0.0, 0
            # Inner bar over the config minibatches — visible intra-epoch progress
            # (each epoch is ~N_train/BATCH_CONFIGS steps). leave=False so it clears
            # when the epoch finishes and the outer epoch_bar stays put.
            batch_bar = tqdm(
                range(0, train_configs.shape[0], BATCH_CONFIGS),
                desc=f"epoch {epoch + 1}",
                leave=False,
            )
            skipped = 0
            for i in batch_bar:
                batch = train_configs[order[i : i + BATCH_CONFIGS]]
                optimizer.zero_grad()
                Obar = network_obar(model, batch, device)
                loss, C0, Rq = rayleigh_loss(Obar)
                # Non-finite guard: one pathological batch must cost one skipped
                # step, not the run. A NaN/inf anywhere here makes
                # clip_grad_norm_'s total norm NaN, which rescales EVERY grad to
                # NaN and bricks the parameters permanently (the epoch-13 NaNs).
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                if not torch.isfinite(grad_norm):
                    skipped += 1
                    continue
                optimizer.step()
                run_loss += loss.item()
                run_C0 += C0.item()
                run_R += Rq.item()
                nb += 1
                batch_bar.set_postfix(loss=f"{loss.item():.4f}", C0=f"{C0.item():.2e}")
            scheduler.step()
            if skipped:
                epoch_bar.write(
                    f"  ⚠ epoch {epoch + 1}: skipped {skipped} non-finite "
                    f"batch(es) — loss/grad blew up there; investigate if frequent."
                )
            train_loss = run_loss / max(nb, 1)
            train_hist.append(train_loss)

            # Validation Rayleigh (whole val set at once, minibatched forward) —
            # this is the model-selection signal, kept off the reported test set.
            val = held_out_obar(model, val_configs, device)
            val_loss, val_C0, val_R = rayleigh_loss(val)
            val_loss = val_loss.item()
            val_hist.append(val_loss)
            # m = −log R; guard the log against a spurious R ≥ 1 (variational bound
            # violated on a finite sample) or R ≤ 0.
            val_m = float("nan")
            if 0.0 < val_R.item() < 1.0:
                val_m = -np.log(val_R.item())

            if not np.isfinite(val_loss):
                epoch_bar.write(
                    f"  ⚠ epoch {epoch + 1}: val loss is non-finite — no checkpoint "
                    f"can be saved this epoch; if this persists the run produces "
                    f"nothing (nan never improves best_val_loss)."
                )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), CHECKPOINT)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            epoch_bar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                C0=f"{run_C0 / nb:.2e}",
                m=f"{val_m:.3f}",
            )
            if run_C0 / nb < 10 * EPS:
                epoch_bar.write(
                    f"  ⚠ epoch {epoch + 1}: C(0)={run_C0 / nb:.2e} near the floor — "
                    f"constant-operator collapse; check the loss guard."
                )
            if epochs_no_improve >= PATIENCE:
                epoch_bar.write(f"Early stopping after {epoch + 1} epochs.")
                break
    except KeyboardInterrupt:
        epoch_bar.write(
            "\nInterrupted — proceeding to evaluation + plot on the best "
            "checkpoint so far."
        )

    # ── Load best (lowest val loss) checkpoint for evaluation. If the run was
    # killed before a single epoch finished, there is no checkpoint yet.
    if not os.path.exists(CHECKPOINT):
        print(
            f"No checkpoint at {CHECKPOINT} (stopped before the first epoch "
            f"completed) — nothing to evaluate."
        )
        return
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    print(f"best val Rayleigh loss: {best_val_loss:.4f}")

    # ── Test effective-mass curves (blocked jackknife) on the UNTOUCHED test
    # split — the only numbers reported. GELT operator on test configs.
    gelt_obar = held_out_obar(model, test_configs, device).double()
    _, _, R_final = rayleigh_loss(gelt_obar)
    m_rayleigh = -np.log(R_final.item()) if 0 < R_final.item() < 1 else float("nan")
    print(f"GELT test Rayleigh mass  m·a_t = −log C(1)/C(0) = {m_rayleigh:.3f}")
    meff_gelt, err_gelt = blocked_jackknife_meff(gelt_obar, JACK_BLOCK)

    # Classical anchors on the SAME test configs: thin, single-smeared, and
    # the multi-level GEVP ground state (float64 for GEVP conditioning).
    print("Building classical smearing basis on test configs …")
    Obar_basis = smearing_operator_basis(
        test_configs, gaugegroup, GEVP_LEVELS, alpha=SMEAR_ALPHA, progress=True
    ).double()
    # Persist the test-split operator time series for offline analysis: the
    # cosh-fit / ground-state-overlap study (scripts/fit_glueball_overlap.py)
    # re-tunes fit windows and figures on these tiny (B, Nt) arrays without
    # another GPU eval pass. Same stem as the checkpoint, so smear-level
    # variants never overwrite each other's dumps.
    obar_dump = os.path.join("datasets", CHECKPOINT.replace(".pth", "_test_obars.pt"))
    os.makedirs("datasets", exist_ok=True)
    torch.save(
        {
            "gelt_obar": gelt_obar,
            "Obar_basis": Obar_basis,
            "meta": {
                "L": L,
                "Lt": LT,
                "beta": BETA,
                "xi": XI,
                "gevp_levels": list(GEVP_LEVELS),
                "gevp_t0": GEVP_T0,
                "jack_block": JACK_BLOCK,
                "input_smear_levels": list(INPUT_SMEAR_LEVELS),
                "checkpoint": CHECKPOINT,
                "best_val_loss": best_val_loss,
            },
        },
        obar_dump,
    )
    print(f"Saved test Ō arrays → {obar_dump}")

    obar_thin, obar_sm = Obar_basis[0], Obar_basis[-1]
    meff_thin, err_thin = blocked_jackknife_meff(obar_thin, JACK_BLOCK)
    meff_sm, err_sm = blocked_jackknife_meff(obar_sm, JACK_BLOCK)
    meff_gevp, err_gevp = blocked_jackknife_gevp_meff(Obar_basis, JACK_BLOCK, GEVP_T0)

    # Report the anchor at several Δ (audit "Moderate": the Δ=1/Δ=2 plateau is a
    # 2-point descent — confirm with Δ=3–4 points).
    print("Classical GEVP ground state (blocked jackknife, test):")
    for dlt in range(GEVP_T0, min(GEVP_T0 + 4, len(meff_gevp))):
        if bool(torch.isfinite(meff_gevp[dlt])):
            print(
                f"     m_eff(Δ={dlt}):  m·a_t = {meff_gevp[dlt].item():.3f} "
                f"± {err_gevp[dlt].item():.3f}"
            )
    print("GELT learned operator (blocked jackknife, test):")
    for dlt in range(1, min(4, len(meff_gelt))):
        if bool(torch.isfinite(meff_gelt[dlt])):
            print(
                f"     m_eff(Δ={dlt}):  m·a_t = {meff_gelt[dlt].item():.3f} "
                f"± {err_gelt[dlt].item():.3f}"
            )

    # ── GELT ↔ classical-ladder diagnostics (Run 4 next steps) ─────────────────
    # (a) Where does the learned operator sit on the smearing ladder? Pearson
    # correlation of the VEV-subtracted Ō fluctuations on test: tracking APE×k
    # means GELT carries ~k smearing steps' worth of staple content.
    d_g = gelt_obar - gelt_obar.mean()
    print("corr(GELT, APE level) of Ō fluctuations on test:")
    for k, lvl in enumerate(GEVP_LEVELS):
        d_c = Obar_basis[k] - Obar_basis[k].mean()
        r = (d_g * d_c).mean() / (d_g.pow(2).mean() * d_c.pow(2).mean()).sqrt()
        print(f"     APE×{lvl}:  r = {r.item():+.3f}")

    # (b) GELT as an extra GEVP operator: the learned operator need not beat the
    # classical basis alone — if it carries overlap the smearing ladder lacks,
    # the enlarged GEVP plateaus lower/earlier than the classical one (the
    # "learned basis" win). Per-operator scale doesn't matter (the GEVP is
    # scale-invariant operator by operator); near-collinearity with a smearing
    # level is handled by the eigenvalue floor in gevp_eigenvalues.
    basis_plus = torch.cat([Obar_basis, gelt_obar.unsqueeze(0)], dim=0)
    meff_gevp_plus, err_gevp_plus = blocked_jackknife_gevp_meff(
        basis_plus, JACK_BLOCK, GEVP_T0
    )
    print("GEVP ground state, basis = classical + GELT (blocked jackknife, test):")
    for dlt in range(GEVP_T0, min(GEVP_T0 + 4, len(meff_gevp_plus))):
        if bool(torch.isfinite(meff_gevp_plus[dlt])):
            print(
                f"     m_eff(Δ={dlt}):  m·a_t = {meff_gevp_plus[dlt].item():.3f} "
                f"± {err_gevp_plus[dlt].item():.3f}"
            )

    # (c) Significance of the GELT-vs-GEVP comparison: jackknife the
    # *difference* on the shared configs (see blocked_jackknife_meff_diff) —
    # negative = GELT below the GEVP (less contamination) at that Δ.
    dmean, derr = blocked_jackknife_meff_diff(
        gelt_obar, Obar_basis, JACK_BLOCK, GEVP_T0
    )
    print("m_eff(GELT) − m_eff(GEVP), blocked jackknife of the difference (test):")
    for dlt in range(GEVP_T0, min(GEVP_T0 + 4, len(dmean))):
        if bool(torch.isfinite(dmean[dlt])):
            sig = abs(dmean[dlt].item()) / max(derr[dlt].item(), 1e-12)
            print(
                f"     Δ={dlt}:  {dmean[dlt].item():+.3f} ± {derr[dlt].item():.3f} "
                f"({sig:.1f}σ)"
            )

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # (0) training curves
    loss_lab = "−mean[" + ", ".join(f"C({d})/C(0)" for d in LOSS_DELTAS) + "]"
    ax[0].plot(train_hist, label=f"train  {loss_lab}")
    ax[0].plot(val_hist, label=f"val  {loss_lab}")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("Rayleigh loss")
    ax[0].set_title(f"Rayleigh training (loss = {loss_lab}; each ratio → e^(−m·Δ))")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # (1) test m_eff(Δ): GELT vs thin/smeared/GEVP, blocked-jackknife bands.
    def _plot(meff, err, lab, col, fmt):
        m, e = meff.numpy(), err.numpy()
        dd = np.arange(len(m))
        ok = np.isfinite(m) & np.isfinite(e) & (dd >= 1)
        ax[1].errorbar(
            dd[ok], m[ok], yerr=e[ok], fmt=fmt, capsize=3, color=col, label=lab,
            alpha=0.85,
        )

    _plot(meff_thin, err_thin, "thin (classical)", "C0", "o-")
    _plot(meff_sm, err_sm, f"smeared ×{GEVP_LEVELS[-1]} (classical)", "C1", "s-")
    # GEVP ground state — only Δ ≥ t0 (per-Δ eigenvalue ordering valid there).
    mg, eg = meff_gevp.numpy(), err_gevp.numpy()
    dd = np.arange(len(mg))
    ok = np.isfinite(mg) & np.isfinite(eg) & (dd >= GEVP_T0)
    ax[1].errorbar(
        dd[ok], mg[ok], yerr=eg[ok], fmt="D-", capsize=3, color="C3", lw=2,
        label=f"classical GEVP ground (levels {GEVP_LEVELS})",
    )
    _plot(meff_gelt, err_gelt, "GELT (learned)", "C2", "^-")
    # Enlarged (classical + GELT) GEVP ground state — same Δ ≥ t0 rule.
    mgp, egp = meff_gevp_plus.numpy(), err_gevp_plus.numpy()
    okp = np.isfinite(mgp) & np.isfinite(egp) & (dd >= GEVP_T0)
    ax[1].errorbar(
        dd[okp], mgp[okp], yerr=egp[okp], fmt="v-", capsize=3, color="C4",
        label="GEVP + GELT (enlarged basis)", alpha=0.85,
    )
    ax[1].axhline(0.33, color="k", ls="--", alpha=0.6, label="anchor m·a_t ≈ 0.33")
    ax[1].set_xlabel("Δ (temporal slices)")
    ax[1].set_ylabel("m_eff(Δ) = m·a_t")
    ax[1].set_title(
        "Test m_eff: learned vs hand-built operators\n"
        "(win = GELT plateaus ≤ and earlier in Δ than the GEVP)"
    )
    ax[1].set_xlim(0, LT // 2)
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"GELT variational 0⁺⁺ glueball — SU(2) L={L} Lt={LT} β={BETA} ξ={XI} "
        f"N_test={test_configs.shape[0]}  (R={R}, layers={GEMHSA_LAYERS}, "
        f"in-smear {list(INPUT_SMEAR_LEVELS)})",
        fontsize=13,
    )
    fig.tight_layout()
    out = "glueball_gelt.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
