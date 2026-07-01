"""GELT as a learned variational 0⁺⁺ glueball operator (deliverable §6.2).

Trains ``GELT(reduction="none")`` on the **Rayleigh loss** ``−C(1)/C(0)`` so that
the converged loss *is* the glueball mass and the trained network *is* the
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
GEMHSA_LAYERS = 3  # small depth to start
NHEAD = 1
D_QKV = 6  # even (RoPE) and ≥ 2·D = 6 so every spatial axis gets a real rotation
#            (caveat 3: pair_axis = p % D leaves axes unrotated when d_qkv < 2D)
D_MODEL = 8
MLP_HIDDEN = 16
INIT_SCALE = 10.0
QK_INIT_SCALE = 1.0
GATE = "softplus"
MODEL_DTYPE = torch.complex64

LR = 3e-3
EPOCHS = 400
PATIENCE = 40
BATCH_CONFIGS = 4  # configs per minibatch; each expands to BATCH_CONFIGS·LT 3D
#                    slices through the network. This is the memory knob (T and
#                    the per-layer K/V scale with it) AND the VEV-estimate knob
#                    (the batch mean ⟨Ō⟩ in the loss is noisier for small
#                    batches — the §4 ratio-estimator bias). Raise it as far as
#                    GPU memory allows.
EPS = 1e-8  # C(0) floor guarding the constant-operator collapse (§4 pitfall 3)

TRAIN_FRACTION = 0.8  # hard train/held-out split, made before any training
JACK_BLOCK = 10  # blocked-jackknife block size (configs) for held-out masses
GEVP_LEVELS = [0, 2, 4, 6]  # classical anchor smearing basis (matches measure_)
GEVP_T0 = 1
SMEAR_ALPHA = 0.5
CHECKPOINT = "best_glueball_gelt.pth"


# ── Rayleigh loss & per-batch operator ────────────────────────────────────────
def network_obar(model, U4_batch, device):
    """Zero-momentum operator Ō(t) of a config minibatch via the 3D GELT.

    ``U4_batch`` : ``(b, 4, Lt, L, L, L, nc, nc)`` full 4D SU(2) configs (time =
    lattice axis 0, i.e. tensor dim 2). Returns ``(b, Lt)`` real Ō[config, t].

    The spatial links (directions 1..3) at each fixed time are pulled out and the
    time axis is *folded into the batch*, so the network only ever sees a single
    timeslice's spatial links — a legitimate single-timeslice operator (audit
    item 1). ``W`` (spatial plaquettes) and ``T`` (3D L1-ball transport) are
    built on the fly per batch (audit item 4).
    """
    b, _, Lt = U4_batch.shape[0], U4_batch.shape[1], U4_batch.shape[2]
    # (b, 3, Lt, L, L, L, nc, nc) → move the lattice time axis in front of the
    # spatial-lattice axes so (config, timeslice) fold contiguously into batch.
    Usp = U4_batch[:, 1:].movedim(2, 1).contiguous()  # (b, Lt, 3, L,L,L, nc,nc)
    U3 = Usp.reshape(b * Lt, 3, L, L, L, NC, NC).to(device)  # 3D slices, batch=b·Lt
    W = plaquette_tensor(U3, gaugegroup)  # (b·Lt, 3, L,L,L, nc,nc) spatial planes
    T = build_transport_average(U3, R, gaugegroup)  # (b·Lt, n_off, L,L,L, nc,nc)
    O = model(W, T)  # (b·Lt, L, L, L) per-site invariant scalar
    Obar = O.sum(dim=(1, 2, 3))  # zero-momentum projection → (b·Lt,)
    return Obar.view(b, Lt)


def rayleigh_loss(Obar):
    """−C(1)/(C(0)+ε) with batch-estimated VEV subtraction and t₀-averaging.

    ``Obar`` : ``(b, Lt)``. Returns ``(loss, C0, R)``; at the optimum m = −log R.
    C(1) is averaged over every time origin (the ``roll``, periodic in time).
    """
    mu = Obar.mean()  # ⟨Ō⟩ — 0⁺⁺ has a NONZERO VEV; must subtract
    d = Obar - mu
    C0 = (d * d).mean()  # connected variance
    C1 = (d.roll(-1, dims=1) * d).mean()  # one-step, summed over all t₀
    Rq = C1 / (C0 + EPS)
    return -Rq, C0, Rq


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

    # ── Hard train/held-out split, BEFORE any training. Every reported mass
    # comes from the held-out split only (audit item 4).
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(0))
    n_train = int(round(TRAIN_FRACTION * N))
    train_idx, held_idx = perm[:n_train], perm[n_train:]
    train_configs = configs[train_idx]
    held_configs = configs[held_idx]
    print(
        f"split: {train_configs.shape[0]} train / {held_configs.shape[0]} held-out "
        f"(block size {JACK_BLOCK} → "
        f"{-(-held_configs.shape[0] // JACK_BLOCK)} jackknife blocks)"
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
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GELT(D=3, R={R}, layers={GEMHSA_LAYERS}, d_qkv={D_QKV}) | params {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)

    # ── Training loop. Minibatch over configs; each config's LT timeslices go
    # through the 3D network together so the temporal correlator can be formed.
    best_held_loss = float("inf")
    train_hist, held_hist = [], []
    epoch_bar = tqdm(range(EPOCHS))
    epochs_no_improve = 0
    for epoch in epoch_bar:
        model.train()
        order = torch.randperm(train_configs.shape[0])
        run_loss, run_C0, run_R, nb = 0.0, 0.0, 0.0, 0
        for i in range(0, train_configs.shape[0], BATCH_CONFIGS):
            batch = train_configs[order[i : i + BATCH_CONFIGS]]
            optimizer.zero_grad()
            Obar = network_obar(model, batch, device)
            loss, C0, Rq = rayleigh_loss(Obar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            run_loss += loss.item()
            run_C0 += C0.item()
            run_R += Rq.item()
            nb += 1
        scheduler.step()
        train_loss = run_loss / nb
        train_hist.append(train_loss)

        # Held-out Rayleigh (whole held-out set at once, minibatched forward).
        held = held_out_obar(model, held_configs, device)
        held_loss, held_C0, held_R = rayleigh_loss(held)
        held_loss = held_loss.item()
        held_hist.append(held_loss)
        # m = −log R; guard the log against a spurious R ≥ 1 (variational bound
        # violated on a finite sample) or R ≤ 0.
        held_m = float("nan")
        if 0.0 < held_R.item() < 1.0:
            held_m = -np.log(held_R.item())

        if held_loss < best_held_loss:
            best_held_loss = held_loss
            torch.save(model.state_dict(), CHECKPOINT)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}",
            held=f"{held_loss:.4f}",
            C0=f"{run_C0 / nb:.2e}",
            m=f"{held_m:.3f}",
        )
        if run_C0 / nb < 10 * EPS:
            epoch_bar.write(
                f"  ⚠ epoch {epoch + 1}: C(0)={run_C0 / nb:.2e} near the floor — "
                f"constant-operator collapse; check the loss guard."
            )
        if epochs_no_improve >= PATIENCE:
            epoch_bar.write(f"Early stopping after {epoch + 1} epochs.")
            break

    # ── Load best (lowest held-out loss) checkpoint for evaluation.
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    print(f"best held-out Rayleigh loss: {best_held_loss:.4f}")

    # ── Held-out effective-mass curves (blocked jackknife) ─────────────────────
    # GELT operator on held-out configs.
    gelt_obar = held_out_obar(model, held_configs, device).double()
    _, _, R_final = rayleigh_loss(gelt_obar)
    m_rayleigh = -np.log(R_final.item()) if 0 < R_final.item() < 1 else float("nan")
    print(f"GELT held-out Rayleigh mass  m·a_t = −log C(1)/C(0) = {m_rayleigh:.3f}")
    meff_gelt, err_gelt = blocked_jackknife_meff(gelt_obar, JACK_BLOCK)

    # Classical anchors on the SAME held-out configs: thin, single-smeared, and
    # the multi-level GEVP ground state (float64 for GEVP conditioning).
    print("Building classical smearing basis on held-out configs …")
    Obar_basis = smearing_operator_basis(
        held_configs, gaugegroup, GEVP_LEVELS, alpha=SMEAR_ALPHA, progress=True
    ).double()
    obar_thin, obar_sm = Obar_basis[0], Obar_basis[-1]
    meff_thin, err_thin = blocked_jackknife_meff(obar_thin, JACK_BLOCK)
    meff_sm, err_sm = blocked_jackknife_meff(obar_sm, JACK_BLOCK)
    meff_gevp, err_gevp = blocked_jackknife_gevp_meff(Obar_basis, JACK_BLOCK, GEVP_T0)

    # Report the anchor at several Δ (audit "Moderate": the Δ=1/Δ=2 plateau is a
    # 2-point descent — confirm with Δ=3–4 points).
    print("Classical GEVP ground state (blocked jackknife, held-out):")
    for dlt in range(GEVP_T0, min(GEVP_T0 + 4, len(meff_gevp))):
        if bool(torch.isfinite(meff_gevp[dlt])):
            print(
                f"     m_eff(Δ={dlt}):  m·a_t = {meff_gevp[dlt].item():.3f} "
                f"± {err_gevp[dlt].item():.3f}"
            )
    print("GELT learned operator (blocked jackknife, held-out):")
    for dlt in range(1, min(4, len(meff_gelt))):
        if bool(torch.isfinite(meff_gelt[dlt])):
            print(
                f"     m_eff(Δ={dlt}):  m·a_t = {meff_gelt[dlt].item():.3f} "
                f"± {err_gelt[dlt].item():.3f}"
            )

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # (0) training curves
    ax[0].plot(train_hist, label="train  −C(1)/C(0)")
    ax[0].plot(held_hist, label="held-out  −C(1)/C(0)")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("Rayleigh loss")
    ax[0].set_title("Rayleigh training (loss = −C(1)/C(0); m = −log(−loss))")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # (1) held-out m_eff(Δ): GELT vs thin/smeared/GEVP, blocked-jackknife bands.
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
    ax[1].axhline(0.33, color="k", ls="--", alpha=0.6, label="anchor m·a_t ≈ 0.33")
    ax[1].set_xlabel("Δ (temporal slices)")
    ax[1].set_ylabel("m_eff(Δ) = m·a_t")
    ax[1].set_title(
        "Held-out m_eff: learned vs hand-built operators\n"
        "(win = GELT plateaus ≤ and earlier in Δ than the GEVP)"
    )
    ax[1].set_xlim(0, LT // 2)
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"GELT variational 0⁺⁺ glueball — SU(2) L={L} Lt={LT} β={BETA} ξ={XI} "
        f"N_held={held_configs.shape[0]}  (R={R}, layers={GEMHSA_LAYERS})",
        fontsize=13,
    )
    fig.tight_layout()
    out = "glueball_gelt.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
