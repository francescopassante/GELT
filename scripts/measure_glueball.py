"""Visual sanity check for the classical 0⁺⁺ glueball baseline (gelt/glueball.py).

The figure answers two *separate* questions, because the answers are different
right now:

  Top row — IS THE PIPELINE CORRECT?
    · synthetic known mass: feed an ensemble whose connected correlator is a
      known single exponential exp(-m_true·Δ); the extracted m_eff(Δ) must
      plateau onto m_true. Validates the correlator + jackknife arithmetic
      independently of any lattice physics.
    · smearing sanity: ⟨O⟩ (mean spatial-plaquette operator) must rise
      monotonically toward 1 as APE steps push the links toward smooth.

  Bottom row — THE REAL ENSEMBLE (the physics):
    · connected correlator C(Δ), thin vs APE-smeared (semilog where positive).
    · effective mass m_eff(Δ) with jackknife bands, thin vs smeared.

The ensemble uses the SU(2) heat-bath + overrelaxation sampler (§8), which beats
Metropolis critical slowing, so the bottom row is no longer autocorrelation-
limited (check_glueball_autocorrelation.py measures τ_int ≈ 2 for the smeared
operator, hence N_SKIP = 5). What remains is operator overlap and raw
statistics: expect the SMEARED operator to develop a short plateau at small Δ
where the thin operator never reaches it; if it is still too noisy, the lever is
N_CONFIGS (see notes/glueball_spectroscopy.md §7).

Run:
    python scripts/measure_glueball.py
"""

import functools
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from gelt.lattice import SU
from gelt.sampler import heatbath_overrelaxation_sweep, mcmc_ensemble
from gelt.glueball import (
    ape_smear,
    connected_correlator,
    effective_mass,
    glueball_operator,
    jackknife_effective_mass,
    zero_momentum,
)

torch.manual_seed(0)
np.random.seed(0)
gaugegroup = SU(2)

# ── Tunable ──────────────────────────────────────────────────────────────────
L = 12  # spatial/temporal extent (cubic L^D lattice)
D = 4  # 3+1 dimensions; time is lattice axis 0
BETA = 2.4  # SU(2); m·a ≈ 1.0 → usable signal at Δ = 0..3 (plateau-friendly)
N_CONFIGS = 2000  # raise for less noise on the bottom row
N_THERM = 300
N_SKIP = 5  # ≳ 2·τ_int (smeared-operator τ_int ≈ 2, per check_glueball_autocorrelation)
SMEAR_ALPHA = 0.5
SMEAR_STEPS = 6
N_OR = 4  # overrelaxation sweeps per heat-bath sweep (decorrelation)

# SU(2) heat-bath + overrelaxation: the exact, no-tuning sampler that beats
# Metropolis critical slowing. This is the prerequisite (§8) for the bottom
# row to develop a resolvable plateau instead of being autocorrelation-limited.
sweep = functools.partial(heatbath_overrelaxation_sweep, n_or=N_OR)

# ── (0,0)  Synthetic validation: known mass in, plateau out ───────────────────
print("1/4  Synthetic correlator validation …")
M_TRUE = 0.5
NT_SYN = 32  # long, so the periodic-estimator bend stays far from small Δ
N_SYN = 4000
VEV = 5.0  # nonzero VEV, to exercise the vacuum subtraction

t = np.arange(NT_SYN)
dist = np.abs(t[:, None] - t[None, :])
cov = np.exp(-M_TRUE * dist) + 1e-6 * np.eye(NT_SYN)  # OU covariance, exp decay
chol = np.linalg.cholesky(cov)
obar_syn = VEV + np.random.randn(N_SYN, NT_SYN) @ chol.T
obar_syn = torch.from_numpy(obar_syn)
meff_syn, err_syn = jackknife_effective_mass(obar_syn)

# ── (0,1)  Smearing sanity: ⟨O⟩ vs number of APE steps ────────────────────────
print("2/4  Smearing monotonicity …")
configs_small, _ = mcmc_ensemble(
    L=6,
    D=D,
    gaugegroup=gaugegroup,
    beta=BETA,
    n_configs=20,
    n_therm=N_THERM,
    n_skip=N_SKIP,
    sweep_fn=sweep,
    progress=False,
)
mean_O_vs_steps = []
for n in range(0, 9):
    Wn = ape_smear(configs_small, gaugegroup, alpha=SMEAR_ALPHA, n_steps=n)
    O = glueball_operator(Wn, gaugegroup)
    mean_O_vs_steps.append(O.mean().item())

# ── Real ensemble (bottom row) ────────────────────────────────────────────────
print(f"3/4  Sampling SU(2) ensemble  L={L} D={D} β={BETA}  N={N_CONFIGS} …")
configs, acc = mcmc_ensemble(
    L=L,
    D=D,
    gaugegroup=gaugegroup,
    beta=BETA,
    n_configs=N_CONFIGS,
    n_therm=N_THERM,
    n_skip=N_SKIP,
    sweep_fn=sweep,
    progress=False,
)
print(f"     acceptance = {acc:.2f}")

print("4/4  Smearing + correlators …")
configs_sm = ape_smear(configs, gaugegroup, alpha=SMEAR_ALPHA, n_steps=SMEAR_STEPS)


def measure(W):
    Obar = zero_momentum(glueball_operator(W, gaugegroup))
    C = connected_correlator(Obar)
    meff, err = jackknife_effective_mass(Obar)
    return Obar, C, meff, err


obar_thin, C_thin, meff_thin, err_thin = measure(configs)
obar_sm, C_sm, meff_sm, err_sm = measure(configs_sm)
print(f"     ⟨Ō⟩ thin = {obar_thin.mean():.2f}   C(0) thin = {C_thin[0]:.3e}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(2, 2, figsize=(12, 9))

# (0,0) synthetic validation
d_syn = np.arange(len(meff_syn))
ax[0, 0].axhline(M_TRUE, color="k", ls="--", label=f"m_true = {M_TRUE}")
ax[0, 0].errorbar(
    d_syn,
    meff_syn.numpy(),
    yerr=err_syn.numpy(),
    fmt="o-",
    capsize=3,
    label="extracted m_eff",
)
ax[0, 0].set_xlim(0, 10)
ax[0, 0].set_title("code check: synthetic known mass → plateau")
ax[0, 0].set_xlabel("Δ")
ax[0, 0].set_ylabel("m_eff(Δ)")
ax[0, 0].legend()

# (0,1) smearing monotonicity
ax[0, 1].plot(range(len(mean_O_vs_steps)), mean_O_vs_steps, "o-")
ax[0, 1].set_title("smearing sanity: ⟨O⟩ rises toward 1")
ax[0, 1].set_xlabel("APE steps")
ax[0, 1].set_ylabel("⟨O⟩ (mean spatial plaquette)")

# (1,0) real correlator, semilog where positive
for C, lab, col in [(C_thin, "thin", "C0"), (C_sm, f"smeared ×{SMEAR_STEPS}", "C1")]:
    C = C.numpy()
    pos = C > 0
    dd = np.arange(len(C))
    ax[1, 0].semilogy(dd[pos], C[pos], "o-", color=col, label=lab)
ax[1, 0].set_title("real ensemble: connected C(Δ)")
ax[1, 0].set_xlabel("Δ")
ax[1, 0].set_ylabel("C(Δ)  (positive part)")
ax[1, 0].legend()

# (1,1) real m_eff with jackknife bands (mask non-finite)
for meff, err, lab, col in [
    (meff_thin, err_thin, "thin", "C0"),
    (meff_sm, err_sm, f"smeared ×{SMEAR_STEPS}", "C1"),
]:
    m = meff.numpy()
    e = err.numpy()
    ok = np.isfinite(m) & np.isfinite(e)
    dd = np.arange(len(m))
    ax[1, 1].errorbar(
        dd[ok], m[ok], yerr=e[ok], fmt="o-", capsize=3, color=col, label=lab
    )
ax[1, 1].set_title("real ensemble: m_eff(Δ)  (look for smeared plateau at small Δ)")
ax[1, 1].set_xlabel("Δ")
ax[1, 1].set_ylabel("m_eff(Δ)")
ax[1, 1].legend()

fig.suptitle(
    f"Glueball baseline — SU(2)  L={L} D={D} β={BETA}  N={N_CONFIGS}  (acc {acc:.2f})",
    fontsize=13,
)
fig.tight_layout()
out = "glueball_validation.png"
fig.savefig(out, dpi=130)
print(f"Saved {out}")
