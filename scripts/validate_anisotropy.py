"""Validation for the anisotropic SU(2) lattice (gelt.lattice / gelt.sampler).

Three checks, in increasing physics content:

  1. REFACTOR CORRECTNESS (ξ = 1).  The anisotropic code path with ξ = 1 must
     reproduce the isotropic 2D SU(2) mean plaquette I₂(β)/I₁(β) — i.e. folding
     the per-plane weight into the staple changed nothing at ξ = 1.

  2. THE ANISOTROPY ACTS (ξ > 1).  With β_t = β·ξ > β_s = β/ξ, temporal
     plaquettes are more ordered than spatial ones, so ⟨P_st⟩ > ⟨P_ss⟩. We scan
     ξ and plot both — the split is the signature that the action is anisotropic.

  3. RENORMALIZED ANISOTROPY (the caveat made visible).  The *bare* ξ ≠ the
     *physical* (renormalized) ξ_R. We estimate ξ_R from the ratio of Creutz
     ratios of spatial-spatial vs space-time Wilson loops:
         χ_ss(R,T) → σ·a_s²,   χ_st(R,T) → σ·a_s·a_t
         ⇒ ξ_R = a_s/a_t ≈ χ_ss / χ_st
     (the string tension σ cancels). This is a noisy, small-lattice *estimate*,
     printed next to ξ_bare so the mismatch is explicit — there is NO auto-tuning.

Time is lattice axis 0 (the sampler / glueball convention).

Run:
    python scripts/validate_anisotropy.py
"""

import functools
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from gelt.lattice import SU, plaquette_tensor, random_links, rectangular_wilson_loop
from gelt.sampler import heatbath_overrelaxation_sweep, mcmc_ensemble

torch.manual_seed(0)
np.random.seed(0)
gaugegroup = SU(2)
TIME_AXIS = 0

# ── Tunable ───────────────────────────────────────────────────────────────────
BETA = 2.0
N_OR = 3
N_THERM = 200
N_CONFIGS = 200
L4 = 8  # spatial extent for the 4D scans


def mean_re_tr_plaquette(configs, temporal):
    """⟨Re Tr P / nc⟩ over the temporal OR spatial plaquette planes.

    Plaquette pairs follow plaquette_tensor's μ<ν order; a plane is temporal iff
    it touches TIME_AXIS (= axis 0), i.e. μ == 0.
    """
    P = plaquette_tensor(configs, gaugegroup)  # (B, n_pairs, *Λ, nc, nc)
    re_tr = P.diagonal(dim1=-2, dim2=-1).sum(-1).real / gaugegroup.nc  # (B, n_pairs, *Λ)
    D = configs.shape[1]
    pairs = [(mu, nu) for mu in range(D) for nu in range(mu + 1, D)]
    idx = [i for i, (mu, nu) in enumerate(pairs) if (mu == TIME_AXIS) == temporal]
    return re_tr[:, idx].mean().item()


def creutz_ratio(configs, mu, nu, R, T):
    """χ(R,T) = -ln[ W(R,T) W(R-1,T-1) / (W(R-1,T) W(R,T-1)) ] in plane (μ, ν)."""

    def W(r, t):
        return rectangular_wilson_loop(configs, gaugegroup, r, t, mu, nu).mean().item()

    return -math.log((W(R, T) * W(R - 1, T - 1)) / (W(R - 1, T) * W(R, T - 1)))


def sample(xi, L, D, Lt=None, n_configs=N_CONFIGS):
    sweep = functools.partial(heatbath_overrelaxation_sweep, n_or=N_OR, xi=xi, time_axis=TIME_AXIS)
    configs, _ = mcmc_ensemble(
        L=L, D=D, gaugegroup=gaugegroup, beta=BETA, n_configs=n_configs,
        n_therm=N_THERM, n_skip=2, sweep_fn=sweep, Lt=Lt, progress=False,
    )
    return configs


# ── 1. ξ = 1 reproduces the exact 2D mean plaquette ───────────────────────────
print("1/3  ξ=1 regression (2D SU(2) mean plaquette) …")
cfg2d = sample(xi=1.0, L=16, D=2)
# 2D has a single plaquette plane (0,1), so the temporal/spatial split is moot —
# average over every plaquette directly.
P = plaquette_tensor(cfg2d, gaugegroup)
meas2d = (P.diagonal(dim1=-2, dim2=-1).sum(-1).real / gaugegroup.nc).mean().item()
b = torch.tensor(BETA, dtype=torch.float64)
exact2d = (torch.special.i0(b) / torch.special.i1(b) - 2.0 / b).item()
print(f"     measured ⟨P⟩ = {meas2d:.4f}   exact I₂/I₁ = {exact2d:.4f}   "
      f"Δ = {abs(meas2d - exact2d):.4f}")

# ── 2. ξ-scan of ⟨P_ss⟩ vs ⟨P_st⟩ (4D) ────────────────────────────────────────
print("2/3  ξ-scan of spatial vs temporal plaquette (4D) …")
xis = [1.0, 1.5, 2.0, 3.0, 4.0]
p_ss, p_st = [], []
for xi in xis:
    cfg = sample(xi=xi, L=L4, D=4)
    p_ss.append(mean_re_tr_plaquette(cfg, temporal=False))
    p_st.append(mean_re_tr_plaquette(cfg, temporal=True))
    print(f"     ξ={xi:>4}:  ⟨P_ss⟩={p_ss[-1]:.4f}   ⟨P_st⟩={p_st[-1]:.4f}")

# ── 3. Renormalized-anisotropy estimate via Creutz-ratio ratio ────────────────
print("3/3  Renormalized anisotropy ξ_R (Creutz-ratio ratio) …")
XI_PROBE = 3.0
cfg = sample(xi=XI_PROBE, L=L4, D=4, n_configs=max(N_CONFIGS, 400))
chi_ss = creutz_ratio(cfg, mu=1, nu=2, R=2, T=2)  # spatial-spatial → σ a_s²
chi_st = creutz_ratio(cfg, mu=TIME_AXIS, nu=1, R=2, T=2)  # space-time → σ a_s a_t
xi_R = chi_ss / chi_st
print(f"     ξ_bare = {XI_PROBE}   →   ξ_R ≈ χ_ss/χ_st = {chi_ss:.4f}/{chi_st:.4f} = {xi_R:.2f}")
print("     (noisy small-lattice estimate; bare ≠ renormalized — no auto-tuning)")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].axhline(exact2d, color="k", ls="--", label="exact I₂(β)/I₁(β)")
ax[0].plot([1.0], [meas2d], "o", ms=9, label="ξ=1 anisotropic path")
ax[0].set_title(f"ξ=1 regression — 2D SU(2)  β={BETA}")
ax[0].set_xlabel("ξ")
ax[0].set_ylabel("⟨Re Tr P / nc⟩")
ax[0].legend()

ax[1].plot(xis, p_ss, "o-", label="⟨P_ss⟩ (spatial)")
ax[1].plot(xis, p_st, "s-", label="⟨P_st⟩ (temporal)")
ax[1].set_title(f"anisotropy splits the plaquette — 4D SU(2)  β={BETA}\n"
                f"ξ_R(ξ_bare={XI_PROBE}) ≈ {xi_R:.2f}")
ax[1].set_xlabel("ξ (bare)")
ax[1].set_ylabel("⟨Re Tr P / nc⟩")
ax[1].legend()

fig.tight_layout()
out = "anisotropy_validation.png"
fig.savefig(out, dpi=130)
print(f"Saved {out}")
