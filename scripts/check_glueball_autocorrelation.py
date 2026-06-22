"""Step-1 pre-flight for glueball spectroscopy: is the ensemble decorrelated?

Before trusting any m_eff plateau (scripts/measure_glueball.py), confirm the
SU(2) heat-bath + overrelaxation chain actually decorrelates the *glueball
operator* — not just the plaquette, which mixes far faster than an extended,
smeared operator. The mean-plaquette agreement in tests/validate proves the
sampler is *correct*; this proves it is *decorrelated enough* for THIS
observable, which is what makes "N configs" really mean N independent samples.

We run one long chain storing every sweep (n_skip=1), build a scalar glueball
observable per config (the zero-momentum operator summed over the lattice),
and measure its integrated autocorrelation time τ_int. The production n_skip
in measure_glueball.py should be ≳ 2·τ_int of the SMEARED operator (the slow
one). Units are combined HB+OR sweeps — the same unit as n_skip.

Run:
    python scripts/check_glueball_autocorrelation.py
"""

import functools

import matplotlib.pyplot as plt
import numpy as np
import torch

from gelt.glueball import ape_smear, glueball_operator
from gelt.lattice import SU, plaquette_tensor
from gelt.sampler import (
    _re_tr,
    heatbath_overrelaxation_sweep,
    integrated_autocorrelation_time,
    mcmc_ensemble,
)

torch.manual_seed(0)
np.random.seed(0)
gaugegroup = SU(2)

# ── Tunable ───────────────────────────────────────────────────────────────────
L = 8  # match the planned measure_glueball.py run
D = 4  # 3+1 dimensions; time is lattice axis 0
BETA = 2.4  # the configuration most likely to show a plateau (m·a ≈ 1.0)
N_CHAIN = 1000  # consecutive sweeps stored — the chain length for the estimate
N_THERM = 300
N_OR = 4  # overrelaxation sweeps per heat-bath sweep
SMEAR_ALPHA = 0.5
SMEAR_STEPS = 6
MAX_LAG = 100

sweep = functools.partial(heatbath_overrelaxation_sweep, n_or=N_OR)

print(f"Sampling a length-{N_CHAIN} chain  SU(2) L={L} D={D} β={BETA}  (n_skip=1) …")
chain, _ = mcmc_ensemble(
    L=L,
    D=D,
    gaugegroup=gaugegroup,
    beta=BETA,
    n_configs=N_CHAIN,
    n_therm=N_THERM,
    n_skip=1,
    sweep_fn=sweep,
)

# ── Scalar-per-config observables along the chain ─────────────────────────────
print("Building observables (plaquette, thin & smeared glueball operator) …")
plaq_ts = (
    _re_tr(plaquette_tensor(chain, gaugegroup)).flatten(1).mean(1) / gaugegroup.nc
).numpy()
glue_thin = glueball_operator(chain, gaugegroup).flatten(1).sum(1).numpy()
chain_sm = ape_smear(chain, gaugegroup, alpha=SMEAR_ALPHA, n_steps=SMEAR_STEPS)
glue_sm = glueball_operator(chain_sm, gaugegroup).flatten(1).sum(1).numpy()

series = {
    "plaquette": plaq_ts,
    "glueball (thin)": glue_thin,
    f"glueball (smeared ×{SMEAR_STEPS})": glue_sm,
}
results = {
    name: integrated_autocorrelation_time(ts, max_lag=MAX_LAG)
    for name, ts in series.items()
}

print("\nIntegrated autocorrelation time  (units = combined HB+OR sweeps):")
for name, (_, tau, w) in results.items():
    print(
        f"  {name:26s} τ_int = {tau:6.2f}  (window {w:3d})  "
        f"→  n_skip ≳ {int(np.ceil(2 * tau))}"
    )

# ── Plot ──────────────────────────────────────────────────────────────────────
COL = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fig, ax = plt.subplots(figsize=(8, 5))
for i, (name, (rho, tau, _)) in enumerate(results.items()):
    ax.plot(
        np.arange(len(rho)), rho, "o-", ms=3, color=COL[i], label=f"{name}:  τ_int={tau:.1f}"
    )
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("Lag  t  (combined HB+OR sweeps)")
ax.set_ylabel("ρ(t)")
ax.set_title(
    f"Glueball-operator autocorrelation — SU(2)  L={L} D={D} β={BETA}\n"
    f"chain length {N_CHAIN};  production n_skip ≳ 2·τ_int of the smeared operator"
)
ax.legend()
ax.set_xlim(-0.5, MAX_LAG)
fig.tight_layout()
out = "glueball_autocorrelation.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")
