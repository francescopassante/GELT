"""Phase A of the attention-range study: does the correlation length move?

The conference abstract promises to study "whether the attention range
correlates with physical correlation length". That plot's x-axis is the
*spatial* correlation length in lattice units,

    ξ_s = 1 / (m · a_s) = 1 / (ξ · m·a_t),

and on the anchored ensemble (β=2.4, ξ=3, m·a_t≈0.33) it is ξ_s ≈ 1 — barely
one lattice spacing. Attention offsets are spatial, so if ξ_s does not move
appreciably across an accessible β window there is nothing for ℓ_att to track
and the whole study is dead on arrival, no matter how good the network is.

This script answers that question *before* any training is committed. It is
purely classical: sample an ensemble at each β and read m·a_t off the
multi-level GEVP, exactly as measure_glueball.py does at the single anchor β.
The ensembles are cached under the SAME key measure_glueball.py and
train_glueball.py use, so nothing sampled here is wasted — the training phase
reuses these files as-is.

Read the printed table. If ξ_s spans a factor of ~2 or more, phase B (one
GELT per β at R=3) is worth its four overnight runs; if it is flat, say so in
the talk and spend the time on the topological-localization clause instead.

Run:
    python scripts/beta_scan.py

Writes ``results/attention/beta_scan.{png,pt}``.
"""

import functools
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from gelt.glueball import jackknife_gevp_effective_mass, smearing_operator_basis
from gelt.lattice import SU
from gelt.sampler import heatbath_overrelaxation_sweep, mcmc_ensemble

# Output artifacts are grouped by study under results/; create the dirs the
# first time this runs in a fresh clone (they hold generated files only).
for _d in ("results/sampler", "results/glueball", "results/attention",
          "results/wilson_regression", "datasets"):
    os.makedirs(_d, exist_ok=True)



# ── Tunables ──────────────────────────────────────────────────────────────────
# β window. The anchor (2.4) sits in the middle so the scan is anchored to a
# known result; the ends are where a_s changes enough to move ξ_s. Going much
# above ~2.7 shrinks a_s until the 0⁺⁺ no longer fits in L=12 (finite volume),
# and much below ~2.1 makes the correlator die before the plateau — the window
# is genuinely narrow, which is the thing being measured here.
# β=2.4 costs nothing: the Run-5 anchor ensemble already sits on disk under
# exactly this cache key, so it loads instead of sampling — and it ties the
# scan to the mass we have independently confirmed twice (m·a_t ≈ 0.33).
BETAS = [2.1, 2.3, 2.4, 2.5, 2.7]

# Everything below MUST match measure_glueball.py so the cache keys agree.
L = 12
D = 4
XI = 3.0
LT = 2 * L
N_CONFIGS = 2000
N_THERM = 300
N_SKIP = 5
N_OR = 4
SMEAR_ALPHA = 0.5
SMEAR_LEVELS = [0, 2, 4, 6]
GEVP_T0 = 1
# Configs per smearing batch. The full N=2000 ensemble is ~10.6 GB as complex64
# and ape_smear clones it, so smearing it in one shot needs >20 GB of device
# memory and OOMs a 32 GB V100. Each config smears independently, so chunking is
# exact, not an approximation. The ensemble stays on the CPU; only the chunk
# moves. (measure_glueball.py sidesteps this by smearing on the CPU entirely.)
SMEAR_CHUNK = 200
# Δ at which the plateau mass is quoted. Δ=2 is where the anchor ensemble's
# GEVP plateau was read (m·a_t = 0.333 ± 0.011); Δ=1 is more contaminated.
QUOTE_DELTA = 2

gaugegroup = SU(2)

# cuda → cpu, deliberately skipping MPS: APE smearing projects onto the group
# with a complex SVD/det, and MPS has no complex linalg.lu_factor — the smear
# raises before it ever gets to the GEVP. (Same reason
# visualize_glueball_attention.py stays on the CPU locally.)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cache_path(beta):
    """The key measure_glueball.py / train_glueball.py build for this ensemble."""
    return f"datasets/glueball_configs_L{L}_Lt{LT}_b{beta}_xi{XI}_N{N_CONFIGS}.pt"


def ensemble(beta):
    """Load the cached ensemble at this β, or sample and cache it."""
    path = cache_path(beta)
    if os.path.exists(path):
        print(f"  loading cached {path}")
        return torch.load(path)
    print(f"  sampling N={N_CONFIGS} at β={beta} (the long pole) …")
    sweep = functools.partial(heatbath_overrelaxation_sweep, n_or=N_OR, xi=XI)
    configs, acc = mcmc_ensemble(
        L=L,
        D=D,
        gaugegroup=gaugegroup,
        beta=beta,
        n_configs=N_CONFIGS,
        n_therm=N_THERM,
        n_skip=N_SKIP,
        sweep_fn=sweep,
        progress=True,
        Lt=LT,
    )
    print(f"  acceptance = {acc:.2f}")
    os.makedirs("datasets", exist_ok=True)
    torch.save(configs, path)
    print(f"  cached → {path}")
    return configs


def gevp_mass(configs):
    """Plateau m·a_t and its jackknife error from the multi-level GEVP.

    Mirrors measure_glueball.py exactly: float64 (the GEVP whitening is
    ill-conditioned in float32 at these statistics) and the ground state is
    column 0 of the (Nt, n_ops) jackknife output.
    """
    parts = []
    for i in tqdm(range(0, configs.shape[0], SMEAR_CHUNK), desc="  smearing"):
        chunk = configs[i : i + SMEAR_CHUNK].to(device)
        parts.append(
            smearing_operator_basis(
                chunk, gaugegroup, SMEAR_LEVELS, alpha=SMEAR_ALPHA
            ).cpu()
        )
        del chunk
        if device.type == "cuda":
            torch.cuda.empty_cache()
    Obar = torch.cat(parts, dim=1).double()  # (n_levels, B, Nt)
    meff, err = jackknife_gevp_effective_mass(Obar, t0=GEVP_T0)
    m_ground, err_ground = meff[:, 0], err[:, 0]
    return m_ground[QUOTE_DELTA].item(), err_ground[QUOTE_DELTA].item(), m_ground.cpu()


def main():
    print(f"device: {device}")
    rows = []
    for beta in BETAS:
        print(f"\n── β = {beta} " + "─" * 40)
        m, err, m_all = gevp_mass(ensemble(beta))
        # ξ_s = 1/(ξ·m·a_t): the spatial correlation length in units of a_s,
        # which is the unit the attention offsets are measured in.
        xi_s = 1.0 / (XI * m)
        xi_s_err = xi_s * err / m  # first-order propagation
        rows.append((beta, m, err, xi_s, xi_s_err, m_all))
        print(f"  m·a_t(Δ={QUOTE_DELTA}) = {m:.4f} ± {err:.4f}   ξ_s = {xi_s:.2f} ± {xi_s_err:.2f}")

    print("\n" + "=" * 62)
    print(f"{'β':>6} {'m·a_t':>16} {'m·a_s':>8} {'ξ_s = 1/(ξ·m·a_t)':>22}")
    print("-" * 62)
    for beta, m, err, xi_s, xi_s_err, _ in rows:
        print(f"{beta:>6} {m:>8.4f} ± {err:.4f} {XI * m:>8.3f} {xi_s:>14.2f} ± {xi_s_err:.2f}")
    span = max(r[3] for r in rows) / min(r[3] for r in rows)
    print("-" * 62)
    print(f"ξ_s spans a factor {span:.2f} across the scan.")
    print(
        "GATE: a factor ≳2 means ℓ_att has something to track and phase B "
        "(one GELT per β at R=3) is worth its overnight runs.\n"
        "      A flat ξ_s means the x-axis does not exist — report that and "
        "spend the time on the topological-localization clause."
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    b = [r[0] for r in rows]
    ax[0].errorbar(b, [r[1] for r in rows], yerr=[r[2] for r in rows], fmt="o-", capsize=4)
    ax[0].set_xlabel(r"$\beta$")
    ax[0].set_ylabel(r"$m\,a_t$ (GEVP plateau)")
    ax[0].set_title(f"Glueball mass vs coupling (Δ={QUOTE_DELTA}, ξ={XI})")
    ax[0].grid(True, alpha=0.3)

    ax[1].errorbar(b, [r[3] for r in rows], yerr=[r[4] for r in rows], fmt="o-", capsize=4)
    ax[1].axhline(1.0, ls=":", color="gray")
    ax[1].set_xlabel(r"$\beta$")
    ax[1].set_ylabel(r"$\xi_s = 1/(\xi\, m\, a_t)$   [lattice spacings $a_s$]")
    ax[1].set_title(f"The x-axis of the attention-range plot\n(spans ×{span:.2f})")
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("results/attention/beta_scan.png", dpi=130, bbox_inches="tight")
    print("\nSaved beta_scan.png")

    torch.save(
        {
            "betas": b,
            "m_at": [r[1] for r in rows],
            "m_at_err": [r[2] for r in rows],
            "xi_s": [r[3] for r in rows],
            "xi_s_err": [r[4] for r in rows],
            "m_eff_curves": [r[5] for r in rows],
            "meta": {"L": L, "Lt": LT, "xi": XI, "N": N_CONFIGS, "quote_delta": QUOTE_DELTA},
        },
        "results/attention/beta_scan.pt",
    )
    print("Saved results/attention/beta_scan.pt")


if __name__ == "__main__":
    main()
