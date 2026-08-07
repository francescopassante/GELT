"""Phase A of the Z₂ attention-range study: find a regime where ξ is large.

The SU(2) glueball cannot answer "does the attention range track the physical
correlation length" (notes/topological_localization.md §6): ξ_s = 1/(m·a_s)
stays ≤ 1.1 lattice spacings across every accessible β, because the 0⁺⁺ is
heavy and ξ_s depends on the lattice *spacing*, not the volume. Enlarging L
does not help; only a finer a_s does, which means β ≈ 2.8 and L ≥ 24 in 4D —
weeks of sampling.

So change system rather than tune parameters. **3D Z₂ gauge theory has a
genuine second-order transition at β_c ≈ 0.7614** (it is dual to the 3D Ising
model), and the mass gap vanishes there: ξ = 1/m diverges as β → β_c⁻. That
gives ξ = 1, 2, 4, 6 … on demand, which is exactly the dynamic range the
attention offsets need in order to have anything to track. Z₂ is also the
cheapest object in the codebase (nc = 1, real links) and the project's declared
testbed.

This script is the gate, structured like scripts/beta_scan.py: it measures the
mass classically — no network — at several β approaching β_c, and reports
ξ = 1/m. Only if ξ spans roughly 1 → 5 is Phase B (one variational operator per
β, reading ℓ_att) worth its training runs.

It also reports τ_int of the smeared operator at each β, because critical
slowing down is the thing most likely to invalidate this scan: local updates
decorrelate like ξ^z with z ≈ 2, so an n_skip that is ample at β = 0.70 can be
far too small at β = 0.758, and an undersampled chain fakes small errors.

Runtime note: the sampling is NOT the long pole here — a Z₂ sweep at 24²×48
costs ~0.010 s, so all five ensembles are ~20 min. The analysis is: ape_smear
loops over configurations in Python, and the jackknife does one GEVP solve per
configuration. Expect the bar under "smearing" to dominate.

Run:
    python scripts/z2_beta_scan.py

Writes ``z2_beta_scan.png`` and ``datasets/z2_beta_scan.pt``.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from gelt.glueball import jackknife_gevp_effective_mass, smearing_operator_basis
from gelt.lattice import Z2
from gelt.sampler import integrated_autocorrelation_time, mcmc_ensemble


# ── Tunables ──────────────────────────────────────────────────────────────────
# 3D Z₂ gauge theory: β_c ≈ 0.7614 (dual to 3D Ising). Approaching from the
# confined side, where the mass gap closes continuously. The closest point is
# deliberately NOT nearer than ~0.758: closer costs critical slowing down and
# risks ξ ≳ L/4, where finite volume bends the mass back up (the lesson from
# the SU(2) scan's β=2.7 point).
BETA_C = 0.7614
# The first scan put points at 0.70 and 0.73 and learned that they are wasted:
# β=0.70 gave ξ = 0.58 ± 0.69 with m_eff turning negative by Δ=5, i.e. the
# correlator dies inside one lattice spacing and the point carries no signal.
# Calibrating Ising scaling ξ ~ |β_c − β|^(−ν), ν ≈ 0.63, against that
# measurement puts ξ ≈ 3 at β_c − β ≈ 0.004 and ξ ≈ 6 at ≈ 0.0016 — so the
# entire usable window is the last ~0.015 below β_c, and the scan has to sit
# inside it. The closest point is held at 0.7600 (ξ ≈ 6 ≈ L/4): nearer than
# that and finite volume bends the mass back up, as it did at SU(2) β=2.7.
BETAS = [0.7450, 0.7520, 0.7560, 0.7585, 0.7600]

L = 24  # spatial extent; keeps ξ ≲ L/4 = 6 measurable without finite volume
D = 3  # 2 spatial + 1 temporal. Time is axis 0
LT = 48  # long temporal extent: near β_c the correlator decays slowly
N_CONFIGS = 2000  # the first scan's N=1000 gave errors larger than the mass
N_THERM = 500
# Metropolis decorrelation. A Z₂ sweep at this size costs ~0.010 s, so skipping
# generously is nearly free and worth it: this is a *fixed* skip across a scan
# whose autocorrelation time grows towards β_c (local updates decorrelate like
# ξ^z, z ≈ 2). The τ_int column exists to tell you whether it was enough —
# raise this, or drop the closest β, if τ_int approaches N_SKIP.
N_SKIP = 50  # measured Metropolis acceptance is only ~0.10 (the Z₂ proposal is
#              a flip, U → −U, and most flips cost too much action), so 20
#              sweeps bought only ~2 accepted updates per link. Sweeps are
#              nearly free here; buy more of them.
SMEAR_ALPHA = 0.5
SMEAR_LEVELS = [0, 2, 4, 8]  # only 2 spatial directions in 3D, so each smearing
#   step is weak (one staple pair) — reach further in levels to compensate, and
#   give the GEVP a wider basis where the signal is weakest
GEVP_T0 = 1
QUOTE_DELTA = 3  # small masses ⇒ read the plateau a little further out

gaugegroup = Z2()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cache_path(beta):
    return f"datasets/z2_configs_L{L}_Lt{LT}_b{beta}_N{N_CONFIGS}.pt"


def ensemble(beta):
    path = cache_path(beta)
    if os.path.exists(path):
        print(f"  loading cached {path}")
        return torch.load(path)
    print(f"  sampling N={N_CONFIGS} at β={beta} …")
    configs, acc = mcmc_ensemble(
        L=L, D=D, gaugegroup=gaugegroup, beta=beta, n_configs=N_CONFIGS,
        n_therm=N_THERM, n_skip=N_SKIP, progress=True, Lt=LT,
    )
    print(f"  acceptance = {acc:.2f}")
    os.makedirs("datasets", exist_ok=True)
    torch.save(configs, path)
    return configs


def measure(configs):
    """Plateau mass, its jackknife error, the full m_eff curve, and τ_int."""
    Obar = smearing_operator_basis(
        configs.to(device), gaugegroup, SMEAR_LEVELS, alpha=SMEAR_ALPHA, progress=True
    ).double().cpu()
    meff, err = jackknife_gevp_effective_mass(Obar, t0=GEVP_T0)
    m_ground, err_ground = meff[:, 0], err[:, 0]
    # Chain observable for τ_int: the most-smeared zero-momentum operator,
    # averaged over time slices — one number per configuration, in chain order.
    series = Obar[-1].mean(dim=1)
    _, tau, _ = integrated_autocorrelation_time(series)
    return (
        m_ground[QUOTE_DELTA].item(),
        err_ground[QUOTE_DELTA].item(),
        m_ground.cpu(),
        float(tau),
    )


def main():
    print(f"device: {device} | Z₂ 3D, β_c ≈ {BETA_C}")
    rows = []
    for beta in BETAS:
        print(f"\n── β = {beta}  (β_c − β = {BETA_C - beta:.4f}) " + "─" * 26)
        m, err, curve, tau = measure(ensemble(beta))
        xi = 1.0 / m if m > 0 else float("inf")
        xi_err = xi * err / m if m > 0 else float("inf")
        rows.append((beta, m, err, xi, xi_err, curve, tau))
        print(f"  m·a(Δ={QUOTE_DELTA}) = {m:.4f} ± {err:.4f}   ξ = {xi:.2f} ± {xi_err:.2f}"
              f"   τ_int = {tau:.1f}")
        print("  m_eff(Δ): " + "  ".join(f"{v:.3f}" for v in curve[:8].tolist()))

    print("\n" + "=" * 72)
    print(f"{'β':>7} {'β_c−β':>8} {'m·a':>18} {'ξ = 1/m':>16} {'τ_int':>8}")
    print("-" * 72)
    for beta, m, err, xi, xi_err, _, tau in rows:
        print(f"{beta:>7} {BETA_C - beta:>8.4f} {m:>9.4f} ± {err:.4f}"
              f" {xi:>8.2f} ± {xi_err:.2f} {tau:>8.1f}")
    span = max(r[3] for r in rows) / min(r[3] for r in rows)
    xi_max = max(r[3] for r in rows)
    tau_max = max(r[6] for r in rows)
    print("-" * 72)
    print(f"ξ spans ×{span:.2f}, reaching {xi_max:.2f} lattice spacings.")
    print(
        "GATE: ξ must reach ≳3–4 spacings (so ℓ_att, quantized on integer\n"
        "      offsets, can resolve it) AND span a factor ≳2. If both hold,\n"
        "      Phase B is worth its training runs.\n"
        f"CHECK: max τ_int = {tau_max:.1f} against N_SKIP = {N_SKIP}. If τ_int is\n"
        "      comparable to or larger than N_SKIP, the chain is undersampled\n"
        "      near β_c and the quoted errors are too small — raise N_SKIP.\n"
        f"CHECK: ξ_max = {xi_max:.2f} against L/4 = {L / 4:.1f}. If ξ approaches\n"
        "      that, finite volume is bending the mass and the point is a bound."
    )

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    b = [r[0] for r in rows]
    ax[0].errorbar(b, [r[1] for r in rows], yerr=[r[2] for r in rows], fmt="o-", capsize=4)
    ax[0].axvline(BETA_C, ls=":", color="red", label=r"$\beta_c$")
    ax[0].set_xlabel(r"$\beta$"); ax[0].set_ylabel(r"$m\,a$")
    ax[0].set_title("Mass gap closes at the transition")
    ax[0].legend(); ax[0].grid(True, alpha=0.3)

    ax[1].errorbar(b, [r[3] for r in rows], yerr=[r[4] for r in rows], fmt="o-", capsize=4)
    ax[1].axhline(L / 4, ls="--", color="gray", label=f"L/4 = {L / 4:.0f} (finite volume)")
    ax[1].axhline(1.0, ls=":", color="red", label="SU(2) glueball ceiling")
    ax[1].set_xlabel(r"$\beta$"); ax[1].set_ylabel(r"$\xi = 1/(m\,a)$  [lattice spacings]")
    ax[1].set_title(f"The x-axis of the attention-range plot\n(reaches {xi_max:.1f}, spans ×{span:.1f})")
    ax[1].legend(); ax[1].grid(True, alpha=0.3)

    for beta, _, _, _, _, curve, _ in rows:
        ax[2].plot(range(1, min(9, len(curve))), curve[1:9], "o-", label=f"β={beta}")
    ax[2].set_xlabel(r"$\Delta$"); ax[2].set_ylabel(r"$m_{\rm eff}(\Delta)$")
    ax[2].set_title("GEVP ground state — check the plateau")
    ax[2].legend(fontsize=8); ax[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"Z₂ 3D β-scan — {L}²×{LT}, N={N_CONFIGS}: does ξ get large enough?",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig("z2_beta_scan.png", dpi=130, bbox_inches="tight")
    print("\nSaved z2_beta_scan.png")

    torch.save(
        {
            "betas": b, "beta_c": BETA_C,
            "m": [r[1] for r in rows], "m_err": [r[2] for r in rows],
            "xi": [r[3] for r in rows], "xi_err": [r[4] for r in rows],
            "m_eff_curves": [r[5] for r in rows], "tau_int": [r[6] for r in rows],
            "meta": {"L": L, "Lt": LT, "D": D, "N": N_CONFIGS,
                     "n_skip": N_SKIP, "quote_delta": QUOTE_DELTA},
        },
        "datasets/z2_beta_scan.pt",
    )
    print("Saved datasets/z2_beta_scan.pt")


if __name__ == "__main__":
    main()
