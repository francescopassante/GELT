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

Runtime note: with N_SKIP raised to cover the measured τ_int, sampling is now
the long pole — ~2000 × 200 sweeps × 0.010 s ≈ 70 min per β, so budget most of
a night for five. The analysis is comparatively cheap: the jackknife is
*blocked*, so it does N/JACK_BLOCK fits rather than one per configuration.

Run:
    python scripts/z2_beta_scan.py

Writes ``results/attention/z2_beta_scan.{png,pt}``.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from gelt.glueball import (
    connected_correlator,
    connected_correlator_matrix,
    fit_cosh_correlator,
    gevp_effective_mass,
    gevp_eigenvalues,
    gevp_ground_vector,
    smearing_operator_basis,
)
from gelt.lattice import Z2
from gelt.sampler import (

    integrated_autocorrelation_time,
    mcmc_ensemble,
    z2_heatbath_sweep,
)

# Output artifacts are grouped by study under results/; create the dirs the
# first time this runs in a fresh clone (they hold generated files only).
for _d in ("results/sampler", "results/glueball", "results/attention",
          "results/wilson_regression", "datasets"):
    os.makedirs(_d, exist_ok=True)



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
N_SKIP = 200  # τ_int measured 43.8 at β=0.760 with N_SKIP=20 — i.e. the chain
#               decorrelated FOUR TIMES SLOWER than it was sampled, so the
#               configurations were not independent and the errors were
#               fiction. This is ~4·τ_int at the worst β. Critical slowing down
#               is intrinsic to approaching β_c with local updates (τ ~ ξ^z,
#               z ≈ 2); the heat-bath removed the rejection problem, not this.
SMEAR_ALPHA = 0.5
SMEAR_LEVELS = [0, 4, 8, 16]  # only 2 spatial directions in 3D, so each smearing
#   step is weak (one staple pair). The ladder has to reach much further to
#   build any ground-state overlap: every scan so far showed m_eff(Δ=1) ≈ 4–5
#   collapsing to ~0.3 by Δ=2, i.e. C(1)/C(2) ≈ 80 — the operator was almost
#   pure excited state, so the ground state only surfaced where the signal had
#   already died.
GEVP_T0 = 1
GEVP_TD = 2  # Δ at which the ground-state eigenvector is defined
FIT_WINDOW = (2, 8)  # cosh-fit window; starts past the contact-term drop at Δ=1
JACK_BLOCK = 20  # blocked jackknife: neighbouring configs are correlated at
#                  large τ_int, so single deletion would understate the error

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
    # Heat-bath, NOT the registry-default Metropolis: the Z₂ flip proposal's
    # acceptance measured 0.05 → 0.02 across this β range, so Metropolis leaves
    # the chain essentially frozen exactly where ξ is largest.
    configs, acc = mcmc_ensemble(
        L=L, D=D, gaugegroup=gaugegroup, beta=beta, n_configs=N_CONFIGS,
        n_therm=N_THERM, n_skip=N_SKIP, progress=True, Lt=LT,
        sweep_fn=z2_heatbath_sweep,
    )
    print(f"  acceptance = {acc:.2f}")
    os.makedirs("datasets", exist_ok=True)
    torch.save(configs, path)
    return configs


def _fit_from_obar(Obar):
    """Mass from a cosh fit to the GEVP-projected correlator.

    Reading a single m_eff(Δ) point throws away every other Δ and inherits the
    noise of one ratio — which is why the first two scans returned errors of the
    same size as the mass. Projecting onto the GEVP ground-state vector and
    fitting the whole window is the way masses are actually quoted (see the
    report's "Quoting the result" section) and uses all the data.
    """
    C = connected_correlator_matrix(Obar)
    v0 = gevp_ground_vector(C, t0=GEVP_T0, td=GEVP_TD)
    proj = torch.einsum("i,ibt->bt", v0, Obar)  # (B, Nt) projected operator
    m, _A, _chi2 = fit_cosh_correlator(connected_correlator(proj), *FIT_WINDOW)
    return m


def measure(configs):
    """Fitted mass, its blocked-jackknife error, the m_eff curve, and τ_int."""
    Obar = smearing_operator_basis(
        configs.to(device), gaugegroup, SMEAR_LEVELS, alpha=SMEAR_ALPHA, progress=True
    ).double().cpu()

    m = _fit_from_obar(Obar)

    # Blocked jackknife over configurations. Blocks, not single deletions,
    # because τ_int is large near β_c and neighbouring configurations are
    # correlated; deleting one at a time would understate the error exactly
    # where it matters. It is also ~N/JACK_BLOCK times cheaper.
    n = Obar.shape[1]
    n_blocks = n // JACK_BLOCK
    vals = []
    for b in range(n_blocks):
        keep = torch.ones(n, dtype=torch.bool)
        keep[b * JACK_BLOCK : (b + 1) * JACK_BLOCK] = False
        vals.append(_fit_from_obar(Obar[:, keep]))
    vals = torch.tensor(vals, dtype=torch.float64)
    err = ((n_blocks - 1) / n_blocks * (vals - vals.mean()).pow(2).sum()).sqrt().item()

    # m_eff curve, kept only so the plateau can be inspected by eye.
    lams = gevp_eigenvalues(connected_correlator_matrix(Obar), t0=GEVP_T0)
    m_curve = gevp_effective_mass(lams)[:, 0]
    # Chain observable for τ_int: the most-smeared zero-momentum operator,
    # averaged over time slices — one number per configuration, in chain order.
    _, tau, _ = integrated_autocorrelation_time(Obar[-1].mean(dim=1))
    return m, err, m_curve.cpu(), float(tau)


def main():
    print(f"device: {device} | Z₂ 3D, β_c ≈ {BETA_C}")
    rows = []
    for beta in BETAS:
        print(f"\n── β = {beta}  (β_c − β = {BETA_C - beta:.4f}) " + "─" * 26)
        m, err, curve, tau = measure(ensemble(beta))
        xi = 1.0 / m if m > 0 else float("inf")
        xi_err = xi * err / m if m > 0 else float("inf")
        rows.append((beta, m, err, xi, xi_err, curve, tau))
        print(f"  m·a (cosh fit Δ∈{FIT_WINDOW}) = {m:.4f} ± {err:.4f}   ξ = {xi:.2f} ± {xi_err:.2f}"
              f"   τ_int = {tau:.1f}")
        print("  m_eff(Δ): " + "  ".join(f"{v:.3f}" for v in curve[:8].tolist()))

    print("\n" + "=" * 72)
    print(f"{'β':>7} {'β_c−β':>8} {'m·a':>18} {'ξ = 1/m':>16} {'τ_int':>8}")
    print("-" * 72)
    for beta, m, err, xi, xi_err, _, tau in rows:
        print(f"{beta:>7} {BETA_C - beta:>8.4f} {m:>9.4f} ± {err:.4f}"
              f" {xi:>8.2f} ± {xi_err:.2f} {tau:>8.1f}")
    # Only RESOLVED points may enter the gate. The first scan "passed" on a
    # point with m = 0.025 ± 0.216 — consistent with zero — whose ξ = 39.7 ± 339
    # single-handedly produced a ×18 span. A ratio built from unresolved
    # numbers is not a measurement.
    resolved = [r for r in rows if r[1] > 0 and r[1] / r[2] >= 3.0]
    tau_max = max(r[6] for r in rows)
    print("-" * 72)
    print(f"resolved points (m/err ≥ 3): {len(resolved)}/{len(rows)}"
          + ("" if not resolved else "  at β = "
             + ", ".join(f"{r[0]}" for r in resolved)))
    if len(resolved) < 3:
        print("GATE FAILS: fewer than 3 resolved points — the scan has no curve to")
        print("      fit. Fix the sampling before reading anything else below.")
        span = xi_max = float("nan")
    else:
        span = max(r[3] for r in resolved) / min(r[3] for r in resolved)
        xi_max = max(r[3] for r in resolved)
        print(f"ξ spans ×{span:.2f}, reaching {xi_max:.2f} lattice spacings"
              " (resolved points only).")
        # The mass must fall monotonically towards β_c. Scatter means noise.
        ms = [r[1] for r in resolved]
        if any(b - a > max(r[2] for r in resolved) for a, b in zip(ms, ms[1:])):
            print("WARNING: m is not monotonically decreasing towards β_c even on the")
            print("      resolved points — that is noise, not critical behaviour.")
    print(
        "GATE: ξ must reach ≳3–4 spacings (so ℓ_att, quantized on integer\n"
        "      offsets, can resolve it) AND span a factor ≳2, on RESOLVED\n"
        "      points. If both hold, Phase B is worth its training runs.\n"
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
    fig.savefig("results/attention/z2_beta_scan.png", dpi=130, bbox_inches="tight")
    print("\nSaved z2_beta_scan.png")

    torch.save(
        {
            "betas": b, "beta_c": BETA_C,
            "m": [r[1] for r in rows], "m_err": [r[2] for r in rows],
            "xi": [r[3] for r in rows], "xi_err": [r[4] for r in rows],
            "m_eff_curves": [r[5] for r in rows], "tau_int": [r[6] for r in rows],
            "meta": {"L": L, "Lt": LT, "D": D, "N": N_CONFIGS,
                     "n_skip": N_SKIP, "fit_window": FIT_WINDOW, "jack_block": JACK_BLOCK},
        },
        "results/attention/z2_beta_scan.pt",
    )
    print("Saved results/attention/z2_beta_scan.pt")


if __name__ == "__main__":
    main()
