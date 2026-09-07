"""What did the network find? — the learned operator decomposed against the
span of the classical basis.

§6.2 reports that the learned operator carries more ground-state weight than
the optimal combination of the classical Morningstar–Peardon basis,
ΔA₀ = +0.078 ± 0.022 (3.6σ). That is a statement that GELT *wins*. It is not a
statement about *what it found*, and a referee who accepts the win asks the next
question immediately: is the advantage new operator content, or is it the same
content in a combination the GEVP failed to locate?

The question has an exact answer, because operators acting on the vacuum are
vectors in a Hilbert space and the equal-time connected correlator is its inner
product:

    ⟨O_a, O_b⟩ ≡ C_ab(0) = ⟨ d_a(t) d_b(t) ⟩,   d = Ō − ⟨Ō⟩,

so "the part of GELT the classical basis cannot express" is a literal orthogonal
projection. Write

    O_GELT = P + r,    P = the orthogonal projection onto span{Ō_i},   r ⟂ span,

and every quantity below is a property of that split. Nothing is fitted to make
it come out any particular way; P is the *best possible* classical approximation
of GELT, better by construction than any classical operator chosen for any other
reason, so ΔA₀(GELT − P) is a lower bound on GELT's advantage over the span.

Why this is the sharper comparison
----------------------------------
The published ΔA₀ compares GELT to the classical GEVP ground vector — a
different operator, correlated with GELT only through sharing configurations.
P is GELT's *own* projection, so the two share every fluctuation that lives in
the span and the jackknifed difference is much better conditioned. It also
answers a different and stronger question: not "is our operator better than
theirs" but "is our operator's advantage outside their reach".

The calibration that makes 13% mean something
---------------------------------------------
A number like "12.9% of the norm² lies outside a four-dimensional span" is
meaningless without a scale — *any* operator outside a 4D subspace has a large
orthogonal component. The scale is the basis's own increments: how much new
content does each rung of the smearing ladder add to the span of the rungs
below it? That is the same statistic applied to the classical operators
themselves, it is free, and it says whether the ladder is still growing or has
saturated.

What each outcome licenses
--------------------------
* **r is a large fraction of the norm AND carries a matching share of the
  ground-state amplitude Z** → the network found operator content the classical
  basis does not span, and that content is what the win is made of. Report the
  decomposition beside the ΔA₀, and quote the ladder increments as the scale.
* **r is large in norm but carries no Z** → the network is decorating the
  classical answer with noise that happens not to hurt; the win is then a
  statistical accident of the GEVP and should be reported as such.
* **r shrinks as the metric time grows** → r is a contact term, i.e. UV content
  that dies by Δ = 2, exactly like the E-irrep contamination in
  `notes/rotational_symmetry.md`. It would then be irrelevant to the mass and
  must not be quoted as physics. The metric scan below is the test.
* **ΔA₀(GELT − P) ≈ 0** → the advantage lives *inside* the span after all and
  the GEVP simply failed to find it. That is still a real result, but it is a
  statement about estimator variance, not about operator content.

What this does NOT establish
----------------------------
Two controls are missing and neither can be run from a dump:

1. **A random-init GELT.** If an untrained network is also ~13% outside the
   span, then being outside is architectural (the transport reaches offsets no
   smeared plaquette does) rather than learned, and the learned part of the
   claim rests on Z_r/Z_G and ΔA₀ alone. One GPU eval pass, no training.
2. **A stronger classical span.** This decomposes against the *published*
   basis. `scripts/su2_fair_fight.py` builds the strengthened ones
   (deep / shapes / full); the decomposition belongs there too, against `full`.

Both are stated here so the numbers below are read with them attached.

Run:
    python scripts/operator_decomposition.py [path/to/…_test_obars.pt ...]
"""

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from gelt.glueball import (  # noqa: E402
    connected_correlator,
    connected_correlator_matrix,
    fit_cosh_correlator,
    gevp_ground_vector,
)

os.makedirs("results/glueball", exist_ok=True)


# ── Tunables ──────────────────────────────────────────────────────────────────
# Defaults are the two independently sampled ensembles behind the §6.2 headline.
DUMPS = sys.argv[1:] or [
    "results/glueball/best_glueball_gelt_sm0-2-4-6_test_obars.pt",
    "results/glueball/best_glueball_gelt_sm0-2-4-6_ens1_test_obars.pt",
]
FIT_WINDOW = (2, 7)  # identical to fit_glueball_overlap.py — comparability
M_RANGE = (0.05, 1.5)
METRIC_TIMES = (0, 1, 2)  # ⟨O_a(t+τ) O_b(t)⟩ metrics; τ>0 down-weights UV noise
OUT_PNG = "results/glueball/operator_decomposition.png"
OUT_PT = "results/glueball/operator_decomposition.pt"


# ── Blocked jackknife (same delete-block scheme as fit_glueball_overlap.py) ───
def _blocks(B, bs):
    idx = torch.arange(B)
    return [idx[i : i + bs] for i in range(0, B, bs)]


def blocked_jackknife(fn, B, bs):
    samples = []
    for bl in _blocks(B, bs):
        mask = torch.ones(B, dtype=torch.bool)
        mask[bl] = False
        samples.append(fn(mask))
    samples = torch.stack(samples)
    n = len(samples)
    mean = samples.mean(dim=0)
    return mean, ((n - 1) / n * ((samples - mean) ** 2).sum(dim=0)).sqrt()


# ── The decomposition ─────────────────────────────────────────────────────────
def vacuum_subtract(O):
    """d = Ō − ⟨Ō⟩ with the VEV taken over configs *and* timeslices, exactly as
    connected_correlator does internally (0⁺⁺ has a nonzero VEV)."""
    return O - O.mean() if O.dim() == 2 else O - O.mean(dim=(1, 2), keepdim=True)


def decompose(target, basis, tau=0):
    """Split ``target`` into its projection onto span(basis) and the remainder.

    The metric is C_ab(τ) symmetrised in (a, b). τ = 0 is the Hilbert-space
    inner product on the states O|vac⟩ and is the one to quote; τ > 0 weights
    the light states more heavily and is the contact-term test.
    """
    dT, dB = vacuum_subtract(target), vacuum_subtract(basis)
    n_t = dB[0].numel()
    M = torch.einsum("ibt,jbt->ij", dB.roll(-tau, dims=2), dB) / n_t
    M = 0.5 * (M + M.T)  # the true matrix is symmetric; the estimator is not
    v = 0.5 * (
        torch.einsum("ibt,bt->i", dB.roll(-tau, dims=2), dT)
        + torch.einsum("ibt,bt->i", dB, dT.roll(-tau, dims=1))
    ) / dT.numel()
    coeff = torch.linalg.solve(M, v)
    P = torch.einsum("i,ibt->bt", coeff, dB)
    return dT, P, dT - P, coeff


def cross_correlator(a, b):
    """⟨a(t+Δ) b(t)⟩ — the off-diagonal partner of connected_correlator."""
    Nt = a.shape[1]
    return torch.stack([(a.roll(-d, dims=1) * b).mean() for d in range(Nt)])


def analyse(dump):
    blob = torch.load(dump, weights_only=False)
    G = blob["gelt_obar"].double()
    basis = blob["Obar_basis"].double()
    meta = blob.get("meta", {})
    t0 = int(meta.get("gevp_t0", 1))
    td = t0 + 1
    jb = int(meta.get("jack_block", 10))
    levels = meta.get("gevp_levels", list(range(basis.shape[0])))
    B, Nt = G.shape
    dmin, dmax = FIT_WINDOW

    print(f"\n{'=' * 78}\n{dump}\n{'=' * 78}")
    print(
        f"  {B} test configs × Nt = {Nt} | classical levels {levels} | "
        f"window Δ ∈ [{dmin}, {dmax}] | block {jb} → {-(-B // jb)} samples"
    )

    def gevp_projected(mask):
        b = basis[:, mask]
        v0 = gevp_ground_vector(connected_correlator_matrix(b), t0=t0, td=td)
        return torch.einsum("i,ibt->bt", v0, b)

    # Diagonal χ² weights are fixed from the full sample so every jackknife
    # sample minimises the same surface (the standard jackknife-of-fit protocol).
    _, sig_G = blocked_jackknife(lambda m: connected_correlator(G[m]), B, jb)
    _, sig_P = blocked_jackknife(
        lambda m: connected_correlator(decompose(G[m], basis[:, m])[1]), B, jb
    )
    _, sig_r = blocked_jackknife(
        lambda m: connected_correlator(decompose(G[m], basis[:, m])[2]), B, jb
    )
    # The GEVP arm exists only as a gate, so its weights follow
    # fit_glueball_overlap.py to the letter: σ from the FULL-sample projection
    # sliced by the mask (while the fit itself re-solves v₀ per sample). Deriving
    # σ from the per-sample projection instead shifts ΔA₀ by ~0.5σ, which is
    # enough to make the gate look like it failed when nothing has changed.
    proj_full = gevp_projected(torch.ones(B, dtype=torch.bool))
    _, sig_V = blocked_jackknife(
        lambda m: connected_correlator(proj_full[m]), B, jb
    )

    def fit_a0(C, sigma):
        m, A, _ = fit_cosh_correlator(C, dmin, dmax, sigma=sigma, m_range=M_RANGE)
        return m, A * (1.0 + math.exp(-m * Nt)) / C[0].item()

    # Ground-state amplitudes are extracted at a COMMON fixed mass. Z is linear
    # in the operator, so Z_G = Z_P + Z_r holds exactly only if all three are
    # read against the same exponential; letting each fit its own m breaks the
    # additivity that makes the amplitude split interpretable.
    m_ref, _ = fit_a0(connected_correlator(G), sig_G)
    cosh_ref = torch.tensor(
        [math.exp(-m_ref * d) + math.exp(-m_ref * (Nt - d)) for d in range(Nt)],
        dtype=torch.float64,
    )
    w = 1.0 / torch.clamp(sig_G, min=1e-30) ** 2
    win = slice(dmin, dmax + 1)

    def amplitude(C):
        """Least-squares A of C(Δ) ≈ A·cosh_ref(Δ) over the window (linear in O)."""
        return (w[win] * cosh_ref[win] * C[win]).sum() / (
            w[win] * cosh_ref[win] ** 2
        ).sum()

    def stats(mask):
        dG, P, r, _ = decompose(G[mask], basis[:, mask])
        m_G, a0_G = fit_a0(connected_correlator(dG), sig_G)
        m_P, a0_P = fit_a0(connected_correlator(P), sig_P)
        m_r, a0_r = fit_a0(connected_correlator(r), sig_r)
        m_V, a0_V = fit_a0(connected_correlator(gevp_projected(mask)), sig_V)
        # Z_r/Z_G from the cross-correlator: C_Gr(Δ) → Z_G Z_r e^{−mΔ} while
        # C_GG(Δ) → Z_G², so the ratio of amplitudes is Z_r/Z_G with its sign.
        z_share = amplitude(cross_correlator(dG, r)) / amplitude(
            connected_correlator(dG)
        )
        out = [
            (r**2).mean() / (dG**2).mean(),  # norm² fraction outside the span
            z_share,
            m_G, a0_G, m_P, a0_P, m_r, a0_r, m_V, a0_V,
            a0_G - a0_P,  # the correlated advantage over the whole span
            a0_G - a0_V,  # the published comparison, for the gate
            m_G - m_P,
        ]
        # The ladder's own increments: new content of rung k over the rungs below.
        for k in range(1, basis.shape[0]):
            _, _, rk, _ = decompose(basis[k][mask], basis[:k][:, mask])
            dk = vacuum_subtract(basis[k][mask])
            out.append((rk**2).mean() / (dk**2).mean())
        # Contact-term test: the same fraction under metrics C(τ), τ > 0.
        for tau in METRIC_TIMES[1:]:
            _, _, rt, _ = decompose(G[mask], basis[:, mask], tau=tau)
            out.append((rt**2).mean() / (dG**2).mean())
        return torch.tensor(out, dtype=torch.float64)

    mu, err = blocked_jackknife(stats, B, jb)
    n_lad = basis.shape[0] - 1

    def show(i, label, fmt="{:+.4f}"):
        s = abs(mu[i].item()) / max(err[i].item(), 1e-12)
        print(
            f"  {label:<50} {fmt.format(mu[i].item())} ± {err[i].item():.4f}"
            f"   ({s:5.1f}σ)"
        )

    print("\n  gate — the published comparison, reproduced from this code path")
    show(2, "m·a_t  GELT")
    show(3, "A₀     GELT")
    show(9, "A₀     classical GEVP ground vector")
    show(11, "ΔA₀    GELT − classical GEVP (published: +0.066/+0.089)")

    print("\n  O_GELT = P + r,  P = orthogonal projection onto the classical span")
    show(0, "norm² fraction of O_GELT outside the span")
    show(1, "Z_r/Z_G — share of the ground-state amplitude in r")
    show(5, "A₀     P  (best classical approximation of GELT)")
    show(7, "A₀     r  (the orthogonal remainder, on its own)")
    show(10, "ΔA₀    GELT − P   ← the advantage that is outside the span")
    show(12, "Δm     GELT − P   (consistent with 0 ⇔ same state)")

    print("\n  scale — new content each rung of the classical ladder adds")
    for k in range(1, basis.shape[0]):
        below = ", ".join(f"×{x}" for x in levels[:k])
        show(12 + k, f"APE×{levels[k]} outside span{{{below}}}")
    show(0, "GELT   outside span{the whole basis}")

    print("\n  contact-term test — the same fraction under the C(τ) metric")
    for j, tau in enumerate(METRIC_TIMES[1:]):
        show(12 + n_lad + 1 + j, f"metric C({tau})")
    print(
        "    (a contact term SHRINKS with τ — it lives in C(0) and is gone by Δ=2)"
    )

    _, P_f, r_f, coeff = decompose(G, basis)
    print("\n  regression coefficients onto " + ", ".join(f"APE×{x}" for x in levels)
          + ": " + ", ".join(f"{c:+.3f}" for c in coeff))
    return dict(
        dump=dump, mu=mu, err=err, levels=levels, Nt=Nt, B=B, m_ref=m_ref,
        n_ladder=n_lad, C_G=connected_correlator(vacuum_subtract(G)),
        C_P=connected_correlator(P_f), C_r=connected_correlator(r_f),
    )


def plot(runs):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # Left: the ladder's increments against GELT's, on a log scale — the point
    # is that the classical increments fall geometrically while GELT's does not
    # sit on that curve.
    for run, mk, col in zip(runs, ("o-", "s--"), ("C0", "C3")):
        mu, err, lv, n = run["mu"], run["err"], run["levels"], run["n_ladder"]
        xs = list(range(1, n + 1))
        ys = [mu[12 + k].item() for k in xs]
        es = [err[12 + k].item() for k in xs]
        tag = os.path.basename(run["dump"]).replace("_test_obars.pt", "")
        tag = tag.rsplit("_", 1)[-1] if tag.endswith(("_ens1", "_ens2")) else "Run 5"
        ax[0].errorbar(xs, ys, yerr=es, fmt=mk, color=col, capsize=3,
                       label=f"classical ladder — {tag}")
        ax[0].errorbar([n + 1], [mu[0].item()], yerr=[err[0].item()], fmt="*",
                       ms=16, color=col, capsize=3)
    n = runs[0]["n_ladder"]
    lv = runs[0]["levels"]
    ax[0].set_xticks(list(range(1, n + 2)))
    ax[0].set_xticklabels([f"APE×{lv[k]}" for k in range(1, n + 1)] + ["GELT"])
    ax[0].set_yscale("log")
    ax[0].set_ylabel("norm² fraction outside the span of everything below")
    ax[0].set_title("New operator content per addition\n(★ = the learned operator)")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(fontsize=8)

    # Right: the overlap panel for the three pieces. ρ(Δ) flat at A₀ ⇔ pure
    # ground state; r is plotted to show it is not a contact term.
    run = runs[0]
    Nt, m = run["Nt"], run["m_ref"]
    ref = torch.tensor(
        [(math.exp(-m * d) + math.exp(-m * (Nt - d)))
         / (1 + math.exp(-m * Nt)) for d in range(Nt)], dtype=torch.float64
    )
    dmax = FIT_WINDOW[1] + 1
    for C, lab, col in ((run["C_G"], "O_GELT", "C0"),
                        (run["C_P"], "P  (in the classical span)", "C1"),
                        (run["C_r"], "r  (orthogonal remainder)", "C3")):
        rho = (C / C[0]) / ref
        ax[1].plot(range(1, dmax + 1), rho[1 : dmax + 1], "o-", color=col, label=lab)
    ax[1].axvspan(FIT_WINDOW[0] - 0.3, FIT_WINDOW[1] + 0.3, color="grey", alpha=0.12)
    ax[1].set_xlabel("Δ")
    ax[1].set_ylabel("ρ(Δ) = [C(Δ)/C(0)] / cosh_ref")
    ax[1].set_title("Ground-state purity of each piece\n(flat ⇔ pure; shaded = fit window)")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(fontsize=8)

    fig.suptitle(
        "The learned operator decomposed against the classical span — "
        "SU(2) 12³×24, β = 2.4, ξ = 3", fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"\nSaved {OUT_PNG}")


def main():
    runs = [analyse(d) for d in DUMPS]
    if len(runs) > 1:
        # Combine the correlated advantages across independent ensembles the
        # same way §6.2 combines its ΔA₀: inverse-variance weighted.
        vals = torch.tensor([r["mu"][10] for r in runs])
        errs = torch.tensor([r["err"][10] for r in runs])
        wgt = 1.0 / errs**2
        comb, cerr = (vals * wgt).sum() / wgt.sum(), (1.0 / wgt.sum()).sqrt()
        print(f"\n{'=' * 78}\ncombined over {len(runs)} independent ensembles")
        print(
            f"  ΔA₀ (GELT − P) = {comb:+.4f} ± {cerr:.4f} "
            f"({abs(comb) / cerr:.1f}σ)  — the advantage outside the classical span"
        )
    plot(runs)
    torch.save({"runs": [{k: v for k, v in r.items()} for r in runs],
                "fit_window": FIT_WINDOW, "metric_times": METRIC_TIMES}, OUT_PT)
    print(f"Saved {OUT_PT}")


if __name__ == "__main__":
    main()
