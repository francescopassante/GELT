"""S1 task 5 (notes/prof_notes.md): two-state fit with the ground (and excited)
mass shared between GELT and the projected GEVP, simultaneous on both
correlators, jackknifed. Offline.

    C_i(Δ) = A_i φ(m₀, Δ) + B_i φ(m₁, Δ),   φ(m, Δ) = e^{−mΔ} + e^{−m(Nt−Δ)},
    i ∈ {GELT, GEVP-projected},  m₀ < m₁ shared,  A_i, B_i free.

For fixed (m₀, m₁) the amplitudes are linear and decouple per operator, so they
are profiled out in closed form (2×2 normal equations, diagonal weights 1/σ_Δ²
from the full-sample blocked jackknife — the repo's convention) and (m₀, m₁)
are found on a grid + a local refinement; no scipy, no convergence failures
inside the jackknife. A₀_i = A_i φ(m₀, 0)/C_i(0) as in the one-state fit, now
with the excited state out of the way and the mass the same for both operators.

Window default [1, 8]; m₁ ∈ (m₀ + 0.1, 1.5]. If m₁ sits at the grid edge in
many jackknife samples the excited mass is not constrained and the script says
so — the prof's instruction is then to say it and stop, and not to read B_i.

--dump=<file> fits one high-statistics dump (scripts/high_stat_eval.py) instead of
the three 400-configuration ensembles; its own Obar_basis is the `published` basis
(so --arm=deep is refused) and --block=<n> is the jackknife block in configs
(default 100; choose it with block_size_scan.py — the chain is n_skip sweeps apart).

Usage:  python scripts/two_state_fit.py [--window=1-8] [--arm=published|deep]
                                         [--dump=<file> [--block=100]]
"""

import math
import os
import sys

import torch

_known = ("--window=", "--arm=", "--dump=", "--block=")
_bad = [a for a in sys.argv[1:] if a.startswith("--") and not a.startswith(_known)]
if _bad:
    raise SystemExit(f"unknown flag(s) {_bad}; known: {list(_known)}")


def _flag(name, default):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{name}=")),
                default)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_glueball_overlap as fgo  # noqa: E402

from gelt.glueball import (  # noqa: E402
    connected_correlator,
    connected_correlator_matrix,
    gevp_ground_vector,
)

WINDOW = tuple(int(x) for x in _flag("window", "1-8").split("-"))
ARM = _flag("arm", "published")
DUMP = _flag("dump", None)
BLOCK = int(_flag("block", "100"))
if DUMP and ARM != "published":
    raise SystemExit("--dump= carries only the published (4-level) basis")
EPS, TRUNC = (1e-12, False) if ARM == "published" else (1e-4, True)
M0 = torch.linspace(0.15, 0.7, 111, dtype=torch.float64)
M1_MAX, M1_GAP, N_M1 = 1.5, 0.1, 141
ENS = {
    "run5": ("best_glueball_gelt_sm0-2-4-6_test_obars.pt", "su2_fair_fight_obars_run5.pt"),
    "ens1": ("best_glueball_gelt_sm0-2-4-6_ens1_test_obars.pt", "su2_fair_fight_obars_ens1.pt"),
    "ens2": ("best_glueball_gelt_sm0-2-4-6_ens2_test_obars.pt", "su2_fair_fight_obars_ens2.pt"),
}
OUT = f"results/glueball/two_state_fit_{ARM}"
if DUMP:
    ENS = {"hs": (DUMP, None)}
    OUT += "_" + os.path.basename(DUMP).replace("_test_obars.pt", "")


def _phi(m, d, Nt):
    return torch.exp(-m[..., None] * d) + torch.exp(-m[..., None] * (Nt - d))


def fit2(Cs, sigs, window):
    """Shared-mass two-state fit. Cs, sigs: lists of (Nt,) → dict of the results."""
    Nt = Cs[0].shape[0]
    dmin, dmax = window
    d = torch.arange(dmin, dmax + 1, dtype=torch.float64)
    y = torch.stack([C[dmin : dmax + 1] for C in Cs]).double()  # (ops, n)
    w = torch.stack([1.0 / s[dmin : dmax + 1].double() ** 2 for s in sigs])

    def chi2(m0, m1):
        """χ² on broadcastable (m0, m1) grids, amplitudes profiled out."""
        f0, f1 = _phi(m0, d, Nt), _phi(m1, d, Nt)  # (..., n)
        tot, amps = 0.0, []
        for i in range(y.shape[0]):
            wi, yi = w[i], y[i]
            a00, a01, a11 = (wi * f0 * f0).sum(-1), (wi * f0 * f1).sum(-1), (wi * f1 * f1).sum(-1)
            b0, b1 = (wi * f0 * yi).sum(-1), (wi * f1 * yi).sum(-1)
            det = (a00 * a11 - a01**2).clamp_min(1e-300)
            A, B = (a11 * b0 - a01 * b1) / det, (a00 * b1 - a01 * b0) / det
            r = yi - A[..., None] * f0 - B[..., None] * f1
            tot = tot + (wi * r * r).sum(-1)
            amps.append((A, B))
        return tot, amps

    def search(m0g, m1g):
        X, Y = torch.meshgrid(m0g, m1g, indexing="ij")
        c, _ = chi2(X, Y)
        c = torch.where(Y > X + M1_GAP, c, torch.full_like(c, float("inf")))
        k = int(torch.argmin(c))
        return X.flatten()[k].item(), Y.flatten()[k].item()

    m1g = torch.linspace(M0[0].item() + M1_GAP, M1_MAX, N_M1, dtype=torch.float64)
    a, b = search(M0, m1g)
    # one local refinement on a 10× finer grid around the coarse optimum
    dm0, dm1 = (M0[1] - M0[0]).item(), (m1g[1] - m1g[0]).item()
    a, b = search(torch.linspace(a - dm0, a + dm0, 41, dtype=torch.float64),
                  torch.linspace(max(b - dm1, a + M1_GAP), min(b + dm1, M1_MAX), 41,
                                 dtype=torch.float64))
    c, amps = chi2(torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64))
    a0 = [amps[i][0].item() * (1 + math.exp(-a * Nt)) / Cs[i][0].item() for i in range(len(Cs))]
    at_edge = b > M1_MAX - 2 * dm1
    return {"m0": a, "m1": b, "chi2": float(c), "A0": a0,
            "B": [amps[i][1].item() for i in range(len(Cs))], "edge": at_edge}


def main():
    dof = 2 * (WINDOW[1] - WINDOW[0] + 1) - 6
    print(f"two-state shared-mass fit, window {WINDOW}, arm `{ARM}`, dof {dof}")
    allres = {}
    for tag, (dump, cache) in ENS.items():
        blob = torch.load(dump if DUMP else os.path.join("dumps", dump),
                          map_location="cpu", weights_only=False)
        gelt = blob["gelt_obar"].double()
        jb = BLOCK if DUMP else int(blob.get("meta", {}).get("jack_block", 10))
        basis = blob["Obar_basis"].double() if DUMP else {n: v.double() for n, v in torch.load(
            os.path.join("dumps", cache), map_location="cpu",
            weights_only=False)["bases"].items()}[ARM]
        B = gelt.shape[0]

        def project(b):
            v0 = gevp_ground_vector(connected_correlator_matrix(b), t0=1, td=2,
                                    eps=EPS, truncate=TRUNC)
            return torch.einsum("i,ibt->bt", v0, b)

        full_p = project(basis)
        _, sg = fgo.blocked_jackknife(lambda m: connected_correlator(gelt[m]), B, jb)
        _, sp = fgo.blocked_jackknife(lambda m: connected_correlator(full_p[m]), B, jb)
        central = fit2([connected_correlator(gelt), connected_correlator(full_p)], [sg, sp],
                       WINDOW)

        def stats(mask):
            r = fit2([connected_correlator(gelt[mask]),
                      connected_correlator(project(basis[:, mask]))], [sg, sp], WINDOW)
            return torch.tensor([r["m0"], r["m1"], r["A0"][0], r["A0"][1],
                                 r["A0"][0] - r["A0"][1], float(r["edge"])],
                                dtype=torch.float64)
        mean, err = fgo.blocked_jackknife(stats, B, jb)
        n_blk = -(-B // jb)
        allres[tag] = (central, mean, err, n_blk)
        print(f"\n{tag}: central m₀ = {central['m0']:.4f}, m₁ = {central['m1']:.3f}"
              f"{' (AT GRID EDGE)' if central['edge'] else ''}, χ²/dof = "
              f"{central['chi2'] / dof:.2f}, B_GELT/A_GELT = "
              f"{central['B'][0] / max(abs(central['A0'][0]), 1e-9):+.2f}")
        print(f"  jackknife: m₀ = {mean[0]:.4f} ± {err[0]:.4f}   m₁ = {mean[1]:.3f} ± "
              f"{err[1]:.3f}   (m₁ at the edge in {mean[5] * n_blk:.0f}/{n_blk} samples)")
        print(f"  A₀ GELT = {mean[2]:.4f} ± {err[2]:.4f}   A₀ GEVP = {mean[3]:.4f} ± "
              f"{err[3]:.4f}   ΔA₀ = {mean[4]:+.4f} ± {err[4]:.4f}")
    w = [1.0 / allres[t][2][4].item() ** 2 for t in allres]
    comb = sum(wi * allres[t][1][4].item() for wi, t in zip(w, allres)) / sum(w)
    print(f"\ncombined ΔA₀ (shared masses, inverse variance, {len(w)} ensembles) = "
          f"{comb:+.4f} ± {1 / math.sqrt(sum(w)):.4f}  ({comb * math.sqrt(sum(w)):.1f}σ)")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(allres, OUT + ".pt")


if __name__ == "__main__":
    main()
