"""Correlated goodness-of-fit check of the shared-mass two-state fit (prof
feedback B, Oct 2026). Offline, high-statistics dump only.

The published two-state fit (scripts/two_state_fit.py, GEVP at (t₀,t_d)=(0,1))
minimises a χ² with diagonal weights, so its χ² says nothing about the model.
Here the joint vector y = [C_GELT(Δ), C_GEVP(Δ)], Δ ∈ window, gets its full
blocked-jackknife covariance Σ (16×16 on [1,8], 400 blocks; v₀ recomputed in
every sample), optionally shrunk toward its diagonal, Σ_λ = (1−λ)Σ + λ diag Σ.
Two numbers per λ:

  (a) the correlated χ² of the PUBLISHED best-fit point (diagonal minimiser),
  (b) a correlated fit: (m₀, m₁) on a grid, the four amplitudes profiled out by
      generalised least squares with Σ_λ⁻¹ (they couple across operators), with
      Σ fixed from the full sample and the fit repeated in every jackknife
      sample for the errors on m₀, m₁, A₀ and ΔA₀.

Usage:  python scripts/two_state_correlated_check.py --dump=<file> [--block=100]
                                                     [--window=1-8]
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_glueball_overlap as fgo  # noqa: E402

from gelt.glueball import (  # noqa: E402
    connected_correlator,
    connected_correlator_matrix,
    gevp_ground_vector,
)


def _flag(name, default):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{name}=")),
                default)


DUMP = _flag("dump", None)
BLOCK = int(_flag("block", "100"))
WINDOW = tuple(int(x) for x in _flag("window", "1-8").split("-"))
T0, TD = 0, 1
SHRINK = (0.0, 0.01, 0.1, 0.5)
M0 = torch.linspace(0.15, 0.7, 111, dtype=torch.float64)
M1_MAX, M1_GAP, N_M1 = 1.5, 0.1, 141
if DUMP is None:
    raise SystemExit("--dump=<high-statistics dump> is required")


def _phi(m, d, Nt):
    return torch.exp(-m * d) + torch.exp(-m * (Nt - d))


def design(m0, m1, d, Nt):
    """(2n, 4) design matrix of the joint vector, amplitudes (A_G, B_G, A_P, B_P)."""
    f0, f1 = _phi(m0, d, Nt), _phi(m1, d, Nt)
    n = d.shape[0]
    X = torch.zeros(2 * n, 4, dtype=torch.float64)
    X[:n, 0], X[:n, 1], X[n:, 2], X[n:, 3] = f0, f1, f0, f1
    return X


def gls(y, W, X):
    """χ² and amplitudes of the GLS fit y ≈ X a with weight matrix W = Σ⁻¹."""
    XtW = X.T @ W
    a = torch.linalg.solve(XtW @ X, XtW @ y)
    r = y - X @ a
    return float(r @ W @ r), a


def _chi2_grid(y, W, d, Nt, m0, m1):
    """Batched GLS χ² over flat grids m0, m1 (G,), amplitudes profiled out."""
    G, n = m0.shape[0], d.shape[0]
    f0 = torch.exp(-m0[:, None] * d) + torch.exp(-m0[:, None] * (Nt - d))
    f1 = torch.exp(-m1[:, None] * d) + torch.exp(-m1[:, None] * (Nt - d))
    X = torch.zeros(G, 2 * n, 4, dtype=torch.float64)
    X[:, :n, 0], X[:, :n, 1], X[:, n:, 2], X[:, n:, 3] = f0, f1, f0, f1
    XtW = X.transpose(1, 2) @ W
    a = torch.linalg.solve(XtW @ X, (XtW @ y)[..., None])[..., 0]
    r = y - (X @ a[..., None])[..., 0]
    return ((r @ W) * r).sum(-1)


def fit_corr(y, W, d, Nt, C0):
    """Correlated two-state fit, grid over (m₀, m₁) plus one local refinement."""
    def search(m0g, m1g):
        X, Y = torch.meshgrid(m0g, m1g, indexing="ij")
        X, Y = X.flatten(), Y.flatten()
        ok = Y > X + M1_GAP
        X, Y = X[ok], Y[ok]
        c = _chi2_grid(y, W, d, Nt, X, Y)
        k = int(torch.argmin(c))
        return c[k].item(), X[k].item(), Y[k].item()

    m1g = torch.linspace(M0[0].item() + M1_GAP, M1_MAX, N_M1, dtype=torch.float64)
    _, a, b = search(M0, m1g)
    dm0, dm1 = (M0[1] - M0[0]).item(), (m1g[1] - m1g[0]).item()
    c, a, b = search(torch.linspace(a - dm0, a + dm0, 41, dtype=torch.float64),
                     torch.linspace(max(b - dm1, a + M1_GAP), min(b + dm1, M1_MAX), 41,
                                    dtype=torch.float64))
    _, amp = gls(y, W, design(torch.tensor(a), torch.tensor(b), d, Nt))
    norm = 1 + math.exp(-a * Nt)
    a0g, a0p = amp[0].item() * norm / C0[0], amp[2].item() * norm / C0[1]
    return {"chi2": c, "m0": a, "m1": b, "A0": (a0g, a0p), "edge": b > M1_MAX - 2 * dm1}


def main():
    blob = torch.load(DUMP, map_location="cpu", weights_only=False)
    gelt = blob["gelt_obar"].double()
    basis = blob["Obar_basis"].double()
    B, Nt = gelt.shape
    dmin, dmax = WINDOW
    d = torch.arange(dmin, dmax + 1, dtype=torch.float64)
    n = d.shape[0]
    dof = 2 * n - 6

    def project(b):
        v0 = gevp_ground_vector(connected_correlator_matrix(b), t0=T0, td=TD, eps=1e-12,
                                truncate=False)
        return torch.einsum("i,ibt->bt", v0, b)

    def joint(mask):
        Cg = connected_correlator(gelt[mask])
        Cp = connected_correlator(project(basis[:, mask]))
        return torch.cat([Cg[dmin : dmax + 1], Cp[dmin : dmax + 1]]), (Cg[0].item(),
                                                                      Cp[0].item())

    full = torch.ones(B, dtype=torch.bool)
    y, C0 = joint(full)
    blocks = fgo._blocks(B, BLOCK)
    nb = len(blocks)
    masks, ys, c0s = [], [], []
    for bl in blocks:
        m = torch.ones(B, dtype=torch.bool)
        m[bl] = False
        yk, c0k = joint(m)
        masks.append(m), ys.append(yk), c0s.append(c0k)
    Y = torch.stack(ys)
    dY = Y - Y.mean(0)
    cov = (nb - 1) / nb * dY.T @ dY
    sig = cov.diagonal().sqrt()
    corr = cov / torch.outer(sig, sig)
    ev = torch.linalg.eigvalsh(corr)
    print(f"dump {os.path.basename(DUMP)}: {B} configs, {nb} jackknife blocks, window "
          f"{WINDOW}, {2 * n} points, dof {dof}, GEVP ({T0},{TD})")
    print(f"correlation matrix: eigenvalues {ev.min():.2e} … {ev.max():.2f} "
          f"(cond {ev.max() / ev.min():.1e})")

    # The published diagonal best fit, re-found here with W = diag(1/σ²).
    Wd = torch.diag(1.0 / sig**2)
    pub = fit_corr(y, Wd, d, Nt, C0)
    Xp = design(torch.tensor(pub["m0"]), torch.tensor(pub["m1"]), d, Nt)
    print(f"\ndiagonal fit (published): m₀ = {pub['m0']:.4f}, m₁ = {pub['m1']:.3f}, "
          f"A₀ = {pub['A0'][0]:.4f} / {pub['A0'][1]:.4f}, diagonal χ²/dof = "
          f"{pub['chi2'] / dof:.3f}")

    from scipy.stats import chi2 as chi2dist

    results = {"published": pub, "eig_corr": ev, "lam": {}}
    for lam in SHRINK:
        S = (1 - lam) * cov + lam * torch.diag(sig**2)
        W = torch.linalg.inv(S)
        # (a) correlated χ² at the published point (amplitudes re-profiled with W)
        c_pub, _ = gls(y, W, Xp)
        # (b) correlated fit, errors from the jackknife with W fixed
        cen = fit_corr(y, W, d, Nt, C0)
        samp = []
        for yk, c0k in zip(ys, c0s):
            r = fit_corr(yk, W, d, Nt, c0k)
            samp.append([r["m0"], r["m1"], r["A0"][0], r["A0"][1],
                         r["A0"][0] - r["A0"][1], float(r["edge"])])
        S_ = torch.tensor(samp, dtype=torch.float64)
        mu = S_.mean(0)
        err = ((nb - 1) / nb * ((S_ - mu) ** 2).sum(0)).sqrt()
        p_pub, p_fit = chi2dist.sf(c_pub, dof), chi2dist.sf(cen["chi2"], dof)
        print(f"\nλ = {lam}:  (a) published point: χ²_corr/dof = {c_pub / dof:.2f} "
              f"(p = {p_pub:.3g})")
        print(f"         (b) correlated fit: χ²/dof = {cen['chi2'] / dof:.2f} (p = {p_fit:.3g}), "
              f"m₀ = {cen['m0']:.4f}, m₁ = {cen['m1']:.3f}")
        print(f"             jackknife: m₀ = {mu[0]:.4f} ± {err[0]:.4f}, m₁ = {mu[1]:.3f} ± "
              f"{err[1]:.3f} (edge {mu[5] * nb:.0f}/{nb}), A₀ G = {mu[2]:.4f} ± {err[2]:.4f}, "
              f"A₀ P = {mu[3]:.4f} ± {err[3]:.4f}, ΔA₀ = {mu[4]:+.4f} ± {err[4]:.4f}")
        results["lam"][lam] = {"chi2_pub": c_pub, "p_pub": p_pub, "central": cen,
                               "p_fit": p_fit, "mean": mu, "err": err}
    out = ("results/glueball/two_state_correlated_"
           + os.path.basename(DUMP).replace("_test_obars.pt", "") + f"_w{dmin}-{dmax}.pt")
    torch.save(results, out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
