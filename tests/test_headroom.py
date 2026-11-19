"""The identity behind scripts/train_distill.py and scripts/variational_headroom.py,
on data whose spectrum is known.

Independent stationary AR(1) modes x_n(t) with lag-1 correlation ρ_n = e^{−E_n}
are an exact transfer-matrix model: operators O_i = Σ_n z_in x_n have
C_ij(Δ) = Σ_n z_in z_jn ρ_n^Δ, and E[x_n(t + a) | slice t] = ρ_n^a x_n(t). So

  - regressing O(t ± a) on a span that resolves every mode must give an operator
    whose correlator is the target's shifted by 2a, C_g(Δ) = C_O(Δ + 2a) — the
    claim that the regression function is the Euclidean-evolved operator;
  - η_S(a) = 1 on such a span, < 1 on an incomplete one, and its large-a limit
    is the largest ground-state overlap the span can reach.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))


def _vh():
    argv = sys.argv
    sys.argv = argv[:1]
    try:
        import variational_headroom
    finally:
        sys.argv = argv
    return variational_headroom


E = torch.tensor([0.3, 0.6, 1.1, 1.9], dtype=torch.float64)
RHO = torch.exp(-E)


def _modes(n_cfg, Nt, seed):
    """(n_modes, n_cfg, Nt) independent stationary AR(1) chains, open in time."""
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(len(E), n_cfg, Nt, dtype=torch.float64)
    x[:, :, 0] = torch.randn(len(E), n_cfg, generator=g, dtype=torch.float64)
    for t in range(1, Nt):
        noise = torch.randn(len(E), n_cfg, generator=g, dtype=torch.float64)
        x[:, :, t] = RHO[:, None] * x[:, :, t - 1] + (1 - RHO[:, None] ** 2).sqrt() * noise
    return x


def _exact_eta(Z, zg, a):
    """η_S(a) from the spectral sums, span rows ``Z (K, n)`` and probe ``zg (n,)``."""
    C0 = Z @ Z.T
    c = Z @ (zg * RHO ** a)
    return (c @ torch.linalg.solve(C0, c)) / (zg ** 2 * RHO ** (2 * a)).sum()


def test_regression_on_the_future_is_euclidean_evolution():
    x = _modes(4000, 64, 0)
    zg = torch.tensor([1.0, 0.7, 0.5, 0.4], dtype=torch.float64)
    G = torch.einsum("n,nct->ct", zg, x)
    a = 2
    # interior slices only: the chain is open, so t ± a must exist
    t = slice(a, 64 - a)
    feats = x[:, :, t].reshape(len(E), -1).T  # a span that resolves every mode
    target = 0.5 * (G[:, 2 * a:] + G[:, : 64 - 2 * a]).reshape(-1)
    w = torch.linalg.lstsq(feats, target[:, None]).solution[:, 0]
    # the fitted operator is the evolved one, mode by mode …
    assert torch.allclose(w, zg * RHO ** a, atol=0.02)
    # … so its correlator is the target's, shifted by 2a
    for d in range(4):
        shifted = (zg ** 2 * RHO ** (d + 2 * a)).sum()
        fitted = (w ** 2 * RHO ** d).sum()
        assert abs(fitted / shifted - 1) < 0.05


def test_eta_is_one_on_a_complete_span_and_the_overlap_bound_on_an_incomplete_one():
    vh = _vh()
    x = _modes(6000, 48, 1)
    zg = torch.tensor([1.0, 0.7, 0.5, 0.4], dtype=torch.float64)
    G = torch.einsum("n,nct->ct", zg, x)
    torch.manual_seed(2)
    Zfull = torch.randn(4, 4, dtype=torch.float64)
    Zpart = torch.randn(2, 4, dtype=torch.float64)
    for Z, complete in ((Zfull, True), (Zpart, False)):
        S = torch.einsum("kn,nct->kct", Z, x)
        got = vh.eta(S, G, 3, 1e-12)
        for a in (1, 2, 3):
            exact = _exact_eta(Z, zg, a)
            assert abs(got[a - 1] - exact) < 0.03, (complete, a, got[a - 1], exact)
            if complete:
                assert abs(exact - 1) < 1e-10
            else:
                assert exact < 1 - 1e-3
    # large a: η → the largest ground-state overlap in the span, ‖P_S e_0‖²
    P = Zpart.T @ torch.linalg.solve(Zpart @ Zpart.T, Zpart)
    assert abs(_exact_eta(Zpart, zg, 80) - P[0, 0]) < 1e-6


def test_single_operator_eta_is_its_true_overlap_at_large_lag():
    zg = torch.tensor([1.0, 0.7, 0.5, 0.4], dtype=torch.float64)
    a0 = zg[0] ** 2 / (zg ** 2).sum()
    assert abs(_exact_eta(zg[None], zg, 80) - a0) < 1e-6


def test_wrap_factor_undoes_the_backward_term_of_a_pure_state():
    vh = _vh()
    m, Nt = 0.31, 24

    def C(d):
        return math.exp(-m * d) + math.exp(-m * (Nt - d))

    # One state on a periodic lattice: every forward factor cancels in
    # C(a)²/(C(0)C(2a)), so what is left is the backward terms alone. The
    # correction removes those of C(a) and C(2a) exactly; C(0)'s, e^{−m·Nt}, is
    # the 6e-4 it leaves behind.
    for a in (2, 4, 5):
        raw = C(a) ** 2 / (C(0) * C(2 * a))
        assert raw < 1 - 1e-3 * a
        assert abs(raw * vh.wrap_factor(a, Nt, m) * (1 + math.exp(-m * Nt)) - 1) < 1e-12
