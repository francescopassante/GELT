"""The K-state variational basis (notes/learned_spectrum.md): the Ky Fan loss.

What the method rests on, each checked against an answer known in advance:
  - at K = 1 the Ky Fan trace is the Rayleigh quotient, and the training loss
    is rayleigh_loss's (so every single-operator run is unchanged);
  - the trace depends on the span only (GL(K) invariance), which is why the K
    outputs need no orthogonality constraint;
  - on an exact spectral model the trace is bounded by the sum of the K largest
    e^{−E_n Δ}, with equality on the top-K span (the Ky Fan principle), and the
    n-th GEVP vector projects onto the n-th state exactly;
  - minimising the training loss over a linear readout of synthetic features
    finds the ground AND the first excited state of a known spectrum — the
    claim that the loss learns excited states, end to end, on data with a
    known answer.
"""

import math
import os
import sys

import pytest
import torch

from gelt.glueball import gevp_ground_vector, gevp_vectors, kyfan_trace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))


@pytest.fixture(scope="module")
def tg():
    """train_glueball, imported with an argv it cannot misread as its own flags."""
    argv = sys.argv
    sys.argv = argv[:1]
    try:
        import train_glueball
    finally:
        sys.argv = argv
    return train_glueball


def _spectral(Z, E, deltas):
    """Exact C(Δ) = Z diag(e^{−EΔ}) Zᵀ for Δ in ``deltas`` → (len, K, K)."""
    w = torch.exp(-E[None, :] * torch.as_tensor(deltas, dtype=E.dtype)[:, None])
    return torch.einsum("in,dn,jn->dij", Z, w, Z)


def test_kyfan_trace_is_the_rayleigh_quotient_at_k1():
    C0 = torch.tensor([[2.5]], dtype=torch.float64)
    C1 = torch.tensor([[1.7]], dtype=torch.float64)
    assert torch.allclose(kyfan_trace(C0, C1), torch.tensor(1.7 / 2.5, dtype=torch.float64),
                          atol=1e-15, rtol=0)


def test_kyfan_trace_depends_on_the_span_only():
    torch.manual_seed(0)
    E = torch.tensor([0.3, 0.5, 0.9, 1.4], dtype=torch.float64)
    Z = torch.randn(3, 4, dtype=torch.float64)
    C = _spectral(Z, E, [0, 1, 2])
    A = torch.randn(3, 3, dtype=torch.float64) + 3 * torch.eye(3, dtype=torch.float64)
    Ca = A @ C @ A.T
    for d in (1, 2):
        assert torch.allclose(kyfan_trace(C[0], C[d]), kyfan_trace(Ca[0], Ca[d]),
                              atol=1e-12, rtol=0)


def test_kyfan_bound_and_equality_on_the_top_span():
    torch.manual_seed(1)
    E = torch.tensor([0.3, 0.5, 0.6, 1.0, 1.5], dtype=torch.float64)
    K = 2
    for d in (1, 2):
        top = torch.exp(-E[:K] * d).sum()
        # equality: any invertible mixing of exactly the two lightest states
        Z = torch.randn(K, K, dtype=torch.float64) @ torch.eye(K, 5, dtype=torch.float64)
        C = _spectral(Z, E, [0, d])
        assert torch.allclose(kyfan_trace(C[0], C[1]), top, atol=1e-12, rtol=0)
        # bound: random spans never exceed it, and a generic one stays below
        for _ in range(20):
            Zr = torch.randn(K, 5, dtype=torch.float64)
            Cr = _spectral(Zr, E, [0, d])
            assert kyfan_trace(Cr[0], Cr[1]) <= top + 1e-12
        # each Ritz value bounds its own level (Hylleraas–Undheim–MacDonald)
        Zr = torch.randn(3, 5, dtype=torch.float64)
        Cr = _spectral(Zr, E, [0, d])
        lam, _ = gevp_vectors(torch.stack([Cr[0], Cr[1]]), t0=0, td=1)
        assert torch.all(lam <= torch.exp(-E[:3] * d) + 1e-12)


def test_gevp_vectors_project_every_state_and_agree_with_the_ground_vector():
    E = torch.tensor([0.3, 0.9], dtype=torch.float64)
    Z = torch.tensor([[1.0, 0.6], [0.4, 1.0]], dtype=torch.float64)
    C = _spectral(Z, E, list(range(10)))
    lam, V = gevp_vectors(C, t0=0, td=1)
    assert torch.allclose(lam, torch.exp(-E), atol=1e-12, rtol=0)
    for n in range(2):
        Cn = torch.einsum("i,tij,j->t", V[:, n], C, V[:, n])
        assert torch.allclose(Cn, torch.exp(-E[n] * torch.arange(10.0, dtype=torch.float64)),
                              atol=1e-10, rtol=0)
    assert torch.allclose(V[:, 0], gevp_ground_vector(C, t0=0, td=1), atol=1e-12, rtol=0)


def test_training_loss_reduces_to_rayleigh_loss_at_k1(tg):
    torch.manual_seed(2)
    Obar = 3.0 + torch.randn(6, 24, dtype=torch.float64).cumsum(dim=1) * 0.1
    l1, c1, r1 = tg.rayleigh_loss(Obar)
    lk, ck, rk = tg.kyfan_loss(Obar[..., None])
    assert abs(lk.item() - l1.item()) < 1e-6  # the 1e-6 ridge is the only difference
    assert abs(ck.item() - c1.item()) < 1e-12
    assert abs(rk.item() - r1.item()) < 1e-6
    assert tg.variational_loss(Obar)[0].item() == l1.item()  # K = 1 takes rayleigh's path


def _periodic_modes(E, B, Nt, gen):
    """B configurations of independent stationary Gaussian modes, periodic in t.

    Mode n has the exactly periodic covariance Σ_k e^{−E_n |Δ + k·Nt|} (the
    periodised exponential, positive definite), drawn through its FFT, so the
    connected correlator of mode n is that of one state of energy E_n on a
    periodic lattice. Returns (B, Nt, n_modes).
    """
    dd = torch.arange(Nt, dtype=torch.float64)
    out = []
    for e in E.tolist():
        gamma = sum(torch.exp(-e * (dd + k * Nt).abs()) for k in range(-4, 5))
        S = torch.fft.fft(gamma).real.clamp_min(0)
        z = torch.randn(B, Nt, generator=gen, dtype=torch.float64)
        x = torch.fft.ifft(torch.fft.fft(z) * S.sqrt()).real
        out.append(x)
    return torch.stack(out, dim=-1)


def test_kyfan_loss_learns_the_first_excited_state(tg):
    """A linear K = 2 readout of 6 noisy mixtures of 5 known states, trained on
    the production loss with the production optimiser: the learned span must
    hold the ground state AND the first excited one (E = 0.3, 0.5), not the
    ground state twice and not the next pair. A K = 1 readout trained the same
    way finds only the ground state — the excited level is what K > 1 adds."""
    gen = torch.Generator().manual_seed(3)
    E = torch.tensor([0.3, 0.5, 0.8, 1.2, 1.8], dtype=torch.float64)
    Nt, B = 24, 1200
    modes = _periodic_modes(E, B, Nt, gen)  # (B, Nt, 5)
    amp = torch.tensor([1.0, 1.0, 1.2, 1.5, 2.0], dtype=torch.float64)
    A = torch.randn(5, 6, generator=gen, dtype=torch.float64)
    feats = (modes * amp) @ A + 0.05 * torch.randn(B, Nt, 6, generator=gen, dtype=torch.float64)
    train, test = feats[:900], feats[900:]

    def fit(K, steps=600):
        torch.manual_seed(4)
        W = torch.nn.Parameter(0.3 * torch.randn(6, K, dtype=torch.float64))
        opt = torch.optim.Adam([W], lr=3e-2)
        for step in range(steps):
            idx = torch.randint(0, train.shape[0], (6,))
            out = train[idx] @ W
            loss = tg.variational_loss(out if K > 1 else out[..., 0])[0]
            opt.zero_grad()
            loss.backward()
            opt.step()
        return W.detach()

    def ritz(W):
        out = test @ W  # (B, Nt, K)
        d = out - out.mean(dim=(0, 1))
        C = torch.stack([torch.einsum("btk,btl->kl", d.roll(-dt, dims=1), d) / (d.shape[0] * Nt)
                         for dt in (0, 1)])
        lam, _ = gevp_vectors(C, t0=0, td=1)
        return -torch.log(lam)  # variational bound on each level, Δ = 1

    e2 = ritz(fit(2))
    e1 = ritz(fit(1))
    # Held-out bounds sit just above the true levels — variational bounds, up to
    # sampling noise ≈ 0.01 at B = 300 (measured: 0.324, 0.539 for K = 2; the
    # K = 3 span, which also holds E = 0.8, brings level 1 to 0.512).
    assert 0.27 < e2[0].item() < 0.34, e2
    assert 0.47 < e2[1].item() < 0.56, e2
    assert 0.27 < e1[0].item() < 0.34, e1
    # a random K = 2 readout of the same features is far from the first excited level
    torch.manual_seed(5)
    e_rand = ritz(torch.randn(6, 2, dtype=torch.float64))
    assert e_rand[1].item() > 0.6, e_rand
    assert math.isfinite(e2[1].item())
