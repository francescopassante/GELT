"""scripts/train_distill.py's objective: the connected loss and the shape loss.

The first distillation run (ens2, lag 2, raw loss) selected its epoch on a
constant offset of the output, which no connected correlator sees, and its whole
loss decrease was a rescale of the field, which no scale-invariant readout sees.
These pin the two properties the rerun rests on: the connected loss is blind to
a constant, and the closed-form shape loss is min_c of the loss over rescales.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))


def _td():
    argv = sys.argv
    sys.argv = argv[:1]
    try:
        import train_distill
    finally:
        sys.argv = argv
    return train_distill


def _fields(seed, b=2, Lt=8, L=6):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(b, Lt, L, L, L, generator=g, dtype=torch.float64)
    t = 0.6 * s + 0.3 * torch.randn(b, Lt, L, L, L, generator=g, dtype=torch.float64)
    return s, t


def test_connected_loss_ignores_a_constant_and_raw_loss_does_not(monkeypatch):
    td = _td()
    s, t = _fields(0)
    monkeypatch.setattr(td, "LOSS", "connected")
    base = td.tmd_loss(s, t)[0]
    shifted = td.tmd_loss(s + 0.12, t)[0]
    assert torch.allclose(base, shifted, rtol=0, atol=1e-12)
    monkeypatch.setattr(td, "LOSS", "raw")
    s = s - (s - t).mean()  # a zero-mean error, so the shift has no cross term
    # an offset b costs V·b² in the zero-momentum term (plus POOL³·b² in the block
    # term): the size of the swings the first run's validation loss showed
    raw, raw_shifted = td.tmd_loss(s, t)[0], td.tmd_loss(s + 0.12, t)[0]
    V = 6 ** 3
    assert abs((raw_shifted - raw).item() - 0.12 ** 2 * (V + td.POOL ** 3)) < 1e-9


def test_quad_stats_reproduce_the_loss(monkeypatch):
    td = _td()
    s, t = _fields(1)
    for loss in ("connected", "raw"):
        monkeypatch.setattr(td, "LOSS", loss)
        q = td.quad_stats(s, t)
        direct = td.tmd_loss(s, t)[0].item()
        assert abs((q[0] - 2 * q[1] + q[2]).item() - direct) < 1e-10


def test_shape_loss_is_the_minimum_over_rescales(monkeypatch):
    td = _td()
    s, t = _fields(2)
    monkeypatch.setattr(td, "LOSS", "connected")
    shape, c = td.shape_loss(td.quad_stats(s, t))
    grid = torch.linspace(c - 0.2, c + 0.2, 401, dtype=torch.float64)
    losses = torch.tensor([td.tmd_loss(float(k) * s, t)[0].item() for k in grid])
    assert abs(losses.min().item() - shape) < 1e-6
    assert abs(grid[losses.argmin()].item() - c) <= 0.2 / 400 + 1e-12
    assert abs(td.tmd_loss(c * s, t)[0].item() - shape) < 1e-10
    # and it is blind to the scale and the offset of the student, as the readout is
    shape2, c2 = td.shape_loss(td.quad_stats(3.0 * s - 55.0, t))
    assert abs(shape2 - shape) < 1e-9 and abs(3.0 * c2 - c) < 1e-9
