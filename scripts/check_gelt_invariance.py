"""Quick gauge-invariance check on the full GELT model: forward(W_g, T_g) ≈ forward(W, T).

This is the sixty-second version of the claim `tests/test_blocks.py` pins
properly, and the one main.tex quotes as "verified to machine precision in
complex128" — so it runs in complex128, and with ``mlp_zero_init=False``:
with the default zero-init readout both sides are *identically* zero and the
check passes without testing anything.
"""

import torch

from gelt import (
    SU,
    build_transport_average,
    link_gauge_transformation,
    plaquette_tensor,
    random_links,
)
from gelt.blocks import GELT

torch.manual_seed(0)
L = 4
D = 2
R = 2
H = 2
layers = 2
nc = 2
gg = SU(nc)
dtype = torch.complex128

# generate random link configuration and compute plaquette tensor
U = random_links(L=L, D=D, gaugegroup=gg, dtype=dtype)
P = plaquette_tensor(U.unsqueeze(0), gg)

# generate random unitary gauge transformation via qr decomposition
raw = torch.randn(L**D, nc, nc, dtype=torch.float64) + 1j * torch.randn(
    L**D, nc, nc, dtype=torch.float64
)
omega, _ = torch.linalg.qr(raw)
omega = omega.reshape(*([L] * D), nc, nc).to(dtype)

# gauge transform the link configuration and compute the new plaquette tensor
U_g = link_gauge_transformation(U, omega, gg)
P_g = plaquette_tensor(U_g.unsqueeze(0), gg)

# compute the transport operators
T = build_transport_average(U.unsqueeze(0), R=R, gaugegroup=gg)
T_g = build_transport_average(U_g.unsqueeze(0), R=R, gaugegroup=gg)

model = GELT(
    gaugegroup=gg,
    L=L,
    D=D,
    R=R,
    nhead=H,
    gemhsa_layers=layers,
    d_qkv=4,
    dtype=dtype,
    # Without this the readout is zero-initialised and both outputs are
    # identically 0.0 — a check that cannot fail and cannot detect anything.
    mlp_zero_init=False,
)

out = model(P, T)
out_g = model(P_g, T_g)
drift = (out_g - out).abs().max().item()
print(
    f"|out_g - out| max = {drift:.3e}    (out = {out.item():.6e}, out_g = {out_g.item():.6e})"
)
