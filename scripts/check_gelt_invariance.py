"""Quick gauge-invariance check on the full GELT model: forward(W_g, T_g) ≈ forward(W, T)."""

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
L, D, R, nc, H, layers = 4, 2, 2, 2, 2, 2
gg = SU(nc)

U = random_links(L=L, D=D, gaugegroup=gg, dtype=torch.complex64)
raw = torch.randn(L**D, nc, nc) + 1j * torch.randn(L**D, nc, nc)
omega, _ = torch.linalg.qr(raw)
omega = omega.reshape(*([L] * D), nc, nc).to(torch.complex64)

U_g = link_gauge_transformation(U, omega, gg)
P = plaquette_tensor(U.unsqueeze(0), gg)
P_g = plaquette_tensor(U_g.unsqueeze(0), gg)

T = build_transport_average(U.unsqueeze(0), R=R, gaugegroup=gg)
T_g = build_transport_average(U_g.unsqueeze(0), R=R, gaugegroup=gg)

model = GELT(gaugegroup=gg, L=L, D=D, R=R, nhead=H, gemhsa_layers=layers, d_qkv=2)

out = model(P, T)
out_g = model(P_g, T_g)
drift = (out_g - out).abs().max().item()
print(f"|out_g - out| max = {drift:.3e}    (out = {out.item():.6e}, out_g = {out_g.item():.6e})")
