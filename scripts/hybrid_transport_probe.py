"""Is the thin-link transport what keeps the net from building smeared rectangles? CPU, ~10 s.

train_glueball.config_inputs feeds plaquettes at several APE levels but takes the
transport T from the least-smeared one (thin links). The loop-doubling step
    T_m(x) P_n(x+mu) T_m(x)^dag . P_n(x)
is the smeared 1x2 rectangle exactly only when the transporting link is at the
plaquettes' own level (m = n): for m != n the factor U_m^dag U_n does not cancel.
This measures how much that costs, classically, with no network: the hybrid
operator against the clean rectangle, site by site and at zero momentum, and
each one's own C(1)/C(0), C(2)/C(0), on a small anisotropic ensemble sampled
here (12 x 6^3, beta = 2.4, xi = 3, 30 configurations — a mechanism check, not a
measurement of the production lattice).

Result of 2026-11-12 (seed 0): the hybrids with thin transport (m = 0) correlate
0.995–0.999 with the clean rectangles at zero momentum (0.975–0.981 site by
site) and lose 0.003–0.010 of C(1)/C(0). The thin transport is NOT the
bottleneck (notes/thesis_record.md §3.8); a transport-level flag is not worth a
training run.

Run:  python scripts/hybrid_transport_probe.py
"""

import functools
import time

import torch
from gelt.lattice import SU, plaquette_tensor, random_links
from gelt.sampler import heatbath_overrelaxation_sweep
from gelt.glueball import ape_smear
torch.manual_seed(0)
g = SU(2); L, Lt, beta, xi = 6, 12, 2.4, 3.0
sweep = functools.partial(heatbath_overrelaxation_sweep, n_or=4, xi=xi)
U = random_links(L, 4, g, Lt=Lt)
t = time.time(); U, _ = sweep(U, g, beta); print("one sweep", time.time() - t, "s", flush=True)
NTH, NCFG, NSKIP = 40, 30, 2
for _ in range(NTH): U, _ = sweep(U, g, beta)
cfgs = []
for _ in range(NCFG):
    for _ in range(NSKIP): U, _ = sweep(U, g, beta)
    cfgs.append(U.clone())
cfgs = torch.stack(cfgs)  # (N,4,Lt,L,L,L,2,2)
print("sampled", time.time() - t, "s", flush=True)
levels = [0, 2, 4, 6]
U3, P = {}, {}
Us, done = cfgs, 0
for lvl in levels:
    if lvl > done:
        Us = ape_smear(Us, g, alpha=0.5, n_steps=lvl - done); done = lvl
    u3 = Us[:, 1:].movedim(2, 1).reshape(NCFG * Lt, 3, L, L, L, 2, 2)
    U3[lvl] = u3; P[lvl] = plaquette_tensor(u3, g)  # (B,3,L,L,L,2,2) planes (0,1),(0,2),(1,2)
print("mean spatial plaquette per level:", {l: round(P[l].diagonal(dim1=-2, dim2=-1).sum(-1).real.mean().item() / 2, 4) for l in levels})
print("<Re Tr U_m^dag U_n>/2 (same link, two levels):")
for m in levels:
    print("  m=%d: " % m + "  ".join("n=%d %.3f" % (n, ((U3[m].conj().transpose(-1, -2) @ U3[n]).diagonal(dim1=-2, dim2=-1).sum(-1).real.mean().item() / 2)) for n in levels))
dag = lambda A: A.conj().transpose(-1, -2)
def hybrid(n, m):
    """sum over the 3 planes of Re Tr[ U_m,mu(x) P_n(x+mu) U_m,mu(x)^dag P_n(x) ]/2 → (B, L,L,L). m == n is the clean 1x2 loop."""
    out = 0
    for k, (mu, nu) in enumerate([(0, 1), (0, 2), (1, 2)]):
        Pn = P[n][:, k]; T = U3[m][:, mu]
        nb = torch.roll(Pn, -1, dims=1 + mu)
        out = out + (T @ nb @ dag(T) @ Pn).diagonal(dim1=-2, dim2=-1).sum(-1).real / 2
    return out
def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); return (a * b).mean() / (a.pow(2).mean() * b.pow(2).mean()).sqrt()
def zero_mom_stats(o):
    """o (N*Lt, L,L,L) → rho(1), rho(2) of the connected zero-momentum correlator."""
    ob = o.sum(dim=(1, 2, 3)).view(NCFG, Lt).double(); d = ob - ob.mean()
    C = [(d * d.roll(-k, 1)).mean() for k in range(3)]
    return (C[1] / C[0]).item(), (C[2] / C[0]).item()
print("\nn = plaquette level, m = transport level.  site-corr / zero-mom corr with the clean rectangle (m = n);  rho(1)=C(1)/C(0), rho(2) of the hybrid itself")
for n in levels:
    clean = hybrid(n, n)
    for m in levels:
        h = hybrid(n, m)
        zc = corr(h.sum(dim=(1, 2, 3)), clean.sum(dim=(1, 2, 3))).item()
        r1, r2 = zero_mom_stats(h)
        print(f"  n={n} m={m}: site {corr(h, clean).item():.3f}  zero-mom {zc:.3f}   rho(1) {r1:.3f}  rho(2) {r2:.3f}" + ("   <- clean" if m == n else "   <- what the net gets" if m == 0 else ""))
print("total", time.time() - t, "s")
