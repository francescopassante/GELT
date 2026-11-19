"""S3 task 3 (notes/prof_notes.md): ΔA0 and Δm_eff(1) after dropping the first g test configs.
Run from the repo root: python scripts/gap_scan.py  (offline, prints only)."""
import math, os, sys
import torch
sys.path.insert(0, "scripts")
import fit_glueball_overlap as fgo
from gelt.glueball import connected_correlator, effective_mass

DUMPS = {"ens0": "dumps/best_glueball_gelt_sm0-2-4-6_test_obars.pt",
         "ens1": "dumps/best_glueball_gelt_sm0-2-4-6_ens1_test_obars.pt",
         "ens2": "dumps/best_glueball_gelt_sm0-2-4-6_ens2_test_obars.pt"}
GAPS = (0, 5, 10, 20)
PUB = {"ens0": (0.066, 0.031), "ens1": (0.089, 0.030), "ens2": (0.100, 0.028)}
BLOCK = 10

def comb(v):
    w = [1 / e**2 for _, e in v]
    return sum(wi * x for wi, (x, _) in zip(w, v)) / sum(w), 1 / math.sqrt(sum(w))

res = {}
for tag, path in DUMPS.items():
    blob = torch.load(path)
    gelt, basis = blob["gelt_obar"].double(), blob["Obar_basis"].double()
    t0 = int(blob.get("meta", {}).get("gevp_t0", 1)); td = t0 + 1
    for g in GAPS:
        G, Bs = gelt[g:], basis[:, g:]
        B = G.shape[0]
        proj = fgo.project_ground(Bs, t0, td)
        _, sg = fgo.blocked_jackknife(lambda m: connected_correlator(G[m]), B, BLOCK)
        _, sp = fgo.blocked_jackknife(lambda m: connected_correlator(proj[m]), B, BLOCK)
        def stats(mask):
            Cg = connected_correlator(G[mask])
            Cp = connected_correlator(fgo.project_ground(Bs[:, mask], t0, td))
            _, ag, _ = fgo.fit_one(Cg, sg); _, ap, _ = fgo.fit_one(Cp, sp)
            return torch.tensor([ag - ap, effective_mass(Cg)[1].item() - effective_mass(Cp)[1].item()],
                                dtype=torch.float64)
        mean, err = fgo.blocked_jackknife(stats, B, BLOCK)
        # central value on the full sub-sample
        full = stats(torch.ones(B, dtype=torch.bool))
        res[(tag, g)] = (full[0].item(), err[0].item(), full[1].item(), err[1].item(), B)
        print(f"{tag} g={g:2d} B={B}: dA0 = {full[0]:+.4f} ± {err[0]:.4f}   dmeff1 = {full[1]:+.4f} ± {err[1]:.4f}")
    got = res[(tag, 0)]
    print(f"  gate g=0 vs published {PUB[tag]}: {got[0]:+.3f} ± {got[1]:.3f}")

print("\ncombined (inverse variance)")
for g in GAPS:
    a = comb([(res[(t, g)][0], res[(t, g)][1]) for t in DUMPS])
    m = comb([(res[(t, g)][2], res[(t, g)][3]) for t in DUMPS])
    print(f"g={g:2d}: dA0 = {a[0]:+.4f} ± {a[1]:.4f}   dmeff1 = {m[0]:+.4f} ± {m[1]:.4f}")
print("\nshift of dA0 vs g=0, in units of the g=0 error")
for t in DUMPS:
    r0 = res[(t, 0)]
    print(t, [f"{(res[(t,g)][0]-r0[0])/r0[1]:+.2f}" for g in GAPS[1:]])
