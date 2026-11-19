"""S2 task 2 (notes/prof_notes.md): stability of the classical arm under the GEVP
reference times (t0, td) ∈ {(0,1), (1,2), (1,3), (2,3)}, offline.

For each pair, on the `published` (4 levels × plaquette) and `deep` (7 levels)
bases of the kept obars caches, the v₀-projected operator is fitted on the
published window [2, 7] exactly as ``fit_glueball_overlap`` does (σ_Δ fixed from
the full sample, v₀ and the whole fit redone in every delete-block jackknife
sample), and compared with the 4-level GELT of the same ensemble:

    ΔA₀ = A₀^GELT − A₀^GEVP,   Δm_eff(1) = m_eff^GELT(1) − m_eff^GEVP(1),

plus the GEVP's own plateau: m_fit, m_eff(1), m_eff(2). `published` uses the
historical floor whitening (so (1,2) is the record's number, which is the gate:
ΔA₀ = +0.066 / +0.089 on run5 / ens1); `deep` has seven operators and uses the
truncated whitening, eps 1e-4. Ensembles are combined by inverse variance.

Reading fixed in prof_notes S2: ΔA₀ vs `published` changing by < 1σ across the
scan → one sentence "stable under the reference times"; larger → the table
reports the scan and the headline uses the (t0, td) that maximises the classical
A₀ (the conservative choice).

Offline, seconds–minutes.  Usage:  python scripts/gevp_time_scan.py
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
    effective_mass,
    gevp_ground_vector,
)

PAIRS = [(0, 1), (1, 2), (1, 3), (2, 3)]
ARMS = {"published": (1e-12, False), "deep": (1e-4, True)}  # eps, truncate
ENS = {
    "run5": ("best_glueball_gelt_sm0-2-4-6_test_obars.pt", "su2_fair_fight_obars_run5.pt"),
    "ens1": ("best_glueball_gelt_sm0-2-4-6_ens1_test_obars.pt", "su2_fair_fight_obars_ens1.pt"),
    "ens2": ("best_glueball_gelt_sm0-2-4-6_ens2_test_obars.pt", "su2_fair_fight_obars_ens2.pt"),
}
PUBLISHED_DA0 = {"run5": 0.066, "ens1": 0.089}
OUT = "results/glueball/gevp_time_scan"
KEYS = ["m_gevp", "A0_gevp", "meff1_gevp", "meff2_gevp", "dA0", "dm", "dmeff1"]


def _combine(vals):
    w = [1.0 / e**2 for _, e in vals]
    return sum(wi * x for wi, (x, _) in zip(w, vals)) / sum(w), 1.0 / math.sqrt(sum(w))


def project(basis, t0, td, eps, trunc):
    v0 = gevp_ground_vector(connected_correlator_matrix(basis), t0=t0, td=td,
                            eps=eps, truncate=trunc)
    return torch.einsum("i,ibt->bt", v0, basis)


def scan(gelt, basis, jb, t0, td, eps, trunc):
    B = gelt.shape[0]
    _, sig_g = fgo.blocked_jackknife(lambda m: connected_correlator(gelt[m]), B, jb)
    full = project(basis, t0, td, eps, trunc)
    _, sig_c = fgo.blocked_jackknife(lambda m: connected_correlator(full[m]), B, jb)

    def stats(mask):
        Cg = connected_correlator(gelt[mask])
        Cc = connected_correlator(project(basis[:, mask], t0, td, eps, trunc))
        m_g, a_g, _ = fgo.fit_one(Cg, sig_g)
        m_c, a_c, _ = fgo.fit_one(Cc, sig_c)
        eg, ec = effective_mass(Cg), effective_mass(Cc)
        return torch.tensor([m_c, a_c, ec[1], ec[2], a_g - a_c, m_g - m_c, eg[1] - ec[1]],
                            dtype=torch.float64)
    return fgo.blocked_jackknife(stats, B, jb)


def main():
    res = {}
    for tag, (dump, cache) in ENS.items():
        blob = torch.load(os.path.join("dumps", dump), map_location="cpu", weights_only=False)
        gelt = blob["gelt_obar"].double()
        jb = int(blob.get("meta", {}).get("jack_block", 10))
        bases = {n: v.double() for n, v in torch.load(
            os.path.join("dumps", cache), map_location="cpu", weights_only=False)["bases"].items()}
        for (t0, td) in PAIRS:
            for arm, (eps, trunc) in ARMS.items():
                res[(tag, t0, td, arm)] = scan(gelt, bases[arm], jb, t0, td, eps, trunc)
        if tag in PUBLISHED_DA0:
            got = res[(tag, 1, 2, "published")][0][4].item()
            ok = abs(got - PUBLISHED_DA0[tag]) < 0.0015
            print(f"gate {tag}: ΔA₀(published, (1,2)) = {got:+.4f}  (record "
                  f"{PUBLISHED_DA0[tag]:+.3f})  {'ok' if ok else 'MISMATCH'}")
            if not ok:
                raise SystemExit("gate failed — the estimator is not the record's")

    lines = []
    tags = list(ENS)
    for arm in ARMS:
        lines.append(f"\n── classical arm `{arm}`, window [2,7], 4-level GELT of each ensemble")
        lines.append(f"{'(t0,td)':<9}{'ens':<6}{'m_fit':>8}{'A0':>8}{'m_eff1':>8}{'m_eff2':>8}"
                     f"{'ΔA0':>18}{'Δm':>18}{'Δm_eff(1)':>18}")
        base = None
        for (t0, td) in PAIRS:
            comb = {k: [] for k in KEYS}
            for tag in tags:
                mean, err = res[(tag, t0, td, arm)]
                for i, k in enumerate(KEYS):
                    comb[k].append((mean[i].item(), err[i].item()))
                lines.append(
                    f"({t0},{td})    {tag:<6}{mean[0]:>8.4f}{mean[1]:>8.4f}{mean[2]:>8.4f}"
                    f"{mean[3]:>8.4f}{mean[4]:>+11.4f}±{err[4]:.4f}{mean[5]:>+11.4f}±{err[5]:.4f}"
                    f"{mean[6]:>+11.4f}±{err[6]:.4f}")
            c = {k: _combine(v) for k, v in comb.items()}
            if (t0, td) == (1, 2):
                base = c["dA0"]
            lines.append(
                f"({t0},{td})    {'comb':<6}{c['m_gevp'][0]:>8.4f}{c['A0_gevp'][0]:>8.4f}"
                f"{c['meff1_gevp'][0]:>8.4f}{c['meff2_gevp'][0]:>8.4f}"
                f"{c['dA0'][0]:>+11.4f}±{c['dA0'][1]:.4f}{c['dm'][0]:>+11.4f}±{c['dm'][1]:.4f}"
                f"{c['dmeff1'][0]:>+11.4f}±{c['dmeff1'][1]:.4f}")
            res[("comb", t0, td, arm)] = c
        # Task reading: ΔA₀ shift from (1,2) in units of the (1,2) combined error.
        for (t0, td) in PAIRS:
            c = res[("comb", t0, td, arm)]
            lines.append(f"  ΔA₀({t0},{td}) − ΔA₀(1,2) = {c['dA0'][0] - base[0]:+.4f} "
                         f"({(c['dA0'][0] - base[0]) / base[1]:+.2f}σ of the (1,2) error)")
    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save({"res": res, "pairs": PAIRS}, OUT + ".pt")
    with open(OUT + ".txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
