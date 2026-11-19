"""P1 (notes/thesis_record.md §11, prof_notes S2 task 3): Δm_eff(1), Δm_eff(2) of
GELT against the strongest classical bases, offline.

A₀ saturates above ≈ 0.9 at `deep`/`full`; m_eff(Δ = 1) does not. This computes

    Δm_eff(Δ) = m_eff^GELT(Δ) − m_eff^classical(Δ),   Δ = 1, 2,   m_eff = log C(Δ)/C(Δ+1)

for the classical arms `published` (4 levels × plaquette), `deep` (7 levels),
`full21` (21 operators — the arm called `full` until 2026-10-02, and in every
number recorded before that date) and `full` (35 operators, 7 levels × all five
shapes), each projected with ``gevp_ground_vector(truncate=True)``
at (t0, td) = (1, 2) and (0, 1) — (0, 1) is the optimal linear operator for the
network's own objective C(1)/C(0).

Two ways to score the classical arm:
  split     v₀ is fitted on one half of the test configurations and applied to the
            other, then swapped (the two halves' vacuum-subtracted correlators
            averaged), so the classical arm is not scored in-sample. This is the
            reading the record prescribes.
  insample  v₀ from all 400 configurations, as every other table in the repo —
            printed beside it so the cost of the split is visible.
In both, v₀ is recomputed inside every delete-block jackknife sample (the
estimator layer is ``fit_glueball_overlap.blocked_jackknife``, block from the
dump's meta), and the difference is jackknifed as a difference, so the
correlation through the shared configurations is automatic. Ensembles are
combined by inverse variance, as ``fit_window_scan.py`` does.

Gates (a wrong number never prints): the in-sample 4-level GELT − `published`
at (1, 2) with the historical floor whitening reproduces Δm_eff(1) =
−0.028(7) (run5) / −0.038(8) (ens1), and the cached `published` equals the dump's
Obar_basis.

Readings fixed in the record before this was run: Δm_eff(1) vs `full` ≤ −3σ
(combined) → "better than the strongest classical basis on an unsaturated
metric"; |Δ| < 2σ → "matches a 21-operator basis from 7 inputs". They were
fixed for the 21-operator arm, now `full21`; the 35-operator `full` row was
added afterwards and is read by the same rule.

Offline, seconds–minutes. Usage:  python scripts/p1_meff_strong_basis.py
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

EPS = 1e-4  # truncation cut, SFF_GEVP_EPS's value
TIMES = [(1, 2), (0, 1)]
ARMS = ["published", "deep", "full21", "full"]
# (tag, 7-level GELT dump, 4-level GELT dump, obars cache)
ENS = {
    "run5": ("best_glueball_gelt_sm0-2-4-6-8-12-16_test_obars.pt",
             "best_glueball_gelt_sm0-2-4-6_test_obars.pt", "su2_fair_fight_obars_run5.pt"),
    "ens1": ("best_glueball_gelt_sm0-2-4-6-8-12-16_ens1_test_obars.pt",
             "best_glueball_gelt_sm0-2-4-6_ens1_test_obars.pt", "su2_fair_fight_obars_ens1.pt"),
    "ens2": (None, "best_glueball_gelt_sm0-2-4-6_ens2_test_obars.pt",
             "su2_fair_fight_obars_ens2.pt"),
}
PUBLISHED_DMEFF1 = {"run5": (-0.028, 0.007), "ens1": (-0.038, 0.008)}
OUT = "results/glueball/p1_meff_strong_basis"


def _combine(vals):
    w = [1.0 / e**2 for _, e in vals]
    return sum(wi * x for wi, (x, _) in zip(w, vals)) / sum(w), 1.0 / math.sqrt(sum(w))


def ground_vector(basis, t0, td, eps, truncate):
    return gevp_ground_vector(connected_correlator_matrix(basis), t0=t0, td=td,
                              eps=eps, truncate=truncate)


def classical_corr(basis, mask, t0, td, mode, eps, truncate):
    """C(Δ) of the v₀-projected classical operator on the configurations in mask."""
    if mode == "insample":
        v0 = ground_vector(basis[:, mask], t0, td, eps, truncate)
        return connected_correlator(torch.einsum("i,ibt->bt", v0, basis[:, mask]))
    B = basis.shape[1]
    half = B // 2
    idx = torch.arange(B)
    Cs = []
    for tr, ev in ((idx < half, idx >= half), (idx >= half, idx < half)):
        v0 = ground_vector(basis[:, tr & mask], t0, td, eps, truncate)
        Cs.append(connected_correlator(torch.einsum("i,ibt->bt", v0, basis[:, ev & mask])))
    return 0.5 * (Cs[0] + Cs[1])


def measure(gelt, basis, jb, t0, td, mode, eps, truncate):
    """(mean, err) of [meff1_g, meff2_g, meff1_c, meff2_c, d1, d2] over blocks."""
    def stats(mask):
        eg = effective_mass(connected_correlator(gelt[mask]))
        ec = effective_mass(classical_corr(basis, mask, t0, td, mode, eps, truncate))
        return torch.stack([eg[1], eg[2], ec[1], ec[2], eg[1] - ec[1], eg[2] - ec[2]])
    return fgo.blocked_jackknife(stats, gelt.shape[0], jb)


def main():
    res = {}
    for tag, (d7, d4, cache) in ENS.items():
        ob = torch.load(os.path.join("dumps", cache), map_location="cpu", weights_only=False)
        bases = {n: v.double() for n, v in ob["bases"].items()}
        gelts = {}
        for lv, f in (("7lv", d7), ("4lv", d4)):
            if f is None:
                continue
            blob = torch.load(os.path.join("dumps", f), map_location="cpu", weights_only=False)
            gelts[lv] = blob["gelt_obar"].double()
            jb = int(blob.get("meta", {}).get("jack_block", 10))
            drift = (blob["Obar_basis"].double() - bases["published"]).abs().max().item()
            scale = bases["published"].abs().max().item()
            if drift / scale > 1e-5:
                raise SystemExit(f"{tag}/{lv}: cached `published` differs from the dump's "
                                 f"Obar_basis by {drift / scale:.1e}")
        print(f"── {tag}: {bases['published'].shape[1]} test configs, block {jb}, "
              f"GELT dumps {list(gelts)}")
        # Gate: the published Δm_eff(1), floor whitening at (1, 2), in-sample.
        if tag in PUBLISHED_DMEFF1:
            m, e = measure(gelts["4lv"], bases["published"], jb, 1, 2, "insample", 1e-12, False)
            got, want = (m[4].item(), e[4].item()), PUBLISHED_DMEFF1[tag]
            ok = abs(got[0] - want[0]) < 0.0015 and abs(got[1] - want[1]) < 0.0015
            print(f"  gate  4-level GELT − published, Δm_eff(1) = {got[0]:+.4f} ± {got[1]:.4f}"
                  f"  (record {want[0]:+.3f} ± {want[1]:.3f})  {'ok' if ok else 'MISMATCH'}")
            if not ok:
                raise SystemExit("gate failed — the estimator is not the record's")
        for lv, gelt in gelts.items():
            for (t0, td) in TIMES:
                for arm in ARMS:
                    for mode in ("split", "insample"):
                        m, e = measure(gelt, bases[arm], jb, t0, td, mode, EPS, True)
                        res[(tag, lv, t0, td, arm, mode)] = (m, e)
    # ── report ────────────────────────────────────────────────────────────────
    lines = []
    for lv in ("7lv", "4lv"):
        for (t0, td) in TIMES:
            for mode in ("split", "insample"):
                lines.append(f"\n{lv} GELT − classical, (t0, td) = ({t0}, {td}), "
                             f"v₀ {'half-split' if mode == 'split' else 'in-sample'}, "
                             f"truncate eps {EPS:g}")
                lines.append(f"  {'arm':<10}{'ens':<6}{'m_eff1 G':>10}{'m_eff1 C':>10}"
                             f"{'Δm_eff(1)':>20}{'Δm_eff(2)':>20}")
                for arm in ARMS:
                    comb = {1: [], 2: []}
                    for tag in ENS:
                        if (tag, lv, t0, td, arm, mode) not in res:
                            continue
                        m, e = res[(tag, lv, t0, td, arm, mode)]
                        comb[1].append((m[4].item(), e[4].item()))
                        comb[2].append((m[5].item(), e[5].item()))
                        lines.append(
                            f"  {arm:<10}{tag:<6}{m[0].item():>10.4f}{m[2].item():>10.4f}"
                            f"{m[4].item():>+12.4f}±{e[4].item():.4f}"
                            f"{m[5].item():>+12.4f}±{e[5].item():.4f}")
                    c1, c2 = _combine(comb[1]), _combine(comb[2])
                    lines.append(
                        f"  {arm:<10}{'comb':<6}{'':>20}"
                        f"{c1[0]:>+12.4f}±{c1[1]:.4f} ({c1[0] / c1[1]:+.1f}σ)"
                        f"{c2[0]:>+9.4f}±{c2[1]:.4f} ({c2[0] / c2[1]:+.1f}σ)")
    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save({"res": {k: (v[0], v[1]) for k, v in res.items()}, "eps": EPS}, OUT + ".pt")
    with open(OUT + ".txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
