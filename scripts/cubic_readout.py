"""S4's readout: every number the thesis quotes about the A₁⁺⁺ projection.

Offline, seconds. Reads ``dumps/cubic/`` (written by ``cubic_projection.py``)
and the source dumps beside them, and prints, per net:

  - the gates the V100 run recorded (anchor, action, classical invariance);
  - ν(g) median and max, and each irrep's share of C(0) on the O_h-augmented
    sample (recomputed from the 48 Ō_g and checked against the stored value);
  - for trained nets, the estimator of ``fit_glueball_overlap.py`` (imported,
    not copied: same window, same fixed diagonal σ, v₀ refitted per jackknife
    sample, block 10) on the original operator and on its A₁ projection — m,
    A₀, ΔA₀ against the GEVP-projected operator — and the paired
    A₀(Ō_A1) − A₀(Ō) and Δm_eff(1) inside one jackknife;
  - for untrained nets, m_eff(1) of both (their free cosh fit does not
    converge, see notes/prof_notes.md "Third ensemble", E6);
  - inverse-variance combinations over the 4-level trained nets;
  - **outliers**: an entry of some Ō_g more than ``OUTLIER_SD`` sd of Ō_e away
    from Ō_e is an exploding output, not a symmetry violation (the L-CNN's p2
    net: three configurations that are ordinary in their own orientation give
    |Ō| up to 4e10 under five of the 48 elements). The net is flagged, and every
    number above is recomputed *also* with those configurations removed from
    all 48 elements — a diagnostic chosen after seeing the outlier, labelled so.

Run:
    python scripts/cubic_readout.py
"""

import glob
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ARGV = sys.argv
sys.argv = sys.argv[:1]
import fit_glueball_overlap as fgo  # noqa: E402

sys.argv = _ARGV

from gelt.cubic import IRREPS, cubic_elements, irrep_fractions  # noqa: E402
from gelt.glueball import connected_correlator, effective_mass  # noqa: E402

CUBIC_DIR = "dumps/cubic"
FRACTION_TOL = 1e-9  # recomputed vs stored irrep shares
OUTLIER_SD = 1e3  # |Ō_g − Ō_e| in units of sd(Ō_e) that marks an exploding output


def _ens(stem):
    return f"ens{stem.split('_ens')[1].split('_')[0]}" if "_ens" in stem else "ens0"


def _kind(stem):
    arch = "L-CNN" if "_lcnn_" in stem else "GELT"
    levels = "7lv" if "-8-12-16" in stem else "4lv"
    return arch, levels, "_rnd" in stem


def estimator(orig, a1, basis, t0, td, jb):
    """fit_glueball_overlap's protocol on (orig, a1) plus their paired differences."""
    B = orig.shape[0]
    sig = {k: fgo.blocked_jackknife(lambda m, o=o: connected_correlator(o[m]), B, jb)[1]
           for k, o in (("o", orig), ("a", a1))}
    proj_full = fgo.project_ground(basis, t0, td)
    sig_p = fgo.blocked_jackknife(lambda m: connected_correlator(proj_full[m]), B, jb)[1]

    def stats(mask):
        m_o, a_o, _ = fgo.fit_one(connected_correlator(orig[mask]), sig["o"])
        m_a, a_a, _ = fgo.fit_one(connected_correlator(a1[mask]), sig["a"])
        proj = fgo.project_ground(basis[:, mask], t0, td)
        m_p, a_p, _ = fgo.fit_one(connected_correlator(proj), sig_p)
        me_o = effective_mass(connected_correlator(orig[mask]))[1].item()
        me_a = effective_mass(connected_correlator(a1[mask]))[1].item()
        me_p = effective_mass(connected_correlator(proj))[1].item()
        return torch.tensor([m_o, a_o, m_a, a_a, m_p, a_p, a_o - a_p, a_a - a_p,
                             a_a - a_o, m_a - m_o, me_o - me_p, me_a - me_p, me_a - me_o],
                            dtype=torch.float64)

    mean, err = fgo.blocked_jackknife(stats, B, jb)
    keys = ["m_o", "A0_o", "m_a", "A0_a", "m_p", "A0_p", "dA0_o", "dA0_a",
            "A0_a-o", "m_a-o", "dmeff1_o", "dmeff1_a", "meff1_a-o"]
    return {k: (mean[i].item(), err[i].item()) for i, k in enumerate(keys)}


def meff1(ob, jb):
    m, s = fgo.blocked_jackknife(
        lambda k: effective_mass(connected_correlator(ob[k]))[1:2], ob.shape[0], jb)
    return m.item(), s.item()


def combine(vals):
    w = [1 / e ** 2 for _, e in vals]
    x = sum(v * wi for (v, _), wi in zip(vals, w)) / sum(w)
    return x, 1 / math.sqrt(sum(w))


def fmt(v, e, sig=False):
    s = f"{v:+.4f} ± {e:.4f}"
    return s + (f" ({abs(v) / e:.1f}σ)" if sig else "")


def outlier_configs(obar_g):
    """Configurations where some Ō_g departs from Ō_e by more than OUTLIER_SD sd."""
    e = obar_g[0]
    dev = (obar_g - e).abs().amax(dim=(0, 2)) / (e - e.mean()).std()
    return sorted((dev > OUTLIER_SD).nonzero().flatten().tolist())


def main():
    elements = cubic_elements()
    files = sorted(f for f in glob.glob(f"{CUBIC_DIR}/cubic_*.pt") if not f.endswith("_gen.pt"))
    if not files:
        raise SystemExit(f"no cubic_*.pt in {CUBIC_DIR}")
    rows, comb = [], {}
    print(f"{'net':<48} {'outside A1g':>11}  largest non-A1 shares            ν med (max)")
    for f in files:
        r = torch.load(f, weights_only=False)
        stem = os.path.basename(f)[len("cubic_"):-len(".pt")]
        frac = irrep_fractions(r["obar_g"], elements)
        drift = max(abs(frac[k] - r["irrep_fraction_C0"][k]) for k in IRREPS)
        if drift > FRACTION_TOL:
            raise SystemExit(f"{stem}: recomputed irrep shares differ from stored by {drift:.1e}")
        a1 = torch.load(f"{CUBIC_DIR}/{stem}_a1_test_obars.pt", weights_only=False)
        if not torch.allclose(a1["gelt_obar"], r["obar_g"].mean(0), rtol=0, atol=1e-9):
            raise SystemExit(f"{stem}: the _a1 dump is not the orbit mean")
        nu = sorted(r["noninvariance"].values())
        top = sorted(((v, k) for k, v in frac.items() if k != "A1g"), reverse=True)[:3]
        gates = (max(r["anchor"].values()), max(r["classical_dev"]), max(r["action_dev"]))
        print(f"{stem:<48} {100 * (1 - frac['A1g']):10.2f}%  "
              + "  ".join(f"{k} {100 * v:.2f}%" for v, k in top)
              + f"   {nu[len(nu) // 2]:.3f} ({nu[-1]:.3f})"
              + f"   gates {gates[0]:.0e}/{gates[1]:.0e}/{gates[2]:.0e}")
        rows.append((stem, r, frac, a1))
        bad = outlier_configs(r["obar_g"])
        if bad:
            keep = torch.tensor([i for i in range(r["obar_g"].shape[1]) if i not in bad])
            og = r["obar_g"][:, keep]
            fr = irrep_fractions(og, elements)
            n_el = int(((r["obar_g"] - r["obar_g"][0]).abs().amax(dim=(1, 2))
                        > OUTLIER_SD * (r["obar_g"][0] - r["obar_g"][0].mean()).std()).sum())
            print(f"  ** {stem}: exploding output on configs {bad} under {n_el} of 48 elements "
                  f"(max |Ō_g| {r['obar_g'].abs().max():.2e}); without them: outside A1g "
                  f"{100 * (1 - fr['A1g']):.2f}%, parity-odd "
                  f"{100 * sum(v for k, v in fr.items() if k.endswith('u')):.2f}%")
            rows.append((stem + "  [outlier configs removed]", {**r, "obar_g": og,
                         "_keep": keep}, fr, {"gelt_obar": og.mean(0)}))

    print("\ntrained nets — the fit_glueball_overlap estimator on Ō and Ō_A1 (window "
          f"{fgo.FIT_WINDOW}, block from the dump's meta)")
    for stem, r, frac, a1 in rows:
        arch, lv, rnd = _kind(stem)
        src = torch.load(r["source_dump"], weights_only=False)
        meta = src["meta"]
        t0 = int(meta.get("gevp_t0", 1))
        td = t0 + 1 if fgo.GEVP_TD is None else fgo.GEVP_TD
        jb = int(meta.get("jack_block", 10))
        orig, a1o = src["gelt_obar"].double(), a1["gelt_obar"].double()
        basis = src["Obar_basis"].double()
        if "_keep" in r:
            orig, basis = orig[r["_keep"]], basis[:, r["_keep"]]
        elif outlier_configs(r["obar_g"]):
            print(f"  {stem:<46} exploding outputs — see the outlier-removed row")
            continue
        if rnd:
            mo, mso = meff1(orig, jb)
            ma, msa = meff1(a1o, jb)
            print(f"  {stem:<46} untrained: m_eff(1) Ō {mo:.3f}({1000 * mso:.0f})  "
                  f"Ō_A1 {ma:.3f}({1000 * msa:.0f})")
            continue
        res = estimator(orig, a1o, basis, t0, td, jb)
        print(f"  {stem:<46} [{arch} {lv} {_ens(stem)}]")
        for tag, mk, ak, dk, ek in (("Ō   ", "m_o", "A0_o", "dA0_o", "dmeff1_o"),
                                   ("Ō_A1", "m_a", "A0_a", "dA0_a", "dmeff1_a")):
            print(f"     {tag}  m {res[mk][0]:.4f}({1e4 * res[mk][1]:.0f})  "
                  f"A0 {res[ak][0]:.4f}({1e4 * res[ak][1]:.0f})  "
                  f"ΔA0 vs GEVP {fmt(*res[dk], True)}  Δm_eff(1) vs GEVP {fmt(*res[ek], True)}")
        print(f"     paired A0(Ō_A1) − A0(Ō) {fmt(*res['A0_a-o'], True)}   "
              f"m {fmt(*res['m_a-o'])}   m_eff(1) {fmt(*res['meff1_a-o'], True)}")
        comb.setdefault((arch, lv) + (("outliers removed",) if "_keep" in r else ()),
                        []).append(res)

    print("\ncombined (inverse variance)")
    lcnn = comb.pop(("L-CNN", "4lv"), []) + comb.pop(("L-CNN", "4lv", "outliers removed"), [])
    if lcnn:
        comb[("L-CNN", "4lv", "ens0 outlier configs removed")] = lcnn
    for key, rs in comb.items():
        if len(rs) < 2:
            continue
        print(f"  {' '.join(key)}, {len(rs)} ensembles:")
        for k, lab in (("A0_a-o", "paired A0(Ō_A1) − A0(Ō)"), ("dA0_o", "ΔA0 vs GEVP, Ō"),
                       ("dA0_a", "ΔA0 vs GEVP, Ō_A1"), ("meff1_a-o", "m_eff(1)(Ō_A1) − m_eff(1)(Ō)"),
                       ("dmeff1_a", "Δm_eff(1) vs GEVP, Ō_A1")):
            print(f"     {lab:<32} {fmt(*combine([r[k] for r in rs]), True)}")
    un = [100 * (1 - fr["A1g"]) for s, _, fr, _ in rows if _kind(s)[2]]
    eg = [100 * fr["Eg"] for s, _, fr, _ in rows if _kind(s)[2]]
    if un:
        print(f"  untrained GELT, {len(un)} nets: outside A1g {min(un):.1f}–{max(un):.1f}%, "
              f"Eg {min(eg):.1f}–{max(eg):.1f}%")


if __name__ == "__main__":
    main()
