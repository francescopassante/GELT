"""Transfer-matrix distillation: student against its teacher, paired. Offline, seconds.

Reads the dumps of scripts/train_distill.py (each carries the student's series as
``gelt_obar`` and the teacher's, on the same test configurations, as
``teacher_obar``) and, with --hs=, a scripts/high_stat_eval.py dump of the
student written with HS_ALSO=<teacher checkpoint> (teacher in ``extra_obars``).
Every difference is formed inside every delete-block jackknife sample; no fit
window, no fit model, no A₀ estimator (scripts/variational_headroom.py is why).

Per dump:
  ΔL        L(student) − L(teacher), L = −mean[C(1)/C(0), C(2)/C(0)] — the loss
            the TEACHER was trained on and the student was not.
  Δm(Δ)     cosh effective mass, student − teacher, Δ = 0..3.
  shift(Δ)  [m_T(Δ) − m_S(Δ)] / [m_T(Δ) − m_T(Δ + 2ā)], ā the mean lag: the share
            of the ideal Euclidean shift C_S(Δ) = C_T(Δ + 2ā) that was realised.
            0 = the student is its teacher, 1 = the exact regression function.
  η(a)      C(a)²/(C(0)C(2a)) with the wrap correction, both nets: the large-a
            value is the true ground-state overlap.
  classical (400-configuration dumps with a fair-fight cache) the student against
            `shapes_sm` and `full`, cross-fitted, and L(full ∪ student) − L(full).

Readings, fixed 2026-11-12 before any student existed (notes/thesis_record.md
§11 P11); "combined" = inverse variance over the ensembles given, and a
high-statistics dump is read on its own:
  R-T1  ΔL < 0 at ≥ 3σ AND Δm(1), Δm(2) each < 0 at ≥ 2σ → regressing the
        Euclidean future yields a purer operator than the Rayleigh loss found in
        the same function class: the training principle, not the architecture,
        was the limit. Quote ΔL, the shift, and η.
  R-T2  in addition L(full ∪ student) − L(full) < 0 at ≥ 3σ, where the 4-level
        teacher's is −0.0006 ± 0.0007 → the student carries ground-state content
        outside all 35 classical operators.
  R-T0  |ΔL| < 2σ → the student is its teacher. With the function class held
        fixed (TMD_INIT=teacher) this says the class, as gradient descent
        explores it from that start, does not contain the evolved operator at
        this lag. It does not say the identity fails (tests/test_headroom.py).
  R-T−  ΔL > 0 at ≥ 3σ → the regression made the operator worse although its
        start was a candidate on validation: a validation/test mismatch, to be
        understood before anything is quoted.

Dumps are combined only within one loss version (meta["tmd"]["loss"]: "raw" for
the first run, "connected" since 2026-11-20 — scripts/train_distill.py says why),
and never two dumps of one ensemble: those are two students, not a replication.

Usage:  python scripts/tmd_readout.py [student dumps …] [--hs=<dump>] [--m-ref=0.313]
"""

import glob
import math
import os
import sys

import torch

_known = ("--hs=", "--m-ref=", "--eps=")
_bad = [a for a in sys.argv[1:] if a.startswith("--") and not a.startswith(_known)]
if _bad:
    raise SystemExit(f"unknown flag(s) {_bad}; known: {list(_known)}")
_ARGV = sys.argv
# variational_headroom parses argv at import and refuses what it does not know:
# hand it only the two flags the scripts share (--hs= means something else there).
sys.argv = sys.argv[:1] + [a for a in sys.argv[1:] if a.startswith(("--m-ref=", "--eps="))]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import variational_headroom as vh  # noqa: E402

sys.argv = _ARGV
HS = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--hs=")), None)
DUMPS = [a for a in sys.argv[1:] if not a.startswith("--")] or sorted(
    p for p in glob.glob("dumps/*_tmd*_test_obars.pt") + glob.glob(
        "results/glueball/*_tmd*_test_obars.pt") if "_hs_" not in p)
KMAX = 11
N_ROW = 2 + 4 + 2 + 5 + 5  # L_S, ΔL | Δm(0..3) | shift(0,1) | η_S(1..5) | η_T(1..5)


def paired(S, T, lag, Nt, keep):
    """The per-dump row for one jackknife sample (see the module docstring)."""
    Cs, Ct = vh.corr1(S[keep], KMAX), vh.corr1(T[keep], KMAX)
    ms = [vh.cosh_mass(Cs, t, Nt) for t in range(4)]
    mt = [vh.cosh_mass(Ct, t, Nt) for t in range(9)]
    row = [vh.loss_of(Cs).item(), (vh.loss_of(Cs) - vh.loss_of(Ct)).item()]
    row += [ms[t] - mt[t] for t in range(4)]
    for t in (0, 1):
        k = min(int(round(t + 2 * lag)), 8)
        den = mt[t] - mt[k]
        row.append((mt[t] - ms[t]) / den if abs(den) > 1e-9 else float("nan"))
    for C in (Cs, Ct):
        row += [(C[a] ** 2 / (C[0] * C[2 * a])).item() * vh.wrap_factor(a, Nt, vh.M_REF)
                for a in range(1, 6)]
    return torch.tensor(row, dtype=torch.float64)


def classical(S, T, basis, keep):
    """[ΔL student − arm (cross-fit), L(arm ∪ student) − L(arm), same for the teacher]."""
    B = vh.unit(basis[:, keep])
    Lc = vh.loss_of(vh.crossfit(B, vh.EPS))
    out = []
    for o in (S, T):
        out.append(vh.loss_of(vh.corr1(o[keep], 3)) - Lc)
        out.append(vh.loss_of(vh.crossfit(torch.cat([B, vh.unit(o[None, keep])]), vh.EPS)) - Lc)
    return torch.stack(out)


def sig(x, e):
    return f"{x:+.4f} ± {e:.4f} ({x / e:+.1f}σ)" if e > 0 else f"{x:+.4f}"


def report(label, S, T, lag, block, Nt):
    n = S.shape[0]
    v, e = vh.jackknife(lambda keep: paired(S, T, lag, Nt, keep), n, block)
    print(f"\n== {label}   ({n} configurations, block {block}, mean lag {lag:g})")
    print(f"   L(student) {v[0]:+.4f}    ΔL = L(student) − L(teacher) = {sig(v[1], e[1])}")
    print("   Δ cosh m_eff(Δ), Δ = 0..3 : " + "   ".join(sig(v[2 + t], e[2 + t])
                                                          for t in range(4)))
    print("   shift realised, Δ = 0, 1  : " + "   ".join(f"{v[6 + t]:+.2f} ± {e[6 + t]:.2f}"
                                                          for t in range(2)))
    print("   η(a) wrap-corrected, a = 1..5   student: " + "  ".join(
        f"{v[8 + a]:.3f}({e[8 + a] * 1e3:.0f})" for a in range(5)))
    print("                                   teacher: " + "  ".join(
        f"{v[13 + a]:.3f}({e[13 + a] * 1e3:.0f})" for a in range(5)))
    return v, e


def verdict(dL, sL, dm1, sm1, dm2, sm2, un=None):
    z = dL / sL
    if z <= -3 and dm1 / sm1 <= -2 and dm2 / sm2 <= -2:
        r = "R-T1: a purer operator than the Rayleigh loss found in the same class"
        if un is not None and un[0] / un[1] <= -3:
            r += "; R-T2: it adds to the 35-operator basis"
        return r
    if z >= 3:
        return "R-T−: worse than its teacher — understand the val/test mismatch first"
    if abs(z) < 2:
        return "R-T0: the student is its teacher"
    return ("between the readings (ΔL %+.1fσ, Δm(1) %+.1fσ, Δm(2) %+.1fσ): report the "
            "numbers, claim nothing" % (z, dm1 / sm1, dm2 / sm2))


def main():
    groups = {}  # loss version → {"rows", "unions", "ens"}
    for path in DUMPS:
        d = torch.load(path, weights_only=False)
        if "teacher_obar" not in d:
            print(f"{path}: no teacher_obar — not a train_distill dump, skipped")
            continue
        meta = d["meta"]
        if meta.get("tmd", {}).get("smoke"):
            print(f"{path}: a plumbing-check dump, skipped")
            continue
        S, T = d["gelt_obar"].double(), d["teacher_obar"].double()
        kind = meta["tmd"].get("loss", "raw")
        g = groups.setdefault(kind, {"rows": [], "unions": [], "ens": []})
        rows, unions = g["rows"], g["unions"]
        lag = sum(meta["tmd"]["lags"]) / len(meta["tmd"]["lags"])
        if torch.equal(S, T):
            # train_distill keeps the start as a candidate: no epoch improved
            # the validation loss, so the saved student is the teacher's weights.
            print(f"\n== {path}\n   the student IS its teacher (the starting checkpoint was "
                  f"selected on validation). reading: R-T0")
            continue
        v, e = report(f"{path}  [{kind} loss]", S, T, lag, meta["jack_block"], meta["Lt"])
        if "test_shape" in meta["tmd"]:
            tm = meta["tmd"]
            print(f"   shape loss on test (after the best rescale): student {tm['test_shape']:.4f} "
                  f"at c* {tm['test_c']:.3f}, teacher {tm['test_shape_teacher']:.4f} at c* "
                  f"{tm['test_c_teacher']:.3f}")
        rows.append((v, e))
        g["ens"].append(meta["ensemble_seed"])
        ens = {0: "run5"}.get(meta["ensemble_seed"], f"ens{meta['ensemble_seed']}")
        cache = f"dumps/su2_fair_fight_obars_{ens}.pt"
        if os.path.exists(cache):
            F = torch.load(cache, weights_only=False)["bases"]
            for arm in ("shapes_sm", "full"):
                if arm not in F or F[arm].shape[1] != S.shape[0]:
                    continue
                c, ce = vh.jackknife(lambda keep, B=F[arm].double(): classical(S, T, B, keep),
                                     S.shape[0], meta["jack_block"])
                print(f"   vs {arm:9s}: ΔL student − arm {sig(c[0], ce[0])} (teacher "
                      f"{sig(c[2], ce[2])});  L(arm ∪ student) − L(arm) {sig(c[1], ce[1])} "
                      f"(teacher {sig(c[3], ce[3])})")
                if arm == "full":
                    unions.append((c[1:2], ce[1:2]))
        else:
            print(f"   (no {cache}: the classical rows are skipped)")
    for kind, g in groups.items():
        rows, unions = g["rows"], g["unions"]
        if not rows:
            continue
        if len(set(g["ens"])) < len(g["ens"]):
            print(f"\n[{kind} loss] two dumps share an ensemble ({g['ens']}): not combined — "
                  f"pass one student per ensemble")
            continue
        m, s = vh.combine(rows)
        un = None
        if unions:
            um, us = vh.combine(unions)
            un = (um[0].item(), us[0].item())
        print(f"\n[{kind} loss] combined over {len(rows)} dump(s): ΔL {sig(m[1], s[1])}   Δm(1) "
              f"{sig(m[3], s[3])}   Δm(2) {sig(m[4], s[4])}"
              + ("" if un is None else f"   union with full {sig(*un)}"))
        print("reading: " + verdict(m[1], s[1], m[3], s[3], m[4], s[4], un))
    if not groups and not HS and not DUMPS:
        print("no train_distill dumps found (pass them, or --hs=)")

    if HS:
        d = torch.load(HS, weights_only=False)
        if not d.get("extra_obars"):
            raise SystemExit(f"{HS} has no extra_obars: rerun high_stat_eval.py with "
                             f"HS_ALSO=<teacher checkpoint> so both nets share one chain")
        S = d["gelt_obar"].double()
        for path, T in d["extra_obars"].items():
            v, e = report(f"{HS}  (teacher {os.path.basename(path)})", S, T.double(),
                          float(os.environ.get("TMD_LAGS", "2").split(",")[0]),
                          max(1, S.shape[0] // 200), d["meta"]["Lt"])
            print("reading (high statistics, one net): "
                  + verdict(v[1], e[1], v[3], e[3], v[4], e[4]))


if __name__ == "__main__":
    main()
