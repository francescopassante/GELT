"""
=========================================================================
The M1 probe's readings — offline, from a directory of run dumps.
=========================================================================

Turns ``results/m1_probe/probe_*_stats.pt`` into the table
``notes/m1_probe.md`` §4 pre-registered. Seconds, no GPU, no checkpoints.

    python scripts/probe_readings.py
    python scripts/probe_readings.py --dir=results/m1_probe --out-tag=_v2

Every ΔR² is **correlated**: the two arms are jackknifed together over the same
configurations in the same order, so the shared ensemble fluctuation cancels
rather than being counted twice. Seeds are combined by **median**, fixed in
advance — ``notes/lcnn_shootout.md`` §9.2 showed a mean over seeds is
meaningless the moment one arm blows up. Ensembles are combined by
inverse-variance weighting, which is what two independent ensembles allow.
"""

import glob
import math
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_common import (  # noqa: E402
    TARGETS,
    env_flag,
    env_str,
    jackknife,
    r2_from_stats,
)

DIR = env_str("PROBE_DIR", "results/m1_probe")
OUT_TAG = env_str("PROBE_OUT_TAG", "")

# The readings, exactly as pre-registered. Each is (label, arm_a, arm_b,
# targets, what a pass means) — every one is ΔR²(a − b).
READINGS = [
    ("R-A  calibration/arch", "gelt", "lcnn", ("T0",),
     "must be within 1σ of zero, or every other reading is void"),
    ("R-A  calibration/M1", "gelt", "frozen", ("T0",),
     "T0 is a convolution: a GELT win here is capacity, not the mechanism"),
    ("R-B  primary (M1)", "gelt", "frozen", ("T1", "T2"),
     ">2σ on T2 confirms M1; within 1σ on both falsifies it here"),
    ("R-B' matched-param", "gelt", "frozen_matched", ("T1", "T2"),
     "read only if R-B favours GELT at >2σ — capacity vs mechanism"),
    ("R-C  architecture", "gelt", "lcnn", ("T1", "T2"),
     "the thesis number; carries the transport confound, read through R-B"),
    ("R-D  M2 control", "lcnn_norm", "lcnn", ("T1", "T2"),
     "if bounding the offset weights closes a gap, the effect was M2"),
    # Post-hoc (notes/m1_probe.md §4.3): added after the gate found the matched
    # L-CNN beating GELT on T2. Softmax does three things at once — it reads the
    # input, it normalises, and it forces non-negativity — and the three signed
    # arms peel them off one at a time. Identical parameter count to `gelt` in
    # all three cases, so no matched-parameter argument is needed for any of
    # them. R-G was originally posed on `signed_bounded`; the 2026-09-16 sweep
    # showed that arm varies sign *and* normalisation at once, so it is re-posed
    # on `signed_l1` (Σ|α| = 1, the softmax's unit gain kept), and T0 rides
    # along because an arm that is simply harder to train shows it there too.
    ("R-G  sign constraint", "signed_l1", "gelt", ("T2", "T0"),
     "α signed at unit L1 gain: does dropping non-negativity alone recover it"),
    ("R-H  normalisation", "signed_l1", "signed_bounded", ("T2", "T0"),
     "Σ|α| = 1 vs a free gain, sign held: what the softmax's normalising buys"),
    ("R-H  boundedness", "signed_bounded", "signed", ("T2",),
     "bounded vs unbounded α, both unnormalised: the M2 cost, inside GELT"),
    # The transport arms (notes/m1_probe.md §4.4). R-C confounds M1 with
    # transport geometry; R-B removed M1 from the comparison, and these remove
    # the *path averaging* half of the geometry — inside GELT, at identical
    # parameters, with the offset set untouched. They are read against `gelt`
    # and never against `lcnn`: a "single" arm is not the L-CNN's transport.
    ("R-I  path averaging", "gelt", "gelt_single", ("T2", "T0"),
     "averaged shortest paths vs one canonical path: what the DP average buys"),
    ("R-I' averaging, on-group", "gelt", "gelt_projected", ("T2", "T0"),
     "same average projected back onto the group: content, or leaving it"),
]


def load_runs(directory, include_tagged=False):
    """``{(arm, target, ens, init, null): dump}`` from a results directory.

    Runs carrying a ``run_tag`` are skipped by default. The LR sweep of
    ``probe_batch.sh`` part 1 writes short runs under ``_sweep_*`` tags at the
    same (arm, target, ensemble, seed) coordinates as the real thing, and
    silently reading one of those as the result would be the worst kind of bug —
    a six-epoch run reported as a twenty-epoch one.
    """
    runs = {}
    for path in sorted(glob.glob(os.path.join(directory, "probe_*_stats.pt"))):
        d = torch.load(path, map_location="cpu", weights_only=False)
        if d.get("run_tag") and not include_tagged:
            continue
        key = (d["arm"], d["target"], d["ensemble_seed"], d["init_seed"],
               bool(d["null"]))
        if key in runs:
            raise SystemExit(
                f"two dumps claim arm={key[0]} target={key[1]} ens={key[2]} "
                f"init={key[3]} null={key[4]}:\n  {runs[key]['_path']}\n  {path}"
            )
        d["_path"] = path
        runs[key] = d
    return runs


def delta_r2(a, b):
    """Correlated ΔR²(a − b) and its blocked-jackknife error.

    The two arms must have been scored on the same test configurations in the
    same order — they are, because ``probe_common.splits`` is deterministic, and
    this checks it rather than trusting it.
    """
    if a["test_configs"] != b["test_configs"]:
        raise SystemExit(
            f"{a['_path']} and {b['_path']} were scored on different test "
            f"configurations — the correlated jackknife would be meaningless."
        )
    val, err, _ = jackknife(
        [a["stats"], b["stats"]],
        lambda sa, sb: r2_from_stats(sa) - r2_from_stats(sb),
        block=a["jack_block"],
    )
    return val, err


def residual_ratio(a, b):
    """Correlated ``(1 − R²_a)/(1 − R²_b)`` and its blocked-jackknife error.

    The *scale-appropriate* form of a between-arm comparison, and the one R-A
    has to be read in. On T0 every arm sits near 0.98 with 0.02 of headroom
    left, so a ΔR² of 0.005 is a quarter of everything still unexplained; on T2
    the same 0.005 would be noise. The residual ratio is invariant to how much
    headroom a target leaves, which is exactly what is needed to compare a
    calibration target against a mechanism target.
    """
    if a["test_configs"] != b["test_configs"]:
        raise SystemExit(
            f"{a['_path']} and {b['_path']} were scored on different test "
            f"configurations — the correlated jackknife would be meaningless."
        )
    val, err, _ = jackknife(
        [a["stats"], b["stats"]],
        lambda sa, sb: (1.0 - r2_from_stats(sa)) / (1.0 - r2_from_stats(sb)),
        block=a["jack_block"],
    )
    return val, err


def calibrated_ratio(a_t, b_t, a_0, b_0):
    """How far the residual ratio moves from the calibration target to this one.

    ``[(1−R²_a)/(1−R²_b)]_T  ÷  [(1−R²_a)/(1−R²_b)]_T0`` — all four runs share
    the same test configurations (``splits`` is deterministic), so the whole
    thing is one correlated jackknife and the generic, mechanism-free part of an
    arm's advantage divides out. **This is what R-A was trying to say.** A value
    of 1 means the pair's relative performance on the mechanism target is
    entirely explained by their relative performance on a pure convolution;
    departures from 1 are what the target was built to expose.
    """
    for x, y in ((a_t, b_t), (a_0, b_0), (a_t, a_0)):
        if x["test_configs"] != y["test_configs"]:
            raise SystemExit("mismatched test splits across the calibration")

    def fn(sa_t, sb_t, sa_0, sb_0):
        rt = (1.0 - r2_from_stats(sa_t)) / (1.0 - r2_from_stats(sb_t))
        r0 = (1.0 - r2_from_stats(sa_0)) / (1.0 - r2_from_stats(sb_0))
        return rt / r0

    val, err, _ = jackknife(
        [a_t["stats"], b_t["stats"], a_0["stats"], b_0["stats"]],
        fn, block=a_t["jack_block"],
    )
    return val, err


# Two ensembles are only combinable if they agree. Beyond this many sigma they
# are measuring different things (or the error is understated) and averaging
# them manufactures a precision neither has.
ENSEMBLE_CONSISTENCY_SIGMA = 3.0


def pooled_pairs(diffs):
    """``(median, min, max, k_positive, n, sign-test p)`` over paired diffs.

    Each ``(ensemble, seed)`` cell gives one difference between two arms trained
    on identical data from identical splits, so the cells are the independent
    units and there are six of them. With n = 6 a median plus a sign test is
    what the data supports; a mean with a jackknife error is not, and the two
    disagree badly when one arm is bimodal — which the per-run table shows the
    L-CNN is on T2.

    The sign test is two-sided and exact: it asks only whether the *direction*
    is consistent, which is the one claim six samples can carry. 6/6 gives
    p = 0.031, 5/6 gives 0.219, 4/6 gives 0.688.
    """
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0, 0, float("nan")
    k = sum(1 for d in diffs if d > 0)
    tail = max(k, n - k)
    p = 2.0 * sum(math.comb(n, i) for i in range(tail, n + 1)) / 2**n
    return (statistics.median(diffs), min(diffs), max(diffs), k, n, min(p, 1.0))


def combine_ensembles(per_ens):
    """Inverse-variance mean of independent per-ensemble ``(value, error)``.

    Returns ``(value, error, note)``. **Refuses to combine mutually
    inconsistent ensembles**: inverse-variance weighting assumes the inputs are
    repeated measurements of one quantity, and two values 60σ apart are not.
    Quoting their weighted mean with the weighted error was producing
    thousand-sigma significances out of a disagreement.
    """
    per_ens = [(v, e) for v, e in per_ens if e > 0]
    if not per_ens:
        return None, None, ""
    if len(per_ens) == 2:
        (v0, e0), (v1, e1) = per_ens
        gap = abs(v0 - v1) / max((e0**2 + e1**2) ** 0.5, 1e-12)
        if gap > ENSEMBLE_CONSISTENCY_SIGMA:
            # Spread-based error, and say so: this is a range, not a precision.
            return (0.5 * (v0 + v1), 0.5 * abs(v0 - v1),
                    f"  ** ensembles disagree at {gap:.0f}σ — "
                    f"midpoint ± half-range, NOT a combined measurement **")
    w = [1.0 / e**2 for _, e in per_ens]
    val = sum(wi * v for wi, (v, _) in zip(w, per_ens)) / sum(w)
    return val, (1.0 / sum(w)) ** 0.5, ""


def median_over_seeds(pairs):
    """Median over init seeds, with an error that **includes the seed spread**.

    Pre-registered (``notes/lcnn_shootout.md`` §9.2): a mean over seeds is not
    robust to one arm blowing up, and the point of a multi-seed protocol is to
    survive exactly that.

    The error is ``max(median seed's jackknife, half the seed range)``. The
    jackknife alone counts only test-configuration noise — with 414k test sites
    it is ~2e-4 — while the dominant variance here is *across initialisations*,
    where one L-CNN seed reaches 0.94 on T2 and another 0.14. Quoting the
    jackknife alone for a median over seeds was reporting 1500σ on quantities
    whose seeds disagree by 0.9, and that was an artifact of the estimator, not
    a measurement. Returns ``(median, error, (min, max), seed_dominated)``.
    """
    if not pairs:
        return None, None, None, False
    vals = [v for v, _ in pairs]
    med = statistics.median(vals)
    jack = min(pairs, key=lambda p: abs(p[0] - med))[1]
    spread = 0.5 * (max(vals) - min(vals))
    return med, max(jack, spread), (min(vals), max(vals)), spread > jack


def main():
    runs = load_runs(DIR, include_tagged=env_flag("PROBE_INCLUDE_TAGGED", False))
    if not runs:
        raise SystemExit(f"no probe dumps in {DIR}/ — run scripts/probe_batch.sh")
    arms = sorted({k[0] for k in runs})
    ens = sorted({k[2] for k in runs})
    seeds = sorted({k[3] for k in runs})
    print("=" * 78)
    print(f"M1 probe readings — {len(runs)} runs from {DIR}/")
    print(f"arms {arms}\nensembles {ens}  init seeds {seeds}")
    print("=" * 78)

    # ── Every run, so bimodality cannot hide behind a median ─────────────────
    print("\nEvery trained run (R² on the held-out split)\n")
    print(f"   {'arm':16s} {'tgt':4s} {'ens':>4s} {'seed':>5s} {'R²':>9s}  flags")
    for k in sorted(runs):
        if k[4]:
            continue
        d = runs[k]
        flags = " ".join(f for f, on in (
            ("DIVERGED", d.get("diverged")), ("COLLAPSED", d.get("collapsed")),
            ("excursion", d.get("max_train", 0) > d.get("divergence_val", 1e9)),
        ) if on)
        print(f"   {k[0]:16s} {k[1]:4s} {k[2]:4d} {k[3]:5d} {d['r2']:+9.4f}  {flags}")

    # ── Dispersion across initialisations — the robustness statistic ─────────
    print("\nSpread across the 6 runs (2 ensembles × 3 seeds) of each cell")
    print("   This is what a median protects against, and what a jackknife over")
    print("   test configurations cannot see.\n")
    print(f"   {'arm':16s} {'tgt':4s} {'min':>9s} {'median':>9s} {'max':>9s}"
          f" {'range':>9s}")
    for arm in arms:
        for t in TARGETS:
            vals = sorted(runs[k]["r2"] for k in runs
                          if k[0] == arm and k[1] == t and not k[4])
            if len(vals) < 2:
                continue
            print(f"   {arm:16s} {t:4s} {vals[0]:+9.4f} "
                  f"{statistics.median(vals):+9.4f} {vals[-1]:+9.4f} "
                  f"{vals[-1] - vals[0]:9.4f}")

    # ── Per-arm R², median over seeds ────────────────────────────────────────
    print("\nR² on the held-out split (median over init seeds, "
          "± the median seed's jackknife)\n")
    header = f"{'arm':16s}" + "".join(f"{t:>22s}" for t in TARGETS)
    print(header)
    table = {}
    for null in (False, True):
        for arm in arms:
            cells, any_cell = [], False
            for t in TARGETS:
                pairs = [
                    (runs[k]["r2"], runs[k]["r2_err"])
                    for k in runs
                    if k[0] == arm and k[1] == t and k[4] == null
                ]
                med, err, spread, dom = median_over_seeds(pairs)
                table[(arm, t, null)] = (med, err, spread, len(pairs), dom)
                if med is None:
                    cells.append(f"{'—':>22s}")
                else:
                    any_cell = True
                    cells.append(f"{med:+13.4f} ± {err:.4f}")
            if any_cell:
                label = arm + (" [null]" if null else "")
                print(f"{label:16s}" + "".join(cells))

    # ── The pre-registered ΔR² readings ──────────────────────────────────────
    results = {}
    for label, a, b, targets, meaning in READINGS:
        print(f"\n── {label}:  ΔR²({a} − {b})")
        print(f"   {meaning}")
        for t in targets:
            per_ens = []
            per_seed_all = []
            for e in ens:
                pairs = []
                for s in seeds:
                    ka, kb = (a, t, e, s, False), (b, t, e, s, False)
                    if ka in runs and kb in runs:
                        pairs.append(delta_r2(runs[ka], runs[kb]))
                med, err, spread, dom = median_over_seeds(pairs)
                if med is None:
                    continue
                per_seed_all += [v for v, _ in pairs]
                per_ens.append((med, err))
                print(f"   {t}  ens{e}: {med:+.4f} ± {err:.4f} "
                      f"({med / err if err else float('nan'):+.1f}σ, "
                      f"{len(pairs)} seeds, spread [{spread[0]:+.4f}, "
                      f"{spread[1]:+.4f}])"
                      + ("  [seed spread dominates]" if dom else ""))
            val, err, note = combine_ensembles(per_ens)
            if val is None:
                print(f"   {t}  — no matched pairs")
                continue
            sig = val / err if err else float("nan")
            print(f"   {t}  combined: {val:+.4f} ± {err:.4f}  ({sig:+.1f}σ){note}")
            # The statistic to actually quote: all six paired differences
            # pooled. Each (ensemble, seed) cell gives one difference between
            # two arms that saw identical data, so with n = 6 the median and a
            # sign test say more than a weighted mean of two disagreeing
            # ensemble medians — and they are robust to the bimodality the
            # per-run table exposes.
            med6, lo, hi, k, n, pv = pooled_pairs(per_seed_all)
            print(f"   {t}  POOLED over {n} (ensemble, seed) cells: "
                  f"median {med6:+.4f}, range [{lo:+.4f}, {hi:+.4f}], "
                  f"{k}/{n} favour {a}, sign test p = {pv:.3f}")
            results[(label, t)] = (val, err, sig, per_seed_all, med6, k, n, pv)

    # ── The residual-ratio view, and R-A's repaired form ─────────────────────
    print("\n── residual ratios  (1−R²_a)/(1−R²_b), and the T0-calibrated move")
    print("   R-A as pre-registered was un-passable: it asked for ΔR² within 1σ")
    print("   of zero, and with 414k test sites σ → 0, so any difference fails.")
    print("   The calibrated column is what it meant — 1.00 = the pair's T2 gap")
    print("   is entirely their generic gap on a pure convolution.")
    for label, a, b, targets, _ in READINGS:
        if label.startswith("R-A"):
            continue
        for t in targets:
            for e in ens:
                pairs, cal = [], []
                for sd in seeds:
                    ka, kb = (a, t, e, sd, False), (b, t, e, sd, False)
                    k0a, k0b = (a, "T0", e, sd, False), (b, "T0", e, sd, False)
                    if ka in runs and kb in runs:
                        pairs.append(residual_ratio(runs[ka], runs[kb]))
                        if k0a in runs and k0b in runs:
                            cal.append(calibrated_ratio(
                                runs[ka], runs[kb], runs[k0a], runs[k0b]))
                med, err, _, _ = median_over_seeds(pairs)
                if med is None:
                    continue
                cmed, cerr, _, _ = median_over_seeds(cal)
                tail = ("" if cmed is None
                        else f"   calibrated ×{cmed:.3f} ± {cerr:.3f}")
                print(f"   {a:>14s}/{b:<14s} {t} ens{e}: "
                      f"×{med:.3f} ± {err:.3f}{tail}")

    # ── R-E, the null ────────────────────────────────────────────────────────
    print("\n── R-E  null: random features (stack frozen at init, head trained)")
    for arm in arms:
        cells = []
        for t in TARGETS:
            med, err, _, n, _ = table.get((arm, t, True), (None, None, None, 0, False))
            cells.append(f"{'—':>22s}" if med is None else f"{med:+13.4f} ± {err:.4f}")
        if any("—" not in c for c in cells):
            print(f"{arm:16s}" + "".join(cells))

    divergence_report(DIR)

    os.makedirs("results/m1_probe", exist_ok=True)
    out = f"results/m1_probe/readings{OUT_TAG}.pt"
    torch.save({"table": table, "results": results, "runs": sorted(runs)}, out)
    print(f"\nwrote {out}")
    write_tex(results, f"results/m1_probe/readings{OUT_TAG}.tex")


def divergence_report(directory):
    """R-F — the failure rate, per architecture family, over *every* run.

    **Post-hoc, and labelled as such** (``notes/m1_probe.md`` §4.1): it was added
    after the part-1 sweep showed both L-CNN arms blowing up at a learning rate
    at which no GELT-family arm did. It is confirmatory rather than exploratory
    only because ``notes/lcnn_shootout.md`` §9.2's 3-of-9 vs 0-of-14 is the
    prior.

    Sweep runs are *included* here, unlike everywhere else in this script. A
    divergence is a divergence whatever the tag, and excluding the short runs
    would throw away most of the evidence — at the cost that the rate is over a
    non-uniform mix of learning rates, so it is a count, never a probability.
    """
    rows, excursions = {}, {}
    for path in sorted(glob.glob(os.path.join(directory, "probe_*_stats.pt"))):
        d = torch.load(path, map_location="cpu", weights_only=False)
        if "diverged" not in d:  # written before the flag existed
            continue
        fam = "GELT family" if d["arch"] == "gelt" else "L-CNN family"
        div, col, tot = rows.get(fam, (0, 0, 0))
        rows[fam] = (div + bool(d["diverged"]),
                     col + bool(d.get("collapsed")), tot + 1)
        # An excursion is a training loss that left the scale entirely. What
        # separates the families is not whether it happens but whether the run
        # comes back, so both are counted.
        if d.get("max_train", 0) > d["divergence_val"]:
            n_exc, n_back = excursions.get(fam, (0, 0))
            came_back = not (d["diverged"] or d.get("collapsed"))
            excursions[fam] = (n_exc + 1, n_back + bool(came_back))
    if not rows:
        return
    print("\n── R-F  training failure (post-hoc), per architecture family")
    print("   counts over every run on disk, sweep runs included — not a rate")
    print(f"   {'family':14s} {'diverged':>9s} {'collapsed':>10s} "
          f"{'excursions':>11s} {'recovered':>10s} {'runs':>6s}")
    for fam, (div, col, tot) in sorted(rows.items()):
        n_exc, n_back = excursions.get(fam, (0, 0))
        print(f"   {fam:14s} {div:9d} {col:10d} {n_exc:11d} {n_back:10d} "
              f"{tot:6d}")
    print("   diverged  = best val never beat 10× the trivial predictor")
    print("   collapsed = ended no better than the trivial predictor, having "
          "once done better")
    print("   excursion = training loss left the scale at some epoch; "
          "'recovered' = and the run still finished usable")


def write_tex(results, path):
    """The ΔR² table as a LaTeX fragment, for reports/."""
    lines = [
        r"\begin{tabular}{llrr}", r"\toprule",
        r"reading & target & median $\Delta R^2$ & sign test \\", r"\midrule",
    ]
    for (label, t), row in results.items():
        name = label.split("  ", 1)[0]
        med6, k, n, pv = row[4], row[5], row[6], row[7]
        lines.append(f"{name} & {t} & ${med6:+.4f}$ & ${k}/{n}$, $p={pv:.3f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
