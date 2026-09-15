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


def combine_ensembles(per_ens):
    """Inverse-variance mean of independent per-ensemble ``(value, error)``."""
    per_ens = [(v, e) for v, e in per_ens if e > 0]
    if not per_ens:
        return None, None
    w = [1.0 / e**2 for _, e in per_ens]
    val = sum(wi * v for wi, (v, _) in zip(w, per_ens)) / sum(w)
    return val, (1.0 / sum(w)) ** 0.5


def median_over_seeds(pairs):
    """Median point estimate over init seeds, with the median seed's error.

    Pre-registered (``notes/lcnn_shootout.md`` §9.2): a mean over seeds is not
    robust to one arm blowing up, and the point of a multi-seed protocol is to
    survive exactly that. Also returns the full spread so a bimodal set of seeds
    cannot hide behind its median.
    """
    if not pairs:
        return None, None, None
    vals = [v for v, _ in pairs]
    med = statistics.median(vals)
    # The error of the seed sitting at (or just below) the median.
    err = min(pairs, key=lambda p: abs(p[0] - med))[1]
    return med, err, (min(vals), max(vals))


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
                med, err, spread = median_over_seeds(pairs)
                table[(arm, t, null)] = (med, err, spread, len(pairs))
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
                med, err, spread = median_over_seeds(pairs)
                if med is None:
                    continue
                per_seed_all += [v for v, _ in pairs]
                per_ens.append((med, err))
                print(f"   {t}  ens{e}: {med:+.4f} ± {err:.4f} "
                      f"({med / err if err else float('nan'):+.1f}σ, "
                      f"{len(pairs)} seeds, spread [{spread[0]:+.4f}, "
                      f"{spread[1]:+.4f}])")
            val, err = combine_ensembles(per_ens)
            if val is None:
                print(f"   {t}  — no matched pairs")
                continue
            sig = val / err if err else float("nan")
            print(f"   {t}  combined: {val:+.4f} ± {err:.4f}  ({sig:+.1f}σ)")
            results[(label, t)] = (val, err, sig, per_seed_all)

    # ── R-E, the null ────────────────────────────────────────────────────────
    print("\n── R-E  null: random features (stack frozen at init, head trained)")
    for arm in arms:
        cells = []
        for t in TARGETS:
            med, err, _, n = table.get((arm, t, True), (None, None, None, 0))
            cells.append(f"{'—':>22s}" if med is None else f"{med:+13.4f} ± {err:.4f}")
        if any("—" not in c for c in cells):
            print(f"{arm:16s}" + "".join(cells))

    os.makedirs("results/m1_probe", exist_ok=True)
    out = f"results/m1_probe/readings{OUT_TAG}.pt"
    torch.save({"table": table, "results": results, "runs": sorted(runs)}, out)
    print(f"\nwrote {out}")
    write_tex(results, f"results/m1_probe/readings{OUT_TAG}.tex")


def write_tex(results, path):
    """The ΔR² table as a LaTeX fragment, for reports/."""
    lines = [
        r"\begin{tabular}{llrr}", r"\toprule",
        r"reading & target & $\Delta R^2$ & significance \\", r"\midrule",
    ]
    for (label, t), (val, err, sig, _) in results.items():
        name = label.split("  ", 1)[0]
        lines.append(f"{name} & {t} & ${val:+.4f} \\pm {err:.4f}$ & "
                     f"${sig:+.1f}\\sigma$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
