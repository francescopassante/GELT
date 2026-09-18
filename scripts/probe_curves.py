"""
=========================================================================
Did the runs converge, or did the horizon cut them off?
=========================================================================

``train_probe.py`` prints ``train``/``val`` every epoch and stores the whole
history in its dump, but in a batch that history lands in one per-phase log
each and nobody reads eighty of them. This reads the dumps instead and answers
the one question a sweep cannot be interpreted without:

  **is the epoch budget the binding constraint rather than the learning rate?**

If a run's best epoch is its last and the curve is still falling, the number in
its log is a statement about how long it trained, not about its rate — and a
sweep read on such runs picks the rate that happens to converge fastest at that
horizon, which is not the same thing as the best rate. The M1 probe hit the
mirror of this and recorded it (``notes/m1_probe.md``: a cosine schedule
converges in exactly the epochs early stopping deletes).

    python scripts/probe_curves.py                    # the current group
    python scripts/probe_curves.py --group=z2
    python scripts/probe_curves.py --group=z2 --filter=V2

The verdict counts runs that ended at their best epoch *and* were still
improving materially over their last quarter. A sweep with many of those should
be re-run at a longer horizon before any rate is fixed.
"""

import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe_common import IS_Z2, env_float, env_str, validate_argv  # noqa: E402

# "Still improving" needs both: the best epoch is the last one, and the curve
# fell by more than this fraction over the final quarter. The second clause
# matters — a run can tick its best on the last epoch by 1e-5 of noise and be
# perfectly converged.
STILL_IMPROVING = env_float("PROBE_CURVES_TAIL", 0.02)
# The classical local ceiling this study has to clear — reading W-E of
# notes/where_attention_can_win.md §9.6. An arm below it has not beaten the best
# bounded-reach classical method and is not an architecture result whatever the
# arm-vs-arm differences say, so the comparison belongs next to the numbers
# rather than in someone's memory. 0.671 is the β = 0.752 pre-flight value;
# β = 0.745 is 0.747. Set to 0 to switch the column off.
CEILING = env_float("PROBE_CEILING", 0.671)
# PROBE_FILTER, not PROBE_CURVES_FILTER: the flag derived from the name is
# what the docstring promises (--filter=), and a flag that does not match its
# documentation is the same silent no-op validate_argv now refuses.
FILTER = env_str("PROBE_FILTER", "")
OUT_DIR = env_str(
    "PROBE_DIR", "results/z2_vortex/probe" if IS_Z2 else "results/m1_probe"
)

BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values, width=24):
    """A coarse picture of the val curve — falling left to right is good."""
    if not values:
        return ""
    if len(values) > width:  # subsample, keeping the endpoints
        step = (len(values) - 1) / (width - 1)
        values = [values[round(i * step)] for i in range(width)]
    finite = [v for v in values if v == v and abs(v) != float("inf")]
    if not finite:
        return "?" * len(values)
    lo, hi = min(finite), max(finite)
    span = (hi - lo) or 1.0
    out = []
    for v in values:
        if v != v or abs(v) == float("inf"):
            out.append("!")
        else:
            out.append(BLOCKS[min(len(BLOCKS) - 1, int((v - lo) / span * len(BLOCKS)))])
    return "".join(out)


def tail_gain(val):
    """Relative fall of the val curve over its last quarter."""
    if len(val) < 4:
        return 0.0
    k = max(1, len(val) // 4)
    start, end = val[-k - 1], val[-1]
    if start != start or start == 0:
        return 0.0
    return (start - end) / abs(start)


def main():
    validate_argv()
    paths = sorted(glob.glob(os.path.join(OUT_DIR, "*_stats.pt")))
    if FILTER:
        paths = [p for p in paths if FILTER in os.path.basename(p)]
    if not paths:
        raise SystemExit(
            f"no dumps under {OUT_DIR}"
            + (f" matching {FILTER!r}" if FILTER else "")
            + " — run train_probe.py (or a probe_batch.sh phase) first."
        )

    print("=" * 100)
    print(f"convergence of {len(paths)} runs under {OUT_DIR}")
    print("'tail' is the relative fall of val over the last quarter; a run whose")
    print("best epoch is its last AND whose tail is still falling was cut off.")
    print("=" * 100)
    print(f"{'arm':16s} {'tgt':4s} {'lr':>7s} {'ep':>3s} {'best':>4s} "
          f"{'val':>9s} {'R²':>8s} {'tail':>7s}  curve")

    cut, rows = [], []
    for path in paths:
        d = torch.load(path, map_location="cpu", weights_only=False)
        hist = d.get("history") or []
        val = [v for _, v in hist]
        best = d.get("best_epoch", -1) + 1
        n = len(hist)
        gain = tail_gain(val)
        r2 = d.get("r2", float("nan"))
        cut_off = n > 0 and best == n and gain > STILL_IMPROVING
        flags = []
        if cut_off:
            flags.append("** still improving **")
        if d.get("diverged"):
            flags.append("DIVERGED")
        if d.get("collapsed"):
            flags.append("COLLAPSED")
        if CEILING and d.get("target") == "V1" and r2 == r2 and r2 < CEILING:
            flags.append(f"below the classical ceiling {CEILING:.3f}")
        print(f"{d.get('arm', '?'):16s} {d.get('target', '?'):4s} "
              f"{d.get('lr', float('nan')):7.1e} {n:3d} {best:4d} "
              f"{(val[-1] if val else float('nan')):9.4f} "
              f"{r2:+8.4f} {gain:+7.1%}  "
              f"{sparkline(val)}  " + "  ".join(flags))
        rows.append((os.path.basename(path), d.get("arm"), d.get("target"),
                     d.get("lr"), n, best, d.get("r2"), gain, cut_off))
        if cut_off:
            cut.append(os.path.basename(path))

    print("=" * 100)
    if cut:
        print(f"** {len(cut)} of {len(paths)} runs ended at their best epoch and were")
        print(f"   still improving by more than {STILL_IMPROVING:.0%} over the last")
        print("   quarter. The epoch budget is the binding constraint, not the rate:")
        print("   the horizon has to be settled before any rate is fixed.")
        horizon = max((r[4] for r in rows if r[8]), default=0)
        print(f"   These runs were cut off at {horizon} epochs. Do **not** guess the")
        print("   next horizon and re-sweep everything — measure it with a few long")
        print("   runs at the rates currently in contention, see where the curve")
        print(f"   flattens, then sweep there:")
        print(f"     python -u scripts/train_probe.py --group=z2 --arm=gelt "
              f"--target=V1 \\\n       --z2-beta=0.752 --epochs={horizon * 3} "
              f"--lr=<best so far> --run-tag=_horizon{horizon * 3}")
        for name in cut[:10]:
            print(f"     {name}")
        if len(cut) > 10:
            print(f"     … and {len(cut) - 10} more")
    else:
        print("No run was cut off by its horizon — the sweep is reading the rate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
