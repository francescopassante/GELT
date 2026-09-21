"""
=========================================================================
Read a glueball LR / init-scale sweep — and say whether it can be trusted
=========================================================================

``lcnn_shootout.sh`` part 1 runs four short trainings and says "compare the four
logs' best val Rayleigh loss, then pick". That instruction is not sufficient,
and ``notes/m1_probe.md`` §8 is where it cost something: **a rate chosen on a
short cosine schedule is not transferable to a long one.** The schedule anneals
to zero at ``EPOCHS``, so a 10-epoch run pulls the rate down almost immediately
while the real run sits near its peak for many epochs — and a sweep read on
runs that were still improving at the horizon picks the rate that descends
fastest early, not the rate that ends lowest.

So this reads three things per run, not one:

* **best val** and the epoch it happened — the number the shootout asks for;
* **was the horizon binding?** The best epoch being the last one, with the curve
  still falling across the final quarter, means the run was cut off. A sweep in
  which the winner was cut off has not measured what it claims to.
* **excursions** — the largest val *after* the best. The M1 probe's ``diverged``
  flag keyed on the best value and was therefore blind to a run that learns,
  blows up and settles back at the trivial predictor; the same shape is possible
  here and the same blindness would follow from reading only the best.

It parses the training logs rather than the dumps, because the dumps of runs
made before 2026-09-21 do not carry the val curve (only ``best_val_loss``).
Runs from now on store ``val_hist`` as well, and this reader prefers it when a
dump is passed.

    python scripts/glueball_sweep_read.py                    # logs/lcnn_ref_sweep_*.log
    python scripts/glueball_sweep_read.py logs/lcnn_sweep_*.log

If the parse comes back empty the log format has moved: it prints what it saw,
and the fallback is ``grep 'best val Rayleigh loss' logs/*_sweep_*.log``.
"""

import glob
import os
import re
import sys

# The outer epoch bar renders as "... N/M [...  train=0.1234, val=0.2345, ...]".
# The completed count is the epoch, which is what makes the series orderable
# even though tqdm redraws the same line many times.
_EPOCH = re.compile(
    r"(\d+)/(\d+)\s*\[[^\]]*?train=([-\d.eE+]+),\s*val=([-\d.eE+]+)"
)
_BEST = re.compile(r"best val Rayleigh loss:\s*([-\d.eE+]+)")


def parse_log(path):
    """``{epoch: val}`` plus the run's own reported best, from one log."""
    text = open(path, errors="replace").read()
    series = {}
    for m in _EPOCH.finditer(text):
        epoch, _, _, val = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        try:
            series[epoch] = float(val)
        except ValueError:
            continue
    reported = _BEST.findall(text)
    return series, (float(reported[-1]) if reported else None)


def verdict(series):
    """Best, where it fell, whether the horizon bound it, and any excursion."""
    if not series:
        return None
    epochs = sorted(series)
    vals = [series[e] for e in epochs]
    best_i = min(range(len(vals)), key=lambda i: vals[i])
    best, best_epoch, last = vals[best_i], epochs[best_i], epochs[-1]

    # Still falling over the final quarter: compare the mean of the last
    # quarter with the quarter before it. A mean rather than an endpoint
    # because the val curve of these runs is visibly noisy epoch to epoch.
    q = max(1, len(vals) // 4)
    falling = len(vals) >= 4 and (sum(vals[-q:]) / q) < (sum(vals[-2 * q:-q]) / q)
    cut_off = best_epoch == last and falling

    after = vals[best_i + 1:]
    excursion = (max(after) / best) if after and best > 0 else 1.0
    return dict(best=best, best_epoch=best_epoch, last=last, falling=falling,
                cut_off=cut_off, excursion=excursion, final=vals[-1])


def label(path):
    name = os.path.basename(path).replace(".log", "")
    m = re.search(r"sweep_lr([\w.+-]+)_is([\w.+-]+)$", name)
    return f"lr={m.group(1)}  init_scale={m.group(2)}" if m else name


def main(argv):
    paths = argv[1:] or sorted(glob.glob("logs/*_sweep_lr*.log"))
    if not paths:
        print("no sweep logs found (logs/*_sweep_lr*.log). Pass paths explicitly.")
        return 1

    print("=" * 86)
    print("Glueball sweep — best val Rayleigh loss, and whether the horizon bound it")
    print("=" * 86)
    print(f"{'run':34s} {'best':>9s} {'@ep':>5s} {'final':>9s} "
          f"{'excursion':>10s}   status")

    rows, parsed = [], 0
    for p in paths:
        series, reported = parse_log(p)
        v = verdict(series)
        if v is None:
            print(f"{label(p):34s} {'—':>9s} {'—':>5s} {'—':>9s} {'—':>10s}   "
                  f"no epoch lines parsed"
                  + (f" (log reports best {reported:.4f})" if reported else ""))
            continue
        parsed += 1
        status = []
        if v["cut_off"]:
            status.append("** CUT OFF — best is the last epoch and still falling")
        if v["excursion"] > 2.0:
            status.append(f"** EXCURSION ×{v['excursion']:.1f} after the best")
        if reported is not None and abs(reported - v["best"]) > 1e-3:
            status.append(f"(log reports {reported:.4f})")
        print(f"{label(p):34s} {v['best']:9.4f} {v['best_epoch']:5d} "
              f"{v['final']:9.4f} {v['excursion']:10.2f}   "
              + ("; ".join(status) if status else "ok"))
        rows.append((v["best"], label(p), v))

    if not rows:
        print("\nNothing parsed. The log format has moved — paste the tail of one "
              "log, or fall back to:\n"
              "  grep 'best val Rayleigh loss' logs/*_sweep_lr*.log")
        return 1

    rows.sort()
    best_val, best_label, best_v = rows[0]
    print("\n" + "-" * 86)
    print(f"lowest best-val: {best_label}  ({best_val:.4f})")

    if best_v["cut_off"]:
        print(
            "\n** The winner was cut off at the horizon.** Its best epoch is its\n"
            "   last and the curve was still falling, so this sweep has ranked\n"
            "   rates by how fast they descend in 10 epochs, not by where they\n"
            "   end. notes/m1_probe.md §8 is the precedent. Re-run the top two\n"
            "   at the full budget before committing part 2 to either."
        )
    if any(r[2]["cut_off"] for r in rows):
        n = sum(1 for r in rows if r[2]["cut_off"])
        print(f"\n   ({n} of {len(rows)} runs were cut off at the horizon.)")
    edge = best_label.split()[0]
    print(
        f"\n   Check the grid edge: if {edge} is the smallest or largest rate\n"
        "   scanned, the optimum is not bracketed and the sweep has not found\n"
        "   it — widen and re-run rather than taking the endpoint."
    )
    print(
        "\n   When it is settled:\n"
        f"     LCNN_ARCH=<arch> LCNN_PARTS=2,3 LCNN_LR=<lr> LCNN_INIT=<scale> \\\n"
        "       bash scripts/lcnn_shootout.sh"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
