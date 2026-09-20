"""
=========================================================================
Which test configuration made an R² negative?
=========================================================================

Two of twelve seed runs came back with R² of −67 and −259, error bars of the
same size, and *val curves that look healthy*. An error bar the size of the
value is the blocked jackknife saying one configuration carries the whole
statistic, so the question is not "did the run diverge" — it plainly did not on
the split it was selected on — but **which held-out configuration the model
explodes on, and whether it is the same one for both arms**.

The dumps already answer it: ``train_probe.py`` stores the six per-configuration
sums an R² is made of, so every per-config R² is exact offline, in seconds, with
no GPU and no re-run.

    python scripts/probe_outliers.py --group=z2 --filter=_seed

A configuration that is an outlier for *both* arms is a property of the
ensemble — that chain segment, that target — and not of either architecture.
One that is an outlier for one arm only is an instability of that arm.
"""

import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe_common import IS_Z2, env_str, r2_from_stats, validate_argv  # noqa: E402

FILTER = env_str("PROBE_FILTER", "")
OUT_DIR = env_str(
    "PROBE_DIR", "results/z2_vortex/probe" if IS_Z2 else "results/m1_probe"
)
# A run is worth decomposing when its error bar is a sizable fraction of its
# value; a healthy run's per-config spread is not interesting.
FLAG_RATIO = float(env_str("PROBE_OUTLIER_RATIO", "0.1"))


def main():
    validate_argv()
    paths = sorted(glob.glob(os.path.join(OUT_DIR, "*_stats.pt")))
    if FILTER:
        paths = [p for p in paths if FILTER in os.path.basename(p)]
    if not paths:
        raise SystemExit(f"no dumps under {OUT_DIR} matching {FILTER!r}")

    for path in paths:
        d = torch.load(path, map_location="cpu", weights_only=False)
        stats = d.get("stats")
        if stats is None:
            continue
        r2, err = d.get("r2", float("nan")), d.get("r2_err", float("nan"))
        if not (abs(err) > FLAG_RATIO * max(abs(r2), 1e-9)):
            continue
        cfgs = d.get("test_configs") or list(range(len(stats)))
        print(f"\n{os.path.basename(path)}")
        print(f"  arm {d.get('arm')}  seed {d.get('init_seed')}  "
              f"R² {r2:+.4f} ± {err:.4f}  over {len(stats)} test configs")
        per = [(c, float(r2_from_stats(stats[i : i + 1])))
               for i, c in enumerate(cfgs)]
        for c, v in per:
            bar = "" if v > -1 else "   ← this one"
            print(f"    config {c:4d}   R² {v:+12.4f}{bar}")
        # Leave-one-out: if dropping the worst config restores the run, the
        # model is fine everywhere else and the number is one configuration.
        worst = min(range(len(per)), key=lambda i: per[i][1])
        keep = torch.cat([stats[:worst], stats[worst + 1:]])
        print(f"  without config {per[worst][0]}: R² = "
              f"{float(r2_from_stats(keep)):+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
