"""Fig. 3 of PRL 128, 032003, assembled offline from the run dumps.

No GPU, no data, no models --- it reads ``dumps/wilson_regression_*.pt`` and
writes ``results/wilson_regression/fig3.{png,pt,tex}`` plus the comparison
table. The L-CNN half of the figure only; see
``notes/wilson_regression_1p1d.md`` §1 on why the baseline CNN is not here.

**Which run is plotted.** The Letter shows "our best L-CNN models", so when more
than one run exists for a loop the panel takes the one with the lowest
**validation** loss --- never the lowest test MSE, which is what the plot is
supposed to be evidence about. The table prints that run and, next to it, the
median and spread over whatever seeds exist; at one seed per loop the three
columns coincide, which is the honest way of saying no best-of-N was taken.

**Dumps carrying a ``run_tag`` are skipped**, the same rule as
``probe_readings.py``: the bracketing runs of ``WR_PARTS=gelt-sweep`` are
ten-epoch and would otherwise be read as the result.

**``epochs`` and ``cut``** answer ``probe_curves.py``'s question, which has to
be asked before any two arms are compared: *was the epoch budget the binding
constraint rather than the architecture?* A run whose best epoch is its last and
whose validation curve is still falling over its final quarter was cut off, and
comparing it with one that converged is comparing budgets, not models.

A GELT arm, if present for a loop, is overlaid on the same panel in a second
colour with its own MSE line. It is the same problem and the same splits, so
the panels are directly comparable; it is *not* a reproduction of the paper's
number and the table labels it as its own row.

**Which MSE.** The lattice-averaged one, Fig. 3's own convention: one point per
test configuration, both sides averaged over the lattice first. See
``wilson_regression_common`` on why.

A CNN arm (``WR_ARCH=cnn``) is overlaid the same way, compared against the
Letter's own CNN number for scale. The table prints **every size** present for
an arm, one row each, with ``*`` on the one the panel plots: the CNN is run at
two sizes, and the smaller one losing is not the claim. ``R2_within`` is R^2
with the coupling's share of the label variance removed
(``wilson_regression_common.r2_within_beta``) --- 0 for a model that reads off
beta and nothing else, which on this dataset is the reading that matters for a
non-equivariant arm.

``--dumps=<glob>`` overrides the default dump pattern, ``--out-tag=<s>`` names
the artifacts, ``--size=<s>`` restricts to one architecture size (by default the
lowest-validation-loss size present is used, which is the Letter's "best model"
reading), ``--fig-targets=W22,W33,W44`` restricts the panels and
``--fig-archs=gelt,cnn`` the arms (panels and table both): the figure is
rebuilt from the dumps, so dropping an arm never needs a rerun.
"""

import glob
import math
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wilson_regression_common import (
    DUMP_DIR,
    LOOPS,
    PAPER_MSE_CNN,
    PAPER_MSE_LCNN,
    RESULT_DIR,
    cfg,
    r2_pair,
    r2_within_beta,
    validate_argv,
)

# The Letter's own green for the L-CNN; a second colour for this repo's arm.
ARCH_STYLE = {
    "lcnn": ("L-CNN", "#2ca02c", "o", 2),
    "gelt": ("GELT", "#d62728", "^", 3),
    "cnn": ("CNN", "#7f7f7f", "s", 1),
}


def load_dumps(pattern):
    runs = []
    for path in sorted(glob.glob(pattern)):
        if path.endswith(".pth"):
            continue
        d = torch.load(path, map_location="cpu", weights_only=False)
        if "mse_avg" not in d:
            continue
        if d.get("run_tag"):
            continue          # a bracketing run, not a result
        d.setdefault("arch", "lcnn")   # dumps written before the GELT arm
        d["_path"] = path
        runs.append(d)
    return runs


def was_cut_off(d):
    """``probe_curves.py``'s rule: best epoch is the last, and still descending.

    Returns ``(epochs_run, cut_off)``. "Still descending" is measured over the
    final quarter of the curve rather than the last step, which is noise.
    """
    val = (d.get("history") or {}).get("val") or []
    n = len(val)
    if n < 4:
        return n, False
    best_is_last = min(range(n), key=lambda i: val[i]) == n - 1
    q = max(2, n // 4)
    return n, bool(best_is_last and val[-1] < val[-q])


def lattice_average(d):
    pred, true = d["test_pred"], d["test_true"]
    dims = tuple(range(1, pred.ndim))
    return pred.mean(dim=dims), true.mean(dim=dims)


def select(runs, target, arch="lcnn", size=None):
    """Best-validation-loss run for one (loop, arch), and every run at that size."""
    cell = [r for r in runs if r["target"] == target and r["arch"] == arch
            and (size is None or r["size"] == size)]
    if not cell:
        return None, []
    if size is None:
        best_size = min(cell, key=lambda r: r["best_val"])["size"]
        cell = [r for r in cell if r["size"] == best_size]
    return min(cell, key=lambda r: r["best_val"]), cell


def main():
    pattern = cfg("WR_DUMPS", os.path.join(DUMP_DIR, "wilson_regression_*.pt"))
    tag = cfg("WR_OUT_TAG", "")
    size = cfg("WR_SIZE", None)
    only = [t.strip().upper()
            for t in str(cfg("WR_FIG_TARGETS", "")).split(",") if t.strip()]
    archs = [a.strip().lower()
             for a in str(cfg("WR_FIG_ARCHS", "")).split(",") if a.strip()]
    archs = archs or ["gelt", "cnn"]   # the figure: GELT and the two CNNs
    unknown = [a for a in archs if a not in ARCH_STYLE]
    if unknown:
        raise SystemExit(f"unknown arch(s) {unknown}; have {list(ARCH_STYLE)}")
    # After every cfg() read, not before: validate_argv only knows the flags
    # cfg() has already been asked for, so called first it refused them all.
    validate_argv()
    runs = load_dumps(pattern)
    if not runs:
        raise SystemExit(f"no run dumps matched {pattern}")
    print(f"{len(runs)} run dump(s) from {pattern}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(RESULT_DIR, exist_ok=True)
    runs = [r for r in runs if r["arch"] in archs]
    targets = [t for t in LOOPS if any(r["target"] == t for r in runs)
               and (not only or t in only)]
    ncol = 1                      # one column, one row per loop
    nrow = len(targets)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2, 3.8 * nrow),
                             squeeze=False)

    table, payload = [], {}
    for i, target in enumerate(targets):
        ax = axes[i // ncol][i % ncol]
        m, n = LOOPS[target]
        lo = hi = None
        note = []
        # GELT plots its best size; the CNN plots both of its sizes.
        series = []
        for arch in ("gelt", "cnn", "lcnn"):
            if arch not in archs:
                continue
            if arch == "cnn":
                szs = sorted({r["size"] for r in runs if r["target"] == target
                              and r["arch"] == "cnn"
                              and (size is None or r["size"] == size)})
                series += [("cnn", sz) for sz in szs]
            else:
                series.append((arch, size))
        for arch, sz_plot in series:
            best, cell = select(runs, target, arch, sz_plot)
            if best is None:
                continue
            label, colour, marker, z = ARCH_STYLE[arch]
            if arch == "cnn":
                label = f"CNN {best['size']}"
                colour = {"matched": "#7f7f7f", "large": "#1f77b4"}.get(
                    best["size"], colour)
                marker = {"matched": "s", "large": "D"}.get(best["size"], marker)
            pred, true = lattice_average(best)
            ax.scatter(pred, true, s=10, c=colour, marker=marker, alpha=0.6,
                       linewidths=0, zorder=z, label=label)
            note.append(f"{label:12} {best['mse_avg']:.1e}")
            ref = (PAPER_MSE_CNN if arch == "cnn"
                   else PAPER_MSE_LCNN).get(target)
            # One row per size present (the panel plots the starred one).
            sizes = sorted({r["size"] for r in runs if r["target"] == target
                            and r["arch"] == arch
                            and (size is None or r["size"] == size)})
            if arch == "cnn":
                sizes = [best["size"]]   # one series per CNN size
            for sz in sizes:
                row_best, row_cell = select(runs, target, arch, sz)
                vals = sorted(r["mse_avg"] for r in row_cell)
                n_ep, cut = was_cut_off(row_best)
                # Recomputed rather than read, so dumps written before r2
                # existed still get the column.
                r2s, r2a = r2_pair(row_best["test_pred"], row_best["test_true"])
                r2w = r2_within_beta(row_best["test_pred"],
                                     row_best["test_true"],
                                     row_best["test_beta"])
                table.append(dict(target=target, arch=arch, size=sz,
                                  plotted=sz == best["size"],
                                  epochs_run=n_ep, cut_off=cut,
                                  r2_site=r2s, r2_avg=r2a, r2_within=r2w,
                                  conv_impl=row_best.get("conv_impl"),
                                  n_param=row_best["n_param"],
                                  paper_n_param=row_best.get("paper_n_param"),
                                  best_mse_avg=row_best["mse_avg"],
                                  best_mse_site=row_best["mse_site"],
                                  paper_mse=ref, n_seeds=len(vals),
                                  median_mse_avg=statistics.median(vals),
                                  min_mse_avg=vals[0], max_mse_avg=vals[-1],
                                  seed=row_best["seed"],
                                  path=row_best["_path"]))
            payload[f"{arch}_{target}" + (f"_{best['size']}" if arch == "cnn"
                                          else "")] = dict(pred=pred, true=true,
                                               mse_avg=best["mse_avg"])
            a = min(true.min().item(), pred.min().item())
            b = max(true.max().item(), pred.max().item())
            lo = a if lo is None else min(lo, a)
            hi = b if hi is None else max(hi, b)
        if lo is None:
            continue
        pad = 0.08 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8,
                zorder=0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(rf"$W^{{({m}\times{n})}}$", fontsize=11)
        ax.set_xlabel("Predicted value")
        ax.set_ylabel("True value")
        # The Letter prints each model's MSE in the panel corner; keeping the
        # paper's own value next to ours is what makes the panel a comparison
        # rather than a picture.
        ax.text(0.03, 0.97, "\n".join(note), transform=ax.transAxes,
                va="top", ha="left", fontsize=7, family="monospace")
        if len(series) > 1:
            ax.legend(loc="lower right", fontsize=8, frameon=False)

    for j in range(len(targets), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.tight_layout()

    stem = "fig3" + (f"_{tag}" if tag else "")
    png = os.path.join(RESULT_DIR, stem + ".png")
    fig.savefig(png, dpi=180)
    print(f"wrote {png}")
    torch.save({"table": table, "panels": payload},
               os.path.join(RESULT_DIR, stem + ".pt"))

    hdr = (f"{'loop':6} {'arch':5} {'size':9} {'N_par':>7} {'MSE_avg':>10} "
           f"{'paper':>9} {'/paper':>8} {'MSE_site':>10} {'R2_site':>13} "
           f"{'R2_within':>13} "
           f"{'site/avg':>9} {'ep':>4} {'cut':>4} {'seeds':>5} {'median':>10} "
           f"{'worst':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in table:
        # site/avg: 64 (the site count) if the per-site errors were
        # independent across the lattice. Well below that means the residual
        # has a coherent, long-wavelength component, which the averaged MSE
        # cannot see and which is a property of the model, not of the task.
        ratio_sa = r["best_mse_site"] / r["best_mse_avg"]
        pm = r["paper_mse"]
        size_s = r["size"] + ("*" if r["plotted"] else "")
        print(f"{r['target']:6} {r['arch']:5} {size_s:9} "
              f"{r['n_param']:7d} "
              f"{r['best_mse_avg']:10.2e} "
              + (f"{pm:9.1e} {r['best_mse_avg'] / pm:8.1e} " if pm is not None
                 else f"{'-':>9} {'-':>8} ")
              + f"{r['best_mse_site']:10.2e} {r['r2_site']:13.9f} "
              f"{r['r2_within']:13.9f} "
              f"{ratio_sa:9.1f} "
              f"{r['epochs_run']:4d} {'YES' if r['cut_off'] else '-':>4} "
              f"{r['n_seeds']:5d} "
              f"{r['median_mse_avg']:10.2e} {r['max_mse_avg']:10.2e}")
    print("\n* = the size the panel plots (lowest validation loss). "
          "R2_within: 0 = knows beta, nothing else.")
    if any(r["cut_off"] for r in table):
        print("\ncut = the best epoch is the last one and the validation curve "
              "is still falling\n      over its final quarter: that run was "
              "stopped by the epoch cap, not by\n      convergence, and its "
              "MSE is an upper bound.")

    tex = os.path.join(RESULT_DIR, stem + ".tex")
    with open(tex, "w") as f:
        f.write("% generated by scripts/wilson_regression_figure.py\n")
        f.write("\\begin{tabular}{llrrrrr}\n\\hline\n")
        f.write("loop & arch & $N_\\mathrm{param}$ & MSE (this work) & "
                "MSE (Favoni et al.) & $R^2_{\\beta}$ & seeds "
                "\\\\\n\\hline\n")
        for r in table:
            f.write(f"$W^{{({LOOPS[r['target']][0]}\\times"
                    f"{LOOPS[r['target']][1]})}}$ & {r['arch']} & "
                    f"{r['n_param']} & "
                    f"{r['best_mse_avg']:.1e} & "
                    + (f"{r['paper_mse']:.1e}" if r['paper_mse'] is not None
                       else "--")
                    + f" & {r['r2_within']:.6f} & "
                    f"{r['n_seeds']} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    print(f"\nwrote {tex}")


if __name__ == "__main__":
    main()
