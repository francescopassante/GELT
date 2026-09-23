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

``--dumps=<glob>`` overrides the default dump pattern, ``--out-tag=<s>`` names
the artifacts, ``--size=<s>`` restricts to one architecture size (by default the
lowest-validation-loss size present is used, which is the Letter's "best model"
reading).
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
    PAPER_MSE_LCNN,
    RESULT_DIR,
    cfg,
    validate_argv,
)

# The Letter's own green for the L-CNN; a second colour for this repo's arm.
ARCH_STYLE = {
    "lcnn": ("L-CNN", "#2ca02c", "o", 2),
    "gelt": ("GELT", "#d62728", "^", 3),
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
    validate_argv()
    pattern = cfg("WR_DUMPS", os.path.join(DUMP_DIR, "wilson_regression_*.pt"))
    tag = cfg("WR_OUT_TAG", "")
    size = cfg("WR_SIZE", None)
    runs = load_dumps(pattern)
    if not runs:
        raise SystemExit(f"no run dumps matched {pattern}")
    print(f"{len(runs)} run dump(s) from {pattern}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(RESULT_DIR, exist_ok=True)
    targets = [t for t in LOOPS if any(r["target"] == t for r in runs)]
    ncol = 2 if len(targets) > 1 else 1
    nrow = math.ceil(len(targets) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.5 * nrow),
                             squeeze=False)

    table, payload = [], {}
    for i, target in enumerate(targets):
        ax = axes[i // ncol][i % ncol]
        m, n = LOOPS[target]
        paper = PAPER_MSE_LCNN[target]
        lo = hi = None
        note = []
        for arch in ("lcnn", "gelt"):
            best, cell = select(runs, target, arch, size)
            if best is None:
                continue
            label, colour, marker, z = ARCH_STYLE[arch]
            pred, true = lattice_average(best)
            ax.scatter(pred, true, s=10, c=colour, marker=marker, alpha=0.6,
                       linewidths=0, zorder=z, label=label)
            vals = sorted(r["mse_avg"] for r in cell)
            note.append(f"{label:5} {best['mse_avg']:.1e}")
            n_ep, cut = was_cut_off(best)
            table.append(dict(target=target, arch=arch, size=best["size"],
                              epochs_run=n_ep, cut_off=cut,
                              conv_impl=best.get("conv_impl"),
                              n_param=best["n_param"],
                              paper_n_param=best.get("paper_n_param"),
                              best_mse_avg=best["mse_avg"],
                              best_mse_site=best["mse_site"],
                              paper_mse=paper, n_seeds=len(vals),
                              median_mse_avg=statistics.median(vals),
                              min_mse_avg=vals[0], max_mse_avg=vals[-1],
                              seed=best["seed"], path=best["_path"]))
            payload[f"{arch}_{target}"] = dict(pred=pred, true=true,
                                               mse_avg=best["mse_avg"])
            a = min(true.min().item(), pred.min().item())
            b = max(true.max().item(), pred.max().item())
            lo = a if lo is None else min(lo, a)
            hi = b if hi is None else max(hi, b)
        if lo is None:
            continue
        note.append(f"paper {paper:.1e}")
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
        if len(note) > 2:
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

    hdr = (f"{'loop':6} {'arch':5} {'size':8} {'N_par':>7} {'MSE_avg':>10} "
           f"{'paper':>9} {'/paper':>8} {'MSE_site':>10} {'site/avg':>9} "
           f"{'ep':>4} {'cut':>4} {'seeds':>5} {'median':>10} {'worst':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in table:
        # site/avg: 64 (the site count) if the per-site errors were
        # independent across the lattice. Well below that means the residual
        # has a coherent, long-wavelength component, which the averaged MSE
        # cannot see and which is a property of the model, not of the task.
        ratio_sa = r["best_mse_site"] / r["best_mse_avg"]
        print(f"{r['target']:6} {r['arch']:5} {r['size']:8} "
              f"{r['n_param']:7d} "
              f"{r['best_mse_avg']:10.2e} {r['paper_mse']:9.1e} "
              f"{r['best_mse_avg'] / r['paper_mse']:8.1e} "
              f"{r['best_mse_site']:10.2e} {ratio_sa:9.1f} "
              f"{r['epochs_run']:4d} {'YES' if r['cut_off'] else '-':>4} "
              f"{r['n_seeds']:5d} "
              f"{r['median_mse_avg']:10.2e} {r['max_mse_avg']:10.2e}")
    if any(r["cut_off"] for r in table):
        print("\ncut = the best epoch is the last one and the validation curve "
              "is still falling\n      over its final quarter: that run was "
              "stopped by the epoch cap, not by\n      convergence, and its "
              "MSE is an upper bound.")

    tex = os.path.join(RESULT_DIR, stem + ".tex")
    with open(tex, "w") as f:
        f.write("% generated by scripts/wilson_regression_figure.py\n")
        f.write("\\begin{tabular}{llrrrr}\n\\hline\n")
        f.write("loop & arch & $N_\\mathrm{param}$ & MSE (this work) & "
                "MSE (Favoni et al.) & seeds \\\\\n\\hline\n")
        for r in table:
            f.write(f"$W^{{({LOOPS[r['target']][0]}\\times"
                    f"{LOOPS[r['target']][1]})}}$ & {r['arch']} & "
                    f"{r['n_param']} & "
                    f"{r['best_mse_avg']:.1e} & {r['paper_mse']:.1e} & "
                    f"{r['n_seeds']} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    print(f"\nwrote {tex}")


if __name__ == "__main__":
    main()
