"""The input↔architecture curve: A₀(x) for three traces, and the three readings.

`notes/fable5.1_10-09_audit.md` WP5b. The spectroscopy chapter rested on one
number, ΔA₀ against one classical basis. This assembles the curve that replaces
it: ground-state overlap as a function of the *input content* x handed to both
methods, for the classical GEVP, a trained GELT and an untrained one.

    x ∈ { thin, 4lv, 7lv }   on the main chain (thin ⊂ 4lv ⊂ 7lv), plus the
    off-chain classical points 4lv+5sh (`shapes_sm`) and 7lv+3sh (`full`).

Everything is read from the per-dump fair-fight outputs — this script fits
nothing and re-estimates nothing, so the numbers here are the numbers that
`su2_fair_fight.py` printed, under one estimator setting for every point
(§8.1's (b): truncate, eps 1e-4). Offline, CPU, seconds:

    python scripts/input_architecture_curve.py                 # after the 26 runs below
    CURVE_EST=_trunc_prune0.99 python scripts/input_architecture_curve.py

Its inputs are produced by, for every dump in `dumps/`:

    SFF_TRUNCATE=1 SFF_BASES=1 python scripts/su2_fair_fight.py <dump>

The pre-registered readings (§1.2 of the note), tested mechanically below:

  R1  the architecture buys input depth — ΔA₀(x) > 2σ at every *resolvable* x,
      and rung-equivalence: A₀_GELT(x_k) ≥ A₀_GEVP(x_{k+1}) within errors.
  R2  an efficient shortcut — ΔA₀(x) consistent with zero at some resolvable x
      while the GEVP still has room (1 − A₀ more than 3σ from zero).
  R3  additive advantage — ΔA₀(x) consistent with a constant c > 0
      (χ²/dof < 2 and c > 3σ).

A point is **resolvable** when the combined classical A₀ < 0.90; above that the
observable is saturated (§1.3) and cannot discriminate the readings. Saturated
points are drawn, shaded, and excluded from the tests.
"""

import glob
import math
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

EST = os.environ.get("CURVE_EST", "_trunc")  # estimator suffix of the inputs
# The curve is drawn at ONE architecture width, so that moving along x changes
# the input content and nothing else; d_model 24 is the only width trained at
# every x (thin, the 4lv width control, 7lv). A net at another width is kept as
# the width control and drawn apart, never averaged into the trace.
WIDTH = int(os.environ.get("CURVE_WIDTH", 24))
OUT = "results/fair_fight/input_architecture_curve"
SATURATED = 0.90  # combined classical A₀ at or above which a point is excluded

# x → (label, the classical arm with the same input content, the next rung's arm)
CHAIN = {
    "thin": ("thin", "thin", "published"),
    "4lv": ("4lv", "published", "deep"),
    "7lv": ("7lv", "deep", "full"),
}
OFFCHAIN = {"4lv+5sh": "shapes_sm", "7lv+3sh": "full"}
ENSEMBLES = ("run5", "ens1")
COLOR = {"classical": "#D55E00", "trained": "#0072B2", "random": "#888888"}


def manifest():
    """Every dump's (x, ensemble, trace, seed, width) and its fair-fight row.

    Keyed off the dump's own metadata, not its filename: `random_init` and
    `d_model` are recorded there, and the smearing ladder is what defines x.
    """
    out = []
    for dump in sorted(glob.glob("dumps/best_glueball_gelt*_test_obars.pt")):
        meta = torch.load(dump, map_location="cpu", weights_only=False)["meta"]
        lv = list(meta.get("input_smear_levels", []))
        x = {1: "thin", 4: "4lv", 7: "7lv"}.get(len(lv))
        if x is None:
            print(f"  skipped (unknown ladder {lv}): {os.path.basename(dump)}")
            continue
        stem = os.path.basename(dump).replace("_test_obars.pt", "")
        res = f"results/fair_fight/su2_fair_fight_{stem}{EST}.pt"
        if not os.path.exists(res):
            print(f"  missing fair-fight output, skipped: {res}")
            continue
        # The two 7-level nets predate the d_model field in the meta (and the
        # _d tag in their names): they are width 24, as they must be — the
        # residual stream cannot be narrower than 3·n_levels = 21 input channels.
        width = meta.get("d_model") or (24 if 3 * len(lv) > 16 else 16)
        rnd = bool(meta.get("random_init", False))
        m = re.search(r"_rnd(\d+)", stem)
        out.append({
            "x": x,
            "ens": "ens1" if meta.get("ensemble_seed", 0) == 1 else "run5",
            "trace": "random" if rnd else "trained",
            "seed": int(m.group(1)) if m else int(meta.get("init_seed", 0)),
            "width": width,
            "row": torch.load(res, weights_only=False)["rows"][0],
        })
    return out


def combine(vals):
    """Inverse-variance combination of (value, error) pairs."""
    w = [1.0 / e ** 2 for _, e in vals if e > 0]
    if not w:
        return float("nan"), float("inf")
    c = sum(v / e ** 2 for v, e in vals if e > 0) / sum(w)
    return c, math.sqrt(1.0 / sum(w))


def random_point(entries):
    """Mean over init seeds, with §1.4's error rule.

    The error is the larger of the seed-to-seed spread and the mean of the
    per-seed jackknife errors: three seeds cannot resolve a spread finely, and a
    poor operator's own fit error is not the whole story either.
    """
    a = [e["row"]["arms"]["gelt"]["A0"] for e in entries]
    err = [e["row"]["arms"]["gelt"]["A0_err"] for e in entries]
    mean = sum(a) / len(a)
    spread = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    return mean, max(spread, sum(err) / len(err))


def collect(man):
    """Per-x, per-ensemble A₀ for the three traces, plus ΔA₀ and the rung."""
    pts = {}
    for x, (_, arm, nxt) in CHAIN.items():
        for ens in ENSEMBLES:
            here = [e for e in man if e["x"] == x and e["ens"] == ens]
            if not here:
                continue
            d = pts.setdefault((x, ens), {})
            row0 = here[0]["row"]
            d["classical"] = (row0["arms"][arm]["A0"], row0["arms"][arm]["A0_err"])
            for name, off in OFFCHAIN.items():
                if off in row0["arms"]:
                    pts.setdefault((name, ens), {})["classical"] = (
                        row0["arms"][off]["A0"], row0["arms"][off]["A0_err"])
            # The trained point of the curve is the net at WIDTH; any other
            # width at the same x is the width control (WP3), carried separately
            # and never averaged in.
            trained = sorted([e for e in here if e["trace"] == "trained"],
                             key=lambda e: (e["width"] != WIDTH, e["width"]))
            if trained:
                main = trained[0]
                if main["width"] != WIDTH:
                    print(f"  no width-{WIDTH} trained net at {x}/{ens} — using "
                          f"d_model {main['width']}")
                a = main["row"]["arms"]["gelt"]
                d["trained"] = (a["A0"], a["A0_err"])
                d["width"] = main["width"]
                dd = main["row"]["delta"][arm]
                d["delta"] = (dd["dA0"], dd["dA0_err"])
                rung = main["row"]["delta"].get(nxt)
                d["rung"] = (rung["dA0"], rung["dA0_err"]) if rung else None
                for extra in trained[1:]:
                    e2 = extra["row"]["arms"]["gelt"]
                    d["width_control"] = (extra["width"], e2["A0"], e2["A0_err"])
                    d["width_control_A0"] = (e2["A0"], e2["A0_err"])
            rnd = [e for e in here if e["trace"] == "random"]
            at_width = [e for e in rnd if e["width"] == WIDTH]
            if at_width and len(at_width) != len(rnd):
                rnd = at_width  # same rule as the trained trace
            if rnd:
                d["random"] = random_point(rnd)
                d["n_seeds"] = len(rnd)
                d["random_width"] = rnd[0]["width"]
    return pts


def readings(pts):
    """R1 / R2 / R3 by their §1.2 definitions, on the resolvable points only."""
    rep, chain = [], list(CHAIN)
    comb = {x: combine([pts[(x, e)]["classical"] for e in ENSEMBLES
                        if (x, e) in pts and "classical" in pts[(x, e)]])
            for x in chain}
    delta = {x: combine([pts[(x, e)]["delta"] for e in ENSEMBLES
                         if (x, e) in pts and "delta" in pts[(x, e)]])
             for x in chain if any((x, e) in pts and "delta" in pts[(x, e)]
                                   for e in ENSEMBLES)}
    resolvable = [x for x in delta if comb[x][0] < SATURATED]
    rep.append(f"resolvable (classical A₀ < {SATURATED}): "
               + ", ".join(f"{x} [A₀ = {comb[x][0]:.3f}]" for x in resolvable)
               + " | saturated: "
               + ", ".join(f"{x} [A₀ = {comb[x][0]:.3f}]"
                           for x in delta if x not in resolvable))

    all_positive = all(delta[x][0] > 2 * delta[x][1] for x in resolvable)
    rungs = {}
    for k in range(len(chain) - 1):
        x = chain[k]
        r = [pts[(x, e)]["rung"] for e in ENSEMBLES
             if (x, e) in pts and pts[(x, e)].get("rung")]
        if r:
            rungs[f"{x}→{chain[k + 1]}"] = combine(r)
    rung_ok = {k: v[0] > -2 * v[1] for k, v in rungs.items()}
    rep.append("R1  ΔA₀ > 2σ at every resolvable x: "
               + ("YES" if all_positive else "NO")
               + " | rung-equivalence: "
               + ", ".join(f"{k} {'holds' if ok else 'FAILS'} "
                           f"({rungs[k][0]:+.3f} ± {rungs[k][1]:.3f})"
                           for k, ok in rung_ok.items())
               + f"  →  R1 {'HOLDS' if all_positive and all(rung_ok.values()) else 'does NOT hold as stated'}")

    # R2 wants a resolvable x where the traces have met AND the GEVP still has
    # room to improve — "met at saturation" is not a shortcut, it is a ceiling.
    r2 = [x for x in resolvable
          if abs(delta[x][0]) < 2 * delta[x][1]
          and (1 - comb[x][0]) > 3 * comb[x][1]]
    rep.append(f"R2  traces meet before saturation at: {r2 or 'nowhere'}  →  "
               f"R2 {'HOLDS' if r2 else 'does NOT hold'}")

    if len(resolvable) > 1:
        v = [delta[x] for x in resolvable]
        c, ce = combine(v)
        chi2 = sum(((a - c) / e) ** 2 for a, e in v)
        dof = len(v) - 1
        ok = chi2 / dof < 2 and c > 3 * ce
        rep.append(f"R3  constant fit c = {c:+.3f} ± {ce:.3f}, χ²/dof = "
                   f"{chi2 / dof:.1f}  →  R3 {'HOLDS' if ok else 'does NOT hold'}")
    else:
        rep.append("R3  needs at least two resolvable points — not testable")

    # The random trace's own reading: the architecture alone must not beat the
    # optimal linear combination of its own inputs.
    bad = [x for x in chain for e in ENSEMBLES
           if (x, e) in pts and "random" in pts[(x, e)]
           and pts[(x, e)]["random"][0] > pts[(x, e)]["classical"][0]]
    rep.append("random trace: architecture alone never beats the classical GEVP"
               if not bad else
               f"random trace: BEATS the classical GEVP at {sorted(set(bad))} — "
               "the 'learned' attribution there rests on trained − random only")
    return rep, comb, delta, rungs, resolvable


def figure(pts, comb, delta, resolvable):
    """Two panels, each a single combined point per x — the ensembles are
    inverse-variance combined here and kept separately in the table. Two
    overlapping ensembles per trace made the earlier version unreadable, and
    every test in :func:`readings` is run on the combined value anyway."""
    chain = list(CHAIN)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for a in ax:
        a.grid(alpha=0.25, lw=0.6)
        a.set_axisbelow(True)
        a.set_xticks(range(len(chain)), chain)
        a.set_xlabel("input content handed to both methods")

    def comb_of(x, key):
        v = [pts[(x, e)][key] for e in ENSEMBLES
             if (x, e) in pts and key in pts[(x, e)]]
        return combine(v) if v else None

    a = ax[0]
    a.axhspan(SATURATED, 1.15, color="#bbb", alpha=0.25, zorder=0)
    a.text(0.02, SATURATED + 0.006, "saturated — excluded from the tests",
           fontsize=8, color="#555", transform=a.get_yaxis_transform())
    rw = sorted({pts[k]["random_width"] for k in pts if "random_width" in pts[k]})
    for trace, label in (("classical", "classical GEVP"),
                         ("trained", f"GELT, trained (d_model {WIDTH})"),
                         ("random", "GELT, untrained (3 seeds, d_model "
                                    + "/".join(str(w) for w in rw) + ")")):
        xs = [i for i, x in enumerate(chain) if comb_of(x, trace)]
        v = [comb_of(chain[i], trace) for i in xs]
        a.errorbar(xs, [q[0] for q in v], yerr=[q[1] for q in v], fmt="o-",
                   ms=8, capsize=4, lw=1.6, color=COLOR[trace], label=label)
    wc = [(i, comb_of(x, "width_control_A0")) for i, x in enumerate(chain)]
    wc = [(i, v) for i, v in wc if v]
    if wc:
        a.errorbar([i + 0.13 for i, _ in wc], [v[0] for _, v in wc],
                   yerr=[v[1] for _, v in wc], fmt="*", ms=14, mfc="none",
                   color=COLOR["trained"], label="GELT, trained (width control)")
    for name in OFFCHAIN:
        v = comb_of(name, "classical")
        if v:
            i = chain.index("4lv" if name.startswith("4lv") else "7lv")
            a.errorbar([i + 0.26], [v[0]], yerr=[v[1]], fmt="^", ms=9, mfc="none",
                       color=COLOR["classical"], label=f"classical, {name}")
    a.legend(fontsize=8, loc="lower right")
    a.set_ylabel(r"$A_0$ — ground-state overlap")
    a.set_title("Ground-state overlap against input content")

    a = ax[1]
    xs = [i for i, x in enumerate(chain) if chain[i] in delta]
    v = [delta[chain[i]] for i in xs]
    a.errorbar(xs, [q[0] for q in v], yerr=[q[1] for q in v], fmt="D-", ms=9,
               capsize=5, lw=1.6, color="k", label="trained GELT − classical GEVP")
    rungs_x = [(i, pts[(chain[i], ENSEMBLES[0])].get("rung")) for i in xs]
    rv = [(i, combine([pts[(chain[i], e)]["rung"] for e in ENSEMBLES
                       if (chain[i], e) in pts and pts[(chain[i], e)].get("rung")]))
          for i, r in rungs_x if r]
    if rv:
        a.errorbar([i + 0.13 for i, _ in rv], [q[0] for _, q in rv],
                   yerr=[q[1] for _, q in rv], fmt="s--", ms=7, capsize=4,
                   color="#009E73", alpha=0.9,
                   label="…− the classical GEVP one rung further")
    for i in xs:
        if chain[i] not in resolvable:
            a.annotate("saturated", (i, delta[chain[i]][0]), fontsize=8,
                       xytext=(4, 10), textcoords="offset points", color="#555")
    a.axhline(0, color="k", lw=1)
    a.set_ylabel(r"$\Delta A_0$")
    a.set_title("Correlated difference, same configurations")
    a.legend(fontsize=8)

    fig.suptitle("One learned operator against the optimal classical combination "
                 "of the same inputs", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT + ".png", dpi=150)
    print(f"saved {OUT}.png")


def table(pts, delta, rungs):
    """Paste-ready LaTeX, one row per x (the same numbers as the figure)."""
    lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
             r"$x$ & ens & $A_0$ classical & $A_0$ trained & $A_0$ random & "
             r"$\Delta A_0$ & combined & rung $\Delta A_0$ \\", r"\midrule"]
    for x in CHAIN:
        for ens in ENSEMBLES:
            d = pts.get((x, ens))
            if not d:
                continue
            def f(key):
                v = d.get(key)
                return f"${v[0]:.3f}({v[1] * 1000:.0f})$" if v else "---"
            dd = d.get("delta")
            comb = (f"${delta[x][0]:+.3f} \\pm {delta[x][1]:.3f}$"
                    if ens == ENSEMBLES[0] and x in delta else "")
            k = next((kk for kk in rungs if kk.startswith(x + "→")), None)
            rung = (f"${rungs[k][0]:+.3f} \\pm {rungs[k][1]:.3f}$"
                    if k and ens == ENSEMBLES[0] else "")
            lines.append(f"{x if ens == ENSEMBLES[0] else ''} & {ens} & {f('classical')} & "
                         f"{f('trained')} & {f('random')} & "
                         + (f"${dd[0]:+.3f} \\pm {dd[1]:.3f}$" if dd else "---")
                         + f" & {comb} & {rung} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(OUT + ".tex", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"saved {OUT}.tex")


def main():
    man = manifest()
    if not man:
        raise SystemExit("no fair-fight outputs found — run su2_fair_fight.py per dump first")
    print(f"{len(man)} dumps with fair-fight outputs (estimator suffix {EST!r})")
    pts = collect(man)
    rep, comb, delta, rungs, resolvable = readings(pts)
    print("\n" + "=" * 78)
    print("THE CURVE — the three pre-registered readings (§1.2 of the plan note)")
    print("=" * 78)
    for line in rep:
        print("  " + line)
    print("\n  " + f"{'x':<8}{'ens':<6}{'classical':>16}{'trained':>16}{'random':>16}"
          f"{'ΔA₀':>18}")
    for x in CHAIN:
        for ens in ENSEMBLES:
            d = pts.get((x, ens), {})
            def s(k):
                v = d.get(k)
                return f"{v[0]:>8.3f} ± {v[1]:.3f}" if v else f"{'—':>16}"
            print(f"  {x:<8}{ens:<6}{s('classical')}{s('trained')}{s('random')}{s('delta')}")
    figure(pts, comb, delta, resolvable)
    table(pts, delta, rungs)
    torch.save({"points": pts, "combined_classical": comb, "delta": delta,
                "rungs": rungs, "resolvable": resolvable, "readings": rep,
                "estimator": EST}, OUT + ".pt")
    print(f"saved {OUT}.pt")


if __name__ == "__main__":
    main()
