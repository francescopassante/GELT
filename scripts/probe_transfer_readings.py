"""
=========================================================================
β-transfer — the pre-registered readings X-A … X-E, offline.
=========================================================================

Design record: ``notes/beta_transfer.md``. Reads the dumps
``probe_transfer.py`` writes and prints the readings fixed in writing before
the first evaluation. No GPU, no ensembles, seconds.

    python scripts/probe_transfer_readings.py --group=z2

**The statistic, and why it is a difference of differences.** §9.9 measured
GELT's spread over initialisations at 6.2× the L-CNN's on this task, which is
an order of magnitude above either arm's jackknife error. Comparing raw R² at
a new coupling would drown in it. Each seed is therefore its own control:
``Δ(β′) = R²(β′) − R²(β₀)`` for *that* checkpoint, and the arms are compared on
their Δ. A uniform inflation — the leave-one-out repair included — cancels.

**The confound this cannot remove, and which is why the per-seed table is
printed above every summary**: regression to the mean. A seed that scored 0.295
at the anchor has room to move up that a seed at 0.61 does not. X-B′ is the
pre-registered sensitivity check — the same contrast over the seeds that clear
the matched-depth classical bar at the anchor.
"""

import glob
import math
import os
import sys
from itertools import combinations

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe_common import IS_Z2, env_str, validate_argv  # noqa: E402

IN_DIR = env_str("PROBE_TRANSFER_DIR", "results/z2_vortex/transfer")
PRE_DIR = env_str("PROBE_PREFLIGHT_DIR", "results/z2_vortex")
TARGET = env_str("PROBE_TARGET", "V1" if IS_Z2 else "T2")
FILTER = env_str("PROBE_FILTER", "")
# The matched-depth classical bar at the anchor coupling
# (notes/where_attention_can_win.md §9.4, BFS capped at 4 hops). Used only by
# X-B′, and only as a *pre-registered* health criterion for which seeds are
# operators at all — it is not a filter chosen after seeing the transfer.
HEALTH_BAR = float(env_str("PROBE_HEALTH_BAR", "0.440"))
# Which column the summaries are built from. "affine" is the primary reading:
# it is the kinder one for the arm with a scale problem, so a GELT advantage
# there is about routing and not about amplitude.
PRIMARY = env_str("PROBE_TRANSFER_PRIMARY", "affine")


def mannwhitney_p(x, y):
    """Exact two-sided Mann–Whitney p for small samples; normal beyond.

    Dependency-free on purpose, for the same reason ``fit_cosh_correlator`` is:
    it has to be callable from anywhere in this repo without adding scipy, and
    at n = 6 against n = 6 the exact enumeration is 924 splits.
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")

    def u_of(a, b):
        return sum((1.0 if p > q else 0.5 if p == q else 0.0) for p in a for q in b)

    u_obs = u_of(x, y)
    mu = n1 * n2 / 2.0
    pooled = list(x) + list(y)
    if math.comb(n1 + n2, n1) <= 200_000:
        idx = range(n1 + n2)
        hits = total = 0
        for pick in combinations(idx, n1):
            s = set(pick)
            u = u_of([pooled[i] for i in pick],
                     [pooled[i] for i in idx if i not in s])
            total += 1
            if abs(u - mu) >= abs(u_obs - mu) - 1e-9:
                hits += 1
        return hits / total
    sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (abs(u_obs - mu) - 0.5) / sd
    return math.erfc(z / math.sqrt(2.0))


def mean_sd(v):
    if not v:
        return float("nan"), float("nan")
    m = sum(v) / len(v)
    if len(v) < 2:
        return m, float("nan")
    return m, math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1))


def median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def classical_bar(beta):
    """``(linear, matched-depth local)`` at one β from the pre-flight dump."""
    path = os.path.join(PRE_DIR, f"preflight_b{beta}.pt")
    if not os.path.exists(path):
        return None
    d = torch.load(path, map_location="cpu", weights_only=False)
    row, meta = d["row"], d["meta"]
    arm = meta.get("matched_arm")
    loc = row["v1_local"].get(arm) if arm in row["v1_local"] else None
    return (row["v1_linear"]["radial"][2], None if loc is None else loc[2], arm)


def main():
    validate_argv("PROBE_TRANSFER_DIR", "PROBE_PREFLIGHT_DIR", "PROBE_TARGET",
                  "PROBE_FILTER", "PROBE_HEALTH_BAR", "PROBE_TRANSFER_PRIMARY")
    if PRIMARY not in ("affine", "raw"):
        raise SystemExit(f"PROBE_TRANSFER_PRIMARY must be affine|raw ({PRIMARY!r})")
    paths = sorted(glob.glob(os.path.join(IN_DIR, f"transfer_{TARGET}_src*_at*.pt")))
    if FILTER:
        paths = [p for p in paths if FILTER in os.path.basename(p)]
    if not paths:
        raise SystemExit(f"no transfer dumps under {IN_DIR}/ for target {TARGET}")

    dumps = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    anchors = [d for d in dumps if d["is_anchor"]]
    if not anchors:
        raise SystemExit(
            "X-A is missing: no anchor dump (the evaluation at the source "
            "coupling). Run probe_transfer.py at PROBE_Z2_BETA = the source β "
            "first — it is the gate that says the pipeline still reproduces the "
            "numbers the checkpoints were written with, and nothing else here "
            "is readable without it."
        )
    if not all(d["anchor_ok"] for d in anchors):
        raise SystemExit(
            "X-A FAILED: the anchor evaluation does not reproduce the source "
            "dumps' R². Splits, mask or target construction have drifted since "
            "the checkpoints were written. Read nothing else."
        )
    anchor = anchors[0]
    src_beta = anchor["src_beta"]
    key = "r2_affine_loo" if PRIMARY == "affine" else "r2_raw_loo"
    other = "r2_raw_loo" if PRIMARY == "affine" else "r2_affine_loo"

    print("=" * 78)
    print(f"β-transfer — target {TARGET}, trained at β = {src_beta}")
    print(f"primary column: {PRIMARY} (leave-worst-one-out), "
          f"secondary: {other.split('_')[1]}")
    print("=" * 78)
    print("\nX-A  anchor gate: PASSED — every checkpoint reproduces its source "
          "dump's R².")

    base = {(r["arm"], r["seed"]): r for r in anchor["rows"]}
    arms = sorted({r["arm"] for r in anchor["rows"]})
    ordered = sorted(dumps, key=lambda d: d["eval_beta"])

    # ── the per-β tables, which are the result; the summaries are derived ──
    for d in ordered:
        beta = d["eval_beta"]
        bar = classical_bar(beta)
        tag = "  [anchor]" if d["is_anchor"] else ""
        print(f"\n─── β = {beta}{tag} " + "─" * 46)
        if bar:
            lin, loc, arm_name = bar
            if loc is not None:
                print(f"    classical at this coupling: linear {lin:+.3f},  "
                      f"matched depth ({arm_name}) {loc:+.3f}")
            else:
                # §9.4's own correction: the uncapped BFS is not a bar a
                # bounded-depth network can be asked to clear, so an old dump
                # without the hop ladder has no usable bar and must not be
                # shown as if it had one.
                print(f"    classical at this coupling: linear {lin:+.3f};  "
                      f"**no matched-depth arm in this dump** — it predates the "
                      f"hop cap. Re-run z2_vortex_preflight.py at this β.")
        else:
            print(f"    (no preflight_b{beta}.pt — run z2_vortex_preflight.py "
                  f"at this coupling for the classical bar)")
        print(f"    {'seed':>4s}" + "".join(f"{a:>22s}" for a in arms))
        rows = {(r["arm"], r["seed"]): r for r in d["rows"]}
        seeds = sorted({s for _, s in rows})
        for s in seeds:
            cells = []
            for a in arms:
                r = rows.get((a, s))
                cells.append("—".rjust(22) if r is None else
                             f"{r[key]:+.4f} (Δ {r[key] - base[(a, s)][key]:+.4f})"
                             .rjust(22))
            print(f"    {s:>4d}" + "".join(cells))
        for a in arms:
            v = [rows[(a, s)][key] for s in seeds if (a, s) in rows]
            m, sd = mean_sd(v)
            above = None if not bar or bar[1] is None else sum(x > bar[1] for x in v)
            print(f"    {a:>10s}  mean {m:+.4f}  sd {sd:.4f}"
                  + (f"  above the matched-depth bar: {above}/{len(v)}"
                     if above is not None else ""))

    # ── X-B / X-C: the contrasts ──
    for label, col, name in (("X-B", key, PRIMARY),
                             ("X-C", other, other.split("_")[1])):
        print(f"\n{label}  degradation contrast on the {name} column "
              f"— Δ = R²(β′) − R²({src_beta}), per seed")
        print(f"    {'β':>8s}" + "".join(f"{a + ' Δ med':>16s}" for a in arms)
              + f"{'contrast':>12s}{'p':>8s}")
        for d in ordered:
            if d["is_anchor"]:
                continue
            rows = {(r["arm"], r["seed"]): r for r in d["rows"]}
            deltas = {
                a: [rows[(a, s)][col] - base[(a, s)][col]
                    for (aa, s) in rows if aa == a and (a, s) in base]
                for a in arms
            }
            meds = [median(deltas[a]) for a in arms]
            if len(arms) == 2 and all(deltas[a] for a in arms):
                contrast = meds[0] - meds[1]
                p = mannwhitney_p(deltas[arms[0]], deltas[arms[1]])
                extra = f"{contrast:>+12.4f}{p:>8.3f}"
            else:
                extra = f"{'—':>12s}{'—':>8s}"
            print(f"    {d['eval_beta']:>8.4f}"
                  + "".join(f"{m:>+16.4f}" for m in meds) + extra)
        if len(arms) == 2:
            print(f"    contrast = median Δ({arms[0]}) − median Δ({arms[1]}); "
                  f"p is an exact two-sided Mann–Whitney on the two sets of Δ.")

    # ── X-B′: the pre-registered sensitivity check ──
    healthy = {a: [s for (aa, s) in base if aa == a and base[(a, s)][key] > HEALTH_BAR]
               for a in arms}
    print(f"\nX-B′ the same contrast over seeds that clear the matched-depth bar "
          f"({HEALTH_BAR:+.3f}) at the anchor")
    n_seeds = {a: sum(1 for (aa, _) in base if aa == a) for a in arms}
    print("    " + ", ".join(f"{a}: {len(healthy[a])}/{n_seeds[a]} seeds "
                             f"{sorted(healthy[a])}" for a in arms))
    for d in ordered:
        if d["is_anchor"]:
            continue
        rows = {(r["arm"], r["seed"]): r for r in d["rows"]}
        deltas = {a: [rows[(a, s)][key] - base[(a, s)][key]
                      for s in healthy[a] if (a, s) in rows] for a in arms}
        meds = [median(deltas[a]) for a in arms]
        line = f"    β = {d['eval_beta']:.4f}  " + "  ".join(
            f"{a} {m:+.4f} (n={len(deltas[a])})" for a, m in zip(arms, meds))
        if len(arms) == 2 and all(deltas[a] for a in arms):
            line += f"   contrast {meds[0] - meds[1]:+.4f}"
        print(line)

    # ── X-D: dispersion of the degradation ──
    print("\nX-D  dispersion of the degradation over initialisations "
          "(§9.9 measured 6.2× against GELT *at* the anchor)")
    for d in ordered:
        if d["is_anchor"]:
            continue
        rows = {(r["arm"], r["seed"]): r for r in d["rows"]}
        sds = {}
        for a in arms:
            v = [rows[(a, s)][key] - base[(a, s)][key]
                 for (aa, s) in rows if aa == a and (a, s) in base]
            sds[a] = mean_sd(v)[1]
        line = f"    β = {d['eval_beta']:.4f}  " + "  ".join(
            f"sd({a}) {sds[a]:.4f}" for a in arms)
        if len(arms) == 2 and all(not math.isnan(sds[a]) for a in arms):
            lo, hi = min(sds.values()), max(sds.values())
            line += f"   ratio {hi / lo:.1f}×" if lo > 0 else "   ratio —"
        print(line)

    # ── X-E: the scale component, which is the reading the two columns make ──
    print("\nX-E  how much of the transfer deficit is scale alone "
          "(affine − raw, per arm, mean over seeds)")
    for d in ordered:
        rows = {(r["arm"], r["seed"]): r for r in d["rows"]}
        gaps = {a: mean_sd([rows[(a, s)]["r2_affine_loo"] - rows[(a, s)]["r2_raw_loo"]
                            for (aa, s) in rows if aa == a])[0] for a in arms}
        print(f"    β = {d['eval_beta']:.4f}  "
              + "  ".join(f"{a} {gaps[a]:+.4f}" for a in arms)
              + ("   [anchor]" if d["is_anchor"] else ""))
    print("    A large gap is an operator whose *shape* transfers and whose "
          "amplitude does not — M2's fingerprint, not M1's.")

    print("\n" + "=" * 78)
    print("Falsification, as pre-registered (notes/beta_transfer.md §4): if X-B "
          "is consistent with 0 at every transfer coupling, the adaptation "
          "hypothesis is dead for this task and the remaining route is M2 on a "
          "tail-dominated metric (where_attention_can_win.md §8).")


if __name__ == "__main__":
    main()
