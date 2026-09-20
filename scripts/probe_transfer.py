"""
=========================================================================
β-transfer — a trained operator, evaluated at a coupling it never saw.
=========================================================================

Design record and pre-registered readings: ``notes/beta_transfer.md``. This is
attempt 5 in the programme ``notes/where_attention_can_win.md`` tracks, and it
is the first one that does not train anything: the twelve V1 checkpoints of
§9.9 already exist, four ensembles already exist, and the question is what
happens when the two are crossed.

**Why this is the experiment.** Every A/B in this repo has trained and tested at
one coupling, with each arm at its own bracketed learning rate. That design is
what makes the accuracy comparison fair, and it is also what makes M1
invisible: a softmax that renormalises the weighting over offsets *per
configuration* has nothing to earn when the configurations never change
distribution. Move β and the vortex gas thins, the cluster-size competition
changes, and the optimal routing moves with it — GELT's aggregation over
offsets still sums to one by construction, the L-CNN's is a free sum over ~13
offsets with weights calibrated at one coupling.

**One coupling per invocation**, because ``probe_common.BETA`` is the cache key
and the printouts, and a second β inside one process would mean a second
meaning for one name. The driver is a shell loop; the join is offline in
``probe_transfer_readings.py``, exactly as ``train_probe.py`` dumps and
``probe_readings.py`` reads.

    # the anchor — the gate, and it must be run first
    PROBE_GROUP=z2 PROBE_Z2_BETA=0.7520 python scripts/probe_transfer.py
    # the transfer couplings
    PROBE_GROUP=z2 PROBE_Z2_BETA=0.7450 python scripts/probe_transfer.py
    PROBE_GROUP=z2 PROBE_Z2_BETA=0.7560 python scripts/probe_transfer.py
    PROBE_GROUP=z2 PROBE_Z2_BETA=0.7585 python scripts/probe_transfer.py

**Two readings per checkpoint, and the difference between them is the point.**

- *raw* — the operator applied as it stands. Its prediction is in the units the
  source run standardised to, so it is mapped back through that run's own
  (μ, σ) and scored against the raw target at the new coupling.
- *affine* — the same prediction after a **two-parameter** recalibration fitted
  on the new coupling's *train* split. One slope, one offset, available to both
  arms identically. This is the kinder reading, and it is the one that isolates
  *routing* from *scale*: an arm that transfers the right shape and the wrong
  amplitude recovers here, an arm that transfers the wrong shape does not.

The gap between the two is a measurement of how much of a transfer deficit is
mere miscalibration — which is M2's fingerprint and not M1's.

**The anchor gate.** Evaluated at the source coupling, this script recomputes a
number the source dump already holds. R² is invariant under a common affine map
of prediction and target, so the recomputed *raw* value must reproduce the
stored ``r2`` to rounding. If it does not, the splits, the mask or the target
construction have drifted since the checkpoints were written and nothing else
here means anything. ``probe_transfer_readings.py`` refuses to print a table
whose anchor is missing or failing.
"""

import glob
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe_common import (  # noqa: E402
    BETA,
    GROUP,
    IS_Z2,
    JACK_BLOCK,
    N_CONFIGS,
    TARGETS,
    accumulate_stats,
    arm_transport,
    build_all_targets,
    build_arm,
    build_mask,
    env_str,
    jackknife,
    load_samples,
    probe_inputs,
    r2_from_stats,
    splits,
    validate_argv,
)

SRC_BETA = float(env_str("PROBE_SRC_BETA", "0.7520"))
TARGET = env_str("PROBE_TARGET", "V1" if IS_Z2 else "T2")
# The arms to transfer. The default is W-D's pair — the two the thesis is about
# and the only two that have six seeds (notes/where_attention_can_win.md §9.9).
TRANSFER_ARMS = tuple(
    a for a in env_str("PROBE_TRANSFER_ARMS", "gelt,lcnn").split(",") if a
)
# Selects one campaign when several tags share an (arm, seed). Not a default:
# picking silently between two checkpoints that differ in learning rate or
# epoch budget is the defect §9.9 found twice.
FILTER = env_str("PROBE_FILTER", "")
DEVICE = env_str("PROBE_DEVICE", "")
IN_DIR = env_str("PROBE_DIR", "results/z2_vortex/probe" if IS_Z2 else "results/m1_probe")
OUT_DIR = env_str("PROBE_TRANSFER_DIR", "results/z2_vortex/transfer")

if not IS_Z2:
    # SU(2)'s second ensemble is a second *chain at the same β*, which is a
    # different seed and not a distribution shift — there is nothing to transfer
    # to. The volume and low-statistics stressors of
    # notes/where_attention_can_win.md §8 are the SU(2) analogues and are a
    # different script.
    raise SystemExit(
        "probe_transfer.py is a Z₂ study: it moves the coupling, and the SU(2) "
        "cache key is a chain seed at one fixed β = 2.4. Run it with "
        "PROBE_GROUP=z2."
    )
if TARGET not in TARGETS:
    raise SystemExit(f"PROBE_TARGET must be one of {TARGETS} (got {TARGET!r})")

# Anchored on ``_b<number>_init<digits>`` as one unit, not on ``_b`` alone:
# ``signed_bounded`` is an arm name containing ``_b``, so splitting on the
# first ``_b`` reads the coupling out of the middle of the arm.
_STEM_RE = re.compile(r"_b(?P<beta>[0-9.]+)_init(?P<seed>\d+)(?P<rest>.*)$")


def _fmt(v, spec=""):
    """Format a dump field that may be absent — older dumps predate two keys."""
    return "—" if v is None else format(v, spec)


def discover(arm):
    """Every checkpoint of one arm at the source coupling, as dicts.

    Discovered by glob rather than constructed from a stem on purpose: the
    learning rate is not in the stem, the six-seed campaign of §9.9 carries a
    run tag and a hand-written loop easily does not, and a stem built here would
    quietly miss whichever convention the run actually used. What the dump
    records is authoritative; what the filename records is how to find it.
    """
    pattern = os.path.join(IN_DIR, f"probe_{arm}_{TARGET}_b*_init*.pth")
    found = []
    for ckpt in sorted(glob.glob(pattern)):
        stem = os.path.basename(ckpt)[: -len(".pth")]
        if "_null" in stem:
            continue  # the random-features null of R-E, not an operator
        if FILTER and FILTER not in stem:
            continue
        m = _STEM_RE.search(stem)
        if m is None or abs(float(m.group("beta")) - SRC_BETA) > 1e-9:
            continue
        dump_path = os.path.join(IN_DIR, f"{stem}_stats.pt")
        if not os.path.exists(dump_path):
            # The (μ, σ) the run standardised to live only in the dump, and
            # without them the prediction cannot be put back into physical
            # units. A checkpoint without its dump is unusable, not optional.
            print(f"  ** {stem}: no {stem}_stats.pt — skipped (no μ, σ)")
            continue
        d = torch.load(dump_path, map_location="cpu", weights_only=False)
        found.append(dict(
            arm=arm, stem=stem, ckpt=ckpt, seed=int(m.group("seed")),
            tag=m.group("rest"), src=d,
        ))
    return found


def _affine_fit(sums):
    """Least-squares ``(a, b)`` for ``y ≈ a·p + b`` from the six sums.

    Degenerate when the prediction is constant over the fit split — a collapsed
    run, or one whose output the mask happens to flatten. Falling back to the
    training mean is the honest answer there: it scores R² ≈ 0 rather than
    dividing by zero, which is what a constant predictor deserves.
    """
    n, sy, syy, sp, spp, syp = sums.tolist()
    den = n * spp - sp * sp
    if den <= 1e-12 * max(1.0, abs(n * spp)):
        return 0.0, sy / n
    a = (n * syp - sy * sp) / den
    b = (sy - a * sp) / n
    return a, b


def r2_affine_from_stats(stats, a, b):
    """``R²`` of ``a·p + b`` against ``y``, from :data:`probe_common.STAT_KEYS`.

    ``Σ(y − a·p − b)² = Σy² − 2aΣyp − 2bΣy + a²Σp² + 2abΣp + n·b²`` — every term
    is one of the six sums, so the affine reading costs no second pass over the
    data and every jackknife sample of it is exact. At ``(a, b) = (1, 0)`` it is
    ``r2_from_stats`` identically, which is the self-check in ``__main__``.
    """
    n, sy, syy, sp, spp, syp = stats.sum(dim=0).tolist()
    sst = syy - sy * sy / n
    sse = syy - 2 * a * syp - 2 * b * sy + a * a * spp + 2 * a * b * sp + n * b * b
    return 1.0 - sse / sst


def drop_worst(stats, fn):
    """``fn`` over all configurations, and over all but the most damaging one.

    §9.9's stated repair: a run whose R² is −67 with an error bar of the same
    size is one held-out configuration, not a training failure, and R²'s
    unbounded tail turns it into a number that swamps the statistic. Computed
    for **every** row rather than for the rows that look bad, so it is a rule
    applied symmetrically and not a judgement applied where it helps — and the
    primary reading is a difference of differences, in which a uniform
    one-in-twenty inflation cancels.
    """
    full = fn(stats)
    best, worst_i = full, -1
    for i in range(stats.shape[0]):
        keep = torch.cat([stats[:i], stats[i + 1:]])
        v = fn(keep)
        if v > best:
            best, worst_i = v, i
    return full, best, worst_i


@torch.no_grad()
def eval_group(entries, arch, transport, U3, y_raw, mask, idx, device):
    """Per-configuration statistics for every entry in one input group.

    The models are grouped by ``(arch, transport)`` because that is what decides
    the **transport**, which is 62.8% of a GELT step
    (``notes/performance_audit.md`` §5.0(v)). Built once per configuration and
    reused across every checkpoint that wants it, instead of once per
    checkpoint: twelve models over ninety configurations is 1080 transport
    builds the naive loop would do and 90 this one does.
    """
    out = {e["stem"]: [] for e in entries}
    for c in idx.tolist():
        U = U3[c:c + 1].reshape(-1, *U3.shape[2:])
        t = y_raw[c:c + 1].reshape(-1, *y_raw.shape[2:]).to(device, torch.float32)
        m = None if mask is None else \
            mask[c:c + 1].reshape(-1, *y_raw.shape[2:]).to(device)
        W, T = probe_inputs(U, arch, device, transport=transport)
        for e in entries:
            pred = e["model"](W, T)
            # Back into physical units through the *source* run's moments: the
            # network predicts (y − μ_src)/σ_src and the target here is raw, so
            # this is what makes an R² at a new coupling mean "does the operator
            # get the cluster size right" rather than "up to an unknown scale".
            p = pred * e["sigma"] + e["mu"]
            p_cpu, t_cpu = p.detach().cpu(), t.detach().cpu()
            if m is not None:
                m_cpu = m.cpu()
                p_cpu, t_cpu = p_cpu[m_cpu], t_cpu[m_cpu]
            out[e["stem"]].append(accumulate_stats(t_cpu, p_cpu))
    return {k: torch.stack(v) for k, v in out.items()}


def main():
    validate_argv("PROBE_SRC_BETA", "PROBE_TRANSFER_ARMS", "PROBE_FILTER",
                  "PROBE_DIR", "PROBE_TRANSFER_DIR", "PROBE_TARGET",
                  "PROBE_DEVICE")
    device = torch.device(
        DEVICE or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    is_anchor = abs(BETA - SRC_BETA) < 1e-9
    print("=" * 78)
    print(f"β-transfer — target {TARGET}   trained at β = {SRC_BETA}   "
          f"evaluated at β = {BETA}" + ("   [ANCHOR — the gate]" if is_anchor else ""))
    print(f"device: {device}   arms: {', '.join(TRANSFER_ARMS)}"
          + (f"   filter: {FILTER!r}" if FILTER else ""))
    print("=" * 78)

    entries = []
    for arm in TRANSFER_ARMS:
        found = discover(arm)
        if not found:
            raise SystemExit(
                f"no checkpoints for arm {arm!r}, target {TARGET}, β = {SRC_BETA} "
                f"under {IN_DIR}/ (pattern probe_{arm}_{TARGET}_b*_init*.pth"
                + (f", filter {FILTER!r}" if FILTER else "") + ").\n"
                f"The transfer study trains nothing: it needs §9.9's checkpoints."
            )
        by_seed = {}
        for e in found:
            by_seed.setdefault(e["seed"], []).append(e)
        clashes = {s: v for s, v in by_seed.items() if len(v) > 1}
        if clashes:
            # Two checkpoints for one (arm, seed) differ in a knob that is not
            # in the stem — the learning rate, the epoch budget, the batch. §9.9
            # found that defect twice; picking one here silently would be the
            # third time.
            raise SystemExit(
                f"arm {arm!r} has more than one checkpoint for seed(s) "
                f"{sorted(clashes)}:\n" + "\n".join(
                    f"  {x['stem']}  (lr {x['src'].get('lr')}, "
                    f"epochs {x['src'].get('epochs_run')}, "
                    f"batch {x['src'].get('batch')})"
                    for v in clashes.values() for x in v
                ) + "\nPass --filter=<substring> to name the campaign."
            )
        entries += found

    print(f"\n{len(entries)} checkpoint(s):")
    for e in entries:
        d = e["src"]
        flags = "".join([
            "  DIVERGED" if d.get("diverged") else "",
            "  COLLAPSED" if d.get("collapsed") else "",
        ])
        print(f"  {e['stem']:<52s} lr {_fmt(d.get('lr'), '<8g')} epochs "
              f"{_fmt(d.get('epochs_run'), '<4')} batch {_fmt(d.get('batch'))}  "
              f"ci {_fmt(d.get('conv_init_scale'))}  "
              f"R²(src) {_fmt(d.get('r2'), '+.4f')}{flags}")

    U3 = load_samples()
    _, targets = build_all_targets(U3, names=(TARGET,))
    mask = build_mask(U3)
    tr, _, te = splits()
    # The target stays **raw**. Standardising it here with this coupling's own
    # moments would hand every arm a free per-β recalibration before the raw
    # reading is even taken, which is the whole quantity X-C measures.
    y_raw = targets[TARGET].double()
    print(f"splits: train {len(tr)}  test {len(te)} configurations of "
          f"{N_CONFIGS} at β = {BETA}")

    for e in entries:
        model, arch = build_arm(e["arm"], seed=e["seed"], grad_checkpoint=False)
        model.load_state_dict(torch.load(e["ckpt"], map_location="cpu"))
        e["model"] = model.to(device).eval()
        e["arch"] = arch
        e["transport"] = arm_transport(e["arm"])
        e["mu"] = e["src"]["target_mu"]
        e["sigma"] = e["src"]["target_sigma"]

    groups = {}
    for e in entries:
        groups.setdefault((e["arch"], e["transport"]), []).append(e)
    print(f"\ninput groups (one transport build each): "
          + ", ".join(f"{a}/{t} ×{len(v)}" for (a, t), v in groups.items()))

    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()
    fit_stats, test_stats = {}, {}
    for (arch, transport), group in groups.items():
        print(f"  {arch}/{transport}: fitting split ({len(tr)} configs) …")
        fit_stats.update(eval_group(group, arch, transport, U3, y_raw, mask, tr, device))
        print(f"  {arch}/{transport}: test split ({len(te)} configs) …")
        test_stats.update(eval_group(group, arch, transport, U3, y_raw, mask, te, device))
    print(f"  ({time.time() - t_start:.0f}s)")

    print(f"\n{'arm':<10s}{'seed':>5s}{'R² raw':>12s}{'±':>9s}"
          f"{'R² affine':>12s}{'±':>9s}{'a':>9s}{'b':>9s}{'loo':>8s}")
    rows, anchor_ok = [], True
    for e in entries:
        st = test_stats[e["stem"]]
        a, b = _affine_fit(fit_stats[e["stem"]].sum(dim=0))
        raw, raw_loo, worst_raw = drop_worst(st, r2_from_stats)
        aff, aff_loo, worst_aff = drop_worst(
            st, lambda s: r2_affine_from_stats(s, a, b)
        )
        _, raw_err, _ = jackknife(st, r2_from_stats)
        _, aff_err, _ = jackknife(st, lambda s: r2_affine_from_stats(s, a, b))
        cfgs = e["src"].get("test_configs") or list(range(st.shape[0]))
        gate = ""
        if is_anchor:
            # R² is invariant under a common affine map of (y, p), so the raw
            # reading here is the source dump's number recomputed by a different
            # route. Anything but agreement means a drift upstream.
            #
            # **The tolerance is relative, and that is not a loosening.** R² has
            # an unbounded tail: §9.9's two pathological seeds score −67 and
            # −259, where an absolute 1e-6 is a demand for ten significant
            # figures on a float64 sum the two routes accumulate in different
            # orders. Those two rows failed a gate they agreed with to seven
            # digits. Real drift — a different split, a different mask — moves
            # R² by O(0.01) at 0.6 and by O(1) at −259, so the gate still bites
            # everywhere: max(1, |ref|) keeps the absolute strictness for the
            # O(1) rows and scales only for the tail.
            ref = e["src"].get("r2")
            tol = 1e-6 * max(1.0, abs(ref)) if ref is not None else 0.0
            ok = ref is not None and abs(raw - ref) <= tol
            anchor_ok &= bool(ok)
            gate = ("  ✓" if ok else
                    f"  ** GATE FAIL: dump says {ref:+.6f}, "
                    f"differs by {abs(raw - ref):.3g} > {tol:.3g} **")
        print(f"{e['arm']:<10s}{e['seed']:>5d}{raw:>+12.4f}{raw_err:>9.4f}"
              f"{aff:>+12.4f}{aff_err:>9.4f}{a:>9.3f}{b:>9.3f}"
              f"{aff_loo:>+8.3f}{gate}")
        rows.append(dict(
            arm=e["arm"], seed=e["seed"], stem=e["stem"], tag=e["tag"],
            r2_raw=raw, r2_raw_err=raw_err, r2_raw_loo=raw_loo,
            worst_raw=(cfgs[worst_raw] if worst_raw >= 0 else None),
            r2_affine=aff, r2_affine_err=aff_err, r2_affine_loo=aff_loo,
            worst_affine=(cfgs[worst_aff] if worst_aff >= 0 else None),
            affine_a=a, affine_b=b,
            stats=st, fit_sums=fit_stats[e["stem"]].sum(dim=0),
            mu=e["mu"], sigma=e["sigma"],
            src_r2=e["src"].get("r2"), src_lr=e["src"].get("lr"),
            src_epochs=e["src"].get("epochs_run"), src_batch=e["src"].get("batch"),
            src_conv_init=e["src"].get("conv_init_scale"),
            src_diverged=e["src"].get("diverged"),
            src_collapsed=e["src"].get("collapsed"),
        ))

    if is_anchor:
        print("\nanchor gate: " + ("PASSED — the pipeline reproduces §9.9's "
              "numbers from the checkpoints" if anchor_ok else
              "** FAILED — splits, mask or targets have drifted since the "
              "checkpoints were written; read nothing else **"))

    dump = os.path.join(
        OUT_DIR, f"transfer_{TARGET}_src{SRC_BETA}_at{BETA}"
        + (f"_{FILTER.strip('_')}" if FILTER else "") + ".pt"
    )
    torch.save({
        "rows": rows, "group": GROUP, "target": TARGET,
        "src_beta": SRC_BETA, "eval_beta": BETA, "is_anchor": is_anchor,
        "anchor_ok": anchor_ok if is_anchor else None,
        "arms": list(TRANSFER_ARMS), "filter": FILTER,
        "n_configs": N_CONFIGS, "jack_block": JACK_BLOCK,
        "test_configs": te.tolist(), "fit_configs": tr.tolist(),
        "masked": mask is not None, "wall_seconds": time.time() - t_start,
    }, dump)
    print(f"wrote {dump}")
    print(json.dumps({
        "eval_beta": BETA, "src_beta": SRC_BETA, "anchor": is_anchor,
        "anchor_ok": anchor_ok if is_anchor else None,
        "rows": [{"arm": r["arm"], "seed": r["seed"],
                  "raw": round(r["r2_raw"], 4),
                  "affine": round(r["r2_affine"], 4)} for r in rows],
    }))


if __name__ == "__main__":
    # The algebra behind the affine reading, checked before it is used: at
    # (a, b) = (1, 0) the expanded form must be r2_from_stats to the bit, and a
    # perfectly recalibrated constant-offset prediction must score exactly what
    # the uncalibrated one scores after the offset is removed.
    _s = torch.tensor([[10.0, 5.0, 7.0, 4.0, 6.0, 4.5],
                       [12.0, 6.0, 9.0, 5.5, 8.0, 6.0]], dtype=torch.float64)
    assert abs(r2_affine_from_stats(_s, 1.0, 0.0) - r2_from_stats(_s)) < 1e-12
    main()
