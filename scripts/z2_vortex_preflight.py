"""
=========================================================================
The Z₂ vortex candidate's pre-flight — the best classical method at reach.
=========================================================================

``notes/where_attention_can_win.md`` §5 makes this the first gate of every
future A/B, and §9 is the candidate it gates:

  > **measure the best classical method at the architecture's own reach**
  > *before* building any training code.

This is ``scripts/probe_preflight.py``'s sibling for the vortex-geometry task,
and it measures two classical arms rather than one, because this task has two:

* the **M1-free ceiling** — the best *linear filter* over the same reach. One
  weight per offset, fixed for every site and every configuration: that is what
  "input-independent offset weighting" means with the nonlinearity stripped
  away, so ``1 − R²`` is the headroom input-dependent weighting has to work in.
* the **local classical ceiling** — the connected component of the vortex line
  through each site, truncated to the same ball
  (``gelt.vortex_targets.local_component_size``). It is *not* a linear filter
  and it is the strongest thing the baseline family could learn at that reach.
  §5's L1 is about being crushed by a classical method with more reach; this
  arm has exactly the architecture's reach, so beating it is a real claim and
  failing to beat it is a real stop.

Five gates, all fixed below before any number existed:

  1. **closure** — the per-cube vortex parity must vanish identically, or the
     dual-loop picture (and every index in it) is wrong;
  2. **non-degeneracy** — the largest cluster must hold a middling fraction of
     the vortices, or "which one is biggest" is a constant;
  3. **calibration** — V2 must be reproduced by a linear filter to R² ≈ 1, or
     it cannot serve as reading W-A;
  4. **linear headroom** — V1's linear ceiling must leave room;
  5. **local headroom** — and so must the *local BFS* ceiling, which is the one
     that decides whether this is an architecture question at all.

It also re-verifies the structural fact §9.3's design rests on: in Z₂ GELT's
path-averaged transport is a hard vortex mask and the single-path one is not.

Run (on the box holding the cached ensembles; no GPU, no checkpoints):

    python scripts/z2_vortex_preflight.py                    # all four β
    Z2V_BETAS=0.7585 Z2V_N_CONFIGS=200 python scripts/z2_vortex_preflight.py
    Z2V_SMOKE=1 python scripts/z2_vortex_preflight.py        # plumbing, seconds

It samples nothing (except in smoke mode): the ensembles are
``scripts/train_z2_glueball.py``'s cached ones, addressed by the identical
cache key, and a missing cache is an error rather than a sampling run.

Environment overrides (each also ``--name=value`` in argv):
    Z2V_BETAS           comma-separated couplings; default all four.
    Z2V_N_CONFIGS       configurations for the linear arms (default 100).
    Z2V_LOCAL_CONFIGS   configurations for the local-BFS arm (default 24). The
                        BFS is per seed and costs seconds per configuration, so
                        the head-to-head table is computed on this subset and
                        says so; the linear arms are reported on both.
    Z2V_REACH           the architecture's Manhattan reach (default 8 = 4
                        layers × R 2 = 4 layers × K 2). Do not change it
                        without changing the architecture.
    Z2V_CHUNK           configurations per streaming chunk of the normal
                        equations (default 2 — the radius-4 ball design matrix
                        is 387 columns over 27 648 sites per configuration).

Writes ``results/z2_vortex/preflight_b<β>.pt``.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe_common import (  # noqa: E402
    validate_argv,
    accumulate_stats,
    env_flag,
    env_int,
    env_str,
    jackknife,
    r2_from_stats,
    splits,
    standardize,
)
from gelt.lattice import Z2, build_transport_average, l1_ball_offsets, plaquette_tensor  # noqa: E402
from gelt.probe_targets import ball_features, ball_reduce  # noqa: E402
from gelt.vortex_targets import (  # noqa: E402
    BALL_RADIUS,
    build_targets,
    closure_defect,
    cluster_labels,
    cluster_size_field,
    largest_cluster_mask,
    local_size_site_feature,
    vortex_field,
)

# ── The gates, written down before the numbers exist ────────────────────────
# A linear filter that already explains this much of V1 has shown the target to
# be a convolution in disguise: nothing whose only extra ingredient is
# input-dependent offset weighting can separate in the remainder.
LINEAR_CEILING_GATE = 0.90
# The mirror condition on the *local* arm, and the one this candidate actually
# risks failing: if a bounded-reach classical algorithm already solves the task,
# the architectures are competing in its residual and §5's L1 applies.
LOCAL_CEILING_GATE = 0.90
# …and on V2, which must be *exactly* linear or it is not a calibration arm.
CALIBRATION_FLOOR = 0.999
# "Which cluster is the biggest" is only a question while the largest cluster
# holds a middling share of the vortices. Outside this window the target is
# effectively constant and the candidate is dead (§9.4).
BASE_RATE_WINDOW = (0.05, 0.95)
# A target whose per-site variation is this small relative to its mean is
# degenerate — the regression would be fitting round-off.
MIN_RELATIVE_SPREAD = 1e-3

# ── The ensemble — train_z2_glueball.py's, by the same cache key ────────────
gaugegroup = Z2()
L, D, LT = 24, 3, 48
N_CONFIGS_CACHE = 2000
BETA_C = 0.7614
ALL_BETAS = [0.7450, 0.7520, 0.7560, 0.7585]

BETAS = [float(b) for b in env_str("Z2V_BETAS", ",".join(str(b) for b in ALL_BETAS)).split(",")]
N_CONFIGS = env_int("Z2V_N_CONFIGS", 100)
LOCAL_CONFIGS = env_int("Z2V_LOCAL_CONFIGS", 24)
REACH = env_int("Z2V_REACH", 8)
CHUNK = env_int("Z2V_CHUNK", 2)
# Hop caps for the classical arm. ``LAYERS`` of the probe is 4, so the 4-hop row
# is the matched-depth ceiling and the uncapped one (None) is what the first
# version of this pre-flight reported — an arm with unbounded iteration inside
# the ball, which no bounded-depth network can be expected to reach.
LAYERS_MATCHED = env_int("Z2V_LAYERS", 4)
HOP_CAPS = [None if x in ("none", "inf", "") else int(x)
            for x in env_str("Z2V_HOP_CAPS", "2,4,8,none").split(",")]
SMOKE = env_flag("Z2V_SMOKE", False)
VERBOSE = env_flag("Z2V_VERBOSE", True)


MATCHED_ARM = f"+ local size, {LAYERS_MATCHED} hops"


def _jack_block(n_test):
    """Blocked-jackknife block size: ~4 blocks, never fewer than two.

    ``probe_common.jackknife`` refuses a single block, which is right — one
    deleted sample has no spread — so a small test split falls back to block 1
    rather than failing. Production splits give 4 blocks.
    """
    block = max(1, n_test // 4)
    return block if n_test >= 2 * block else 1


def _r2(stats, block):
    """``(R², error)``, with a NaN error where the split cannot be jackknifed.

    A one-configuration test split has no deleted-block spread to report. That
    happens only in smoke mode; printing NaN is the honest form, and inventing
    a zero error bar is not.
    """
    if stats.shape[0] < 2 * block:
        return r2_from_stats(stats), float("nan")
    value, err, _ = jackknife(stats, r2_from_stats, block=block)
    return value, err


def cache_path(beta):
    """``train_z2_glueball.py``'s ensemble cache path — the identical key."""
    return f"datasets/z2_configs_L{L}_Lt{LT}_b{beta}_N{N_CONFIGS_CACHE}.pt"


def load_configs(beta, n):
    """``(n, D, *Λ, 1, 1)`` links. Raises rather than sampling for 24 hours."""
    if SMOKE:
        # Production *geometry*, short chain: Z₂ heat-bath is ~20 ms a sweep at
        # this volume, so a geometry-faithful plumbing pass costs seconds. The
        # statistics are not production — 200 + 20 sweeps against the cache's
        # 500 + 200 — and nothing read off a smoke run is a measurement.
        from gelt.lattice import random_links
        from gelt.sampler import z2_heatbath_sweep

        torch.manual_seed(0)
        U = random_links(L, D, gaugegroup, dtype=torch.float64, Lt=LT)
        for _ in range(200):
            U, _ = z2_heatbath_sweep(U, gaugegroup, beta)
        out = []
        for _ in range(n):
            for _ in range(20):
                U, _ = z2_heatbath_sweep(U, gaugegroup, beta)
            out.append(U.clone())
        return torch.stack(out)
    path = cache_path(beta)
    if not os.path.exists(path):
        raise SystemExit(
            f"Ensemble cache {path} not found. This pre-flight reuses "
            f"train_z2_glueball.py's ensembles and samples nothing itself — run "
            f"`python scripts/train_z2_glueball.py {beta}` on the box that holds "
            f"them first, or set Z2V_SMOKE=1 for a plumbing pass."
        )
    if VERBOSE:
        print(f"loading cached {path} …")
    return torch.load(path, map_location="cpu")[:n].to(torch.float64)


# ── Design matrices ─────────────────────────────────────────────────────────
def _features(v_chunk, kind, reach):
    """``(n_sites, n_features)`` design matrix for one chunk of configurations.

    Built from the **per-plane** vortex indicator, three channels, because that
    is what the networks are fed — not from the site-summed count, which would
    hand the linear arm a reduction it has not earned.

    ``"radial"`` is one column per (plane, shell) out to ``reach``: the
    convolution that is blind to direction, and the M1 probe measured it to
    equal the full-ball filter to four decimals. ``"ball"`` is one column per
    (plane, offset) in the L1-ball of radius :data:`BALL_RADIUS` — the general
    convolution, at the radius the calibration target is defined on. Both get an
    intercept, so the trivial predictor is nested.
    """
    cols = []
    for p in range(v_chunk.shape[1]):
        f = v_chunk[:, p].to(torch.float64)
        if kind == "radial":
            S, _ = ball_reduce(f, reach)
            X = torch.cat([S[:1], S[1:] - S[:-1]], dim=0)  # shells
        else:
            X = ball_features(f, BALL_RADIUS)
        cols.append(X.reshape(X.shape[0], -1).t())
    X = torch.cat(cols, dim=1).double()
    return torch.cat([X, torch.ones(X.shape[0], 1, dtype=torch.float64)], dim=1)


def _augment(X, extra):
    """Append the local-connectivity column to a design matrix, before the
    intercept stays last only by convention — the solve does not care."""
    return torch.cat([X[:, :-1], extra.reshape(-1, 1).double(), X[:, -1:]], dim=1)


def fit_linear_filter(v, y, train_idx, test_idx, kind, reach, extra=None,
                      mask=None, ridge=1e-8):
    """OLS over the chosen features; returns the all-sites and masked readings.

    The normal equations are accumulated in float64 over configuration chunks,
    so the ``(n_sites × n_features)`` design matrix is never materialised whole:
    at the production size the radius-4 ball arm alone would be 2.8 M × 388.

    ``extra`` is an optional per-site column (the local-connectivity feature),
    passed as a full field so the chunking can slice it the same way.

    ``mask`` is an optional per-site bool field. When given, a **second** fit is
    accumulated over the masked rows in the same pass and evaluated on the
    masked rows of the test split — fit and evaluation on the same sites, so it
    is a ceiling rather than a transplanted predictor. That is the reading that
    matters for V1: only ~10% of sites carry a vortex at all, the rest have V1
    exactly 0, and an all-sites R² mostly scores "is there a vortex here", which
    is not the question and is not where an architecture can separate. It is
    also what a masked-loss training arm would optimise, so the two stay
    comparable.

    Returns ``((stats, pred), (stats_masked, pred_masked))``; the second is
    ``None`` when no mask is given.
    """
    acc = {"all": None, "masked": None} if mask is not None else {"all": None}

    def _accumulate(key, X, t):
        if acc[key] is None:
            n_feat = X.shape[1]
            acc[key] = [
                torch.zeros(n_feat, n_feat, dtype=torch.float64),
                torch.zeros(n_feat, dtype=torch.float64),
            ]
        acc[key][0] += X.t() @ X
        acc[key][1] += X.t() @ t

    for lo in range(0, len(train_idx), CHUNK):
        idx = train_idx[lo : lo + CHUNK]
        X = _features(v[idx], kind, reach)
        if extra is not None:
            X = _augment(X, extra[idx])
        t = y[idx].reshape(-1).double()
        _accumulate("all", X, t)
        if mask is not None:
            m = mask[idx].reshape(-1)
            _accumulate("masked", X[m], t[m])

    betas = {}
    for key, (XtX, Xty) in acc.items():
        # A ridge proportional to the trace keeps the solve well posed when two
        # offsets carry near-identical information; at 1e-8 it is conditioning,
        # not regularisation, and it cannot flatter the fit.
        n_feat = XtX.shape[0]
        XtX = XtX + ridge * XtX.diagonal().mean() * torch.eye(
            n_feat, dtype=torch.float64
        )
        betas[key] = torch.linalg.solve(XtX, Xty)

    out = {key: ([], []) for key in betas}
    for i in test_idx.tolist():
        idx = torch.tensor([i])
        X = _features(v[idx], kind, reach)
        if extra is not None:
            X = _augment(X, extra[idx])
        t = y[idx].reshape(-1).double()
        pred = X @ betas["all"]
        out["all"][0].append(accumulate_stats(t, pred))
        out["all"][1].append(pred)
        if mask is not None:
            m = mask[idx].reshape(-1)
            pred_m = X[m] @ betas["masked"]
            out["masked"][0].append(accumulate_stats(t[m], pred_m))
            out["masked"][1].append(pred_m)
    packed = {k: (torch.stack(v_[0]), torch.cat(v_[1])) for k, v_ in out.items()}
    return packed["all"], packed.get("masked")


def auc(scores, labels):
    """Rank AUC with exact tie handling — the binary reading of the same fit."""
    scores = scores.reshape(-1).double()
    labels = labels.reshape(-1).to(torch.bool)
    n_pos = int(labels.sum())
    n_neg = labels.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores, stable=True)
    s, y = scores[order], labels[order]
    ranks = torch.empty(s.numel(), dtype=torch.float64)
    i = 0
    while i < s.numel():
        j = i
        while j + 1 < s.numel() and s[j + 1] == s[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)).item()


# ── The structural check §9.3's design rests on ─────────────────────────────
def transport_mask_check(U_one):
    """Is the path-averaged transport a vortex mask, and the single-path one not?

    Returns ``(mask_rate, single_is_trivial, p_neg)``. In Z₂ the adjoint action
    of a transport is ``T² W``, so ``T² = 0`` switches a neighbour off entirely
    and ``T² = 1`` leaves it alone. The average over the two shortest paths of a
    diagonal offset is ``(1 + P_enclosed)/2``, so the mask must fire at exactly
    the vortex density — which is the check.
    """
    U = U_one.unsqueeze(0)
    P = plaquette_tensor(U, gaugegroup)
    p_neg = (P[..., 0, 0].real < 0).double().mean().item()
    T_avg = build_transport_average(U, 2, gaugegroup, mode="average")
    T_one = build_transport_average(U, 2, gaugegroup, mode="single")
    offs = l1_ball_offsets(D, 2)
    diag = [i for i, dx in enumerate(offs) if sum(int(c) != 0 for c in dx) == 2]
    mask_rate = (T_avg[:, diag] == 0).double().mean().item()
    single_trivial = bool(((T_one.abs() - 1).abs() < 1e-12).all())
    return mask_rate, single_trivial, p_neg


# ── One coupling ────────────────────────────────────────────────────────────
def run_beta(beta):
    print("\n" + "=" * 78)
    print(f"β = {beta}   (β_c − β = {BETA_C - beta:+.4f})")
    print("=" * 78)
    configs = load_configs(beta, N_CONFIGS)
    n = configs.shape[0]
    v = vortex_field(configs, gaugegroup)
    lattice = tuple(v.shape[2:])
    print(f"  {n} configurations, lattice {lattice}, "
          f"{v.shape[1]} planes → {v[0].numel()} plaquettes each")

    row = {"beta": beta, "n_configs": n, "lattice": lattice}
    verdict = {}

    # G1 — closure. Hard: everything downstream indexes the same cubes.
    defect = int(closure_defect(v).max())
    verdict["closure"] = defect == 0
    print(f"\n  G1  closure defect (max over cubes) ........ {defect}"
          f"          {'PASS' if defect == 0 else 'FAIL'}")
    if defect:
        raise SystemExit(
            "the per-cube vortex parity is nonzero: these are not Z₂ gauge "
            "configurations, or gelt.vortex_targets.CUBE_FACES is wrong."
        )

    # G2 — the vortex gas and the cluster competition.
    lab = cluster_labels(v)
    sizes = cluster_size_field(v, lab)
    biggest = largest_cluster_mask(v, lab)
    p_neg = v.double().mean().item()
    per_cfg_base = [
        (biggest[b].sum() / v[b].sum().clamp_min(1)).item() for b in range(n)
    ]
    base_rate = sum(per_cfg_base) / len(per_cfg_base)
    n_clusters = [int(torch.unique(lab[b][v[b]]).numel()) for b in range(n)]
    lo, hi = BASE_RATE_WINDOW
    verdict["base_rate"] = lo <= base_rate <= hi
    print(f"  G2  vortex density p_neg .................. {p_neg:.4f}")
    print(f"      vortex plaquettes per configuration ... "
          f"{int(v.sum()) // n} in {sum(n_clusters) // n} clusters")
    print(f"      largest-cluster share (mean over cfg) . {base_rate:.3f}"
          f"     {'PASS' if verdict['base_rate'] else 'FAIL'}"
          f"  (window {lo}–{hi})")
    print(f"      largest cluster (median over cfg) ..... "
          f"{int(torch.median(torch.tensor([float(sizes[b].max()) for b in range(n)])))}")
    row.update(p_neg=p_neg, base_rate=base_rate, per_cfg_base=per_cfg_base,
               n_clusters=n_clusters)

    # G3 — the transport asymmetry the 2×2 design rests on (§9.3).
    mask_rate, single_trivial, p_neg_one = transport_mask_check(configs[0])
    agrees = abs(mask_rate - p_neg_one) < 1e-9
    print(f"\n  G3  path-averaged transport: masked diagonal offsets "
          f"{mask_rate:.4f}")
    print(f"      vortex density on the same configuration ........... "
          f"{p_neg_one:.4f}   {'match' if agrees else 'MISMATCH'}")
    print(f"      single-path transport is ±1 (adjoint action trivial) "
          f"{single_trivial}")
    verdict["transport_mask"] = agrees and single_trivial
    row.update(mask_rate=mask_rate, single_trivial=single_trivial)

    # Targets and splits.
    targets = build_targets(v, lab=lab)
    tr, va, te = splits(n)
    print(f"\n  splits: train {len(tr)}  val {len(va)}  test {len(te)} configurations")
    jack_block = _jack_block(len(te))

    # G4 — V2, the calibration arm: a linear filter must reproduce it exactly.
    y2_raw = targets["V2"]
    y2, _, _ = standardize(y2_raw, tr)
    (stats, _), _ = fit_linear_filter(v, y2, tr, te, "ball", REACH)
    r2_cal, err_cal = _r2(stats, jack_block)
    verdict["calibration"] = r2_cal >= CALIBRATION_FLOOR
    print(f"\n  G4  V2 (calibration), linear ball filter at radius {BALL_RADIUS}:")
    print(f"        R² = {r2_cal:+.6f} ± {err_cal:.6f}"
          f"     {'PASS' if verdict['calibration'] else 'FAIL'}"
          f"  (needs ≥ {CALIBRATION_FLOOR})")
    row.update(v2_r2=(r2_cal, err_cal))

    # G5 — V1, the primary: the two classical ceilings.
    #
    # Two R² per arm, and **the masked one is the reading**. Over half the sites
    # carry no vortex, their V1 is exactly 0, and a linear filter gets them from
    # the local count — so an all-sites R² mostly scores "is there a vortex
    # here", which is not the question and is not where an architecture can
    # separate. The masked column restricts to sites that touch a vortex, i.e.
    # to the sites where "how big is its cluster" is actually asked.
    y1_raw = targets["V1"]
    spread = (y1_raw.std() / y1_raw.abs().mean().clamp_min(1e-30)).item()
    y1, _, _ = standardize(y1_raw, tr)
    site_has_vortex = v.any(dim=1)
    site_in_biggest = biggest.any(dim=1)

    print(f"\n  G5  V1 (primary): std/|mean| = {spread:.3e}   "
          f"sites carrying a vortex: {site_has_vortex.double().mean():.3f}")
    print(f"        {'arm':32s} {'R² all sites':>16s} {'R² vortex sites':>19s}"
          f" {'AUC':>7s}")
    lin = {}
    for kind, label in (("radial", f"linear radial, reach {REACH}"),
                        ("ball", f"linear ball,  radius {BALL_RADIUS}")):
        (stats, _), (stats_m, pred_m) = fit_linear_filter(
            v, y1, tr, te, kind, REACH, mask=site_has_vortex
        )
        r2, err = _r2(stats, jack_block)
        r2m, errm = _r2(stats_m, jack_block)
        sel = site_has_vortex[te].reshape(-1)
        a = auc(pred_m, site_in_biggest[te].reshape(-1)[sel])
        lin[kind] = (r2, err, r2m, errm, a)
        print(f"        {label:32s} {r2:+8.4f} ± {err:.4f} {r2m:+11.4f} ± {errm:.4f}"
              f" {a:7.3f}")
    row["v1_linear"] = lin

    # The local classical arm, on its own (smaller) sub-ensemble: the BFS is per
    # seed and costs seconds per configuration, so it is measured on a subset and
    # the linear arm is refitted there too, to keep the comparison paired.
    n_loc = min(LOCAL_CONFIGS, n)
    tr_l, va_l, te_l = splits(n_loc)
    jack_l = _jack_block(len(te_l))
    print(f"\n      local classical arm, first {n_loc} configurations "
          f"(train {len(tr_l)} / test {len(te_l)}), BFS at reach {REACH}:")
    print(f"      **hop caps** {HOP_CAPS} — a k-layer network gets k rounds of"
          f" message passing, so the uncapped arm is not in its function class.")
    y1_loc, _, _ = standardize(y1_raw[:n_loc], tr_l)
    arms = [("linear radial (same subset)", None)]
    for cap in HOP_CAPS:
        label = "unlimited hops" if cap is None else f"+ local size, {cap} hops"
        arms.append((label, local_size_site_feature(v[:n_loc], REACH, max_hops=cap)))
    local_rows = {}
    for name, extra in arms:
        (stats, _), (stats_m, pred_m) = fit_linear_filter(
            v[:n_loc], y1_loc, tr_l, te_l, "radial", REACH, extra=extra,
            mask=site_has_vortex[:n_loc],
        )
        r2, err = _r2(stats, jack_l)
        r2m, errm = _r2(stats_m, jack_l)
        sel = site_has_vortex[:n_loc][te_l].reshape(-1)
        a = auc(pred_m, site_in_biggest[:n_loc][te_l].reshape(-1)[sel])
        local_rows[name] = (r2, err, r2m, errm, a)
        print(f"        {name:32s} {r2:+8.4f} ± {err:.4f} {r2m:+11.4f} ± {errm:.4f}"
              f" {a:7.3f}")
    row["v1_local"] = local_rows

    # The gates read the *masked* numbers, for the reason above.
    r2_lin = lin["radial"][2]
    r2_loc = local_rows[MATCHED_ARM][2]
    r2_lin_sub = local_rows["linear radial (same subset)"][2]
    print(f"\n      the gate below reads the **matched-depth** arm "
          f"({MATCHED_ARM}); the uncapped row is context, not a ceiling.")
    verdict["linear_headroom"] = (r2_lin <= LINEAR_CEILING_GATE) and (
        spread >= MIN_RELATIVE_SPREAD
    )
    verdict["local_headroom"] = r2_loc <= LOCAL_CEILING_GATE
    print(f"\n      headroom over the M1-free linear filter ... {1 - r2_lin:+.4f}"
          f"   {'PASS' if verdict['linear_headroom'] else 'FAIL'}"
          f"  (linear R² ≤ {LINEAR_CEILING_GATE})")
    print(f"      headroom over the local classical arm ..... {1 - r2_loc:+.4f}"
          f"   {'PASS' if verdict['local_headroom'] else 'FAIL'}"
          f"  (local R² ≤ {LOCAL_CEILING_GATE})")
    print(f"      what connectivity is worth over density ... "
          f"{r2_loc - r2_lin_sub:+.4f}")

    row["verdict"] = verdict
    row["spread"] = spread
    return row


def main():
    validate_argv()
    torch.manual_seed(0)
    print("=" * 78)
    print("Z₂ vortex geometry — pre-flight: the best classical method at reach")
    print("notes/where_attention_can_win.md §9.4")
    print("=" * 78)
    print(f"reach = Manhattan {REACH} (4 layers × R 2 = 4 layers × K 2) | "
          f"calibration radius {BALL_RADIUS} | {N_CONFIGS} configs "
          f"({LOCAL_CONFIGS} for the local arm)" + ("  [SMOKE]" if SMOKE else ""))

    rows = [run_beta(beta) for beta in BETAS]

    print("\n" + "=" * 78)
    print("R² columns are the masked reading — vortex-carrying sites only.")
    print("'routing' is R²(local) − R²(linear): how much of the task is "
          "connectivity rather than density.")
    print(f"{'β':>8s} {'p_neg':>7s} {'base':>6s} {'R² lin':>17s} {'R² local':>17s}"
          f" {'headroom':>9s} {'routing':>8s}  gates")
    for r in rows:
        failed = [k for k, ok in r["verdict"].items() if not ok]
        r2_lin, e_lin = r["v1_linear"]["radial"][2], r["v1_linear"]["radial"][3]
        loc = r["v1_local"]["+ local component size"]
        r2_loc, e_loc = loc[2], loc[3]
        print(f"{r['beta']:8.4f} {r['p_neg']:7.4f} {r['base_rate']:6.3f}"
              f" {r2_lin:+9.4f} ± {e_lin:.4f} {r2_loc:+9.4f} ± {e_loc:.4f}"
              f" {1 - r2_loc:+9.4f} {r2_loc - r2_lin:+8.4f}"
              f"  {'all pass' if not failed else 'FAIL: ' + ','.join(failed)}")

    os.makedirs("results/z2_vortex", exist_ok=True)
    for r in rows:
        out = f"results/z2_vortex/preflight_b{r['beta']}.pt"
        torch.save({"row": r,
                    "gates": dict(linear_ceiling=LINEAR_CEILING_GATE,
                                  local_ceiling=LOCAL_CEILING_GATE,
                                  calibration_floor=CALIBRATION_FLOOR,
                                  base_rate_window=BASE_RATE_WINDOW,
                                  min_spread=MIN_RELATIVE_SPREAD),
                    "meta": dict(reach=REACH, ball_radius=BALL_RADIUS,
                                 n_configs=N_CONFIGS, local_configs=LOCAL_CONFIGS,
                                 L=L, Lt=LT, smoke=SMOKE)}, out)
        print(f"wrote {out}")

    failed = [r["beta"] for r in rows if not all(r["verdict"].values())]
    print("\nVERDICT: " + (
        "every coupling clears the pre-flight — the candidate is a go, and the "
        "β ladder of §9.6 can be fixed from the table above"
        if not failed else
        f"couplings {failed} FAIL a gate; read §9.8 before building anything"))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
