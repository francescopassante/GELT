"""Exact ground truth for the Z₂ mass gap, from the dual Ising model.

The question this closes
------------------------
`notes/attention_as_operator.md` §6.1.2 — and the paper, twice — states the
central number as a *candidate* rather than a result:

    the trained attention field reads ξ ~21% above the classical smeared
    operator (slope 1.21); the classical operator is not ground truth, it is
    one more contaminated operator, and contamination biases ξ low; so the
    trained ξ is the candidate for the true gap, **pending an uncontaminated
    reference**.

The note prices that reference at an L = 32 gauge run: new ensembles *and* new
checkpoints at all five β, ≈40 h. This script buys it for ~30 minutes of GPU,
and buys a *better* one, by measuring the same mass gap in the exactly dual
description.

Why the dual reference is not just another operator
---------------------------------------------------
Wegner's duality maps 3D Z₂ gauge theory at β onto the 3D Ising model at
β* = −½ ln tanh β, plaquettes onto dual bonds, and the confined phase onto the
**broken** phase. Three consequences, all load-bearing:

1. The mass gap is *the same number*, not a proxy — the 0⁺⁺ glueball is the
   lightest state of the Z₂-even sector, which is the lightest Ising excitation.
2. In the broken phase the Ising **order parameter** σ interpolates that state
   with near-unit overlap. On the gauge side σ is the 't Hooft disorder
   operator: non-local in the links, and therefore not a member of any
   smeared-loop variational basis that the classical measurement could have
   chosen. The reference genuinely comes from outside the gauge side's toolbox,
   which is what "uncontaminated" has to mean here.
3. It is ~10⁻⁴ the cost, so statistics and volumes out of reach of the gauge
   ensembles are cheap — including the L = 48 run that answers §7's
   finite-volume question with no new gauge sweeps.

What is measured
----------------
Per β, on two volumes:

* **matched** (48×24², the dual of the gauge torus) — the accuracy yardstick.
  The gauge operators were measuring *this* system, so this is what their ξ
  should be compared against; comparing them to an infinite-volume number would
  charge them for finite volume as if it were contamination.
* **large** (96×48²) — the same physics with finite volume pushed back by 2×,
  which separates "the classical operator is contaminated" from "the box is
  squeezing ξ", and settles whether the non-monotonic ξ(0.7585) > ξ(0.760)
  excursion is physics, volume, or a fluctuation of the gauge ensembles.

Two dual operators are fitted, through *exactly* the gauge-side analysis code
(`connected_correlator`, `fit_cosh_correlator`, same Δ ∈ [2,8] window, same
blocked jackknife), so nothing in the comparison differs except the operator:

* ``m``   — the wall magnetisation: the order parameter, best possible overlap;
* ``e_t`` — the temporal-bond wall energy: the *literal* dual of the spatial
  plaquette the glueball operator is built from.

They must agree on the mass and disagree on A₀. That is the internal check.

The external check, and why it is decisive
------------------------------------------
Before any of this is believed, the duality itself is verified with a
prediction that has no free parameter:

    ⟨P⟩(β) = tanh β + [1 − ⟨s_i s_j⟩(β*)] / sinh 2β

with ⟨P⟩ measured on the production gauge ensembles already in `datasets/` and
⟨ss⟩ measured here. Nothing is fitted; the two sides are simulated by unrelated
code. (`tests/test_ising.py` runs the same check at β = 0.5 on a small lattice.)

Pre-registered outcomes
-----------------------
Let d_op = ξ_op − ξ_dual at matched volume.

* **|d_class| > |d_trained| ≈ 0** → the headline. The learned attention field is
  the more accurate measurement of the mass gap, the A₀-implied direction of the
  bias is confirmed against an external standard, and §6.1.2's candidate becomes
  a result: *training removes excited-state contamination, and the resulting
  operator is right where the standard one is low.*
* **d_trained ≈ d_class ≈ 0, dual ξ between them** → both operators are fine and
  the 21% gap is a window/fit systematic. Report and drop the accuracy claim.
* **|d_trained| > |d_class|, dual ≈ classical** → the overshoot is a systematic
  of the attention correlator, not de-contamination. §6.1.2's reading is wrong
  and must be retracted; the structural claim (Pearson 0.9966) is untouched.

And independently, on the excursion: if the large-volume dual is monotonic while
the matched-volume dual reproduces ξ(0.7585) > ξ(0.760), the excursion is finite
volume. If *both* dual volumes are monotonic, the excursion belongs to the gauge
ensembles themselves — a shared fluctuation of the configurations both operators
were measured on — and the paper's "survives a non-monotonic excursion" argument
weakens to "agrees configuration by configuration", which is still true and
still worth something, but is a different sentence.

Run:
    python scripts/dual_ground_truth.py            # full run, ~30 min on a V100
    DGT_SMOKE=1 python scripts/dual_ground_truth.py  # 2 min, tiny lattices

Environment overrides: ``DGT_REPLICAS``, ``DGT_MEASURE``, ``DGT_SKIP``,
``DGT_THERM``, ``DGT_VOLUMES`` (``matched``/``large``/``both``), ``DGT_SMOKE``.

Writes ``results/dual/dual_ground_truth.{png,pt}``.
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from gelt.glueball import connected_correlator, fit_cosh_correlator
from gelt.ising import (
    GAUGE_BETA_C,
    ISING_BETA_C,
    dual_beta,
    ising_measure,
    predicted_plaquette,
)

# 3D Ising correlation-length exponent, Kos–Poland–Simmons-Duffin–Vichi
# conformal bootstrap (2016): ν = 0.629971(4). Quoted to six digits because the
# scaling test below is a *prediction*, not a fit, and the input has to be
# better determined than the thing it is testing.
NU_ISING = 0.629971
from gelt.lattice import Z2, plaquette_tensor
from gelt.sampler import integrated_autocorrelation_time

RESULTS = "results/dual"
os.makedirs(RESULTS, exist_ok=True)

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# --------------------------------------------------------------------------
# Tunables. The five β are the gauge scan's, unchanged — the whole point is to
# land on exactly the couplings the gauge measurements were made at.
# --------------------------------------------------------------------------
BETAS = [0.7450, 0.7520, 0.7560, 0.7585, 0.7600]

# Λ = (Lt, L, L), time on axis 0, matching the gauge convention.
VOLUMES = {
    "matched": (48, 24, 24),  # dual of the production gauge torus
    "large": (96, 48, 48),  # finite volume pushed back 2×
}

# Same fit window and jackknife block as z2_attention_correlator.py, so the
# dual number and the gauge numbers differ only in the operator.
FIT_WINDOW = (2, 8)
# A second, later window. If ξ from [2,8] and from [6,16] agree on an operator
# with A₀ ≈ 1, the window is not what separates the gauge-side operators.
LATE_WINDOW = (6, 16)
SMOKE = os.environ.get("DGT_SMOKE", "0") == "1"


def _env(name: str, full: int, smoke: int) -> int:
    """Env override, else the smoke or production default.

    Smoke supplies *defaults*, so an explicit override still wins — otherwise
    `DGT_SMOKE=1 DGT_THERM=2000` would silently run 100 sweeps.
    """
    return int(os.environ.get(name, smoke if SMOKE else full))


if SMOKE:
    VOLUMES = {"matched": (24, 12, 12), "large": (32, 16, 16)}

N_REPLICAS = _env("DGT_REPLICAS", 32, 8)
N_MEASURE = _env("DGT_MEASURE", 500, 40)
# n_skip ≳ 2·τ_int. Local updates decorrelate like ξ^z with z ≈ 2.02, and ξ
# reaches ~6 here, so ~40 sweeps is the scale; τ_int is measured and reported
# per β so the choice can be audited rather than trusted.
N_SKIP = _env("DGT_SKIP", 50, 5)
# Fixed from the cold-vs-hot start test: at both extreme β on the matched
# volume the two starts converge on ⟨ss⟩ by ~800 sweeps (they approach the
# plateau from opposite sides, which is the point of running both), so 1500 is
# ~2× the measured requirement. `DGT_THERM_CHECK=1` re-runs that test.
N_THERM = _env("DGT_THERM", 1500, 100)
JACK_BLOCK = _env("DGT_JACK", 40, 10)
THERM_CHECK = os.environ.get("DGT_THERM_CHECK", "0") == "1"

_want = os.environ.get("DGT_VOLUMES", "both")
if _want != "both":
    VOLUMES = {_want: VOLUMES[_want]}

# Gauge-side results to compare against: the diagonal run of §6.1 (1200 unseen
# configurations per β), which is where the paper's table comes from.
GAUGE_SIDE = "results/attention/z2_attention_correlator_diag_R6.pt"
GAUGE_ENSEMBLE = "datasets/z2_configs_L24_Lt48_b{b}_N2000.pt"

# Okabe–Ito subset; CVD separation validated (worst adjacent ΔE 11.0 deutan).
# Ground truth is deliberately *not* one of these — it is neutral ink, because
# it is the reference the coloured series are measured against, not a peer.
COL = {
    "classical": "#0072B2",
    "trained": "#D55E00",
    "random": "#009E73",
    "dual_large": "#E69F00",
}
INK = "#1a1a1a"


# --------------------------------------------------------------------------
# analysis — deliberately the gauge side's own code path
# --------------------------------------------------------------------------
def _fit(obar, window):
    """(m, A₀) from a cosh fit to the connected correlator of ``obar`` (B, Nt).

    Same estimator as `z2_attention_correlator._fit`: A₀ is the
    Morningstar–Peardon ground-state overlap fraction A(1+e^{−m·Nt})/C(0).
    """
    C = connected_correlator(obar)
    nt = obar.shape[1]
    if not torch.isfinite(C[: window[1] + 1]).all() or C[0] <= 0:
        return float("nan"), float("nan")
    m, A, _chi2 = fit_cosh_correlator(C, *window)
    if not np.isfinite(m) or m <= 0:
        return float("nan"), float("nan")
    return float(m), float(A * (1.0 + math.exp(-m * nt)) / C[0].item())


def _jack(obar, window, block=JACK_BLOCK):
    """Blocked-jackknife (m, A₀) with errors.

    Blocks, not single deletions, and — because `ising_measure` returns its
    measurements chain-major — a contiguous block is a contiguous stretch of one
    Markov chain, which is what the block is supposed to absorb. ``n_measure``
    is a multiple of the block so no block straddles two chains.
    """
    obar = obar.double()
    B = obar.shape[0]
    m0, a0 = _fit(obar, window)
    reps = []
    for i in range(0, B, block):
        keep = torch.ones(B, dtype=torch.bool)
        keep[i : i + block] = False
        reps.append(_fit(obar[keep], window))
    reps = torch.tensor(reps, dtype=torch.float64)
    reps = reps[torch.isfinite(reps).all(dim=1)]
    n = reps.shape[0]
    err = (((n - 1) / n) * (reps - reps.mean(0)).pow(2).sum(0)).sqrt()
    return m0, float(err[0]), a0, float(err[1])


def _xi(m, m_err):
    """ξ = 1/m with propagated error."""
    if not np.isfinite(m) or m <= 0:
        return float("nan"), float("nan")
    return 1.0 / m, m_err / m**2


# --------------------------------------------------------------------------
# the parameter-free duality check
# --------------------------------------------------------------------------
def gauge_plaquette(beta):
    """⟨P⟩ measured on the production gauge ensemble, or None if not cached."""
    tag = ("%g" % beta).rstrip("0").rstrip(".") if beta != 0.76 else "0.76"
    path = GAUGE_ENSEMBLE.format(b=tag)
    if not os.path.exists(path):
        return None, None
    cfg = torch.load(path, map_location="cpu")
    g = Z2()
    per_cfg = []
    for i in range(0, cfg.shape[0], 100):
        P = plaquette_tensor(cfg[i : i + 100], g)[..., 0, 0].real
        per_cfg.append(P.mean(dim=tuple(range(1, P.dim()))))
    per_cfg = torch.cat(per_cfg).double()
    # τ-corrected, like the dual side: the gauge chains reach τ_int ≈ 6–8 near
    # β_c (notes/attention_as_operator.md §6.1), so std/√N understates the error
    # by a factor of ~4 exactly where the duality check is most interesting.
    tau = float(integrated_autocorrelation_time(per_cfg)[1])
    n_eff = per_cfg.numel() / max(1.0, 2.0 * tau)
    return float(per_cfg.mean()), float(per_cfg.std() / n_eff**0.5)


# --------------------------------------------------------------------------
# gauge-side numbers to compare against
# --------------------------------------------------------------------------
def load_gauge_side():
    """{beta: {'classical': (xi, err, A0), 'trained': …, 'random': …}}."""
    if not os.path.exists(GAUGE_SIDE):
        print(f"  ! {GAUGE_SIDE} missing — dual numbers will stand alone")
        return {}
    d = torch.load(GAUGE_SIDE, map_location="cpu", weights_only=False)
    out = {}
    for row in d["rows"]:
        cell = {}
        c = row["classical"]
        cell["classical"] = (*_xi(c["m"], c["m_err"]), c["A0"], c["A0_err"])
        for name, net in row["nets"].items():
            a = net["attention_single"]
            key = "random" if name == "random" else "trained"
            cell[key] = (
                *_xi(a["m"], a.get("m_err", float("nan"))),
                a.get("A0", float("nan")),
                a.get("A0_err", float("nan")),
            )
        out[round(row["beta"], 4)] = cell
    return out


# --------------------------------------------------------------------------
# Sampling adequacy. The 2026-08-14 run failed here and it is worth being
# explicit about why, because the failure was invisible in every observable
# except the duality check.
#
# In the *broken* phase the slowest mode is not the critical one. Tunnelling
# between the two magnetisation sectors has a barrier set by the interface
# area, so its τ is exponential in the volume rather than ξ^z: measured on
# 48×24² at β = 0.760, ξ² ≈ 76 sweeps against τ_int(M) = 1575. Thermalising for
# 1500 sweeps was therefore *one* autocorrelation time, the cold start had not
# relaxed, ⟨ss⟩ read high, and the duality check went to −18σ at exactly the
# three β where τ(M) blew up. The tell in the ξ table was that the matched
# volume came out *above* the large volume at two β — incoherent, since finite
# volume can only squeeze ξ.
#
# So the run gates on τ_int of the magnetisation and escalates itself rather
# than reporting a cell it cannot support. τ is measured in units of
# measurements; ×n_skip converts to sweeps.
THERM_TAU = 20.0  # require n_therm ≥ 20·τ(M)
SKIP_TAU = 2.0  # require n_skip  ≥  2·τ(M)
MAX_ESCALATIONS = int(os.environ.get("DGT_MAX_ESCALATE", 3))


def _sampling_verdict(tau_mag_sweeps, n_therm, n_skip):
    """(ok, needed_therm, needed_skip) against the τ(M) gate."""
    need_t = int(math.ceil(THERM_TAU * tau_mag_sweeps))
    need_s = int(math.ceil(SKIP_TAU * tau_mag_sweeps))
    return (n_therm >= need_t and n_skip >= need_s), need_t, need_s


def _run_once(beta, shape, n_therm, n_skip, n_measure, tag=0):
    bs = dual_beta(beta)
    out = ising_measure(
        bs, shape,
        n_replicas=N_REPLICAS, n_measure=n_measure,
        n_therm=n_therm, n_skip=n_skip,
        device=device,
        seed=abs(hash((round(beta, 6), shape, tag))) % (2**31),
        ordered_start=True, progress=True,
    )
    mg = out["magnetisation"].reshape(N_REPLICAS, n_measure)
    e = out["bond_energy"].reshape(N_REPLICAS, n_measure)
    taus = {}
    for key, series in (("tau_int", e), ("tau_int_mag", mg)):
        taus[key] = float(
            np.mean([
                float(integrated_autocorrelation_time(series[r])[1])
                for r in range(N_REPLICAS)
            ])
        )
    return out, mg, taus


def measure(beta, shape, label):
    """Run the dual Ising at β*, fit both operators, escalate if undersampled.

    The escalation loop is the point: a first pass at the configured settings
    measures τ(M), and if the gate fails the *same* β is re-run with n_therm and
    n_skip taken from the measured τ (capped by MAX_ESCALATIONS). Cost therefore
    scales with the β that need it rather than with the worst one, which matters
    because τ(M) spans 25 → 1575 sweeps across this scan.
    """
    n_therm, n_skip, n_measure = N_THERM, N_SKIP, N_MEASURE
    history = []
    for attempt in range(MAX_ESCALATIONS + 1):
        out, mg, taus = _run_once(beta, shape, n_therm, n_skip, n_measure, attempt)
        tau_sw = taus["tau_int_mag"] * n_skip
        ok, need_t, need_s = _sampling_verdict(tau_sw, n_therm, n_skip)
        history.append(
            {"n_therm": n_therm, "n_skip": n_skip, "n_measure": n_measure,
             "tau_mag_sweeps": tau_sw, "ok": ok}
        )
        if ok or attempt == MAX_ESCALATIONS:
            break
        # Keep the chain length bounded: buying decorrelation is worth more than
        # buying samples once the samples are correlated anyway.
        n_therm, n_skip = need_t, need_s
        n_measure = max(100, N_MEASURE // 2 ** (attempt + 1))
        print(
            f"    ↑ escalating β={beta}: τ(M)={tau_sw:.0f} sweeps →"
            f" n_therm={n_therm} n_skip={n_skip} n_measure={n_measure}"
        )

    res = {"beta": beta, "beta_star": dual_beta(beta), "shape": shape,
           "label": label, "sampling": history, "sampling_ok": history[-1]["ok"],
           "n_therm": n_therm, "n_skip": n_skip, "n_measure": n_measure}
    res.update(taus)
    res["tau_mag_sweeps"] = history[-1]["tau_mag_sweeps"]

    # Measurements that straddle a magnetisation sign flip are the ones the
    # sign-fixed σ correlator cannot represent: the wall array is flipped
    # wholesale by the global sign, so a configuration caught mid-tunnelling
    # contributes a spurious large fluctuation. Dropping the neighbours of every
    # flip is cheap; the effect of dropping them is *reported* rather than
    # assumed negligible.
    flip = torch.sign(mg[:, 1:]) != torch.sign(mg[:, :-1])
    res["tunnel_rate"] = float(flip.double().mean())
    keep = torch.ones_like(mg, dtype=torch.bool)
    keep[:, 1:] &= ~flip
    keep[:, :-1] &= ~flip
    keep = keep.reshape(-1)

    for op in ("m", "e_t"):
        m, me, a0, a0e = _jack(out[op], FIT_WINDOW)
        xi, xie = _xi(m, me)
        res[op] = {"m": m, "m_err": me, "xi": xi, "xi_err": xie,
                   "A0": a0, "A0_err": a0e}
        m2, me2, _, _ = _jack(out[op], LATE_WINDOW)
        res[op]["xi_late"], res[op]["xi_late_err"] = _xi(m2, me2)
        if keep.sum() > JACK_BLOCK * 4 and keep.sum() < keep.numel():
            mc, mce, _, _ = _jack(out[op][keep], FIT_WINDOW)
            res[op]["xi_notunnel"], res[op]["xi_notunnel_err"] = _xi(mc, mce)
        C = connected_correlator(out[op].double())
        res[op]["profile"] = (C[:10] / C[0]).tolist()

    res["bond_energy"] = float(out["bond_energy"].mean())
    # τ-corrected: the naive std/√N ignores that consecutive measurements are
    # correlated, and quoting it turned a ~0.6% systematic into "−18σ".
    n_eff = out["bond_energy"].numel() / max(1.0, 2.0 * res["tau_int"])
    res["bond_energy_err"] = float(out["bond_energy"].std() / n_eff**0.5)
    res["pred_plaquette"] = predicted_plaquette(beta, res["bond_energy"])
    res["pred_plaquette_err"] = res["bond_energy_err"] / math.sinh(2 * beta)
    return res


def thermalisation_check(shape):
    """Cold vs hot start on ⟨ss⟩ — they must converge, from opposite sides.

    The one systematic that would silently corrupt everything downstream: an
    ordered start biases ⟨ss⟩ *high* and a random start biases it *low*, so if
    the two agree, thermalisation is established rather than assumed. This is
    how N_THERM was fixed; it is kept runnable so the choice can be re-audited
    on a new volume instead of inherited.
    """
    from gelt.ising import heatbath_sweep, mean_bond_energy, random_spins

    print(f"\nTHERMALISATION CHECK on {shape} — cold vs hot start")
    for beta in (BETAS[0], BETAS[-1]):
        bs = dual_beta(beta)
        torch.manual_seed(0)
        cold = random_spins(shape, 8, device=device, ordered=True)
        hot = random_spins(shape, 8, device=device, ordered=False)
        marks = {}
        for n in range(N_THERM + 1):
            if n % max(1, N_THERM // 6) == 0:
                marks[n] = (
                    float(mean_bond_energy(cold).mean()),
                    float(mean_bond_energy(hot).mean()),
                )
            cold, hot = heatbath_sweep(cold, bs), heatbath_sweep(hot, bs)
        print(f"  β={beta}  β*={bs:.6f}")
        for n, (c, h) in marks.items():
            print(f"    sweep {n:>5}  cold {c:.5f}  hot {h:.5f}  gap {c - h:+.5f}")


def main():
    print(f"device: {device} | 3D Z₂ ↔ 3D Ising, β_c = {GAUGE_BETA_C:.9f}")
    print(
        f"replicas={N_REPLICAS} measure={N_MEASURE} skip={N_SKIP} therm={N_THERM}"
        f" → {N_REPLICAS * N_MEASURE} measurements per (β, volume)"
    )
    gauge = load_gauge_side()
    if THERM_CHECK:
        for shape in VOLUMES.values():
            thermalisation_check(shape)

    results = {name: [] for name in VOLUMES}
    for name, shape in VOLUMES.items():
        print(f"\n{'=' * 74}\nvolume '{name}' = {shape}\n{'=' * 74}")
        for beta in BETAS:
            r = measure(beta, shape, name)
            results[name].append(r)
            print(
                f"  β={beta:<7} β*={r['beta_star']:.6f}"
                f"  ξ(σ)={r['m']['xi']:.3f}±{r['m']['xi_err']:.3f}"
                f" [A₀={r['m']['A0']:.3f}]"
                f"  ξ(ε_t)={r['e_t']['xi']:.3f}±{r['e_t']['xi_err']:.3f}"
                f" [A₀={r['e_t']['A0']:.3f}]"
                f"  τ(M)={r['tau_mag_sweeps']:.0f}sw"
                f" tun={r['tunnel_rate']:.3f}"
                f" {'' if r['sampling_ok'] else '  ⚠ UNDERSAMPLED'}"
            )

    # ------------------------------------------------------- sampling gate
    print(f"\n{'=' * 74}\nSAMPLING GATE — τ_int of the MAGNETISATION (the slow"
          f" mode of the broken phase)\n{'=' * 74}")
    print(f"{'volume':>9} {'β':>8} {'τ(M) sweeps':>12} {'n_therm/τ':>10}"
          f" {'n_skip/τ':>9} {'tunnel':>8} {'verdict':>14}")
    any_bad = False
    for name, rows in results.items():
        for r in rows:
            tau = max(r["tau_mag_sweeps"], 1e-9)
            any_bad |= not r["sampling_ok"]
            print(
                f"{name:>9} {r['beta']:>8} {tau:>12.0f} {r['n_therm'] / tau:>10.1f}"
                f" {r['n_skip'] / tau:>9.1f} {r['tunnel_rate']:>8.3f}"
                f" {'ok' if r['sampling_ok'] else 'UNDERSAMPLED':>14}"
            )
    if any_bad:
        print(
            "\n  ⚠ Cells above marked UNDERSAMPLED did not reach"
            f" n_therm ≥ {THERM_TAU:.0f}τ(M) and n_skip ≥ {SKIP_TAU:.0f}τ(M)"
            " even after escalation."
            "\n    Their ξ is NOT usable: a cold start that has not relaxed"
            " reads too ordered, which biases"
            "\n    ⟨ss⟩ high and ξ(σ) long. Raise DGT_MAX_ESCALATE, or accept"
            " that the matched volume"
            "\n    cannot be sampled at that β with a local algorithm."
        )

    # ---------------------------------------------------------------- checks
    print(f"\n{'=' * 74}\nDUALITY CHECK — parameter-free, vs the gauge ensembles"
          f"\n{'=' * 74}")
    print(f"{'β':>8} {'⟨ss⟩(β*)':>12} {'⟨P⟩ predicted':>16} {'⟨P⟩ measured':>16} {'pull':>8}")
    checks = []
    ref = results.get("matched") or next(iter(results.values()))
    for r in ref:
        pm, pe = gauge_plaquette(r["beta"])
        if pm is None:
            print(f"{r['beta']:>8}  (gauge ensemble not cached)")
            continue
        s = (r["pred_plaquette_err"] ** 2 + (pe or 0) ** 2) ** 0.5
        pull = (r["pred_plaquette"] - pm) / s if s > 0 else float("nan")
        checks.append(pull)
        print(
            f"{r['beta']:>8} {r['bond_energy']:>12.6f} "
            f"{r['pred_plaquette']:>10.6f}±{r['pred_plaquette_err']:.6f} "
            f"{pm:>10.6f}±{pe:.6f} {pull:>+8.1f}σ"
        )
    if checks:
        print(
            f"\n  max |pull| = {max(abs(c) for c in checks):.1f}σ — the Ising sweep,"
            " the duality map and the derivation all have to be right"
            "\n  simultaneously for this to pass, and nothing in it is fitted."
        )

    # ------------------------------------------------------------- the table
    if gauge and "matched" in results:
        print(f"\n{'=' * 74}\nACCURACY vs the matched-volume dual ground truth"
              f"\n{'=' * 74}")
        print(
            f"{'β':>8} {'ξ dual(σ)':>14} {'ξ classical':>14} {'ξ attn train':>14}"
            f" {'ξ attn rand':>14}"
        )
        dev = {"classical": [], "trained": [], "random": []}
        for r in results["matched"]:
            g = gauge.get(round(r["beta"], 4), {})
            t = r["m"]["xi"]
            line = f"{r['beta']:>8} {t:>8.3f}±{r['m']['xi_err']:.3f}"
            for k in ("classical", "trained", "random"):
                if k in g:
                    line += f" {g[k][0]:>8.3f}±{g[k][1]:.3f}"
                    dev[k].append((g[k][0] - t) / t)
                else:
                    line += f" {'—':>14}"
            print(line)
        print(f"\n{'operator':>14} {'mean fractional deviation from truth':>40}")
        for k in ("classical", "trained", "random"):
            if dev[k]:
                a = np.array(dev[k])
                print(
                    f"{k:>14} {a.mean():>+20.1%}  (rms {np.sqrt((a**2).mean()):.1%},"
                    f" worst {a[np.argmax(abs(a))]:+.1%})"
                )
        best = min(
            (k for k in dev if dev[k]),
            key=lambda k: np.sqrt((np.array(dev[k]) ** 2).mean()),
        )
        print(f"\n  closest to the dual ground truth: **{best}**")

    # ------------------------------------------------------- finite volume
    if len(results) == 2:
        print(f"\n{'=' * 74}\nFINITE VOLUME — is the ξ(0.7585) > ξ(0.760) excursion real?"
              f"\n{'=' * 74}")
        print(f"{'β':>8} {'ξ matched':>16} {'ξ large':>16} {'shift':>10}")
        for a, b in zip(results["matched"], results["large"]):
            s = (b["m"]["xi"] - a["m"]["xi"]) / a["m"]["xi"]
            print(
                f"{a['beta']:>8} {a['m']['xi']:>10.3f}±{a['m']['xi_err']:.3f}"
                f" {b['m']['xi']:>10.3f}±{b['m']['xi_err']:.3f} {s:>+10.1%}"
            )
        for name in ("matched", "large"):
            xis = [r["m"]["xi"] for r in results[name]]
            mono = all(x < y for x, y in zip(xis, xis[1:]))
            print(f"  {name:>8}: {'monotonic in β' if mono else 'NON-monotonic'}"
                  f"  ({', '.join(f'{x:.2f}' for x in xis)})")

    scaling_test(results)

    torch.save(
        {"results": results, "gauge": gauge, "betas": BETAS,
         "beta_c": GAUGE_BETA_C, "volumes": VOLUMES,
         "meta": {"fit_window": FIT_WINDOW, "late_window": LATE_WINDOW,
                  "jack_block": JACK_BLOCK, "n_replicas": N_REPLICAS,
                  "n_measure": N_MEASURE, "n_skip": N_SKIP, "n_therm": N_THERM}},
        f"{RESULTS}/dual_ground_truth.pt",
    )
    plot(results, gauge)
    print(f"\nwrote {RESULTS}/dual_ground_truth.{{png,pt}}")


def scaling_test(results):
    """Do the dual measurements reproduce 3D Ising criticality? A prediction.

    Near β*_c, ξ = ξ₀ · t*^(−ν) with t* = (β* − β*_c)/β*_c and ν = 0.629971 known
    from the conformal bootstrap to six digits. Fixing ξ₀ from the **single
    lowest-β point** — the one furthest from criticality, hence least affected
    by finite volume — turns every other point into a parameter-free
    prediction. This is the strongest available validation of the dual
    measurements, and it is independent of the gauge side entirely.

    It is also the sharpest available *diagnosis*: a point that misses the
    prediction badly while its neighbours hit it is a point whose chain, fit
    window or box has failed, not a point that has discovered new physics.

    A two-parameter fit is reported alongside, but only as a consistency check
    — with ξ ≤ L/4 enforced at just a few points, an effective exponent here is
    not a measurement of ν and must not be quoted as one (the same caution
    `notes/attention_as_operator.md` §7 already applies to the gauge scan).
    """
    print(f"\n{'=' * 74}\nSCALING TEST — is the dual reproducing 3D Ising"
          f" criticality?\n{'=' * 74}")
    for name, rows in results.items():
        ok = [r for r in rows if r["sampling_ok"] and np.isfinite(r["m"]["xi"])]
        if len(ok) < 3:
            print(f"  {name}: fewer than 3 usable points — skipped")
            continue
        t = np.array([
            (dual_beta(r["beta"]) - ISING_BETA_C) / ISING_BETA_C for r in ok
        ])
        xi = np.array([r["m"]["xi"] for r in ok])
        err = np.array([r["m"]["xi_err"] for r in ok])
        amp = xi[0] * t[0] ** NU_ISING  # fixed from the lowest-β point alone
        print(f"\n  {name}:  ξ₀ = {amp:.4f} fixed from β = {ok[0]['beta']},"
              f" ν = {NU_ISING} (bootstrap)")
        print(f"  {'β':>8} {'t*':>10} {'ξ predicted':>12} {'ξ measured':>18}"
              f" {'dev':>8}")
        for r, ti, xm, xe in zip(ok, t, xi, err):
            p = amp * ti**-NU_ISING
            print(f"  {r['beta']:>8} {ti:>10.6f} {p:>12.3f}"
                  f" {xm:>11.3f}±{xe:.3f} {100 * (xm - p) / p:>+7.1f}%")
        # unweighted log-log slope, consistency only
        nu_fit = -np.polyfit(np.log(t), np.log(xi), 1)[0]
        print(f"  effective exponent from a two-parameter fit: {nu_fit:.3f}"
              f"  (bootstrap {NU_ISING}) — consistency only, not a measurement")


def plot(results, gauge):
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))
    matched = results.get("matched", [])
    betas = [r["beta"] for r in matched]

    # -- panel 1: ξ(β), everything against the ground-truth band -------------
    a = ax[0]
    if matched:
        t = np.array([r["m"]["xi"] for r in matched])
        te = np.array([r["m"]["xi_err"] for r in matched])
        a.fill_between(betas, t - te, t + te, color=INK, alpha=0.18, lw=0)
        a.plot(betas, t, "-", color=INK, lw=2.2, zorder=5,
               label="dual ground truth (24²×48)")
    if "large" in results:
        lg = [r["m"]["xi"] for r in results["large"]]
        a.plot(betas, lg, "--", color=COL["dual_large"], lw=2, marker="s",
               ms=7, label="dual, 48²×96")
    for k, lab, mk in (
        ("classical", "classical smeared basis", "o"),
        ("trained", "attention, trained", "D"),
        ("random", "attention, random init", "^"),
    ):
        xs = [b for b in betas if k in gauge.get(round(b, 4), {})]
        if not xs:
            continue
        ys = [gauge[round(b, 4)][k][0] for b in xs]
        es = [gauge[round(b, 4)][k][1] for b in xs]
        a.errorbar(xs, ys, yerr=es, fmt=mk + "-", color=COL[k], ms=7,
                   capsize=3, lw=1.6, label=lab)
    a.axvline(GAUGE_BETA_C, ls=":", color="gray", lw=1)
    a.set_xlabel(r"$\beta$")
    a.set_ylabel(r"$\xi$  [lattice spacings]")
    a.set_title("Correlation length against an exact reference")
    a.legend(fontsize=8, frameon=False)
    a.grid(alpha=0.25, lw=0.6)

    # -- panel 2: fractional deviation from truth ----------------------------
    a = ax[1]
    a.axhline(0, color=INK, lw=2)
    for k, lab, mk in (
        ("classical", "classical", "o"),
        ("trained", "attention, trained", "D"),
        ("random", "attention, random", "^"),
    ):
        xs, ys, es = [], [], []
        for r in matched:
            g = gauge.get(round(r["beta"], 4), {})
            if k not in g:
                continue
            xs.append(r["beta"])
            ys.append(100 * (g[k][0] - r["m"]["xi"]) / r["m"]["xi"])
            es.append(100 * g[k][1] / r["m"]["xi"])
        if xs:
            a.errorbar(xs, ys, yerr=es, fmt=mk + "-", color=COL[k], ms=7,
                       capsize=3, lw=1.6, label=lab)
    a.set_xlabel(r"$\beta$")
    a.set_ylabel(r"$(\xi_{\rm op}-\xi_{\rm true})\,/\,\xi_{\rm true}$   [%]")
    a.set_title("Who is right?  (0 = the dual ground truth)")
    a.legend(fontsize=8, frameon=False)
    a.grid(alpha=0.25, lw=0.6)

    # -- panel 3: the parameter-free duality check ---------------------------
    a = ax[2]
    pred = [r["pred_plaquette"] for r in matched]
    meas, merr = [], []
    for r in matched:
        pm, pe = gauge_plaquette(r["beta"])
        meas.append(pm if pm is not None else np.nan)
        merr.append(pe if pe is not None else 0.0)
    lim = [min(pred + [m for m in meas if np.isfinite(m)] + [0.88]) - 0.005,
           max(pred + [m for m in meas if np.isfinite(m)] + [0.95]) + 0.005]
    a.plot(lim, lim, "-", color="gray", lw=1, zorder=0)
    a.errorbar(meas, pred, xerr=merr, fmt="o", color=COL["classical"], ms=8,
               capsize=3)
    for b, x, y in zip(betas, meas, pred):
        a.annotate(f"{b}", (x, y), textcoords="offset points", xytext=(6, -10),
                   fontsize=8, color=INK)
    a.set_xlim(lim)
    a.set_ylim(lim)
    a.set_xlabel(r"$\langle P\rangle$ measured (gauge ensembles)")
    a.set_ylabel(r"$\langle P\rangle$ predicted from $\langle ss\rangle(\beta^*)$")
    a.set_title("Duality check: no free parameter")
    a.grid(alpha=0.25, lw=0.6)

    fig.suptitle(
        "3D Z₂ mass gap against its exactly dual Ising model — "
        "is the learned attention operator the accurate one?",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/dual_ground_truth.png", dpi=150)


if __name__ == "__main__":
    main()
