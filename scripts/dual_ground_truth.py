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
    dual_beta,
    ising_measure,
    predicted_plaquette,
)
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
    return float(per_cfg.mean()), float(per_cfg.std() / per_cfg.numel() ** 0.5)


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
def measure(beta, shape, label):
    """Run the dual Ising at β* and fit both operators."""
    bs = dual_beta(beta)
    out = ising_measure(
        bs, shape,
        n_replicas=N_REPLICAS, n_measure=N_MEASURE,
        n_therm=N_THERM, n_skip=N_SKIP,
        device=device, seed=abs(hash((round(beta, 6), shape))) % (2**31),
        ordered_start=True, progress=True,
    )

    res = {"beta": beta, "beta_star": bs, "shape": shape, "label": label}
    for op in ("m", "e_t"):
        m, me, a0, a0e = _jack(out[op], FIT_WINDOW)
        xi, xie = _xi(m, me)
        res[op] = {"m": m, "m_err": me, "xi": xi, "xi_err": xie,
                   "A0": a0, "A0_err": a0e}
        m2, me2, _, _ = _jack(out[op], LATE_WINDOW)
        res[op]["xi_late"], res[op]["xi_late_err"] = _xi(m2, me2)
        res[op]["profile"] = (
            connected_correlator(out[op].double())[:10]
            / connected_correlator(out[op].double())[0]
        ).tolist()

    # diagnostics: τ_int of each chain observable, and tunnelling.
    # Both are reported because they are different modes: the energy is fast,
    # the magnetisation is the *slow* mode of the broken phase and is what sets
    # whether n_skip was enough. Under-decorrelation inflates errors rather than
    # biasing ξ (the jackknife blocks absorb it), but it has to be visible.
    mg = out["magnetisation"].reshape(N_REPLICAS, N_MEASURE)
    e = out["bond_energy"].reshape(N_REPLICAS, N_MEASURE)
    for key, series in (("tau_int", e), ("tau_int_mag", mg)):
        res[key] = float(
            np.mean([
                float(integrated_autocorrelation_time(series[r])[1])
                for r in range(N_REPLICAS)
            ])
        )
    flips = (torch.sign(mg[:, 1:]) != torch.sign(mg[:, :-1])).double().mean()
    res["tunnel_rate"] = float(flips)
    res["bond_energy"] = float(out["bond_energy"].mean())
    res["bond_energy_err"] = float(
        out["bond_energy"].std() / out["bond_energy"].numel() ** 0.5
    )
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
                f"  τ(ε)={r['tau_int']:.1f} τ(M)={r['tau_int_mag']:.1f}"
                f" tun={r['tunnel_rate']:.3f}"
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
