"""Exact spin-2 ground truth from the dual Ising model.

Companion to ``scripts/z2_rotation_irreps.py``. That script decomposes the
attention field into irreps of the lattice rotation group D₄ and finds a
non-scalar component; the obvious next question is whether that component
couples to a *physical* state, and if so which one. This script measures the
answer exactly, on the dual side, with no network involved.

3D Z₂ gauge theory is Kramers–Wannier–Wegner dual to the 3D Ising model
(``notes/dual_ground_truth.md``), and the duality is a lattice identity: it maps
the cubic lattice to itself, so **spin channels are preserved**. A gauge-side
2⁺⁺ glueball is an Ising state in the same D₄ irrep, and on the Ising side the
interpolating operator is elementary — the difference of the two spatial bond
energies,

    O_B1(t) = Σ_x [ s(x)s(x+ê₁) − s(x)s(x+ê₂) ]          (B₁ ~ x²−y²)
    O_B2(t) = Σ_x [ s(x)s(x+ê₁+ê₂) − s(x)s(x+ê₁−ê₂) ]    (B₂ ~ xy)

against the scalar channel's wall magnetisation ``m`` and temporal bond energy
``e_t``. All five are measured on the **same configurations**, so the ratio
m_B/m_A₁ — the number the gauge side is compared against — is a correlated
jackknife rather than a quotient of two independent runs.

Two nulls are built in and both are exact:

  * ⟨O_B⟩ = 0 identically, by the D₄ symmetry of the ensemble. A nonzero mean
    beyond its own error means the sampler is not equilibrated.
  * the B correlator has **no** overlap with the scalar ground state, again by
    symmetry — so m_B > m_A₁ is guaranteed if the measurement is sound, and
    m_B ≈ m_A₁ would mean the projection is leaking rather than that the states
    are degenerate.

Conventions are copied from ``scripts/dual_ground_truth.py`` (volume, sweeps,
jackknife block, ``fit_cosh_correlator`` through the gauge side's own code
path) so the scalar-channel numbers here must reproduce the ones recorded in
``notes/dual_ground_truth.md`` — that agreement is the run's validation.

Only the **large** 96×48² volume is used: ``dual_ground_truth.md`` §7 records
that the matched 48×24² volume tunnels between magnetisation sectors at 6–48%
for β ≥ 0.756, which is a real light state in the σ channel and not something
more sweeps can fix.

Run (``-u``: without it stdout block-buffers when piped while tqdm writes to
stderr, so the bars appear ahead of the prints and the log reads out of order):

    python -u scripts/dual_spin2.py
    DS2_SMOKE=1 python -u scripts/dual_spin2.py   # 2 min plumbing check

**Cost, and how to tell whether it is on the GPU.** Each β needs
``N_THERM + N_MEASURE·N_SKIP`` = 26,500 sweeps of a 32 × 96 × 48 × 48 spin array.
A sweep is ~24 elementwise kernels over 7.07M spins: on a V100 that is ~1 ms, so
expect order 10² sweeps/s and a few minutes per β. At ~17 sweeps/s — the CPU rate
for this size — it is 25 min per β and ~1.8 h for the scan. The header line
prints the device; if it says ``cpu`` that is why. The measurement bar ticks once
per ``N_SKIP`` sweeps, so it looks frozen on CPU even when it is fine.

There is also a **silent fit phase** after each β: ``JACK_BLOCK`` = 40 on
32 × 500 = 16,000 measurements is 400 jackknife replicas × 12 operators, each
recomputing the correlator over ~16k rows, with no progress bar. Minutes on CPU.

If it has to be cut, ``DS2_MEASURE`` is the linear wall-time knob and
``DS2_REPLICAS`` is linear on CPU only (replicas are batched, hence nearly free
on a GPU). Cut those before ``DS2_SKIP``: n_skip is set against the
*magnetisation* autocorrelation, and under-skipping there biases ⟨ss⟩ high and
shows up as a failure of the duality check — see ``gelt.ising.heatbath_sweep``.

Environment overrides: ``DS2_REPLICAS``, ``DS2_MEASURE``, ``DS2_SKIP``,
``DS2_THERM``, ``DS2_JACK``, ``DS2_SHAPE`` (e.g. ``96,48,48``).

Writes ``results/rotation/dual_spin2.{pt,png}``.
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from gelt.glueball import connected_correlator, fit_cosh_correlator  # noqa: E402
from gelt.ising import (  # noqa: E402
    bond_field,
    dual_beta,
    heatbath_sweep,
    random_spins,
)
from gelt.sampler import integrated_autocorrelation_time  # noqa: E402

os.makedirs("results/rotation", exist_ok=True)

SMOKE = os.environ.get("DS2_SMOKE", "0") == "1"


def _env(name, full, smoke):
    return int(os.environ.get(name, smoke if SMOKE else full))


# The four couplings the gauge-side scan keeps; β = 0.760 is dropped there for
# finite volume (notes/attention_as_operator.md §8) and there is nothing on the
# dual side for it to be compared against.
BETAS = [0.7450, 0.7520, 0.7560, 0.7585]
SHAPE = tuple(int(x) for x in os.environ.get("DS2_SHAPE", "96,48,48").split(","))
N_REPLICAS = _env("DS2_REPLICAS", 32, 4)
N_MEASURE = _env("DS2_MEASURE", 500, 30)
N_SKIP = _env("DS2_SKIP", 50, 5)
N_THERM = _env("DS2_THERM", 1500, 60)
JACK_BLOCK = _env("DS2_JACK", 40, 5)

# Scalar channel: the window dual_ground_truth.py uses, so its numbers are
# reproducible here. Non-scalar channel: a state 2–3× heavier is gone before
# Δ = 8, so the window is shorter — and m_eff is printed either way, which is
# what makes a missing plateau visible rather than hidden inside a fit.
A_WINDOW = (2, 8)
B_WINDOW = (1, 6)
MIN_FIT_POINTS = 4

if SMOKE:
    SHAPE = (24, 12, 12)
    BETAS = BETAS[:2]
    A_WINDOW, B_WINDOW, MIN_FIT_POINTS = (1, 5), (1, 4), 3

OUT_PT = "results/rotation/dual_spin2.pt"
OUT_PNG = "results/rotation/dual_spin2.png"

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))

# Λ = (Nt, L, L) with time on lattice axis 0, so tensor axes 2 and 3 are the
# two spatial directions of the slice the gauge-side network sees.
AX_T, AX_1, AX_2 = 1, 2, 3
# A thin bond-energy difference is a very pointlike operator: its correlator is
# almost all contact term, and the smoke run measured C(1)/C(0) ≈ 0.04 with
# nothing resolvable beyond. The cure is the one the gauge side already uses for
# exactly this reason (notes/glueball_spectroscopy.md §7): smear the field the
# operator is built from, **spatially only** — smearing in time would void the
# transfer-matrix interpretation that makes the decay a mass at all.
SMEAR_LEVELS = (0, 2, 4, 8, 16)
SMEAR_ALPHA = 0.5
A1_OPS = ("m", "e_t")
B_OPS = tuple(f"{b}@{n}" for n in SMEAR_LEVELS for b in ("b1", "b2"))


def _smear(s, n_steps, alpha=SMEAR_ALPHA):
    """Spatial-only local averaging — the Ising analogue of APE smearing.

    The result is no longer ±1, which is fine: an interpolating operator only
    has to have the right quantum numbers and a good overlap, not to be a spin
    configuration.
    """
    for _ in range(n_steps):
        nb = (s.roll(1, AX_1) + s.roll(-1, AX_1)
              + s.roll(1, AX_2) + s.roll(-1, AX_2)) / 4
        s = (1 - alpha) * s + alpha * nb
    return s


def observables(s):
    """Zero-momentum operators per timeslice, ``(R, Nt)`` each.

    ``m`` and ``e_t`` are the scalar channel (as in ``ising.wall_observables``,
    reimplemented here so this script needs nothing added to the library);
    ``b1`` (~ x²−y², from the axis bonds) and ``b2`` (~ xy, from the diagonals)
    are the two spin-2 partners, built at each smearing level.
    """
    wall = lambda f: f.sum(dim=(AX_1, AX_2))  # noqa: E731
    m = wall(s)
    # Sign-fixed per replica by the global magnetisation — the standard
    # restricted-ensemble treatment of the broken phase. Tunnelling is reported
    # as a diagnostic rather than assumed absent.
    sign = torch.sign(m.sum(dim=1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    out = {"m": m * sign, "e_t": wall(bond_field(s, AX_T))}
    sm, done = s, 0
    for lvl in SMEAR_LEVELS:
        if lvl > done:
            sm, done = _smear(sm, lvl - done), lvl
        out[f"b1@{lvl}"] = wall(bond_field(sm, AX_1) - bond_field(sm, AX_2))
        out[f"b2@{lvl}"] = wall(sm * sm.roll((-1, -1), dims=(AX_1, AX_2))
                                - sm * sm.roll((-1, +1), dims=(AX_1, AX_2)))
    return out


def sample(beta_star, seed):
    """Thermalise, then collect ``N_MEASURE`` measurements per replica."""
    gen = None
    try:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        torch.rand(1, device=device, generator=gen)
    except (RuntimeError, TypeError):  # MPS has no device generator
        gen = None
        torch.manual_seed(seed)

    s = random_spins(SHAPE, N_REPLICAS, device=device, generator=gen, ordered=True)
    for _ in tqdm(range(N_THERM), desc="  therm", leave=False):
        s = heatbath_sweep(s, beta_star, generator=gen)

    acc = {k: [] for k in A1_OPS + B_OPS}
    mag = []
    for _ in tqdm(range(N_MEASURE), desc="  measure", leave=False):
        for _ in range(N_SKIP):
            s = heatbath_sweep(s, beta_star, generator=gen)
        o = observables(s)
        for k in acc:
            # .cpu() before .double(): MPS has no float64.
            acc[k].append(o[k].cpu().double())
        mag.append(s.flatten(1).mean(dim=1).cpu().double())
    # Chain-major: a contiguous block of rows is a contiguous stretch of one
    # Markov chain, so a blocked jackknife blocks within a chain.
    out = {k: torch.stack(v, dim=1).reshape(-1, SHAPE[0]) for k, v in acc.items()}
    out["_mag"] = torch.stack(mag, dim=1)  # (R, N_MEASURE), before sign fixing
    return out


# ── Statistics (mirrors z2_rotation_irreps.py so the two sides are comparable) ─
def _window(C, bounds):
    lo, hi = bounds
    if not torch.isfinite(C[0]) or C[0] <= 0:
        return None
    d = lo - 1
    for k in range(lo, min(hi, len(C) - 1) + 1):
        if not torch.isfinite(C[k]) or C[k] <= 0:
            break
        d = k
    return (lo, d) if d - lo + 1 >= MIN_FIT_POINTS else None


def _fit(series, nt, window):
    C = connected_correlator(series)
    if not torch.isfinite(C[: window[1] + 1]).all() or C[0] <= 0:
        return float("nan"), float("nan")
    m, A, _ = fit_cosh_correlator(C, *window)
    if not np.isfinite(m) or m <= 0:
        return float("nan"), float("nan")
    return float(m), float(A * (1.0 + math.exp(-m * nt)) / C[0].item())


def _err(vals):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float("nan")
    return float(np.sqrt((len(v) - 1) / len(v) * ((v - v.mean()) ** 2).sum()))


def _blocks(B):
    n = max(2, B // JACK_BLOCK)
    for i in range(n):
        keep = torch.ones(B, dtype=torch.bool)
        keep[i * JACK_BLOCK: (i + 1) * JACK_BLOCK] = False
        yield keep


def measure_ops(data, nt):
    """Every operator's (m, A₀), plus the **correlated** m_B/m_A₁ ratios.

    The ratio is the point of the whole script, and computing it from two
    independently-errored masses would throw away the cancellation between two
    operators measured on identical configurations.
    """
    wins, out = {}, {}
    for k in A1_OPS + B_OPS:
        C = connected_correlator(data[k])
        w = _window(C, A_WINDOW if k in A1_OPS else B_WINDOW)
        wins[k] = w
        prof = (C / C[0]).tolist()[:12] if C[0] > 0 else None
        meff = [float(torch.log(C[d] / C[d + 1])) if C[d] > 0 and C[d + 1] > 0
                else float("nan") for d in range(min(8, len(C) - 1))]
        mean = float(data[k].mean())
        rec = {"window": w, "profile": prof, "m_eff": meff, "mean": mean,
               "signal": float(C[2] / C[0]) if C[0] > 0 and len(C) > 2 else float("-inf"),
               "mean_err": float(data[k].mean(dim=1).std() / math.sqrt(len(data[k])))}
        if w is None:
            rec.update({"m": float("nan"), "m_err": float("nan"),
                        "A0": float("nan"), "A0_err": float("nan")})
        else:
            m, a0 = _fit(data[k], nt, w)
            ms, a0s = zip(*(_fit(data[k][b], nt, w) for b in _blocks(len(data[k]))))
            rec.update({"m": m, "m_err": _err(ms), "A0": a0, "A0_err": _err(a0s)})
        out[k] = rec

    ref = "m"  # the order parameter: near-unit overlap in the broken phase
    for k in B_OPS:
        if wins[k] is None or wins[ref] is None:
            continue

        def ratio(mask):
            mb, _ = _fit(data[k][mask], nt, wins[k])
            ma, _ = _fit(data[ref][mask], nt, wins[ref])
            return mb / ma if np.isfinite(mb) and np.isfinite(ma) and ma > 0 else float("nan")

        full = ratio(torch.ones(len(data[k]), dtype=torch.bool))
        reps = [ratio(b) for b in _blocks(len(data[k]))]
        out[k]["ratio"] = full
        out[k]["ratio_err"] = _err(reps)
    return out


def main():
    print(f"dual spin-2 ground truth | shape {SHAPE} | {N_REPLICAS} replicas × "
          f"{N_MEASURE} meas × {N_SKIP} skip | device {device}")
    rows = []
    for beta in BETAS:
        bs = dual_beta(beta)
        print(f"\nβ = {beta}  →  β* = {bs:.6f}")
        data = sample(bs, seed=int(beta * 1e6))
        mag = data.pop("_mag")
        # Tunnelling between magnetisation sectors is a real light σ-channel
        # state, not a sampling artefact; report it rather than assume it away.
        flips = (mag.sign().diff(dim=1) != 0).double().mean().item()
        tau = integrated_autocorrelation_time(mag[0].numpy())[1]
        res = measure_ops(data, SHAPE[0])
        print(f"  tunnel rate {flips:.3f} | τ_int(M) {tau:.2f} measurements")
        for k, r in res.items():
            xi = 1 / r["m"] if np.isfinite(r["m"]) and r["m"] > 0 else float("nan")
            extra = ""
            if "ratio" in r:
                extra = f"  m/m_σ = {r['ratio']:.3f} ± {r['ratio_err']:.3f}"
            print(f"    {k:<4s} m = {r['m']:.4f} ± {r['m_err']:.4f}  "
                  f"ξ = {xi:6.3f}  A₀ = {r['A0']:.3f} ± {r['A0_err']:.3f}"
                  f"  ⟨O⟩ = {r['mean']:+.3e} ± {r['mean_err']:.1e}{extra}")
        rows.append({"beta": beta, "beta_star": bs, "shape": SHAPE,
                     "tunnel_rate": flips, "tau_int_mag": float(tau), "ops": res})
        torch.save({"rows": rows, "betas": BETAS, "shape": SHAPE,
                    "meta": {"n_replicas": N_REPLICAS, "n_measure": N_MEASURE,
                             "n_skip": N_SKIP, "n_therm": N_THERM,
                             "jack_block": JACK_BLOCK, "a_window": A_WINDOW,
                             "b_window": B_WINDOW}}, OUT_PT)

    print("\n" + "=" * 74)
    print("SPIN-2 GROUND TRUTH — what the gauge-side B channel must reproduce")
    print("=" * 74)
    print("  Most-smeared resolved level per family, with a convergence verdict.")
    print("  β       operator   m               m/m_σ           A₀        conv")
    for r in rows:
        o = r["ops"]
        for fam in ("b1", "b2"):
            cand = [k for k in B_OPS if k.startswith(fam) and np.isfinite(o[k]["m"])]
            if not cand:
                print(f"  {r['beta']:<6.4f}  {fam:<9s} unresolved at every smearing level")
                continue
            # Ordered by smearing level, not by C(2)/C(0): the most-smeared
            # operator always wins on signal, so ranking that way would quote a
            # mass that is still drifting as though it were the answer.
            cand.sort(key=lambda x: int(x.split("@")[1]))
            k = cand[-1]
            # The mass must stop moving between the last two levels. It did not
            # at β = 0.745 (1.467 → 1.296, nine times its own error, with A₀
            # still climbing), which means the lightest state in the channel has
            # not been isolated and the true mass is *below* the quoted one.
            verdict = "  ?  "
            if len(cand) > 1:
                prev, cur = o[cand[-2]], o[k]
                sig = abs(cur["m"] - prev["m"]) / max(cur["m_err"], 1e-30)
                verdict = "yes" if sig < 2 else f"NO {sig:.0f}σ"
            g = lambda f, d="  —  ": (f"{o[k][f]:.3f}"  # noqa: E731
                                      if f in o[k] and np.isfinite(o[k][f]) else d)
            print(f"  {r['beta']:<6.4f}  {k:<9s} {g('m')} ± {g('m_err')}   "
                  f"{g('ratio')} ± {g('ratio_err')}   {g('A0')} ± {g('A0_err')}"
                  f"   {verdict}")
    print("\n  conv = did the mass stop moving between the last two smearing")
    print("  levels? 'NO' means the ladder has not isolated the lightest state and")
    print("  the true mass is below the quoted one — read it as an upper bound.")
    print("  The two-particle threshold is m/m_σ = 2.0: two σ in a relative D-wave")
    print("  carry spin 2, so a ratio converging to 2 is the continuum rather than")
    print("  a state, and only a ratio settling *below* 2 is a 2⁺⁺ bound state.")
    print("\n  The ratio is the transferable number: it is a pure spectrum quantity,")
    print("  so the gauge-side attention B channel must land on it if the")
    print("  non-scalar content couples to the physical spin-2 state at all.")

    def best(r, fam):
        cand = [k for k in B_OPS if k.startswith(fam) and np.isfinite(r["ops"][k]["m"])]
        return max(cand, key=lambda x: r["ops"][x]["signal"]) if cand else None

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    betas = [r["beta"] for r in rows]
    for k, mk in zip(A1_OPS, ("o", "s")):
        y = [r["ops"][k]["m"] for r in rows]
        e = [r["ops"][k]["m_err"] for r in rows]
        ax[0].errorbar(betas, y, yerr=e, marker=mk, capsize=3, label=k)
    for fam, mk in (("b1", "^"), ("b2", "v")):
        y = [r["ops"][best(r, fam)]["m"] if best(r, fam) else np.nan for r in rows]
        e = [r["ops"][best(r, fam)]["m_err"] if best(r, fam) else np.nan for r in rows]
        ax[0].errorbar(betas, y, yerr=e, marker=mk, capsize=3, label=fam)
    ax[0].set_xlabel("β (gauge)")
    ax[0].set_ylabel("m")
    ax[0].set_title("Dual Ising masses by channel")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    for fam, mk in (("b1", "^"), ("b2", "v")):
        y = [r["ops"][best(r, fam)].get("ratio", np.nan) if best(r, fam) else np.nan
             for r in rows]
        e = [r["ops"][best(r, fam)].get("ratio_err", np.nan) if best(r, fam) else np.nan
             for r in rows]
        ax[1].errorbar(betas, y, yerr=e, marker=mk, capsize=3, label=f"{fam}/σ")
    ax[1].axhline(2.0, color="k", ls=":", lw=1, label="2-particle threshold")
    ax[1].set_xlabel("β (gauge)")
    ax[1].set_ylabel("m_B / m_σ")
    ax[1].set_title("Spin-2 to scalar mass ratio")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.suptitle(f"Exact spin-2 ground truth from the dual Ising model, Λ = {SHAPE}")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"\nwrote {OUT_PT} and {OUT_PNG}")


if __name__ == "__main__":
    main()
