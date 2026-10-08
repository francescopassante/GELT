"""The SU(2) attention-as-operator study, read with the 0⁺⁺ chapter's estimator.

Reads the raw series written by ``scripts/su2_attention_dump.py`` (GPU half) and
does every statistic on CPU, in minutes:

* **Estimator = fit_glueball_overlap.py's**, not the Z₂ run's: one-state cosh fit
  on the fixed window Δ ∈ [2, 7] with diagonal weights σ_Δ fixed on the full
  sample, A₀ = A(1 + e^{−m Nt})/C(0), GEVP projection at (t0, td) = (1, 2) with
  v₀ recomputed in every replica, blocked jackknife with b = 10. Network bases
  (24 attention channels, 16 residual channels per stage) are standardised and
  the GEVP truncates the directions of C(t0) below 1e-4 · s_max (the fair-fight
  convention for bases larger than four); the classical 4-level basis goes
  through the same projection (the chapter checked truncation and floor agree
  there to the third decimal).
* **Arms per network**: its own output Ō; the attention field (all 24 channels,
  and per block, 6 channels); the residual stream per stage (Re Tr W_c / N,
  16 channels, stage ℓ = stream entering block ℓ, last = entering the readout).
  Each basis is quoted by its GEVP projection and by its best single channel
  (chosen once on the full sample).
* **Trained − random** with two separate errors: (conf) the blocked jackknife
  of the difference of the means on the shared configurations, and (init) the
  spread between networks, √(s²_tr/n_tr + s²_rnd/n_rnd). The second is the one
  the original single-random-seed table could not have.
* **The inheritance control** (``notes/thesis_record.md`` §11 P4, arm (a)):
  per block, ΔA₀(trained − random) of the attention field against that of the
  residual stream entering the same block — the features Q and K are computed
  from. The output-regressed arm (b) is deliberately absent: Ō's A₀ is ≈ 0.9,
  so removing the component along Ō removes almost all ground-state weight from
  any operator, and (b) would lose the gain under both readings.
* **Checks**: config-scramble null on the trained attention projection (max
  |C(Δ)/C(0)|, Δ = 1…7, must be ≈ 0); δf/f per arm; the legacy Z₂-estimator
  number for (trained_ens0, random_s0), which must reproduce the original
  +0.286(56) since random_s0 is the original random arm's seed.

    ../GELT/.venv/bin/python scripts/su2_attention_analysis.py [dump] [--block=10]

Writes ``results/attention/su2_attention_analysis.pt``.
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gelt.glueball import (  # noqa: E402
    connected_correlator,
    connected_correlator_matrix,
    fit_cosh_correlator,
    gevp_ground_vector,
)

_POS = [a for a in sys.argv[1:] if not a.startswith("--")]
DUMP = _POS[0] if _POS else "dumps/su2_attention_series.pt"
_FLAGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
BLOCK = int(_FLAGS.get("block", 10))
WINDOW = (2, 7)
M_RANGE = (0.05, 1.5)
T0, TD = 1, 2
EPS = 1e-4
OUT = os.environ.get("SAA_OUT") or (
    "results/attention/su2_attention_analysis"
    + ("_smoke" if "smoke" in os.path.basename(DUMP) else "") + ".pt")


# ── Estimator ─────────────────────────────────────────────────────────────────
def standardize(basis):
    s = basis.reshape(basis.shape[0], -1).std(dim=1).clamp_min(1e-30)
    return basis / s.view(-1, 1, 1)


def project(basis, mask=None, v0=None):
    """(B, Nt) ground-state projection of an (n, B, Nt) basis; n = 1 passes through."""
    b = basis if mask is None else basis[:, mask]
    if b.shape[0] == 1:
        return b[0]
    if v0 is None:
        C = connected_correlator_matrix(b)
        v0 = gevp_ground_vector(C, t0=T0, td=TD, eps=EPS, truncate=True)
    return torch.einsum("i,ibt->bt", v0, b)


def fit(obar, sig):
    C = connected_correlator(obar)
    m, A, _ = fit_cosh_correlator(C, *WINDOW, sigma=sig, m_range=M_RANGE)
    a0 = A * (1.0 + math.exp(-m * C.shape[0])) / C[0].item()
    return m, a0


def blocks(B):
    return [torch.arange(i, min(i + BLOCK, B)) for i in range(0, B, BLOCK)]


def jackknife(fn, B):
    samples = []
    for bl in blocks(B):
        mask = torch.ones(B, dtype=torch.bool)
        mask[bl] = False
        samples.append(fn(mask))
    s = torch.stack(samples)
    n = len(samples)
    mean = s.mean(0)
    return mean, ((n - 1) / n * ((s - mean) ** 2).sum(0)).sqrt(), s


def sigma_of(obar, B):
    return jackknife(lambda m: connected_correlator(obar[m]), B)[1]


def alive(basis):
    """Drop channels with no fluctuation (a dead gate, a constant reduction)."""
    s = basis.reshape(basis.shape[0], -1).std(dim=1)
    return basis[s > 1e-12 * s.max().clamp_min(1e-300)]


# ── Arms ──────────────────────────────────────────────────────────────────────
def arm_specs(blob):
    """name → (n, B, Nt) basis, for every network and the classical basis."""
    meta = blob["meta"]
    labels = meta["attn_labels"]
    n_layers = len(meta["resid_stages"]) - 1
    specs = {("classical", "gevp"): blob["classical"].double()}
    specs[("classical", "APE_last")] = blob["classical"][-1:].double()
    for net, d in blob["nets"].items():
        specs[(net, "out")] = d["out"].double().unsqueeze(0)
        specs[(net, "attn")] = d["attn"].double()
        for l in range(n_layers):
            idx = [i for i, s in enumerate(labels) if s.startswith(f"L{l + 1}h")]
            specs[(net, f"attn_L{l + 1}")] = d["attn"][idx].double()
        for s in range(n_layers + 1):
            tag = f"resid_L{s + 1}" if s < n_layers else "resid_out"
            specs[(net, tag)] = d["resid"][s].double()
    return specs


def prepare(specs, B):
    """Standardise, drop dead channels, fix σ_Δ and the best single channel."""
    prep = {}
    for key, basis in specs.items():
        basis = alive(basis)
        if basis.shape[0] == 0:
            continue
        basis = standardize(basis)
        proj = project(basis)
        entry = {"basis": basis, "sig": sigma_of(proj, B)}
        if basis.shape[0] > 1:
            best, best_a0 = None, -np.inf
            for i in range(basis.shape[0]):
                C = connected_correlator(basis[i])
                if C[0] <= 0:
                    continue
                try:
                    _, a0 = fit(basis[i], None)
                except Exception:  # noqa: BLE001 — a channel that cannot be fitted is skipped
                    continue
                if np.isfinite(a0) and a0 > best_a0:
                    best, best_a0 = i, a0
            if best is not None:
                single = basis[best]
                entry["single_idx"] = best
                entry["single_sig"] = sigma_of(single, B)
        prep[key] = entry
    return prep


def run(blob):
    B, Nt = blob["classical"].shape[1:]
    print(f"{DUMP}: {B} configs × Nt = {Nt}, block {BLOCK} → {-(-B // BLOCK)} "
          f"jackknife blocks; window {WINDOW}, GEVP ({T0},{TD}), truncate eps {EPS:g}")
    prep = prepare(arm_specs(blob), B)
    keys = list(prep)
    # Slot layout of the packed statistic: (m, A₀) per arm for the GEVP
    # projection, then (m, A₀) for the best single channel where there is one.
    slots = []
    for k in keys:
        slots.append((k, "gevp"))
        if "single_idx" in prep[k]:
            slots.append((k, "single"))

    def stat(mask):
        vals = []
        for k, kind in slots:
            e = prep[k]
            if kind == "gevp":
                ob = project(e["basis"], mask)
                m, a0 = fit(ob, e["sig"])
            else:
                m, a0 = fit(e["basis"][e["single_idx"]][mask], e["single_sig"])
            vals += [m, a0]
        return torch.tensor(vals, dtype=torch.float64)

    full = stat(torch.ones(B, dtype=torch.bool))
    _, err, samples = jackknife(stat, B)
    res = {}
    for j, (k, kind) in enumerate(slots):
        res[(k[0], k[1], kind)] = {
            "m": full[2 * j].item(), "m_err": err[2 * j].item(),
            "A0": full[2 * j + 1].item(), "A0_err": err[2 * j + 1].item(),
            "A0_samples": samples[:, 2 * j + 1],
            "n_ops": prep[k]["basis"].shape[0],
        }
    return res, prep, B, Nt


# ── Reporting ─────────────────────────────────────────────────────────────────
def fmt(v, e):
    if not np.isfinite(e) or e < 1e-9:
        return f"{v:.3f}"
    d = max(0, -int(math.floor(math.log10(e))) + 1)
    return f"{v:.{d}f}({round(e * 10**d):d})"


def delta(res, trained, randoms, arm, kind="gevp"):
    """Trained − random on one arm: (Δ, err_conf, err_init, n_tr, n_rnd)."""
    tr = [res[(n, arm, kind)] for n in trained if (n, arm, kind) in res]
    rn = [res[(n, arm, kind)] for n in randoms if (n, arm, kind) in res]
    if not tr or not rn:
        return None
    a_tr = np.array([r["A0"] for r in tr])
    a_rn = np.array([r["A0"] for r in rn])
    d = a_tr.mean() - a_rn.mean()
    s = torch.stack([r["A0_samples"] for r in tr]).mean(0) - \
        torch.stack([r["A0_samples"] for r in rn]).mean(0)
    n = len(s)
    e_conf = math.sqrt((n - 1) / n * ((s - s.mean()) ** 2).sum().item())
    var_tr = a_tr.var(ddof=1) / len(a_tr) if len(a_tr) > 1 else 0.0
    var_rn = a_rn.var(ddof=1) / len(a_rn) if len(a_rn) > 1 else 0.0
    return {"d": d, "conf": e_conf, "init": math.sqrt(var_tr + var_rn),
            "n_tr": len(tr), "n_rnd": len(rn),
            "tr": a_tr.tolist(), "rnd": a_rn.tolist()}


def scramble_null(obar, seed=0):
    """Max |C(Δ)/C(0)| over Δ = 1…7 after an independent config permutation per t."""
    g = torch.Generator().manual_seed(seed)
    B, Nt = obar.shape
    sc = torch.stack([obar[torch.randperm(B, generator=g), t] for t in range(Nt)], dim=1)
    C = connected_correlator(sc)
    return (C[1:8] / C[0]).abs().max().item()


def rel_fluct(mom):
    """Median δf/f over channels from per-site (⟨f⟩, ⟨f²⟩)."""
    s1, s2 = mom[..., 0], mom[..., 1]
    r = (s2 - s1**2).clamp_min(0).sqrt() / s1.abs().clamp_min(1e-30)
    return float(r.median())


def legacy(blob):
    """The original Z₂-estimator number for (trained_ens0, random_s0)."""
    if not {"trained_ens0", "random_s0"} <= set(blob["nets"]):
        return None
    import z2_attention_correlator as zac

    nt = blob["meta"]["Nt"]
    out = {}
    series = {}
    for n in ("trained_ens0", "random_s0"):
        chan = blob["nets"][n]["attn"].double()
        mom = blob["nets"][n]["attn_mom"]
        rel = (mom[:, 1] - mom[:, 0] ** 2).clamp_min(0).sqrt() / mom[:, 0].abs().clamp_min(1e-30)
        chan = chan[rel > zac.MIN_REL_FLUCT]
        out[n] = zac._jack(chan, nt)
        series[n] = zac._resolved_series(chan, out[n])
    dd = zac._corr_delta(series["trained_ens0"], series["random_s0"], nt,
                         wa=out["trained_ens0"].get("window"),
                         wb=out["random_s0"].get("window"))
    return {"A0_tr": out["trained_ens0"]["A0"], "A0_rnd": out["random_s0"]["A0"],
            "dA0": dd["dA0"] if dd else float("nan"),
            "dA0_err": dd["dA0_err"] if dd else float("nan")}


def main():
    blob = torch.load(DUMP, weights_only=False)
    res, prep, B, Nt = run(blob)
    nets = list(blob["nets"])
    trained = [n for n in nets if n.startswith("trained")]
    randoms = [n for n in nets if n.startswith("random")]
    n_layers = len(blob["meta"]["resid_stages"]) - 1
    arms = (["out", "attn"] + [f"attn_L{l + 1}" for l in range(n_layers)]
            + [f"resid_L{s + 1}" for s in range(n_layers)] + ["resid_out"])

    c = res[("classical", "gevp", "gevp")]
    c6 = res[("classical", "APE_last", "gevp")]
    print(f"\nclassical GEVP {blob['meta']['smear_levels']}: m = {fmt(c['m'], c['m_err'])}"
          f"  A₀ = {fmt(c['A0'], c['A0_err'])}   |   APE×{blob['meta']['smear_levels'][-1]}:"
          f" A₀ = {fmt(c6['A0'], c6['A0_err'])}")

    print(f"\nA₀ per network (GEVP projection | best single channel); m from the GEVP arm")
    head = f"  {'arm':<11}" + "".join(f"{n:>22}" for n in nets)
    print(head)
    for arm in arms:
        row = f"  {arm:<11}"
        for n in nets:
            g = res.get((n, arm, "gevp"))
            s = res.get((n, arm, "single"))
            cell = "—" if g is None else fmt(g["A0"], g["A0_err"])
            if s is not None:
                cell += f" | {s['A0']:.2f}"
            row += f"{cell:>22}"
        print(row)
    print(f"  {'m (attn)':<11}" + "".join(
        f"{fmt(res[(n, 'attn', 'gevp')]['m'], res[(n, 'attn', 'gevp')]['m_err']):>22}"
        for n in nets))
    print(f"  {'m (out)':<11}" + "".join(
        f"{fmt(res[(n, 'out', 'gevp')]['m'], res[(n, 'out', 'gevp')]['m_err']):>22}"
        for n in nets))

    deltas = {}
    if trained and randoms:
        print(f"\ntrained − random, ΔA₀ = mean over {len(trained)} trained − mean over "
              f"{len(randoms)} random   [± conf (jackknife) ± init (between nets)]")
        for arm in arms:
            for kind in ("gevp", "single"):
                d = delta(res, trained, randoms, arm, kind)
                if d is None:
                    continue
                deltas[(arm, kind)] = d
                tot = math.hypot(d["conf"], d["init"])
                print(f"  {arm:<11} {kind:<6}  ΔA₀ = {d['d']:+.3f} ± {d['conf']:.3f} "
                      f"± {d['init']:.3f}   ({d['d'] / tot if tot else float('nan'):+.1f}σ)")
        print("\ninheritance control — per block, attention vs the residual stream "
              "entering it (GEVP arms):")
        for l in range(n_layers):
            a = deltas.get((f"attn_L{l + 1}", "gevp"))
            r = deltas.get((f"resid_L{l + 1}", "gevp"))
            if a and r:
                print(f"  block {l + 1}:  ΔA₀(attention) = {a['d']:+.3f}   "
                      f"ΔA₀(residual in) = {r['d']:+.3f}   "
                      f"A₀ trained: attention {np.mean(a['tr']):.3f}, residual "
                      f"{np.mean(r['tr']):.3f}")
        # Per-pair differences: the spread a single random seed hides.
        print("\nper-pair ΔA₀ on the full attention field (GEVP):")
        for t in trained:
            print("  " + t + ": " + "  ".join(
                f"{res[(t, 'attn', 'gevp')]['A0'] - res[(r, 'attn', 'gevp')]['A0']:+.3f}"
                for r in randoms))

    print("\nchecks:")
    nulls = {}
    for n in nets:
        if (n, "attn") in prep:
            ob = project(prep[(n, "attn")]["basis"])
            nulls[n] = scramble_null(ob)
    print("  config-scramble null, max |C(Δ)/C(0)|, Δ=1…7: " +
          "  ".join(f"{n} {v:.3f}" for n, v in nulls.items()))
    fl = {n: (rel_fluct(blob["nets"][n]["attn_mom"]),
              rel_fluct(blob["nets"][n]["resid_mom"])) for n in nets}
    print("  median δf/f (attention | residual): " +
          "  ".join(f"{n} {a:.3g}|{r:.3g}" for n, (a, r) in fl.items()))
    leg = legacy(blob)
    if leg:
        print(f"  legacy Z₂ estimator (trained_ens0 vs random_s0): A₀ {leg['A0_tr']:.2f}/"
              f"{leg['A0_rnd']:.2f}, ΔA₀ = {leg['dA0']:+.3f} ± {leg['dA0_err']:.3f} "
              f"(published +0.286 ± 0.056)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save({
        "res": {k: {kk: vv for kk, vv in v.items() if kk != "A0_samples"}
                for k, v in res.items()},
        "deltas": deltas, "nulls": nulls, "rel_fluct": fl, "legacy": leg,
        "meta": {**blob["meta"], "block": BLOCK, "window": WINDOW,
                 "t0": T0, "td": TD, "eps": EPS, "dump": DUMP, "B": B},
    }, OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
