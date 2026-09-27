"""Which cosh fit is biased on A₀? — the synthetic test behind S1 task 2.

``fit_window_scan.py`` found that at [2, 7] the fully correlated fit gives
ΔA₀ = +0.031 ± 0.021 where the diagonal fit every table uses gives +0.077 ±
0.022, with the whole drop at covariance shrinkage λ < 0.1. Residuals of one
sign along the covariance's flat common mode are *not* by themselves a sign
of failure — a correlated fit that knows the points move together should
decline to follow a common shift — so the question is empirical: at this
noise, with the covariance estimated from 40 blocks, which estimator recovers
a known ΔA₀?

The test (a parametric bootstrap, per ensemble, [2, 7] only):

- **Truth.** For each operator, C(Δ ≥ 1) is a pure cosh A·[e^{−mΔ} + e^{−m(Nt−Δ)}]
  and C(0) is the observed one, so A₀_true = A(1 + e^{−mNt})/C(0) is known and
  the model is exact in every window. (m, A) are taken from the real data's
  diagonal fits (truth **T_diag**, ΔA₀_true = the published number) and, as a
  cross-check that the answer does not depend on which estimator set the
  truth, from its correlated fits (**T_corr**).
- **Noise.** The measured blocked-jackknife covariance Σ of the joint vector
  [C_GELT(Δ), C_GEVP(Δ)] — so C(0)–C(Δ) and GELT–GEVP correlations are the
  data's. One synthetic experiment draws 40 Gaussian block vectors whose mean
  has covariance Σ, then does exactly what the scan does: the jackknife
  covariance Σ̂ from those 40 blocks, σ = √diag Σ̂, and ``fgo.fit_one`` at every
  λ of ``fws.SHRINK`` through ``fws.shrunk`` (λ = 1 is the published
  estimator, λ = 0 the correlated fit).
- **Not modelled**: v₀ refitted per jackknife sample (the projected operator is
  fixed here), non-Gaussian tails, and the fact that Σ itself is a 40-sample
  estimate of the true covariance — its small eigenvalues are the synthetic
  world's truth. The gate below checks the first two cannot matter much.

Readings (fixed before the first run), on the **combined** numbers under
**T_diag**; T_corr is printed beside it and any reading it flips is named:

- **Gate.** The synthetic spread of the diagonal-fit ΔA₀ must be within 25%
  of its real jackknife error on each ensemble, or the noise model is not the
  data's and nothing below is read.
- **R0.** |bias of the diagonal fit| ≥ 0.01 → the published estimator is
  biased by that much at this noise; the headline is corrected by it.
- **R1.** bias(corr) − bias(diag) has the sign of the observed difference
  D_obs = ΔA₀(corr) − ΔA₀(diag), is at least half of it, and |bias(corr)| ≥
  0.01 → the drop is an estimator bias of the correlated fit at 40 blocks:
  +0.031 is an artifact, the diagonal fit stays, the caveat is one sentence.
- **R2.** Otherwise, if the observed D_obs is inside the null (two-sided
  p ≥ 0.05 around the synthetic mean of D) → the disagreement is noise-sized.
  Then RMSE(corr) ≤ 0.8 · RMSE(diag) → the correlated fit is the more precise
  estimator and +0.031 is quoted beside +0.077 as an equally valid estimate;
  else → the diagonal fit stays and +0.031 is quoted as a noisier alternative.
- **R3.** Otherwise (p < 0.05 and no estimator bias explains it) → neither
  noise nor covariance-estimation bias under a single-state truth; what is
  left is model misspecification inside the window, and S1 task 5 (the
  two-state fit) is required before the headline is quoted.

Offline, ~15 s on a laptop CPU, no GPU; reads the tracked dumps.

Run:
    python scripts/fit_estimator_mc.py
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_glueball_overlap as fgo  # noqa: E402
import fit_window_scan as fws  # noqa: E402  (its argv check: this script takes no flags)

from gelt.glueball import connected_correlator  # noqa: E402


# ── Tunables ──────────────────────────────────────────────────────────────────
N_EXP = 1000  # synthetic experiments per (ensemble, truth); MC error on a bias ≈ sd/32
SEED = 0
WINDOW = fws.REFERENCE  # (2, 7)
SHRINK = fws.SHRINK  # (0, 0.1, 0.25, 0.5, 1)
GATE_TOL = 0.25  # synthetic sd of ΔA₀(diag) vs its real jackknife error
BIAS_MATERIAL = 0.01  # half the combined statistical error of the headline
P_NULL = 0.05
RMSE_RATIO = 0.8
OUT = "results/glueball/fit_estimator_mc.pt"


def estimate(C, Sigma, Nt):
    """(n_λ, 4) = (m, A₀) of GELT then GEVP at every λ, as the scan fits them."""
    rows = []
    for lam in SHRINK:
        row = []
        for sl in (slice(0, Nt), slice(Nt, 2 * Nt)):
            S = Sigma[sl, sl]
            m, a0, _ = fgo.fit_one(C[sl], fws.shrunk(S, S.diagonal().sqrt(), lam), WINDOW)
            row += [m, a0]
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float64)


def real_inputs(path):
    """Full-sample joint correlator, its jackknife covariance, and the scan's rows."""
    blob = torch.load(path)
    g = blob["gelt_obar"].double()
    basis = blob["Obar_basis"].double()
    meta = blob.get("meta", {})
    t0 = int(meta.get("gevp_t0", 1))
    td = t0 + 1 if fgo.GEVP_TD is None else fgo.GEVP_TD
    jb = int(meta.get("jack_block", 10))
    B, Nt = g.shape
    proj = fgo.project_ground(basis, t0, td)  # fixed v₀, as σ_Δ in the scan

    def joint(m):
        return torch.cat([connected_correlator(g[m]), connected_correlator(proj[m])])

    J = fws.jackknife_samples(joint, B, jb)
    return joint(torch.ones(B, dtype=torch.bool)), fws.jackknife_cov(J), J.shape[0], Nt


def truth(C, Nt, est_row):
    """C with Δ ≥ 1 replaced by the pure cosh of one estimator's (m, A₀) row."""
    mu = C.clone()
    dd = torch.arange(Nt, dtype=torch.float64)
    for i, (m, a0) in enumerate((est_row[0:2].tolist(), est_row[2:4].tolist())):
        c0 = C[i * Nt].item()
        A = a0 * c0 / (1.0 + math.exp(-m * Nt))  # fit_one's A₀ map, inverted
        f = A * (torch.exp(-m * dd) + torch.exp(-m * (Nt - dd)))
        mu[i * Nt + 1 : (i + 1) * Nt] = f[1:]
    return mu


def simulate(mu, Sigma, n_blk, Nt, gen):
    """N_EXP synthetic experiments → (N_EXP, n_λ, 4) estimates."""
    ev, V = torch.linalg.eigh(Sigma)
    keep = ev > 1e-12 * ev[-1]
    L = V[:, keep] * ev[keep].sqrt()  # Σ = L Lᵀ on its support (rank ≤ n_blk − 1)
    out = torch.empty(N_EXP, len(SHRINK), 4, dtype=torch.float64)
    for e in range(N_EXP):
        z = torch.randn(n_blk, L.shape[1], generator=gen, dtype=torch.float64)
        y = mu + math.sqrt(n_blk) * z @ L.T  # 40 block vectors; their mean has cov Σ
        ybar = y.mean(0)
        J = (n_blk * ybar - y) / (n_blk - 1)  # delete-one-block samples
        out[e] = estimate(ybar, fws.jackknife_cov(J), Nt)
    return out


def main():
    gen = torch.Generator().manual_seed(SEED)
    i_d, i_c = SHRINK.index(1.0), SHRINK.index(0.0)
    res, obs, jk_err = {}, {}, {}
    for path in fws.DUMPS:  # every default dump present, ens2 included once copied
        tag = fws._tag(path)
        C, Sigma, n_blk, Nt = real_inputs(path)
        est = estimate(C, Sigma, Nt)  # the real data's full-sample fits
        obs[tag] = est
        # The real jackknife errors of ΔA₀ per λ: the gate's reference, and the
        # weights the real combination used.
        scan = fws.scan_dump(path)["windows"][WINDOW]["shrink"]
        jk_err[tag] = torch.tensor([scan[lam]["dA0"][1] for lam in SHRINK])
        res[tag] = {}
        for tname, row in (("T_diag", est[i_d]), ("T_corr", est[i_c])):
            mu = truth(C, Nt, row)
            sim = simulate(mu, Sigma, n_blk, Nt, gen)
            edge = ((sim[..., [0, 2]] - fgo.M_RANGE[0]).abs() < 1e-6).sum() + (
                (sim[..., [0, 2]] - fgo.M_RANGE[1]).abs() < 1e-6).sum()
            res[tag][tname] = {"sim": sim, "true": row, "edge": int(edge)}

    tags = list(res)
    w = {t: 1.0 / jk_err[t] ** 2 for t in tags}  # (n_λ,) per ensemble
    wsum = sum(w.values())

    def dA0(x):  # (..., n_λ, 4) → (..., n_λ)
        return x[..., 1] - x[..., 3]

    report = {}
    for tname in ("T_diag", "T_corr"):
        print(f"\n══ truth {tname}: C(Δ ≥ 1) = the real data's "
              f"{'diagonal' if tname == 'T_diag' else 'correlated'} cosh fit, C(0) observed; "
              f"{N_EXP} experiments per ensemble, window {list(WINDOW)}")
        per = {}
        for t in tags:
            sim, tr = res[t][tname]["sim"], res[t][tname]["true"]
            per[t] = {"d": dA0(sim), "true": (tr[1] - tr[3]).item(),
                      "a0g": sim[..., 1] - tr[1], "a0p": sim[..., 3] - tr[3],
                      "D_obs": (dA0(obs[t])[i_c] - dA0(obs[t])[i_d]).item()}
            if res[t][tname]["edge"]:
                print(f"  WARNING {t}: {res[t][tname]['edge']} synthetic fits at the m grid edge")
        # Combined exactly as the real numbers are: inverse variance with the
        # real jackknife errors, per λ.
        comb_d = sum(w[t] * per[t]["d"] for t in tags) / wsum
        comb_true = sum(w[t] * per[t]["true"] for t in tags) / wsum  # (n_λ,)
        comb_Dobs = (sum(w[t][i_c] * dA0(obs[t])[i_c] for t in tags) / wsum[i_c]
                     - sum(w[t][i_d] * dA0(obs[t])[i_d] for t in tags) / wsum[i_d]).item()
        rows = {t: (per[t]["d"], torch.full((len(SHRINK),), per[t]["true"]), per[t]["D_obs"])
                for t in tags}
        rows["combined"] = (comb_d, comb_true, comb_Dobs)

        report[tname] = {}
        for t, (d, true, D_obs) in rows.items():
            bias = d.mean(0) - true
            sd = d.std(0)
            rmse = ((d - true) ** 2).mean(0).sqrt()
            D = d[:, i_c] - d[:, i_d]
            p = ((D - D.mean()).abs() >= abs(D_obs - D.mean().item())).double().mean().item()
            report[tname][t] = {"bias": bias, "sd": sd, "rmse": rmse, "true": true[i_d].item(),
                                "D_mean": D.mean().item(), "D_sd": D.std().item(),
                                "D_obs": D_obs, "p_D": p}
            print(f"\n  {t}: ΔA₀_true = {true[i_d].item():+.4f}")
            print(f"    {'λ':>6} {'bias ± MC':>18} {'sd':>8} {'RMSE':>8}"
                  + (f" {'real jk err':>12}" if t in jk_err else ""))
            for k, lam in enumerate(SHRINK):
                tag_ = "  ← the estimator" if lam == 1.0 else ("  ← correlated fit" if lam == 0.0 else "")
                print(f"    {lam:>6g} {bias[k].item():+.4f} ± {sd[k].item() / math.sqrt(N_EXP):.4f}"
                      f" {sd[k].item():8.4f} {rmse[k].item():8.4f}"
                      + (f" {jk_err[t][k].item():12.4f}" if t in jk_err else "") + tag_)
            if t in per:
                for k, lam in ((i_c, 0.0), (i_d, 1.0)):
                    print(f"    A₀ bias at λ = {lam:g}:  GELT {per[t]['a0g'][:, k].mean().item():+.4f}"
                          f"   GEVP {per[t]['a0p'][:, k].mean().item():+.4f}")
            print(f"    D = ΔA₀(corr) − ΔA₀(diag): null {D.mean().item():+.4f} ± {D.std().item():.4f}"
                  f"   observed {D_obs:+.4f}   two-sided p = {p:.3f}")

    # ── Gate and readings ──────────────────────────────────────────────────────
    print("\n══ gate: synthetic sd of ΔA₀(diag) against its real jackknife error")
    ok = True
    for t in tags:
        sd, je = report["T_diag"][t]["sd"][i_d].item(), jk_err[t][i_d].item()
        good = abs(sd / je - 1) <= GATE_TOL
        ok &= good
        print(f"  {t}: {sd:.4f} vs {je:.4f}  (ratio {sd / je:.2f}, need within ±{GATE_TOL:.0%})"
              f"  {'ok' if good else 'FAILED'}")
    if not ok:
        raise SystemExit("gate failed: the noise model is not the data's; no reading.")

    def reading(r):
        b_d, b_c = r["bias"][i_d].item(), r["bias"][i_c].item()
        D_obs, shift = r["D_obs"], b_c - b_d
        if abs(b_d) >= BIAS_MATERIAL:
            yield "R0", f"the diagonal fit is biased by {b_d:+.4f} — correct the headline by it"
        if (shift * D_obs > 0 and abs(shift) >= abs(D_obs) / 2 and abs(b_c) >= BIAS_MATERIAL):
            yield "R1", (f"the correlated fit is biased ({b_c:+.4f}, shift {shift:+.4f} vs observed "
                         f"{D_obs:+.4f}): +0.031 is an estimator artifact; the diagonal fit stays")
        elif r["p_D"] >= P_NULL:
            ratio = r["rmse"][i_c].item() / r["rmse"][i_d].item()
            if ratio <= RMSE_RATIO:
                yield "R2", (f"noise-sized (p = {r['p_D']:.2f}) and the correlated fit is the more "
                             f"precise (RMSE ratio {ratio:.2f}): quote +0.031 beside +0.077")
            else:
                yield "R2", (f"noise-sized (p = {r['p_D']:.2f}), RMSE ratio {ratio:.2f} > "
                             f"{RMSE_RATIO}: the diagonal fit stays, +0.031 is a noisier alternative")
        else:
            yield "R3", (f"not noise (p = {r['p_D']:.3f}) and no estimator bias explains it: "
                         f"misspecification inside the window — S1 task 5 is required")

    readings = {tn: list(reading(report[tn]["combined"])) for tn in ("T_diag", "T_corr")}
    print("\n══ readings (combined; T_diag is the reading, T_corr the cross-check)")
    for tn, rs in readings.items():
        for code, text in rs:
            print(f"  {tn}  {code}: {text}")
    if [c for c, _ in readings["T_diag"]] != [c for c, _ in readings["T_corr"]]:
        print("  → the two truths give different readings: say so wherever this is quoted")

    torch.save({"n_exp": N_EXP, "seed": SEED, "window": WINDOW, "shrink": SHRINK,
                "report": report, "readings": readings,
                "observed": {t: obs[t] for t in tags}, "jk_err": jk_err}, OUT)
    print(f"\n  saved {OUT}")


if __name__ == "__main__":
    main()
