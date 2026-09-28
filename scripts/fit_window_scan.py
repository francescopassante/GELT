"""Fit robustness of the 0⁺⁺ headline — the supervisor's S1, tasks 1, 2 and 4.

The 3.6σ of ΔA₀ = +0.078 ± 0.022 (notes/thesis_record.md §3.2) is a statistical
error *given* the analysis choices; it does not contain the systematic from the
cosh-fit window, which ``fit_glueball_overlap.FIT_WINDOW`` fixes at Δ ∈ [2, 7]
and whose docstring says "should be spot-checked by editing it" — it never was,
systematically. This scan repeats the headline measurement on every window of
notes/prof_notes.md S1 and applies the readings fixed there.

The estimator is **imported**, not copied: ``fit_one`` (the cosh fit + A₀),
``project_ground`` (v₀ at (t0, td), refitted in every jackknife sample),
``blocked_jackknife`` and ``M_RANGE`` all come from ``fit_glueball_overlap``,
so the [2, 7] row is the published measurement by construction — and is gated
on reproducing it before any other row is read. σ_Δ is fixed from the full
sample (window-independent, as there), so every window minimises the same
diagonal χ² surface it would there.

Task 1 — per window and per ensemble: m and A₀ for GELT and the projected
GEVP, the jackknifed differences Δm and ΔA₀ (correlated, same configurations).
The two ensembles are combined by inverse variance, as the record combines them.

Task 2 — the correlated χ². The covariance of C(Δ) is the blocked-jackknife
one, from the same delete-block samples as σ_Δ (its diagonal is σ_Δ², checked).
Printed per window and operator:

- χ²/dof of the diagonal fit, as fit_glueball_overlap prints it;
- the correlated χ² = rᵀ Σ⁻¹ r **at the diagonal fit's (m, A)** — the one the
  supervisor asks for, since those are the fits quoted — with its p-value
  (conservative: χ² at a non-minimising point is stochastically larger than
  χ²(dof));
- the correlated fit's own minimum and p-value;
- cond of the window's *correlation* matrix (the covariance's condition number
  mostly measures C's own decay over the window).

Σ⁻¹ is multiplied by the Hartlap factor (n − p − 2)/(n − 1), n = jackknife
blocks, p = window points (32/39 for [2,7] at 40 blocks), the standard
debiasing of an inverted sample covariance; it rescales χ² and leaves the
minimiser alone. The correlated fit — covariance fixed from the full sample,
refitted in every jackknife sample, like σ_Δ — is printed as a check of the
diagonal one and **does not replace it** (S1 task 2: "do not change the
central estimator silently").

The shrinkage scan was added after the first run, and is labelled so: the
correlated fit moved ΔA₀ from +0.077 to +0.031, and the correlation matrix of
C over the window is ~85% one common mode (top eigenvector uniform in Δ,
eigenvalues ≈ 5.0, 0.5–1.0, 0.1–0.3, then ≤ 0.06). The correlated fit sits
off the data along that mode — every residual one sign — and fitting C/C(0)
instead does not remove it (+0.036 ± 0.020), so it is the weighting and not
the A₀ ratio. That this was a failure of the correlated fit at 40 blocks was
the first guess and is **wrong**: ``fit_estimator_mc.py`` finds both fits
unbiased at this noise and the gap a p = 0.057 fluctuation. ``SHRINK`` fits with
Σ_λ = (1 − λ)Σ + λ·diag Σ, which is the fully correlated fit at λ = 0 and
**the estimator itself** at λ = 1 (routed through the σ path, bit-identical),
so the scan shows where between the two the number sits. No λ is picked.

Task 4 — m_eff(1) against the fitted mass. In every jackknife sample
d = m_eff(Δ) − m_fit([2, 7]) for the same operator, so the correlations are
automatic; GELT at Δ = 1 is the reading, GELT at Δ = 2 and the GEVP at Δ = 1,
2 are the contrast. m_eff is the log ratio (``effective_mass``), as main.tex
plots it and as the published Δm_eff(1) is defined; on a pure cosh at the
fitted masses it reads 0.0006 (run5) / 0.0003 (ens1) below m at Δ = 1 — the
backward wave — which is noise at these errors. The GEVP's m_eff is the
projected operator's: at Δ = t0 = 1 with td = 2 it equals the principal
correlator's log λ₀(1)/λ₀(2) identically (v₀ is the td eigenvector and
v₀ᵀC(t0)v₀ = 1), which is how Δm_eff(1)(GELT − GEVP) is gated on reproducing
the record's −0.028(7) / −0.038(8) with train_glueball.py's definition.

Readings (notes/prof_notes.md S1, fixed before the numbers existed), applied
to the **combined** values — the ones the abstract quotes:

- task 1: ΔA₀ ≥ 2σ in every admitted window and max − min of its central
  values below the [2, 7] error → "stable under the fit window": keep 3.6σ as
  the headline and quote the spread as a systematic. Otherwise → quote ΔA₀ as
  the [2, 7] value ± stat ± (half the spread across windows) and rewrite the
  abstract's "3.6σ".
- task 4: |d|/σ_d < 1.5 for GELT at Δ = 1 → "compatible with a plateau from
  Δ = 1 within errors". Otherwise → main.tex l. 59, 406, 427, 436, 505 say
  "reduced excited-state contamination already at Δ = 1" instead of
  "plateaus at Δ = 1".

Admission: [2, 7], [3, 7] and [2, 6] are the supervisor's required set and are
always admitted. The Δmax = 8 windows are "if the signal allows", which is
fixed here as: C(8) at ≥ 2σ from zero (full sample, blocked-jackknife σ) for
both operators on every ensemble. They are printed either way.

Task 3 — the thesis artifacts, written beside the .pt: ``…fit_window_scan.tex``
(a booktabs tabular, one block per ensemble and one combined; rows = windows)
and ``…fit_window_scan.png`` (ΔA₀ against the window, the diagonal fit per
ensemble and combined). Both are regenerated by every run, so they cannot drift from
the numbers printed above them.

Offline, seconds, no GPU: reads the tracked ``dumps/*_test_obars.pt``.

Run:
    python scripts/fit_window_scan.py [dump.pt ...] [--windows=2-7,3-7,2-6]
"""

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

# Refuse a flag this script does not read, before anything else sees argv: an
# unrecognised --name=value would otherwise run the defaults while the command
# line claimed otherwise (the probe's validate_argv discipline).
_KNOWN_FLAGS = ("--windows=",)
_bad = [a for a in sys.argv[1:] if a.startswith("--") and not a.startswith(_KNOWN_FLAGS)]
if _bad:
    raise SystemExit(f"unknown flag(s) {_bad}; known: {list(_KNOWN_FLAGS)}")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_glueball_overlap as fgo  # noqa: E402

from gelt.glueball import connected_correlator, effective_mass  # noqa: E402


# ── Tunables ──────────────────────────────────────────────────────────────────
REFERENCE = fgo.FIT_WINDOW  # (2, 7) — the published window
REQUIRED = [(2, 7), (3, 7), (2, 6)]  # the supervisor's minimum set
EXTENDED = [(3, 8), (2, 8)]  # "if the signal allows" — gated on SNR_MIN at Δmax
SNR_MIN = 2.0
PLATEAU_SIGMA = 1.5  # task 4's threshold on |d|/σ_d
# Covariance shrinkage toward the diagonal: 0 = fully correlated fit, 1 = the
# diagonal fit every table uses. Both endpoints are always in.
SHRINK = (0.0, 0.1, 0.25, 0.5, 1.0)
_flag = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--windows=")), None)
WINDOWS = (
    [tuple(int(x) for x in w.split("-")) for w in _flag.split(",")]
    if _flag
    else REQUIRED + EXTENDED
)
if REFERENCE not in WINDOWS:
    WINDOWS = [REFERENCE] + WINDOWS  # every row is read against it

_POSITIONAL = [a for a in sys.argv[1:] if not a.startswith("--")]
DEFAULT_DUMPS = [
    "dumps/best_glueball_gelt_sm0-2-4-6_test_obars.pt",
    "dumps/best_glueball_gelt_sm0-2-4-6_ens1_test_obars.pt",
    # The third ensemble (scripts/ens2_batch.sh): read when its dump is in dumps/.
    "dumps/best_glueball_gelt_sm0-2-4-6_ens2_test_obars.pt",
]
DUMPS = _POSITIONAL or [d for d in DEFAULT_DUMPS if os.path.exists(d)]
# The gates: record §3.2's per-ensemble ΔA₀ at the reference window and
# Δm_eff(1)(GELT − GEVP), as printed (three decimals). Checked only when the
# default dumps are scanned.
PUBLISHED_DA0 = {"run5": (0.066, 0.031), "ens1": (0.089, 0.030)}
PUBLISHED_DMEFF1 = {"run5": (-0.028, 0.007), "ens1": (-0.038, 0.008)}
PUBLISHED_TOL = 1.5e-3  # the printed rounding, with room for the 4e-7 drift
# A custom run must not overwrite the canonical output.
OUT = "results/glueball/fit_window_scan" + (
    "_custom" if (_flag or _POSITIONAL) else "") + ".pt"

FIT_KEYS = ["m_g", "A0_g", "m_p", "A0_p", "dm", "dA0"]
MEFF_KEYS = ["meff1_g", "meff2_g", "meff1_p", "meff2_p", "mfit_g", "mfit_p",
             "d1_g", "d2_g", "d1_p", "d2_p", "dmeff1"]


def _tag(dump):
    """Short ensemble name: the Run-5 anchor, or a replication seed."""
    b = os.path.basename(dump)
    return f"ens{b.split('_ens')[1].split('_')[0]}" if "_ens" in b else "run5"


def _combine(vals):
    """Inverse-variance mean and error of ``[(x, σ), …]``."""
    w = [1.0 / e**2 for _, e in vals]
    return sum(wi * x for wi, (x, _) in zip(w, vals)) / sum(w), 1.0 / math.sqrt(sum(w))


def jackknife_samples(fn, B, jb):
    """The delete-block samples ``fgo.blocked_jackknife`` averages, kept."""
    out = []
    for bl in fgo._blocks(B, jb):
        mask = torch.ones(B, dtype=torch.bool)
        mask[bl] = False
        out.append(fn(mask))
    return torch.stack(out)


def jackknife_cov(samples):
    """Blocked-jackknife covariance of the rows of ``samples`` (n, Nt)."""
    n = samples.shape[0]
    d = samples - samples.mean(dim=0)
    return (n - 1) / n * d.T @ d


def hartlap(n, p):
    """Debiasing factor of an inverted n-sample covariance of dimension p."""
    return (n - p - 2) / (n - 1)


def chi2_corr(C, m, a0, cov, w):
    """rᵀ Σ⁻¹ r on window ``w`` at the cosh (m, A₀) ``fgo.fit_one`` returned."""
    Nt = C.shape[0]
    A = a0 * C[0].item() / (1.0 + math.exp(-m * Nt))  # fit_one's A₀ map, inverted
    dd = torch.arange(w[0], w[1] + 1, dtype=torch.float64)
    r = C[w[0] : w[1] + 1] - A * (torch.exp(-m * dd) + torch.exp(-m * (Nt - dd)))
    return (r @ torch.linalg.inv(cov[w[0] : w[1] + 1, w[0] : w[1] + 1]) @ r).item()


def shrunk(cov, sig, lam):
    """The fit weight at shrinkage λ: Σ_λ = (1 − λ)Σ + λ·diag Σ.

    λ = 1 goes through the σ path, so it IS the estimator, not a copy of it.
    """
    return sig if lam == 1.0 else (1 - lam) * cov + lam * torch.diag(sig**2)


def pvalue(chi2, dof):
    """Upper tail of χ²(dof)."""
    return torch.special.gammaincc(
        torch.tensor(dof / 2, dtype=torch.float64), torch.tensor(chi2 / 2, dtype=torch.float64)
    ).item()


def scan_dump(path):
    """Every window's fits on one dump, the goodness of fit, and the m_eff tests."""
    blob = torch.load(path)
    gelt_obar = blob["gelt_obar"].double()
    basis = blob["Obar_basis"].double()
    meta = blob.get("meta", {})
    t0 = int(meta.get("gevp_t0", 1))
    td = t0 + 1 if fgo.GEVP_TD is None else fgo.GEVP_TD
    jb = int(meta.get("jack_block", 10))
    B, Nt = gelt_obar.shape
    n_blk = -(-B // jb)
    print(f"── {os.path.basename(path)} ({_tag(path)}): {B} test configs × Nt = {Nt}, "
          f"GEVP (t0, td) = ({t0}, {td}), block {jb} → {n_blk} jackknife blocks")
    if (t0, td) != (1, 2):
        raise SystemExit("the m_eff(1) identity with the principal correlator needs "
                         f"(t0, td) = (1, 2); this dump has ({t0}, {td})")

    proj_full = fgo.project_ground(basis, t0, td)
    # σ_Δ exactly as the estimator takes it, and the full covariance from the
    # same delete-block samples (fixed full-sample v₀, as σ_Δ is).
    _, sig_g = fgo.blocked_jackknife(lambda m: connected_correlator(gelt_obar[m]), B, jb)
    _, sig_p = fgo.blocked_jackknife(lambda m: connected_correlator(proj_full[m]), B, jb)
    cov_g = jackknife_cov(jackknife_samples(
        lambda m: connected_correlator(gelt_obar[m]), B, jb))
    cov_p = jackknife_cov(jackknife_samples(
        lambda m: connected_correlator(proj_full[m]), B, jb))
    for cov, sig in ((cov_g, sig_g), (cov_p, sig_p)):
        if not torch.allclose(cov.diagonal().sqrt(), sig, rtol=1e-10, atol=0):
            raise SystemExit("jackknife covariance diagonal ≠ σ_Δ² — the blocks differ")
    C_g, C_p = connected_correlator(gelt_obar), connected_correlator(proj_full)
    i_ref = WINDOWS.index(REFERENCE)
    edges = fgo.M_RANGE
    weights = [(shrunk(cov_g, sig_g, lam), shrunk(cov_p, sig_p, lam)) for lam in SHRINK]
    i_diag, i_corr = SHRINK.index(1.0), SHRINK.index(0.0)

    # One jackknife pass: v₀ and both correlators are window-independent, so each
    # sample is projected once and fitted per window, diagonal and correlated.
    def stats(mask):
        Cg = connected_correlator(gelt_obar[mask])
        Cp = connected_correlator(fgo.project_ground(basis[:, mask], t0, td))
        rows, at_edge = [], 0
        for w in WINDOWS:
            row = []
            for s_g, s_p in weights:
                m_g, a_g, _ = fgo.fit_one(Cg, s_g, w)
                m_p, a_p, _ = fgo.fit_one(Cp, s_p, w)
                row.append([m_g, a_g, m_p, a_p, m_g - m_p, a_g - a_p])
                at_edge += sum(min(abs(x - edges[0]), abs(x - edges[1])) < 1e-6
                               for x in (m_g, m_p))
            rows.append(row)
        fits = torch.tensor(rows, dtype=torch.float64)  # (window, λ, 6)
        eg, ep = effective_mass(Cg), effective_mass(Cp)
        mg, mp = fits[i_ref, i_diag, 0], fits[i_ref, i_diag, 2]
        meff = torch.stack([eg[1], eg[2], ep[1], ep[2], mg, mp,
                            eg[1] - mg, eg[2] - mg, ep[1] - mp, ep[2] - mp, eg[1] - ep[1]])
        return torch.cat([fits.flatten(), meff, torch.tensor([float(at_edge)])])

    mean, err = fgo.blocked_jackknife(stats, B, jb)
    shape = (len(WINDOWS), len(SHRINK), 6)
    nf = math.prod(shape)
    fit_m, fit_e = mean[:nf].view(shape), err[:nf].view(shape)
    if mean[-1] > 0:
        print(f"  WARNING: {mean[-1].item() * n_blk:.0f} jackknife fits hit the m grid "
              f"edge {edges} — those rows are not measurements")

    out = {"n_blocks": n_blk, "windows": {}, "meff": {}}
    for i, w in enumerate(WINDOWS):
        p = w[1] - w[0] + 1
        dof, h = p - 2, hartlap(n_blk, p)
        row = {"shrink": {lam: {k: (fit_m[i, l, j].item(), fit_e[i, l, j].item())
                                for j, k in enumerate(FIT_KEYS)}
                          for l, lam in enumerate(SHRINK)},
               "dof": dof, "hartlap": h}
        row["diag"], row["corr"] = row["shrink"][1.0], row["shrink"][0.0]
        for op, C, sig, cov in (("g", C_g, sig_g, cov_g), ("p", C_p, sig_p, cov_p)):
            m, a0, chi2_d = fgo.fit_one(C, sig, w)
            chi2_c = h * chi2_corr(C, m, a0, cov, w)
            chi2_cf = h * fgo.fit_one(C, cov, w)[2]
            sd = cov[w[0] : w[1] + 1, w[0] : w[1] + 1].diagonal().sqrt()
            rho = cov[w[0] : w[1] + 1, w[0] : w[1] + 1] / (sd[:, None] * sd[None, :])
            row[op] = {
                "chi2dof_diag": chi2_d / dof,
                "chi2dof_corr": chi2_c / dof, "p_corr": pvalue(chi2_c, dof),
                "chi2dof_corrfit": chi2_cf / dof, "p_corrfit": pvalue(chi2_cf, dof),
                "chi2_corr": chi2_c,
                "cond_rho": torch.linalg.cond(rho).item(),
                "snr": (C[w[1]] / sig[w[1]]).item(),
            }
        out["windows"][w] = row
    for j, k in enumerate(MEFF_KEYS):
        out["meff"][k] = (mean[nf + j].item(), err[nf + j].item())
    return out


def _fmt(v):
    return f"{v[0]:.4f}({round(v[1] * 1e4):>3d})"


def _pm(v, sig=True):
    s = f"{v[0]:+.4f} ± {v[1]:.4f}"
    return s + (f" ({v[0] / v[1]:+.1f}σ)" if sig else "")


def _w(w):
    return f"[{w[0]},{w[1]}]{'*' if w == REFERENCE else ' '}"


def print_ensemble(tag, res):
    rows = res["windows"]
    print(f"\n  {tag} — diagonal cosh fits, the estimator every table uses "
          f"(* = the published window; errors in units of 1e-4)")
    print(f"  {'window':<8} {'m GELT':>12} {'A₀ GELT':>12} {'m GEVP':>12} {'A₀ GEVP':>12}"
          f" {'Δm':>18} {'ΔA₀':>25}")
    for w, r in rows.items():
        d = r["diag"]
        print(f"  {_w(w)}  {_fmt(d['m_g']):>12} {_fmt(d['A0_g']):>12} {_fmt(d['m_p']):>12} "
              f"{_fmt(d['A0_p']):>12}  {_pm(d['dm'], False):>18}  {_pm(d['dA0']):>25}")

    print(f"\n  {tag} — goodness of fit (correlated χ² × Hartlap; p in brackets)")
    print(f"  {'window':<8} {'op':<5} {'diag':>6}   {'corr @ diag fit':>16}   "
          f"{'corr fit (min)':>16}   {'cond ρ':>8}   S/N C(Δmax)")
    for w, r in rows.items():
        for op, nm in (("g", "GELT"), ("p", "GEVP")):
            g = r[op]
            print(f"  {_w(w) if op == 'g' else '':<8} {nm:<5} {g['chi2dof_diag']:6.2f}   "
                  f"{g['chi2dof_corr']:7.2f} ({g['p_corr']:.2f})   "
                  f"{g['chi2dof_corrfit']:7.2f} ({g['p_corrfit']:.2f})   "
                  f"{g['cond_rho']:8.1e}   {g['snr']:5.1f}")

    print(f"\n  {tag} — correlated cosh fits (a check; not the estimator)")
    print(f"  {'window':<8} {'m GELT':>12} {'A₀ GELT':>12} {'m GEVP':>12} {'A₀ GEVP':>12}"
          f" {'ΔA₀ corr fit':>25} {'ΔA₀ diag fit':>25}")
    for w, r in rows.items():
        c, d = r["corr"], r["diag"]
        print(f"  {_w(w)}  {_fmt(c['m_g']):>12} {_fmt(c['A0_g']):>12} {_fmt(c['m_p']):>12} "
              f"{_fmt(c['A0_p']):>12}  {_pm(c['dA0']):>25} {_pm(d['dA0']):>25}")


# Figure colours: one categorical slot (dataviz reference blue, ≥ 3:1 on the
# light surface) for the combined fit, the per-ensemble points in neutral grey
# so they read as context. Every series also has its own marker, so identity
# is never colour-alone.
C_DIAG, C_ENS = "#2a78d6", "#898781"
C_TEXT, C_TEXT2, C_GRID, C_AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
ENS_LABEL = {"run5": "ens0 (Run 5)", "ens1": "ens1"}


def _tex_v(v, dig=3):
    """0.3324 ± 0.0268 → 0.332(27), in math mode."""
    return f"${v[0]:.{dig}f}({round(v[1] * 10**dig)})$"


def _tex_d(v):
    """A signed difference with its significance: +0.077(22) [3.6σ]."""
    return f"${v[0]:+.3f}({round(v[1] * 1000)})\\;[{v[0] / v[1]:.1f}\\sigma]$"


def table(per, combined, admitted, stem):
    """S1 task 3: the window table, paste-ready (the same numbers as the printout)."""
    order = sorted(WINDOWS)
    lines = [
        "% fit_window_scan.py — S1 of notes/prof_notes.md. Diagonal cosh fits (the",
        "% estimator); chi2/dof is the CORRELATED one at those fits (blocked-jackknife",
        "% covariance, Hartlap-corrected; summed over ensembles in the combined block);",
        "% the last column is the correlated fit, an unbiased but noisier alternative",
        "% (scripts/fit_estimator_mc.py). * = the published window; † = admitted by",
        f"% the S/N rule (C(Δmax) ≥ {SNR_MIN:g}σ for both operators on both ensembles).",
        r"\begin{tabular}{lccccccc}", r"\toprule",
        r"window & $m_\mathrm{GELT}$ & $A_0^\mathrm{GELT}$ & $m_\mathrm{GEVP}$ & "
        r"$A_0^\mathrm{GEVP}$ & $\Delta A_0$ & $\chi^2/\mathrm{dof}$ (G / P) & "
        r"$\Delta A_0$, corr.\ fit \\",
    ]

    def wlab(w):
        mark = "^*" if w == REFERENCE else ("^\\dagger" if w not in REQUIRED else "")
        return f"$[{w[0]},{w[1]}]{mark}$"

    for tag, res in per.items():
        lines += [r"\midrule", rf"\multicolumn{{8}}{{l}}{{\emph{{{ENS_LABEL.get(tag, tag)}}}}} \\"]
        for w in order:
            if w not in admitted:
                continue
            r = res["windows"][w]
            d, c = r["diag"], r["corr"]
            lines.append(
                f"{wlab(w)} & {_tex_v(d['m_g'])} & {_tex_v(d['A0_g'])} & {_tex_v(d['m_p'])} & "
                f"{_tex_v(d['A0_p'])} & {_tex_d(d['dA0'])} & "
                f"{r['g']['chi2dof_corr']:.2f} / {r['p']['chi2dof_corr']:.2f} & "
                f"{_tex_d(c['dA0'])} \\\\")
    if len(per) > 1:
        lines += [r"\midrule", r"\multicolumn{8}{l}{\emph{combined (inverse variance)}} \\"]
        for w in order:
            if w not in admitted:
                continue
            c = combined[w]
            lines.append(
                f"{wlab(w)} & --- & --- & --- & --- & {_tex_d(c['diag']['dA0'])} & "
                f"{c['g']['chi2dof_corr']:.2f} / {c['p']['chi2dof_corr']:.2f} & "
                f"{_tex_d(c['corr']['dA0'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(stem + ".tex", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  saved {stem}.tex")


def figure(per, combined, admitted, stem):
    """S1 task 3: ΔA₀ against the fit window.

    Windows ordered by (Δmin, Δmax), so the Δmin = 2 → 3 step — where all the
    movement is — reads as one step. Per-ensemble points are context (grey);
    the combined diagonal fit is the headline. The correlated fit is in the
    table only.
    """
    order = [w for w in sorted(WINDOWS) if w in admitted]
    x = range(len(order))
    fig, ax = plt.subplots(figsize=(7.6, 4.6), facecolor="white")
    ax.axhline(0.0, color=C_AXIS, lw=1.0, zorder=1)
    # Where Δmin changes: the only step the scan finds.
    for i in range(1, len(order)):
        if order[i][0] != order[i - 1][0]:
            ax.axvline(i - 0.5, color=C_GRID, lw=1.0, zorder=0)

    series = []
    markers = {"run5": "s", "ens1": "^"}
    names = {"run5": "ens0"}  # the figure drops the "(Run 5)" the table keeps
    for k, tag in enumerate(per):
        series.append((names.get(tag, tag),
                       [per[tag]["windows"][w]["diag"]["dA0"] for w in order],
                       dict(fmt=markers.get(tag, "v"), color=C_ENS, ms=5, mfc=C_ENS)))
    if len(per) > 1:
        series.append(("combined",
                       [combined[w]["diag"]["dA0"] for w in order],
                       dict(fmt="o", color=C_DIAG, ms=8, mfc=C_DIAG)))
    n = len(series)
    for i, (label, vals, kw) in enumerate(series):
        off = (i - (n - 1) / 2) * 0.14
        ax.errorbar([xi + off for xi in x], [v[0] for v in vals], yerr=[v[1] for v in vals],
                    capsize=3, elinewidth=1.2, lw=0, label=label, zorder=3, **kw)

    ax.set_xticks(list(x), [f"[{a},{b}]" for a, b in order])
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_xlabel(r"cosh-fit window $[\Delta_\mathrm{min}, \Delta_\mathrm{max}]$",
                  color=C_TEXT2)
    ax.set_ylabel(r"$\Delta A_0 = A_0^\mathrm{GELT} - A_0^\mathrm{GEVP}$", color=C_TEXT2)
    ax.set_title("Ground-state overlap advantage against the fit window",
                 color=C_TEXT, fontsize=11)
    ax.grid(axis="y", color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_AXIS)
    ax.tick_params(colors=C_TEXT2, labelsize=9)
    ax.legend(fontsize=8, loc="upper left", frameon=False, labelcolor=C_TEXT2)
    fig.tight_layout()
    fig.savefig(stem + ".png", dpi=200)
    plt.close(fig)
    print(f"  saved {stem}.png")


def main():
    per = {}
    for d in DUMPS:
        per[_tag(d)] = scan_dump(d)
    for tag, res in per.items():
        print_ensemble(tag, res)

    # ── Gates: the reference window must be the published measurement ─────────
    if not _POSITIONAL:
        for name, pub, get in (
            ("ΔA₀ at [2,7]", PUBLISHED_DA0, lambda t: per[t]["windows"][REFERENCE]["diag"]["dA0"]),
            ("Δm_eff(1) GELT − GEVP", PUBLISHED_DMEFF1, lambda t: per[t]["meff"]["dmeff1"]),
        ):
            for tag, (x, e) in pub.items():
                if tag not in per:
                    raise SystemExit(f"GATE: the published {tag} dump is missing")
                got = get(tag)
                if abs(got[0] - x) > PUBLISHED_TOL or abs(got[1] - e) > PUBLISHED_TOL:
                    raise SystemExit(
                        f"GATE FAILED: {tag} {name} = {got[0]:+.4f} ± {got[1]:.4f}, the "
                        f"record says {x:+.3f} ± {e:.3f}. The estimator has drifted from "
                        f"the published one; nothing below is readable.")
        print("\n  gates: ΔA₀ at [2,7] and Δm_eff(1) reproduce record §3.2 on run5 "
              "and ens1 — ok")

    # ── Combination over the ensembles ─────────────────────────────────────────
    tags = list(per)
    combined = {}
    for w in WINDOWS:
        combined[w] = {
            fit: {k: _combine([per[t]["windows"][w][fit][k] for t in tags]) for k in FIT_KEYS}
            for fit in ("diag", "corr")
        }
        combined[w]["shrink"] = {
            lam: {k: _combine([per[t]["windows"][w]["shrink"][lam][k] for t in tags])
                  for k in FIT_KEYS}
            for lam in SHRINK
        }
        for op in ("g", "p"):  # independent ensembles: χ² and dof add
            chi2 = sum(per[t]["windows"][w][op]["chi2_corr"] for t in tags)
            dof = sum(per[t]["windows"][w]["dof"] for t in tags)
            combined[w][op] = {"chi2dof_corr": chi2 / dof, "p_corr": pvalue(chi2, dof)}
    admitted = []
    for w in WINDOWS:
        if w in REQUIRED or all(
            min(per[t]["windows"][w]["g"]["snr"], per[t]["windows"][w]["p"]["snr"]) >= SNR_MIN
            for t in tags
        ):
            admitted.append(w)
    print(f"\n  combined ({' + '.join(tags)}, inverse variance; correlated χ² summed "
          f"over ensembles, at the diagonal fits)")
    print(f"  {'window':<8} {'Δm':>18} {'ΔA₀ (diag fit)':>25} {'ΔA₀ (corr fit)':>25}"
          f"   {'χ²/dof corr G (p)':>17} {'P (p)':>11}   admitted")
    for w in WINDOWS:
        c = combined[w]
        why = ("required" if w in REQUIRED else
               f"S/N ≥ {SNR_MIN:g} at Δ = {w[1]}" if w in admitted else
               f"no — S/N < {SNR_MIN:g} at Δ = {w[1]}")
        print(f"  {_w(w)}  {_pm(c['diag']['dm'], False):>18} {_pm(c['diag']['dA0']):>25} "
              f"{_pm(c['corr']['dA0']):>25}   {c['g']['chi2dof_corr']:9.2f} ({c['g']['p_corr']:.2f})"
              f" {c['p']['chi2dof_corr']:5.2f} ({c['p']['p_corr']:.2f})   {why}")

    print(f"\n  combined ΔA₀ against the covariance shrinkage λ "
          f"(Σ_λ = (1 − λ)Σ + λ·diag Σ; λ = 0 fully correlated, λ = 1 the estimator)")
    print(f"  {'window':<7}" + "".join(f"{'λ = ' + format(lam, 'g'):>24}" for lam in SHRINK))
    for w in WINDOWS:
        print(f"  {_w(w)}" + "".join(
            f"{_pm(combined[w]['shrink'][lam]['dA0']).replace(' ± ', '±'):>24}"
            for lam in SHRINK))

    # ── Task 4: m_eff against the fitted mass ──────────────────────────────────
    print(f"\n  m_eff(Δ) against the cosh-fit mass on {list(REFERENCE)}, same operator, "
          f"same jackknife sample (d = m_eff − m_fit)")
    print(f"  {'':<24}" + "".join(f"{t:>28}" for t in tags) + f"{'combined':>28}")
    rows = [("GELT  m_eff(1)", "meff1_g", False), ("GELT  m_fit", "mfit_g", False),
            ("GEVP  m_eff(1)", "meff1_p", False), ("GEVP  m_fit", "mfit_p", False),
            ("d  GELT  Δ = 1", "d1_g", True), ("d  GELT  Δ = 2", "d2_g", True),
            ("d  GEVP  Δ = 1", "d1_p", True), ("d  GEVP  Δ = 2", "d2_p", True),
            ("Δm_eff(1) GELT − GEVP", "dmeff1", True)]
    meff_comb = {}
    for label, k, comb in rows:
        vals = [per[t]["meff"][k] for t in tags]
        cells = [(f"{v[0]:.4f} ± {v[1]:.4f}" if not comb else _pm(v)) for v in vals]
        if comb:
            meff_comb[k] = _combine(vals)
            cells.append(_pm(meff_comb[k]))
        else:
            cells.append("—")
        print(f"  {label:<24}" + "".join(f"{c:>28}" for c in cells))

    # ── The readings, as fixed in notes/prof_notes.md S1 ──────────────────────
    da = {w: combined[w]["diag"]["dA0"] for w in admitted}
    ref_x, ref_e = da[REFERENCE]
    centrals = [x for x, _ in da.values()]
    spread = max(centrals) - min(centrals)
    min_sig = min(x / e for x, e in da.values())
    stable = min_sig >= 2.0 and spread < ref_e
    print(f"\n  reading, task 1, on {len(admitted)} admitted windows "
          f"({', '.join(f'[{a},{b}]' for a, b in admitted)}):")
    print(f"    smallest significance {min_sig:.1f}σ (need ≥ 2)   "
          f"spread max − min = {spread:.4f} (need < the [2,7] error {ref_e:.4f})")
    tot = math.sqrt(ref_e**2 + (spread / 2) ** 2)
    print(f"    stat ⊕ window = ±{tot:.4f} → ΔA₀ at {ref_x / tot:.1f}σ including the "
          f"fit-window systematic")
    if len(tags) > 1:  # do the ensembles scatter more than their own errors say?
        xs = [per[t]["windows"][REFERENCE]["diag"]["dA0"] for t in tags]
        chi2 = sum(((x - ref_x) / e) ** 2 for x, e in xs)
        print(f"    ensemble consistency at {list(REFERENCE)}: χ² = {chi2:.2f} on "
              f"{len(xs) - 1} dof, p = {pvalue(chi2, len(xs) - 1):.2f} "
              f"(per ensemble: " + ", ".join(f"{t} {x:+.3f}({round(e * 1000)})"
                                              for t, (x, e) in zip(tags, xs)) + ")")
    if stable:
        print(f"    → STABLE under the fit window. Headline unchanged: ΔA₀ = "
              f"{ref_x:+.3f} ± {ref_e:.3f} ({ref_x / ref_e:.1f}σ), window "
              f"systematic ± {spread / 2:.3f} (half the spread).")
    else:
        print(f"    → NOT stable by the pre-registered test. Quote ΔA₀ = "
              f"{ref_x:+.3f} ± {ref_e:.3f} (stat) ± {spread / 2:.3f} (window) "
              f"and rewrite the abstract's σ count.")

    d1 = meff_comb["d1_g"]
    plateau = abs(d1[0]) / d1[1] < PLATEAU_SIGMA
    print(f"\n  reading, task 4: GELT d(Δ = 1) = {_pm(d1)} combined "
          f"(need |d|/σ_d < {PLATEAU_SIGMA:g})")
    if plateau:
        print("    → \"compatible with a plateau from Δ = 1 within errors\".")
    else:
        print("    → NOT a plateau at Δ = 1: main.tex l. 59, 406, 427, 436, 505 say "
              "\"reduced excited-state contamination already at Δ = 1\" instead of "
              "\"plateaus at Δ = 1\".")

    c_ref = combined[REFERENCE]
    print(f"\n  task 2: at [2,7] the correlated χ²/dof of the quoted fits is "
          f"{c_ref['g']['chi2dof_corr']:.2f} (p = {c_ref['g']['p_corr']:.2f}) for GELT, "
          f"{c_ref['p']['chi2dof_corr']:.2f} (p = {c_ref['p']['p_corr']:.2f}) for the GEVP; "
          f"the correlated fit gives ΔA₀ = {_pm(c_ref['corr']['dA0'])} against the "
          f"diagonal {_pm(c_ref['diag']['dA0'])}. The diagonal fit stays the estimator.")

    torch.save({"dumps": DUMPS, "windows": WINDOWS, "reference": REFERENCE,
                "admitted": admitted, "per_ensemble": per, "combined": combined,
                "meff_combined": meff_comb, "spread": spread, "stable": stable,
                "plateau": plateau, "snr_min": SNR_MIN}, OUT)
    print(f"\n  saved {OUT}")
    stem = os.path.splitext(OUT)[0]
    table(per, combined, admitted, stem)
    figure(per, combined, admitted, stem)


if __name__ == "__main__":
    main()
