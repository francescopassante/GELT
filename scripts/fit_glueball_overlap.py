"""Cosh fit + ground-state overlap A₀ — presentation-grade §6.2 spectroscopy.

The m_eff(Δ) point cloud (glueball_gelt.png) is how lattice results are
*plotted*; a mass is *quoted* from a fit of C(Δ) to the periodic single-state
(cosh) form A·[e^{−mΔ} + e^{−m(Nt−Δ)}], drawn as a horizontal band over the
fit window. And the GELT-vs-GEVP claim is really about *operator quality*,
whose standard metric is the **ground-state overlap fraction**

    A₀ = A·(1 + e^{−m·Nt}) / C(0)  ∈ (0, 1],

the fraction of the operator's spectral weight sitting on the ground state
(Morningstar–Peardon rank their smeared/fuzzed operators by exactly this). A
perfect variational operator has A₀ = 1 — and A₀ is precisely what the §6.2
Rayleigh loss maximises, so the training objective and the reported quality
metric are the same physical quantity.

Everything here runs on the tiny (B, Nt) test-split Ō arrays dumped by
scripts/train_glueball.py (set EVAL_ONLY = True there to reproduce the dump
from an existing checkpoint in one GPU eval pass) — CPU-trivial, no GPU, no
ensemble, so fit windows and figure styling can be re-tuned locally at will.

Protocol
--------
- Operators: **GELT** (learned), the **projected classical GEVP operator**
  (fixed ground-state eigenvector v₀ at (t0, t0+1) applied to the smearing
  basis — ``gevp_ground_vector``, the optimal *single* operator in the
  classical span, so the comparison with the single learned operator is
  apples-to-apples), and **APE×max** (the best single smearing level). Thin
  links appear in the overlap panel as points only (no plateau to fit).
- One SHARED fit window (``FIT_WINDOW``): comparability beats per-operator
  optimality; window stability should be spot-checked by editing it. σ_Δ
  weights are fixed from the full-sample blocked jackknife of C(Δ); diagonal
  χ² only (see ``fit_cosh_correlator``).
- Errors: the ENTIRE fit — including v₀ — is redone inside every delete-block
  jackknife sample, and the differences m_GELT − m_GEVP and A₀_GELT − A₀_GEVP
  are jackknifed directly so the shared-ensemble fluctuations cancel (same
  logic as blocked_jackknife_meff_diff in train_glueball.py: these are the
  significance statements).

Figure (glueball_overlap.png)
-----------------------------
- Left: m_eff(Δ) for GELT and the projected GEVP with their fitted-mass bands
  over the window — the standard plateau-plus-fit-band presentation.
- Right: ρ(Δ) = [C(Δ)/C(0)] / cosh_ref(Δ), cosh_ref the fitted GELT ground
  state normalised to 1 at Δ = 0. A pure ground-state operator is FLAT at
  height A₀; excited-state contamination is the excess toward small Δ. Dashed
  lines mark each fitted A₀ — the panel that *shows* why GELT wins.

Run:
    python scripts/fit_glueball_overlap.py [path/to/…_test_obars.pt]
"""

import os
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from gelt.glueball import (

    connected_correlator,
    connected_correlator_matrix,
    effective_mass,
    fit_cosh_correlator,
    gevp_ground_vector,
)

# Output artifacts are grouped by study under results/; create the dirs the
# first time this runs in a fresh clone (they hold generated files only).
for _d in ("results/sampler", "results/glueball", "results/attention",
          "results/wilson_regression", "datasets"):
    os.makedirs(_d, exist_ok=True)


# ── Tunables ──────────────────────────────────────────────────────────────────
_POSITIONAL = [a for a in sys.argv[1:] if not a.startswith("--")]
DUMP = (
    _POSITIONAL[0]
    if _POSITIONAL
    else "results/glueball/best_glueball_gelt_sm0-2-4-6_test_obars.pt"
)
# --vs=<dump>: a second learned operator from the SAME ensemble and test split,
# added as a fourth row and — the point of the option — differenced against the
# first inside every jackknife sample. That is the matched-architecture
# comparison (GELT vs L-CNN): both operators see the same 400 configurations, so
# the shared-ensemble fluctuations cancel exactly as they do against the GEVP,
# and the two ΔA₀-against-GEVP numbers must NOT be subtracted by hand instead —
# they are correlated through the classical arm.
VS = next(
    (a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--vs=")), None
)
FIT_WINDOW = (2, 7)  # shared cosh-fit window [Δmin, Δmax]. Starts at 2 so the
#                      classical GEVP's residual Δ=1 contamination (the Run-5
#                      3.9σ point) does not tilt ITS fit — the contamination is
#                      what A₀ reports, not what the mass fit should absorb.
M_RANGE = (0.05, 1.5)  # m grid for the profiled-A scan (anchor is ≈ 0.33)
GEVP_TD = None  # diagonalisation time for v₀; None → t0 + 1 (standard)
MEFF_DMAX = 10  # last Δ drawn in the m_eff panel
RHO_DMAX = 8  # last Δ drawn in the overlap panel (cosh_ref shrinks ~e^{−mΔ},
#               so the ratio's noise blows up beyond the fit window)
ANCHOR = 0.33  # classical GEVP plateau m·a_t (the Run-3 anchor)
OUT = "results/glueball/glueball_overlap.png"


# ── Blocked jackknife of an arbitrary statistic ───────────────────────────────
def _blocks(B, block_size):
    idx = torch.arange(B)
    return [idx[i : i + block_size] for i in range(0, B, block_size)]


def blocked_jackknife(fn, B, block_size):
    """Delete-block jackknife mean/err of any statistic ``fn(mask) → tensor``.

    Same delete-block scheme as train_glueball.py (residual autocorrelation on
    the chain-ordered test split is not fully characterized, so consecutive
    blocks of configs are deleted, not single configs).
    """
    blocks = _blocks(B, block_size)
    samples = []
    for bl in blocks:
        mask = torch.ones(B, dtype=torch.bool)
        mask[bl] = False
        samples.append(fn(mask))
    samples = torch.stack(samples)
    n = len(blocks)
    mean = samples.mean(dim=0)
    err = ((n - 1) / n * ((samples - mean) ** 2).sum(dim=0)).sqrt()
    return mean, err


def outlier_share(obar):
    """How much of C(0) one configuration carries: (share, index, × median).

    An operator's A₀ is a ratio to C(0), so a single configuration that blows up
    destroys the measurement while leaving the Δ ≥ 1 shape — and the mass — fine.
    It is not hypothetical: one trained L-CNN and two untrained ones put 36–98%
    of C(0) on a single configuration that every classical operator finds
    unremarkable (notes/lcnn_shootout.md §9.2), where all 14 GELT operators
    measured sit at 0.5–1.0%, i.e. the 1/400 a healthy operator should show.
    Printed for every arm, so the check is uniform rather than something applied
    to an arm one dislikes.
    """
    d = obar - obar.mean()
    per_cfg = d.pow(2).mean(dim=1)
    top = per_cfg.argmax().item()
    return (
        (per_cfg[top] / per_cfg.sum()).item(),
        top,
        (per_cfg[top] / per_cfg.median()).item(),
    )


def arch_label(meta):
    """How to name a dump's learned operator: the architecture that made it."""
    arch = str(meta.get("arch", "gelt")).lower()
    name = {"lcnn": "L-CNN", "lcnn_ref": "L-CNN (authors')"}.get(arch, "GELT")
    return f"{name} (random init)" if meta.get("random_init") else f"{name} (learned)"


def project_ground(basis, t0, td):
    """v₀-projected scalar operator (B, Nt) of an (n_ops, B, Nt) basis."""
    v0 = gevp_ground_vector(connected_correlator_matrix(basis), t0=t0, td=td)
    return torch.einsum("i,ibt->bt", v0, basis)


def main():
    blob = torch.load(DUMP)
    gelt_obar = blob["gelt_obar"].double()
    Obar_basis = blob["Obar_basis"].double()
    meta = blob.get("meta", {})
    t0 = int(meta.get("gevp_t0", 1))
    td = t0 + 1 if GEVP_TD is None else GEVP_TD
    jb = int(meta.get("jack_block", 10))
    levels = meta.get("gevp_levels", list(range(Obar_basis.shape[0])))
    B, Nt = gelt_obar.shape
    dmin, dmax = FIT_WINDOW
    dof = (dmax - dmin + 1) - 2
    print(f"loaded {DUMP}: {B} test configs × Nt={Nt}, basis levels {levels}")
    print(
        f"fit window Δ ∈ [{dmin}, {dmax}] (cosh), GEVP (t0, td) = ({t0}, {td}), "
        f"block {jb} → {-(-B // jb)} jackknife blocks"
    )

    label = arch_label(meta)
    vs_obar = vs_label = None
    if VS is not None:
        vs_blob = torch.load(VS)
        vs_obar = vs_blob["gelt_obar"].double()
        vs_label = arch_label(vs_blob.get("meta", {}))
        # Same ensemble and same split, or the difference below is meaningless.
        # The fingerprint is the THIN level of the classical basis: it is a
        # plaquette sum over the raw configurations, with no smearing and no
        # projection, so two dumps of the same split agree on it bit-exactly.
        # The smeared levels need a tolerance — dumps made before and after
        # SU(2).project went closed-form (performance_audit §3.1) differ by
        # ~4e-7 there, which is a different code path, not a different ensemble.
        vs_basis = vs_blob["Obar_basis"].double()
        if vs_obar.shape != gelt_obar.shape or vs_basis.shape != Obar_basis.shape:
            raise SystemExit(f"--vs dump has a different shape from {DUMP}.")
        if not torch.equal(vs_basis[0], Obar_basis[0]):
            raise SystemExit(
                f"--vs dump is not the same test split as {DUMP}: the thin-link "
                f"operators differ, so these are different configurations and a "
                f"correlated difference would be nonsense."
            )
        drift = (
            (vs_basis - Obar_basis).abs().max() / Obar_basis.abs().max()
        ).item()
        if drift > 1e-5:
            raise SystemExit(
                f"--vs dump shares the configurations but its smeared basis "
                f"differs by {drift:.1e} — too much for a projection-route "
                f"change; check SMEAR_ALPHA and the smearing levels."
            )
        print(f"        vs {VS}: {vs_label}" + (
            f"   (smeared basis agrees to {drift:.1e} — the closed-form "
            f"SU(2).project route)" if drift else ""
        ))

    thin_obar, sm_obar = Obar_basis[0], Obar_basis[-1]
    proj_full = project_ground(Obar_basis, t0, td)

    # σ_Δ(C) from the full sample — FIXED diagonal weights for every fit,
    # including inside the jackknife samples, so all samples minimise the same
    # χ² surface (the standard jackknife-of-fit protocol).
    _, sig_gelt = blocked_jackknife(
        lambda m: connected_correlator(gelt_obar[m]), B, jb
    )
    _, sig_proj = blocked_jackknife(
        lambda m: connected_correlator(proj_full[m]), B, jb
    )
    _, sig_sm = blocked_jackknife(lambda m: connected_correlator(sm_obar[m]), B, jb)
    sig_vs = (
        None
        if vs_obar is None
        else blocked_jackknife(lambda m: connected_correlator(vs_obar[m]), B, jb)[1]
    )

    def fit_one(C, sig):
        """(m, A₀, χ²) of one operator's correlator on the shared window."""
        m, A, chi2 = fit_cosh_correlator(C, dmin, dmax, sigma=sig, m_range=M_RANGE)
        a0 = A * (1.0 + math.exp(-m * Nt)) / C[0].item()
        return m, a0, chi2

    # Full-sample fits: central χ²/dof and the reference mass for cosh_ref.
    _, _, chi2_g = fit_one(connected_correlator(gelt_obar), sig_gelt)
    _, _, chi2_p = fit_one(connected_correlator(proj_full), sig_proj)
    _, _, chi2_s = fit_one(connected_correlator(sm_obar), sig_sm)

    # Jackknife of the whole fit — v₀ is recomputed per sample so its noise
    # propagates; the packed differences give the correlated significances.
    def fit_stats(mask):
        m_g, a0_g, _ = fit_one(connected_correlator(gelt_obar[mask]), sig_gelt)
        proj = project_ground(Obar_basis[:, mask], t0, td)
        m_p, a0_p, _ = fit_one(connected_correlator(proj), sig_proj)
        m_s, a0_s, _ = fit_one(connected_correlator(sm_obar[mask]), sig_sm)
        packed = [m_g, a0_g, m_p, a0_p, m_s, a0_s, m_g - m_p, a0_g - a0_p]
        if vs_obar is not None:
            m_v, a0_v, _ = fit_one(connected_correlator(vs_obar[mask]), sig_vs)
            packed += [m_v, a0_v, m_g - m_v, a0_g - a0_v]
        return torch.tensor(packed, dtype=torch.float64)

    stats, stats_err = blocked_jackknife(fit_stats, B, jb)

    names = [label, "GEVP-projected", f"APE×{levels[-1]}"]
    chis = [chi2_g, chi2_p, chi2_s]
    if vs_obar is not None:
        _, _, chi2_v = fit_one(connected_correlator(vs_obar), sig_vs)
        names.append(vs_label)
        chis.append(chi2_v)
    print(f"\ncosh fits on Δ ∈ [{dmin}, {dmax}]  (χ²/dof from the full sample, dof={dof}):")
    print(f"  {'operator':<16} {'m·a_t (fit)':<20} {'A₀ (ground overlap)':<22} χ²/dof")
    # The --vs operator is packed after the three differences, so its (m, A₀)
    # live at 8/9 rather than at 2·i.
    slots = [0, 2, 4] + ([8] if vs_obar is not None else [])
    for nm, c2, k in zip(names, chis, slots):
        m, me = stats[k].item(), stats_err[k].item()
        a, ae = stats[k + 1].item(), stats_err[k + 1].item()
        print(f"  {nm:<18} {m:.4f} ± {me:.4f}      {a:.4f} ± {ae:.4f}        {c2 / dof:.2f}")
    # Single-configuration dominance of C(0) — the A₀ denominator.
    print("\nshare of C(0) carried by its single largest configuration "
          "(healthy ≈ 1/N_cfg):")
    arms = [(names[0], gelt_obar), ("GEVP-projected", proj_full),
            (f"APE×{levels[-1]}", sm_obar)]
    if vs_obar is not None:
        arms.append((vs_label, vs_obar))
    flagged = []
    for nm, ob in arms:
        share, idx, rel = outlier_share(ob)
        mark = "   ** DOMINATED — A₀ below is not a measurement **" if share > 0.10 else ""
        print(f"  {nm:<18} {100 * share:5.1f}%   (config {idx}, {rel:.0f}× the median){mark}")
        if share > 0.10:
            flagged.append(nm)
    if flagged:
        print(
            "  → " + ", ".join(flagged) + ": one configuration owns the A₀ "
            "denominator. Re-run that arm; do not quote its A₀, and do not "
            "silently drop the configuration either — see "
            "notes/lcnn_shootout.md §9.2."
        )

    dm, dme = stats[6].item(), stats_err[6].item()
    da, dae = stats[7].item(), stats_err[7].item()
    print(f"\n{label} − GEVP-projected, correlated (same-configs) jackknife of the difference:")
    print(
        f"  Δm  = {dm:+.4f} ± {dme:.4f}  ({abs(dm) / max(dme, 1e-12):.1f}σ)"
        f"   — same physics ⇔ consistent with 0"
    )
    print(
        f"  ΔA₀ = {da:+.4f} ± {dae:.4f}  ({abs(da) / max(dae, 1e-12):.1f}σ)"
        f"   — > 0 ⇔ the learned operator carries more ground-state weight"
    )
    if vs_obar is not None:
        dmv, dmve = stats[10].item(), stats_err[10].item()
        dav, dave = stats[11].item(), stats_err[11].item()
        print(f"\n{label} − {vs_label}, correlated (same-configs) jackknife:")
        print(
            f"  Δm  = {dmv:+.4f} ± {dmve:.4f}  ({abs(dmv) / max(dmve, 1e-12):.1f}σ)"
        )
        print(
            f"  ΔA₀ = {dav:+.4f} ± {dave:.4f}  ({abs(dav) / max(dave, 1e-12):.1f}σ)"
            f"   — 0 ⇔ the two architectures are the same operator quality"
        )

    # ── m_eff curves for the fit-band panel ────────────────────────────────────
    meff_gelt, meff_gelt_err = blocked_jackknife(
        lambda m: effective_mass(connected_correlator(gelt_obar[m])), B, jb
    )
    meff_proj, meff_proj_err = blocked_jackknife(
        lambda m: effective_mass(
            connected_correlator(project_ground(Obar_basis[:, m], t0, td))
        ),
        B,
        jb,
    )

    # ── Normalised correlators for the overlap panel ───────────────────────────
    def rho_fn(get):
        def fn(mask):
            C = connected_correlator(get(mask))
            return C / C[0]

        return fn

    rho = {
        "thin": blocked_jackknife(rho_fn(lambda m: thin_obar[m]), B, jb),
        "sm": blocked_jackknife(rho_fn(lambda m: sm_obar[m]), B, jb),
        "proj": blocked_jackknife(
            rho_fn(lambda m: project_ground(Obar_basis[:, m], t0, td)), B, jb
        ),
        "gelt": blocked_jackknife(rho_fn(lambda m: gelt_obar[m]), B, jb),
    }
    # Reference ground-state shape: the GELT fitted mass (the fitted masses
    # agree within errors — the Δm line above — so the choice is cosmetic;
    # it is stated on the panel).
    m_ref = stats[0].item()
    dd = np.arange(Nt)
    cosh_ref = (np.exp(-m_ref * dd) + np.exp(-m_ref * (Nt - dd))) / (
        1.0 + math.exp(-m_ref * Nt)
    )

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # (0) m_eff with fitted-mass bands over the window.
    def _meff_pts(meff, err, first_d, lab, col, fmt):
        m_, e_ = meff.numpy(), err.numpy()
        d_ = np.arange(len(m_))
        ok = np.isfinite(m_) & np.isfinite(e_) & (d_ >= first_d) & (d_ <= MEFF_DMAX)
        ax[0].errorbar(
            d_[ok], m_[ok], yerr=e_[ok], fmt=fmt, capsize=3, color=col, label=lab,
            alpha=0.85,
        )

    _meff_pts(meff_proj, meff_proj_err, t0, "GEVP-projected (fixed v₀)", "C3", "D-")
    _meff_pts(meff_gelt, meff_gelt_err, 1, "GELT (learned)", "C2", "^-")
    for i, col, nm in [(0, "C2", "GELT"), (2, "C3", "GEVP")]:
        m_, e_ = stats[i].item(), stats_err[i].item()
        ax[0].fill_between(
            [dmin - 0.3, dmax + 0.3], m_ - e_, m_ + e_, color=col, alpha=0.18,
            label=f"{nm} fit  m·a_t = {m_:.3f} ± {e_:.3f}",
        )
    ax[0].axhline(ANCHOR, color="k", ls="--", alpha=0.6, label=f"anchor m·a_t ≈ {ANCHOR}")
    ax[0].set_xlabel("Δ (temporal slices)")
    ax[0].set_ylabel("m_eff(Δ) = m·a_t")
    ax[0].set_xlim(0, MEFF_DMAX + 0.5)
    ax[0].set_ylim(0.0, 0.8)
    ax[0].set_title(
        f"m_eff with cosh-fit bands (window Δ ∈ [{dmin}, {dmax}])\n"
        "(the quoted mass is the fitted band, not any single m_eff point)"
    )
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # (1) ρ(Δ) = [C(Δ)/C(0)] / cosh_ref(Δ): flat at A₀ ⇔ pure ground state.
    specs = [
        ("thin", "thin links", "C0", "o", None),
        ("sm", f"APE×{levels[-1]}", "C1", "s", (stats[5], stats_err[5])),
        ("proj", "GEVP-projected", "C3", "D", (stats[3], stats_err[3])),
        ("gelt", "GELT (learned)", "C2", "^", (stats[1], stats_err[1])),
    ]
    for key, lab, col, mk, a0 in specs:
        mean, err = rho[key]
        r_, e_ = mean.numpy() / cosh_ref, err.numpy() / cosh_ref
        d_ = np.arange(Nt)
        ok = np.isfinite(r_) & np.isfinite(e_) & (d_ <= RHO_DMAX)
        if a0 is not None:
            lab = lab + f"   A₀ = {a0[0].item():.3f} ± {a0[1].item():.3f}"
            ax[1].axhline(a0[0].item(), color=col, ls="--", alpha=0.5)
        ax[1].errorbar(
            d_[ok], r_[ok], yerr=e_[ok], fmt=mk + "-", capsize=3, color=col,
            label=lab, alpha=0.85,
        )
    ax[1].set_xlabel("Δ (temporal slices)")
    ax[1].set_ylabel(f"ρ(Δ) = [C(Δ)/C(0)] / cosh_ref(Δ; m = {m_ref:.3f})")
    ax[1].set_xlim(-0.3, RHO_DMAX + 0.5)
    ax[1].set_ylim(0.0, 1.25)
    ax[1].set_title(
        "Ground-state fraction of each operator\n"
        "(flat at A₀ ⇔ pure ground state; small-Δ excess = excited-state "
        "contamination)"
    )
    ax[1].legend(loc="lower right")
    ax[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"0⁺⁺ cosh fits & ground-state overlap — SU(2) L={meta.get('L', '?')} "
        f"Lt={meta.get('Lt', Nt)} β={meta.get('beta', '?')} ξ={meta.get('xi', '?')}  "
        f"N_test={B}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
