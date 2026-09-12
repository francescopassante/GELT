"""What did the network find? — the learned operator decomposed against the
span of the classical basis.

§6.2 reports that the learned operator carries more ground-state weight than
the optimal combination of the classical Morningstar–Peardon basis,
ΔA₀ = +0.078 ± 0.022 (3.6σ). That is a statement that GELT *wins*. It is not a
statement about *what it found*, and a referee who accepts the win asks the next
question immediately: is the advantage new operator content, or is it the same
content in a combination the GEVP failed to locate?

The question has an exact answer, because operators acting on the vacuum are
vectors in a Hilbert space and the equal-time connected correlator is its inner
product:

    ⟨O_a, O_b⟩ ≡ C_ab(0) = ⟨ d_a(t) d_b(t) ⟩,   d = Ō − ⟨Ō⟩,

so "the part of GELT the classical basis cannot express" is a literal orthogonal
projection. Write

    O_GELT = P + r,    P = the orthogonal projection onto span{Ō_i},   r ⟂ span,

and every quantity below is a property of that split. Nothing is fitted to make
it come out any particular way; P is the *best possible* classical approximation
of GELT, better by construction than any classical operator chosen for any other
reason, so ΔA₀(GELT − P) is a lower bound on GELT's advantage over the span.

Why this is the sharper comparison
----------------------------------
The published ΔA₀ compares GELT to the classical GEVP ground vector — a
different operator, correlated with GELT only through sharing configurations.
P is GELT's *own* projection, so the two share every fluctuation that lives in
the span and the jackknifed difference is much better conditioned. It also
answers a different and stronger question: not "is our operator better than
theirs" but "is our operator's advantage outside their reach".

The calibration that makes 13% mean something
---------------------------------------------
A number like "12.9% of the norm² lies outside a four-dimensional span" is
meaningless without a scale — *any* operator outside a 4D subspace has a large
orthogonal component. The scale is the basis's own increments: how much new
content does each rung of the smearing ladder add to the span of the rungs
below it? That is the same statistic applied to the classical operators
themselves, it is free, and it says whether the ladder is still growing or has
saturated.

What each outcome licenses
--------------------------
* **r is a large fraction of the norm AND carries a matching share of the
  ground-state amplitude Z** → the network found operator content the classical
  basis does not span, and that content is what the win is made of. Report the
  decomposition beside the ΔA₀, and quote the ladder increments as the scale.
* **r is large in norm but carries no Z** → the network is decorating the
  classical answer with noise that happens not to hurt; the win is then a
  statistical accident of the GEVP and should be reported as such.
* **r shrinks as the metric time grows** → r is a contact term, i.e. UV content
  that dies by Δ = 2, exactly like the E-irrep contamination measured in the
  (retired) rotational-symmetry study. It would then be irrelevant to the mass and
  must not be quoted as physics. The metric scan below is the test.
* **ΔA₀(GELT − P) ≈ 0** → the advantage lives *inside* the span after all and
  the GEVP simply failed to find it. That is still a real result, but it is a
  statement about estimator variance, not about operator content.

The two controls (run 2026-09-12, audit §8.3)
--------------------------------------------
Both were missing when this script was written and both are now flags:

1. **A stronger classical span** — `--basis=<obars.pt>:<arm>` takes the span
   from an arm of `su2_fair_fight.py`'s cache (`published`, `deep`, `shapes`,
   `shapes_sm`, `full`) instead of the dump's own four-level `Obar_basis`. The
   GEVP gate arm then reads with audit §8.1's adopted whitening (truncate at
   eps 1e-4), because a long ladder's C(t0) has null directions. The
   ladder-increment loop is skipped for an arm that mixes loop shapes, since
   its members are not nested.
2. **A random-init GELT** — nothing to add to the script but `--m-ref=<float>`:
   an untrained net's own cosh fit is not a usable reference for the amplitude
   split, so the trained net's mass on the same ensemble is passed in.

And one statement that came free with the first:

3. `--shape-span=<obars.pt>:<arm>` projects the residual r onto span{arm} and
   reports the fraction of its norm² that lands inside. With arm = `full` ⊃
   `deep`, that is "is the out-of-span content rectangular loops the network
   rediscovered, or something no planar loop expresses?"

`--basis`, `--shape-span` and `--m-ref` are repeatable: one value applies to
every dump, N values pair with the N dumps in order, so the two ensembles —
each with its own obars cache and its own reference mass — combine inside one
run. Runs of the same ensemble (init seeds) average under §1.4's rule before
the ensembles combine inverse-variance.

Run:
    python scripts/operator_decomposition.py [path/to/…_test_obars.pt ...]
    python scripts/operator_decomposition.py dumps/…_test_obars.pt \
        --basis=dumps/su2_fair_fight_obars_run5.pt:deep \
        --shape-span=dumps/su2_fair_fight_obars_run5.pt:full
"""

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from gelt.glueball import (  # noqa: E402
    connected_correlator,
    connected_correlator_matrix,
    fit_cosh_correlator,
    gevp_ground_vector,
)

os.makedirs("results/glueball", exist_ok=True)


# ── Tunables ──────────────────────────────────────────────────────────────────
FIT_WINDOW = (2, 7)  # identical to fit_glueball_overlap.py — comparability
M_RANGE = (0.05, 1.5)
METRIC_TIMES = (0, 1, 2)  # ⟨O_a(t+τ) O_b(t)⟩ metrics; τ>0 down-weights UV noise
PROJ_EPS = 1e-12  # machine-precision guard on the C(0) Gram (see span_solve)
GEVP_EPS_TRUNC = 1e-4  # audit §8.1's adopted whitening, used with --basis


# ── Command line ──────────────────────────────────────────────────────────────
def _spec(arg):
    """``<obars.pt>:<arm>`` → (path, arm)."""
    path, _, arm = arg.rpartition(":")
    if not path or not arm:
        raise SystemExit(f"expected <obars.pt>:<arm>, got {arg!r}")
    return path, arm


# --basis / --shape-span may be repeated: one spec applies to every dump, N
# specs pair with the N dumps in order, which is how the two ensembles (each
# with its own obars cache) are combined inside a single run.
DUMPS, BASIS_SPECS, SHAPE_SPECS, M_REFS, OUT_TAG = [], [], [], [], ""
for _a in sys.argv[1:]:
    if _a.startswith("--basis="):
        BASIS_SPECS.append(_spec(_a.split("=", 1)[1]))
    elif _a.startswith("--shape-span="):
        SHAPE_SPECS.append(_spec(_a.split("=", 1)[1]))
    elif _a.startswith("--m-ref="):
        M_REFS.append(float(_a.split("=", 1)[1]))
    elif _a.startswith("--proj-eps="):
        PROJ_EPS = float(_a.split("=", 1)[1])
    elif _a.startswith("--out-tag="):
        OUT_TAG = _a.split("=", 1)[1]
    elif _a.startswith("--"):
        raise SystemExit(f"unknown option {_a}")
    else:
        DUMPS.append(_a)

# Defaults are the two independently sampled ensembles behind the §6.2 headline.
DUMPS = DUMPS or [
    "results/glueball/best_glueball_gelt_sm0-2-4-6_test_obars.pt",
    "results/glueball/best_glueball_gelt_sm0-2-4-6_ens1_test_obars.pt",
]

# One output file per (span, shape span, dump), so the runs of audit §3 WP5a do
# not overwrite each other or the published decomposition.
_TAG = OUT_TAG or (
    ("_" + BASIS_SPECS[0][1] if BASIS_SPECS else "")
    + ("_shape-" + SHAPE_SPECS[0][1] if SHAPE_SPECS else "")
    + (
        "_" + os.path.basename(DUMPS[0]).replace("best_glueball_gelt_", "")
        .replace("_test_obars.pt", "")
        if len(DUMPS) == 1
        else ""
    )
)
for _name, _opt in (("--basis", BASIS_SPECS), ("--shape-span", SHAPE_SPECS),
                    ("--m-ref", M_REFS)):
    if len(_opt) not in (0, 1, len(DUMPS)):
        raise SystemExit(f"{_name}: give one value, or one per dump")
OUT_PNG = f"results/glueball/operator_decomposition{_TAG}.png"
OUT_PT = f"results/glueball/operator_decomposition{_TAG}.pt"


# ── Blocked jackknife (same delete-block scheme as fit_glueball_overlap.py) ───
def _blocks(B, bs):
    idx = torch.arange(B)
    return [idx[i : i + bs] for i in range(0, B, bs)]


def blocked_jackknife(fn, B, bs):
    samples = []
    for bl in _blocks(B, bs):
        mask = torch.ones(B, dtype=torch.bool)
        mask[bl] = False
        samples.append(fn(mask))
    samples = torch.stack(samples)
    n = len(samples)
    mean = samples.mean(dim=0)
    return mean, ((n - 1) / n * ((samples - mean) ** 2).sum(dim=0)).sqrt()


# ── The decomposition ─────────────────────────────────────────────────────────
def vacuum_subtract(O):
    """d = Ō − ⟨Ō⟩ with the VEV taken over configs *and* timeslices, exactly as
    connected_correlator does internally (0⁺⁺ has a nonzero VEV)."""
    return O - O.mean() if O.dim() == 2 else O - O.mean(dim=(1, 2), keepdim=True)


def span_solve(M, v, eps=None):
    """Solve ``M c = v`` on the directions of M above ``eps · λ_max``.

    The C(0) Gram of a long smearing ladder is near-degenerate by construction
    — APE×12 and APE×16 are almost the same operator, and `deep`'s Gram has
    cond ≈ 1e5, `full`'s ≈ 7e8 — so a plain solve returns huge cancelling
    coefficients. Dropping the numerically null directions leaves the
    projection where it was and keeps the coefficients readable; the default
    eps = 1e-12 is a machine-precision guard, not the audit §8.1 whitening cut
    (that one applies to the GEVP's C(t0), a different matrix).
    """
    w, Q = torch.linalg.eigh(M)
    keep = w > (PROJ_EPS if eps is None else eps) * w[-1]
    Qk = Q[:, keep]
    return Qk @ ((Qk.T @ v) / w[keep])


def decompose(target, basis, tau=0):
    """Split ``target`` into its projection onto span(basis) and the remainder.

    The metric is C_ab(τ) symmetrised in (a, b). τ = 0 is the Hilbert-space
    inner product on the states O|vac⟩ and is the one to quote; τ > 0 weights
    the light states more heavily and is the contact-term test.
    """
    dT, dB = vacuum_subtract(target), vacuum_subtract(basis)
    n_t = dB[0].numel()
    M = torch.einsum("ibt,jbt->ij", dB.roll(-tau, dims=2), dB) / n_t
    M = 0.5 * (M + M.T)  # the true matrix is symmetric; the estimator is not
    v = 0.5 * (
        torch.einsum("ibt,bt->i", dB.roll(-tau, dims=2), dT)
        + torch.einsum("ibt,bt->i", dB, dT.roll(-tau, dims=1))
    ) / dT.numel()
    coeff = span_solve(M, v)
    P = torch.einsum("i,ibt->bt", coeff, dB)
    return dT, P, dT - P, coeff


def cross_correlator(a, b):
    """⟨a(t+Δ) b(t)⟩ — the off-diagonal partner of connected_correlator."""
    Nt = a.shape[1]
    return torch.stack([(a.roll(-d, dims=1) * b).mean() for d in range(Nt)])


def load_arm(spec, n_cfg):
    """``bases[arm]``, ``labels[arm]`` from a fair-fight obars cache."""
    path, arm = spec
    blob = torch.load(path, weights_only=False)
    if arm not in blob["bases"]:
        raise SystemExit(f"{path}: no arm {arm!r} (has {list(blob['bases'])})")
    b = blob["bases"][arm].double()
    if b.shape[1] != n_cfg:
        raise SystemExit(
            f"{path}:{arm} has {b.shape[1]} configs, the dump has {n_cfg} — "
            "wrong ensemble?"
        )
    return b, list(blob["labels"][arm])


def _lab(x):
    """Row label for a basis member: an APE level from a dump, a fair-fight
    label ('n4·2x2') from an arm."""
    return x if isinstance(x, str) else f"APE×{x}"


def analyse(dump, i=0):
    basis_spec = BASIS_SPECS[i if len(BASIS_SPECS) > 1 else 0] if BASIS_SPECS else None
    shape_spec = SHAPE_SPECS[i if len(SHAPE_SPECS) > 1 else 0] if SHAPE_SPECS else None
    m_ref_fixed = M_REFS[i if len(M_REFS) > 1 else 0] if M_REFS else None
    blob = torch.load(dump, weights_only=False)
    G = blob["gelt_obar"].double()
    basis = blob["Obar_basis"].double()
    meta = blob.get("meta", {})
    t0 = int(meta.get("gevp_t0", 1))
    td = t0 + 1
    jb = int(meta.get("jack_block", 10))
    levels = meta.get("gevp_levels", list(range(basis.shape[0])))
    B, Nt = G.shape
    dmin, dmax = FIT_WINDOW

    # --basis: take the classical span from a fair-fight arm instead of the
    # dump's own four-level Obar_basis (audit §3 WP5a). The dump's jack_block,
    # t0 and fit window are unchanged — only the span moves.
    if basis_spec is not None:
        basis, levels = load_arm(basis_spec, B)
    # The GEVP gate arm's whitening follows audit §8.1: the published path
    # keeps the floor it was measured with, so the gate reproduces bit-for-bit;
    # a strengthened arm has null C(t0) directions and is read with the adopted
    # truncation at eps 1e-4.
    gevp_kw = dict(eps=GEVP_EPS_TRUNC, truncate=True) if basis_spec else {}
    # The ladder-increment loop assumes nested rungs — true of a pure smearing
    # ladder, false of an arm that mixes loop shapes (audit §3 WP5a).
    nested = len({str(x).split("·")[1] for x in levels if "·" in str(x)}) <= 1
    shape_basis = None
    if shape_spec is not None:
        shape_basis, _ = load_arm(shape_spec, B)

    print(f"\n{'=' * 78}\n{dump}\n{'=' * 78}")
    print(
        f"  {B} test configs × Nt = {Nt} | classical levels {levels} | "
        f"window Δ ∈ [{dmin}, {dmax}] | block {jb} → {-(-B // jb)} samples"
    )
    if basis_spec is not None:
        print(
            f"  span from {basis_spec[0]}:{basis_spec[1]} "
            f"({basis.shape[0]} operators) | GEVP whitening: truncate at "
            f"eps {GEVP_EPS_TRUNC:g}"
        )
    if shape_basis is not None:
        print(
            f"  shape span from {shape_spec[0]}:{shape_spec[1]} "
            f"({shape_basis.shape[0]} operators)"
        )

    def gevp_projected(mask):
        b = basis[:, mask]
        v0 = gevp_ground_vector(
            connected_correlator_matrix(b), t0=t0, td=td, **gevp_kw
        )
        return torch.einsum("i,ibt->bt", v0, b)

    # Diagonal χ² weights are fixed from the full sample so every jackknife
    # sample minimises the same surface (the standard jackknife-of-fit protocol).
    _, sig_G = blocked_jackknife(lambda m: connected_correlator(G[m]), B, jb)
    _, sig_P = blocked_jackknife(
        lambda m: connected_correlator(decompose(G[m], basis[:, m])[1]), B, jb
    )
    _, sig_r = blocked_jackknife(
        lambda m: connected_correlator(decompose(G[m], basis[:, m])[2]), B, jb
    )
    # The GEVP arm exists only as a gate, so its weights follow
    # fit_glueball_overlap.py to the letter: σ from the FULL-sample projection
    # sliced by the mask (while the fit itself re-solves v₀ per sample). Deriving
    # σ from the per-sample projection instead shifts ΔA₀ by ~0.5σ, which is
    # enough to make the gate look like it failed when nothing has changed.
    proj_full = gevp_projected(torch.ones(B, dtype=torch.bool))
    _, sig_V = blocked_jackknife(
        lambda m: connected_correlator(proj_full[m]), B, jb
    )

    def fit_a0(C, sigma):
        m, A, _ = fit_cosh_correlator(C, dmin, dmax, sigma=sigma, m_range=M_RANGE)
        return m, A * (1.0 + math.exp(-m * Nt)) / C[0].item()

    # Ground-state amplitudes are extracted at a COMMON fixed mass. Z is linear
    # in the operator, so Z_G = Z_P + Z_r holds exactly only if all three are
    # read against the same exponential; letting each fit its own m breaks the
    # additivity that makes the amplitude split interpretable.
    m_ref, _ = fit_a0(connected_correlator(G), sig_G)
    if m_ref_fixed is not None:
        # --m-ref: an untrained net's own cosh fit is not a usable reference
        # (audit §3 WP5a), so the amplitude split is read against the trained
        # net's mass on the same ensemble.
        m_ref = m_ref_fixed
    cosh_ref = torch.tensor(
        [math.exp(-m_ref * d) + math.exp(-m_ref * (Nt - d)) for d in range(Nt)],
        dtype=torch.float64,
    )
    w = 1.0 / torch.clamp(sig_G, min=1e-30) ** 2
    win = slice(dmin, dmax + 1)

    def amplitude(C):
        """Least-squares A of C(Δ) ≈ A·cosh_ref(Δ) over the window (linear in O)."""
        return (w[win] * cosh_ref[win] * C[win]).sum() / (
            w[win] * cosh_ref[win] ** 2
        ).sum()

    names = [
        "frac_out", "z_share", "m_G", "a0_G", "m_P", "a0_P", "m_r", "a0_r",
        "m_V", "a0_V", "dA0_GP", "dA0_GV", "dm_GP",
    ]
    if nested:
        names += [f"ladder{k}" for k in range(1, basis.shape[0])]
    names += [f"metric{tau}" for tau in METRIC_TIMES[1:]]
    if shape_basis is not None:
        names.append("shape_frac")
    ix = {n: i for i, n in enumerate(names)}

    def _metric(tau):
        """The symmetrised C_ab(τ) Gram of the span, on the full sample."""
        dB = vacuum_subtract(basis)
        M = torch.einsum("ibt,jbt->ij", dB.roll(-tau, dims=2), dB) / dB[0].numel()
        return 0.5 * (M + M.T)

    def stats(mask):
        dG, P, r, _ = decompose(G[mask], basis[:, mask])
        m_G, a0_G = fit_a0(connected_correlator(dG), sig_G)
        m_P, a0_P = fit_a0(connected_correlator(P), sig_P)
        m_r, a0_r = fit_a0(connected_correlator(r), sig_r)
        m_V, a0_V = fit_a0(connected_correlator(gevp_projected(mask)), sig_V)
        # Z_r/Z_G from the cross-correlator: C_Gr(Δ) → Z_G Z_r e^{−mΔ} while
        # C_GG(Δ) → Z_G², so the ratio of amplitudes is Z_r/Z_G with its sign.
        z_share = amplitude(cross_correlator(dG, r)) / amplitude(
            connected_correlator(dG)
        )
        out = [
            (r**2).mean() / (dG**2).mean(),  # norm² fraction outside the span
            z_share,
            m_G, a0_G, m_P, a0_P, m_r, a0_r, m_V, a0_V,
            a0_G - a0_P,  # the correlated advantage over the whole span
            a0_G - a0_V,  # the published comparison, for the gate
            m_G - m_P,
        ]
        # The ladder's own increments: new content of rung k over the rungs below.
        if nested:
            for k in range(1, basis.shape[0]):
                _, _, rk, _ = decompose(basis[k][mask], basis[:k][:, mask])
                dk = vacuum_subtract(basis[k][mask])
                out.append((rk**2).mean() / (dk**2).mean())
        # Contact-term test: the same fraction under metrics C(τ), τ > 0.
        for tau in METRIC_TIMES[1:]:
            _, _, rt, _ = decompose(G[mask], basis[:, mask], tau=tau)
            out.append((rt**2).mean() / (dG**2).mean())
        if shape_basis is not None:
            # --shape-span: how much of r a *richer* span reaches. span{arm} ⊇
            # span{basis} for full ⊃ deep, and r ⟂ span{basis}, so what lands
            # inside is exactly the loop-shape directions — "is the out-of-span
            # content rectangular loops the network rediscovered?"
            _, Ps, _, _ = decompose(r, shape_basis[:, mask])
            out.append((Ps**2).mean() / (r**2).mean())
        return torch.tensor(out, dtype=torch.float64)

    mu, err = blocked_jackknife(stats, B, jb)
    n_lad = basis.shape[0] - 1 if nested else 0

    def show(name, label, fmt="{:+.4f}"):
        i = ix[name]
        s = abs(mu[i].item()) / max(err[i].item(), 1e-12)
        print(
            f"  {label:<50} {fmt.format(mu[i].item())} ± {err[i].item():.4f}"
            f"   ({s:5.1f}σ)"
        )

    gate = "the published comparison" if basis_spec is None else (
        f"the GEVP over span{{{basis_spec[1]}}}"
    )
    print(f"\n  gate — {gate}, reproduced from this code path")
    show("m_G", "m·a_t  GELT")
    show("a0_G", "A₀     GELT")
    show("a0_V", "A₀     classical GEVP ground vector")
    show("dA0_GV", "ΔA₀    GELT − classical GEVP (published: +0.066/+0.089)")

    print("\n  O_GELT = P + r,  P = orthogonal projection onto the classical span")
    show("frac_out", "norm² fraction of O_GELT outside the span")
    show("z_share", "Z_r/Z_G — share of the ground-state amplitude in r")
    show("a0_P", "A₀     P  (best classical approximation of GELT)")
    show("a0_r", "A₀     r  (the orthogonal remainder, on its own)")
    show("dA0_GP", "ΔA₀    GELT − P   ← the advantage that is outside the span")
    show("dm_GP", "Δm     GELT − P   (consistent with 0 ⇔ same state)")

    if nested:
        print("\n  scale — new content each rung of the classical ladder adds")
        for k in range(1, basis.shape[0]):
            below = ", ".join(f"×{x}" for x in levels[:k])
            show(f"ladder{k}", f"{_lab(levels[k])} outside span{{{below}}}")
        show("frac_out", "GELT   outside span{the whole basis}")
    else:
        print(
            "\n  (no ladder increments: this span mixes loop shapes, so its "
            "members are not nested)"
        )

    print("\n  contact-term test — the same fraction under the C(τ) metric")
    for tau in METRIC_TIMES[1:]:
        show(f"metric{tau}", f"metric C({tau})")
        # C(τ>0) is only a metric if it is positive definite, and for a long
        # near-degenerate ladder it is not: `deep`'s C(2) Gram has a negative
        # eigenvalue on the full sample, so the "projection" is undefined and
        # the jackknife error explodes. Say so rather than print the number
        # as if it meant something.
        ev = torch.linalg.eigvalsh(_metric(tau))
        if ev[0] <= 0 or ev[-1] / ev[0] > 1e6:
            print(
                f"      ↑ NOT USABLE on this span: the C({tau}) Gram has "
                f"λ_min = {ev[0]:+.2e}, λ_max/λ_min = {ev[-1] / ev[0]:.1e}"
            )
    print(
        "    (a contact term SHRINKS with τ — it lives in C(0) and is gone by Δ=2)"
    )

    if shape_basis is not None:
        print(f"\n  is r loop shapes? — r projected onto span{{{shape_spec[1]}}}")
        show("shape_frac", "fraction of r's norm² the richer span reaches")

    _, P_f, r_f, coeff = decompose(G, basis)
    print("\n  regression coefficients onto " + ", ".join(_lab(x) for x in levels)
          + ": " + ", ".join(f"{c:+.3f}" for c in coeff))
    return dict(
        dump=dump, mu=mu, err=err, levels=levels, Nt=Nt, B=B, m_ref=m_ref,
        n_ladder=n_lad, ix=ix, nested=nested,
        basis_arm=None if basis_spec is None else ":".join(basis_spec),
        shape_arm=None if shape_spec is None else ":".join(shape_spec),
        m_ref_fixed=m_ref_fixed,
        C_G=connected_correlator(vacuum_subtract(G)),
        C_P=connected_correlator(P_f), C_r=connected_correlator(r_f),
    )


def ensemble_of(dump):
    """run5 (seed 0) or ens1 (seed 1) — the tag is in the filename, as
    `dumps/README.md` says it must be."""
    return "ens1" if "_ens1" in os.path.basename(dump) else "run5"


def plot(runs):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # Left: the ladder's increments against GELT's, on a log scale — the point
    # is that the classical increments fall geometrically while GELT's does not
    # sit on that curve.
    for run, mk, col in zip(runs, ("o-", "s--"), ("C0", "C3")):
        mu, err, lv, n = run["mu"], run["err"], run["levels"], run["n_ladder"]
        tag = os.path.basename(run["dump"]).replace("_test_obars.pt", "")
        tag = tag.rsplit("_", 1)[-1] if tag.endswith(("_ens1", "_ens2")) else "Run 5"
        if run["nested"]:
            xs = list(range(1, n + 1))
            ys = [mu[run["ix"][f"ladder{k}"]].item() for k in xs]
            es = [err[run["ix"][f"ladder{k}"]].item() for k in xs]
            ax[0].errorbar(xs, ys, yerr=es, fmt=mk, color=col, capsize=3,
                           label=f"classical ladder — {tag}")
        ax[0].errorbar([n + 1], [mu[0].item()], yerr=[err[0].item()], fmt="*",
                       ms=16, color=col, capsize=3)
    n = runs[0]["n_ladder"]
    lv = runs[0]["levels"]
    ax[0].set_xticks(list(range(1, n + 2)))
    ax[0].set_xticklabels([_lab(lv[k]) for k in range(1, n + 1)] + ["GELT"])
    ax[0].set_yscale("log")
    ax[0].set_ylabel("norm² fraction outside the span of everything below")
    ax[0].set_title("New operator content per addition\n(★ = the learned operator)")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(fontsize=8)

    # Right: the overlap panel for the three pieces. ρ(Δ) flat at A₀ ⇔ pure
    # ground state; r is plotted to show it is not a contact term.
    run = runs[0]
    Nt, m = run["Nt"], run["m_ref"]
    ref = torch.tensor(
        [(math.exp(-m * d) + math.exp(-m * (Nt - d)))
         / (1 + math.exp(-m * Nt)) for d in range(Nt)], dtype=torch.float64
    )
    dmax = FIT_WINDOW[1] + 1
    for C, lab, col in ((run["C_G"], "O_GELT", "C0"),
                        (run["C_P"], "P  (in the classical span)", "C1"),
                        (run["C_r"], "r  (orthogonal remainder)", "C3")):
        rho = (C / C[0]) / ref
        ax[1].plot(range(1, dmax + 1), rho[1 : dmax + 1], "o-", color=col, label=lab)
    ax[1].axvspan(FIT_WINDOW[0] - 0.3, FIT_WINDOW[1] + 0.3, color="grey", alpha=0.12)
    ax[1].set_xlabel("Δ")
    ax[1].set_ylabel("ρ(Δ) = [C(Δ)/C(0)] / cosh_ref")
    ax[1].set_title("Ground-state purity of each piece\n(flat ⇔ pure; shaded = fit window)")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(fontsize=8)

    span = ("the classical span" if not BASIS_SPECS
            else f"span{{{BASIS_SPECS[0][1]}}}")
    fig.suptitle(
        f"The learned operator decomposed against {span} — "
        "SU(2) 12³×24, β = 2.4, ξ = 3", fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"\nSaved {OUT_PNG}")


def combine(runs, key):
    """Combine one statistic over runs: seeds of the same ensemble average
    under §1.4's rule (the larger of the seed spread and the mean error), and
    independent ensembles combine inverse-variance, as §6.2 does."""
    groups = {}
    for r in runs:
        groups.setdefault(ensemble_of(r["dump"]), []).append(
            (r["mu"][r["ix"][key]].item(), r["err"][r["ix"][key]].item())
        )
    per_ens = []
    for ens, vals in groups.items():
        v = torch.tensor([x for x, _ in vals])
        e = torch.tensor([y for _, y in vals])
        if len(vals) == 1:
            per_ens.append((ens, v[0].item(), e[0].item()))
        else:
            spread = v.std(unbiased=True).item() / math.sqrt(len(vals))
            per_ens.append((ens, v.mean().item(), max(spread, e.mean().item())))
    vals = torch.tensor([x for _, x, _ in per_ens])
    errs = torch.tensor([y for _, _, y in per_ens])
    wgt = 1.0 / errs**2
    comb = ((vals * wgt).sum() / wgt.sum()).item()
    return comb, (1.0 / wgt.sum()).sqrt().item(), per_ens


def main():
    runs = [analyse(d, i) for i, d in enumerate(DUMPS)]
    if len(runs) > 1:
        # Combine the correlated advantages across independent ensembles the
        # same way §6.2 combines its ΔA₀: inverse-variance weighted.
        print(f"\n{'=' * 78}\ncombined over {len(runs)} runs")
        for key, label in (
            ("dA0_GP", "ΔA₀ (GELT − P)  — the advantage outside the classical span"),
            ("frac_out", "norm² fraction outside the span"),
            ("z_share", "Z_r/Z_G"),
        ) + ((("shape_frac", "fraction of r inside the richer span"),)
             if SHAPE_SPECS else ()):
            comb, cerr, per_ens = combine(runs, key)
            detail = ", ".join(f"{e} {v:+.4f}±{s:.4f}" for e, v, s in per_ens)
            print(f"  {label}\n    = {comb:+.4f} ± {cerr:.4f} "
                  f"({abs(comb) / cerr:.1f}σ)   [{detail}]")
    plot(runs)
    torch.save({"runs": [{k: v for k, v in r.items()} for r in runs],
                "fit_window": FIT_WINDOW, "metric_times": METRIC_TIMES,
                "basis": BASIS_SPECS, "shape_span": SHAPE_SPECS, "m_ref": M_REFS},
               OUT_PT)
    print(f"Saved {OUT_PT}")


if __name__ == "__main__":
    main()
