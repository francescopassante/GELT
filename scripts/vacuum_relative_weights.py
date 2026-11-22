"""Exact relative weights of the vacuum along a configuration ladder — ground truth.

For two slices U, U' the reweighting identity is exact (Greensite–Iwasaki's
relative weights, here by the free-energy-perturbation route):

    ψ₀²(U')/ψ₀²(U) = e^{−[S_s(U') − S_s(U)]} · E_{rest | U}[ e^{−ΔS_temp} ],
    ΔS_temp = −(β_t/nc) Σ_x,i Re Tr[(U'_i − U_i)(x) · A_i(x)],

A the temporal staples of the slice (they depend on the rest only). So with
log ψ₀² = −S_s + Φ, the true Φ(U') − Φ(U) is log E_U[e^{−ΔS_temp}], measured by
a chain that samples the rest with the slice frozen at U (gelt.sampler's
``frozen=``) — no model, no fit, no training data.

A ladder U(s) = exp(i s X·σ/2) U* (s = 0, δ, 2δ, …) on top of a test-split slice
U*, with δ chosen from a pilot so that ΔS_temp fluctuates by VAC_RW_TARGET_STD
per rung (the overlap condition). Each rung's difference is estimated forward
from chain k and backward from chain k+1; their mean is the estimate and their
disagreement is the gate (G-RW: more than max(1, 5%) of the rungs off by > 3σ
⇒ the ladder is not read). The rung estimate itself is Bennett's acceptance
ratio, the minimum-variance combination of the two. Families (VAC_FAMILY):

  noise   X ~ N(0, 1) per link and colour: the ultraviolet direction.
  plane   an abelian plane wave, X³ = cos(2π x₁/L) on the links along axis 0 of
          the slice (Greensite's family, here on top of a typical slice).
  smear   X = log(V U*†), V the 3D-APE-smeared slice (4 steps): s = 1 is the
          smooth configuration. Far off the training distribution — reported,
          read separately.

Every model is then scored on the same ladder: each classical arm of
vacuum_classical.py, and every network checkpoint in VAC_RW_NETS (comma list;
its residual arm is added back). Per model: the cumulative Φ(s) − Φ(0) against
the truth, χ²/dof with the truth's errors, the worst deviation, and R².

Env / --vac-*: VAC_FAMILY (noise), VAC_RW_CONFIG (0: index into the test split),
VAC_RW_TARGET_STD (1.0), VAC_RW_SMAX (0.3 noise, 1.0 plane and smear),
VAC_RW_MAX_RUNGS (200), VAC_RW_MEAS (32 measurements per rung, one sweep apart),
VAC_RW_EQ (4 sweeps after each switch), VAC_RW_SEED (0), VAC_N_OR_COND (2),
VAC_RW_NETS, VAC_SMOKE. Output: rw_<family>_cfg<k>[_std<s>]<ens>.pt, the
_std tag only off the default spread.

    VAC_FAMILY=plane VAC_RW_NETS=results/vacuum/net_gelt_sm0_resfull_init0_ens0.pth \\
        python scripts/vacuum_relative_weights.py
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vacuum_common as vc  # noqa: E402

vc.validate_argv("rw-smear-steps")

import torch  # noqa: E402

from gelt.glueball import ape_smear  # noqa: E402
from gelt.sampler import heatbath_overrelaxation_sweep  # noqa: E402
from gelt.vacuum import (  # noqa: E402
    delta_temporal_action,
    frozen_slice_mask,
    slice_spatial_action,
    slice_temporal_staples,
    su2_exp,
    su2_log,
)

FAMILY = str(vc.cfg("VAC_FAMILY", "noise")).lower()
if FAMILY not in ("noise", "plane", "smear"):
    raise SystemExit(f"VAC_FAMILY must be noise | plane | smear, got {FAMILY!r}")
CONF = int(vc.cfg("VAC_RW_CONFIG", 0))
TARGET_STD = float(vc.cfg("VAC_RW_TARGET_STD", 1.0))
SMAX = float(vc.cfg("VAC_RW_SMAX", 0.3 if FAMILY == "noise" else 1.0))
MAX_RUNGS = int(vc.cfg("VAC_RW_MAX_RUNGS", 12 if vc.SMOKE else 200))
MEAS = int(vc.cfg("VAC_RW_MEAS", 8 if vc.SMOKE else 32))
EQ = int(vc.cfg("VAC_RW_EQ", 2 if vc.SMOKE else 4))
SEED = int(vc.cfg("VAC_RW_SEED", 0))
N_OR = int(vc.cfg("VAC_N_OR_COND", 2))
SMEAR_STEPS = int(vc.cfg("VAC_RW_SMEAR_STEPS", 4))
NETS = [p for p in str(vc.cfg("VAC_RW_NETS", "")).split(",") if p]
JB = 4  # jackknife block, in measurements (they are one sweep apart)
# the target spread is in the name when it is not the default, so a rerun of a
# ladder that failed G-RW at a smaller step sits beside it instead of over it
STD_TAG = "" if TARGET_STD == 1.0 else f"_std{TARGET_STD:g}"
OUT_PATH = f"{vc.OUT}/rw_{FAMILY}_cfg{CONF}{STD_TAG}{vc.ENS}.pt"


def direction(Ustar):
    """X ``(3, *Λ, 3)`` float64 of the family, on the device."""
    g = torch.Generator(device="cpu").manual_seed(SEED)
    shape = Ustar.shape[:-2] + (3,)
    if FAMILY == "noise":
        return torch.randn(shape, generator=g, dtype=torch.float64).to(vc.device)
    if FAMILY == "plane":
        X = torch.zeros(shape, dtype=torch.float64, device=vc.device)
        x1 = torch.arange(vc.L, dtype=torch.float64, device=vc.device)
        X[0, :, :, :, 2] = torch.cos(2 * math.pi * x1 / vc.L)[None, :, None]
        return X
    V = ape_smear(Ustar[None].to(torch.complex128), vc.G, alpha=vc.tg.SMEAR_ALPHA,
                  n_steps=SMEAR_STEPS, directions=range(3))[0]
    return su2_log(V @ vc.G.dagger(Ustar.to(torch.complex128)))


def _exp_fwd(wf, wr):
    """Forward exponential average: Φ_{k+1} − Φ_k = log ⟨e^{−w_F}⟩_k."""
    return float(torch.logsumexp(-wf, 0) - math.log(wf.shape[0]))


def _exp_bwd(wf, wr):
    """Backward: Φ_{k+1} − Φ_k = −log ⟨e^{−w_R}⟩_{k+1}."""
    return -float(torch.logsumexp(-wr, 0) - math.log(wr.shape[0]))


def _bar(wf, wr):
    """Bennett's acceptance ratio for Φ_{k+1} − Φ_k (equal sample sizes).

    w_F = S_{k+1} − S_k on chain k, w_R = S_k − S_{k+1} on chain k+1, F the free
    energy (= −Φ). Solves Σ σ(−(w_F − ΔF)) = Σ σ(−(w_R + ΔF)) by bisection —
    the minimum-variance combination of the two exponential averages.
    """
    def h(dF):
        return float(torch.sigmoid(-(wf - dF)).sum() - torch.sigmoid(-(wr + dF)).sum())
    lo, hi = -_exp_bwd(wf, wr) - 50.0, -_exp_fwd(wf, wr) + 50.0
    lo, hi = min(lo, hi), max(lo, hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if h(mid) > 0:
            hi = mid
        else:
            lo = mid
    return -0.5 * (lo + hi)


def rung_jack(fn, wf, wr):
    """An estimator of one rung and its blocked-jackknife error over both chains."""
    wf = torch.as_tensor(wf, dtype=torch.float64)
    wr = torch.as_tensor(wr, dtype=torch.float64)
    n = min(wf.shape[0], wr.shape[0])
    nb = max(2, n // JB)
    edges = [round(k * n / nb) for k in range(nb + 1)]
    cut = lambda x, a, b: torch.cat([x[:a], x[b:]])  # noqa: E731
    samp = torch.tensor([fn(cut(wf, a, b), cut(wr, a, b)) for a, b in zip(edges[:-1], edges[1:])])
    return fn(wf, wr), math.sqrt((nb - 1) / nb * float(((samp - samp.mean()) ** 2).sum()))


def main():
    vc.require_cuda("vacuum_relative_weights")
    configs = vc.load_ensemble()
    _, _, test = vc.splits(configs.shape[0])
    if not 0 <= CONF < len(test):
        raise SystemExit(f"VAC_RW_CONFIG={CONF}: the test split has {len(test)} configurations")
    i = list(test)[CONF]
    t0 = vc.LT // 2
    U = configs[i].to(vc.device)
    dt = U.dtype
    Ustar = U[1:, t0].to(torch.complex128)
    X = direction(Ustar)
    frozen = frozen_slice_mask(U.shape[:-2], t0, device=vc.device)
    slice_at = lambda s: su2_exp(s * X) @ Ustar  # noqa: E731  (complex128)

    def sweeps(U, n):
        for _ in range(n):
            U, _ = heatbath_overrelaxation_sweep(U, vc.G, vc.BETA, n_or=N_OR, xi=vc.XI,
                                                 frozen=frozen)
        return U

    def staples(U):
        return slice_temporal_staples(U[None].to(torch.complex128), vc.G, t0)[0]

    # Pilot: the spread of ΔS_temp over one step sets δ — the overlap between
    # neighbouring rungs depends on it. Linear in δ only to first order, so the
    # step is re-measured at the proposed size and rescaled until the spread is
    # within [0.7, 1.4] × the target (the plane wave's first guess was 4× off).
    def spread(d):
        nonlocal U
        dS = []
        for _ in range(max(8, MEAS // 2)):
            U = sweeps(U, 1)
            dS.append(float(delta_temporal_action((slice_at(d) - Ustar)[None],
                                                  staples(U)[None], vc.BETA_T)))
        return float(torch.tensor(dS).std())

    delta = 1e-3
    for it in range(8):
        sd = spread(delta)
        if 0.7 * TARGET_STD <= sd <= 1.4 * TARGET_STD:
            break
        delta *= TARGET_STD / max(sd, 1e-12) if it == 0 else min(
            2.0, max(0.5, TARGET_STD / max(sd, 1e-12)))
    n_rungs = min(MAX_RUNGS, max(2, math.ceil(SMAX / delta)))
    smax = n_rungs * delta
    print(f"{FAMILY} ladder on test config {i}, slice {t0}: δ = {delta:.4g} "
          f"(ΔS spread {sd:.3g} after {it + 1} pilot rounds), {n_rungs} rungs, "
          f"s_max = {smax:.4g}"
          + ("" if smax >= SMAX - 1e-12 else f"  (capped by VAC_RW_MAX_RUNGS; asked {SMAX})"))
    s = [k * delta for k in range(n_rungs + 1)]
    ladder = [slice_at(x) for x in s]

    fwd, bwd = [[] for _ in s], [[] for _ in s]
    t = time.time()
    for k in range(len(s)):
        U[1:, t0] = ladder[k].to(dt)
        U = sweeps(U, EQ)
        for _ in range(MEAS):
            U = sweeps(U, 1)
            A = staples(U)[None]
            if k + 1 < len(s):
                fwd[k].append(float(delta_temporal_action((ladder[k + 1] - ladder[k])[None],
                                                          A, vc.BETA_T)))
            if k > 0:
                bwd[k].append(float(delta_temporal_action((ladder[k - 1] - ladder[k])[None],
                                                          A, vc.BETA_T)))
        if (k + 1) % 20 == 0:
            print(f"  rung {k + 1}/{len(s)}  {(time.time() - t) / (k + 1):.1f} s/rung", flush=True)

    dphi, err, bad = [], [], 0
    for k in range(n_rungs):
        f, fe = rung_jack(_exp_fwd, fwd[k], bwd[k + 1])
        b, be = rung_jack(_exp_bwd, fwd[k], bwd[k + 1])
        m, me = rung_jack(_bar, fwd[k], bwd[k + 1])
        dphi.append(m)
        err.append(me)
        if abs(f - b) > 3 * math.sqrt(fe * fe + be * be) + 1e-12:
            bad += 1
    true = torch.tensor([0.0] + list(torch.tensor(dphi).cumsum(0)), dtype=torch.float64)
    terr = torch.tensor([0.0] + list(torch.tensor(err).pow(2).cumsum(0).sqrt()),
                        dtype=torch.float64)
    Ss = slice_spatial_action(torch.stack(ladder), vc.G, vc.BETA_S).cpu()
    # Each rung's error is itself a jackknife over ~8 blocks, so a 3σ excursion
    # is likelier than the Gaussian 0.3%: allow one on a short ladder, 5% on a long one.
    gate = bad <= max(1, 0.05 * n_rungs)
    print(f"G-RW: {bad}/{n_rungs} rungs with forward ≠ backward at 3σ → "
          + ("pass" if gate else "FAIL — lower VAC_RW_TARGET_STD; the ladder is not read"))
    print(f"true Φ(s_max) − Φ(0) = {float(true[-1]):+.3f} ± {float(terr[-1]):.3f};  "
          f"log ψ₀² change = {float(true[-1] - (Ss[-1] - Ss[0])):+.3f}")

    models = predictions(torch.stack(ladder))
    res = {}
    print(f"\n{'model':38s} {'Φ(s_max)−Φ(0)':>14s} {'χ²/dof':>8s} {'max|Δ|/σ':>9s} {'R²':>7s}")
    for name, pred in models.items():
        pred = pred - pred[0]
        dev = pred - true
        sig = terr[1:].clamp_min(1e-9)
        chi2 = float(((dev[1:] / sig) ** 2).mean())
        r2 = 1 - float((dev ** 2).sum() / ((true - true.mean()) ** 2).sum().clamp_min(1e-30))
        res[name] = dict(pred=pred, chi2=chi2, maxdev=float((dev[1:].abs() / sig).max()), r2=r2)
        print(f"{name:38s} {float(pred[-1]):+14.3f} {chi2:8.2f} {res[name]['maxdev']:9.1f} {r2:7.4f}")
    torch.save(dict(family=FAMILY, config=i, t0=t0, s=s, delta=delta, true=true, true_err=terr,
                    dphi=dphi, dphi_err=err, bad_rungs=bad, gate=gate, S_s=Ss, models=res,
                    fwd=fwd, bwd=bwd, meta=dict(target_std=TARGET_STD, meas=MEAS, eq=EQ,
                                                n_or=N_OR, seed=SEED, nets=NETS,
                                                smoke=vc.SMOKE)), OUT_PATH)
    print(f"→ {OUT_PATH}")


@torch.no_grad()
def _phi_classical(arm, coef, slices):
    out = []
    for j in range(0, slices.shape[0], 16):
        out.append(vc.classical_phi(arm, coef)(slices[j:j + 16]))
    return torch.cat(out).cpu()


def predictions(slices):
    """Φ_model on every rung, per model: the classical arms, then the nets."""
    out = {}
    cl = None
    if os.path.exists(vc.classical_path()):
        cl = torch.load(vc.classical_path(), weights_only=False)
        out["null (Φ ≡ 0)"] = torch.zeros(slices.shape[0], dtype=torch.float64)
        for arm in vc.ARM_ORDER:
            out[f"classical {arm}"] = _phi_classical(arm, cl["arms"][arm]["coef"], slices)
    else:
        print(f"** no {vc.classical_path()} — classical arms skipped")
    for path in NETS:
        model, c = vc.load_net(path)
        phi = []
        with torch.no_grad():
            for j in range(0, slices.shape[0], 8):
                phi.append(vc.phi_net(model, c, slices[j:j + 8].to(vc.tg.MODEL_DTYPE)).double().cpu())
        phi = torch.cat(phi)
        if c["residual"] != "none":
            if cl is None:
                raise SystemExit(f"{path} is residual over {c['residual']} but there is "
                                 f"no {vc.classical_path()}")
            phi = phi + _phi_classical(c["residual"], cl["arms"][c["residual"]]["coef"], slices)
        out[os.path.basename(path).replace(".pth", "")] = phi
    return out


if __name__ == "__main__":
    main()
