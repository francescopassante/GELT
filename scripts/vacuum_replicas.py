"""Conditional replicas on the test split: the vacuum score's signal and noise floor.

For each of VAC_REPLICA_CONFIGS test configurations (evenly spaced), one slice
t0 is frozen and every other link is resampled by the exact SU(2) heat-bath +
overrelaxation chain conditioned on it (gelt.sampler's ``frozen=``). The chain
starts from the configuration itself, which is already a draw from that
conditional, so nothing thermalises: the replicas are exact at every sweep and
only their correlation with F⁰ decays.

Stored per config: the slice U, F⁰ (the temporal force on it in the measured
configuration), F̄ (mean of the replicas at sweeps ≥ VAC_LATE), the curve
c(n) = F⁰·F(n), and the spatial force's ‖F_s‖², F_s·F⁰, F_s·F̄ (to normalise ε
against the full score). The curve is gate V0: it must plateau, and its
plateau is ‖E[F|U]‖² — the part of the temporal force the slice determines.
Also checked here, on the production ensemble: Stein's identity (the test of
tests/test_vacuum.py) on the measured slices.

Resumable (a partial dump is extended). Each chain is seeded from
(VAC_REPLICA_SEED, config index), so a resumed run is the run it would have
been in one pass, and a second seed gives independent chains on the same
configurations. V100, ~0.7 min per sweep of the grid's longest chain (200
configs): ~90 min at the defaults.

The defaults are run 2's (2026-11-21). Run 1 (2026-11-20, unseeded, sweeps
"1,2,3,4,6,8,12,16,24,32", late 12) failed V0 at c(12) − c(32) = +0.0017 ±
0.0005 with the curve flat from n = 3 to 24 — notes/thesis_record.md §11 P12.
Its dump is kept as replicas<ens>_run1.pt.

Env / --vac-*: VAC_ENSEMBLE_SEED (0), VAC_REPLICA_CONFIGS (200),
VAC_REPLICA_SWEEPS ("1,2,3,4,6,8,12,16,24,32,48,64,96,128"), VAC_LATE (32),
VAC_N_OR_COND (2), VAC_REPLICA_SEED (1), VAC_SMOKE, VAC_DEVICE.

    python scripts/vacuum_replicas.py
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vacuum_common as vc  # noqa: E402

vc.validate_argv()

import torch  # noqa: E402

from gelt.lattice import plaquette_tensor  # noqa: E402
from gelt.sampler import heatbath_overrelaxation_sweep  # noqa: E402
from gelt.vacuum import (  # noqa: E402
    casimir_fundamental,
    frozen_slice_mask,
    spatial_force,
    temporal_force,
)

N_REP = int(vc.cfg("VAC_REPLICA_CONFIGS", 12 if vc.SMOKE else 200))
SWEEPS = vc.levels(vc.cfg("VAC_REPLICA_SWEEPS", "1,2,4,8" if vc.SMOKE
                          else "1,2,3,4,6,8,12,16,24,32,48,64,96,128"))
LATE = int(vc.cfg("VAC_LATE", 4 if vc.SMOKE else 32))
N_OR = int(vc.cfg("VAC_N_OR_COND", 2))
SEED = int(vc.cfg("VAC_REPLICA_SEED", 1))
if not [n for n in SWEEPS if n >= LATE]:
    raise SystemExit(f"no replica sweep ≥ VAC_LATE={LATE} in {SWEEPS}")


def slice_force(U4c, t0):
    """Temporal force on slice t0 of one config, ``(3, *Λ, 3)`` float64."""
    return temporal_force(U4c[None], vc.G, vc.BETA, vc.XI)[t0]


def main():
    vc.require_cuda("vacuum_replicas")
    configs = vc.load_ensemble()
    _, _, test = vc.splits(configs.shape[0])
    idx = vc.evenly(test, N_REP)
    path = vc.replica_path()
    rep = torch.load(path, weights_only=False) if os.path.exists(path) else None
    if rep is not None and (rep["meta"]["sweeps"] != SWEEPS or rep["meta"]["late"] != LATE
                            or rep["meta"]["n_or"] != N_OR
                            or rep["meta"].get("seed") != SEED):
        m = rep["meta"]
        raise SystemExit(f"{path} was written with sweeps {m['sweeps']}, late {m['late']}, "
                         f"n_or {m['n_or']}, seed {m.get('seed')}; move it away to change "
                         "the grid, the chain or the seed")
    if rep is None:
        rep = dict(idx=[], t0=[], U3=[], F0=[], Flate=[], curve=[], FsFs=[], FsF0=[],
                   FsFl=[], stein=[],
                   meta=dict(sweeps=SWEEPS, late=LATE, n_or=N_OR, seed=SEED, L=vc.L, Lt=vc.LT,
                             beta=vc.BETA, xi=vc.XI, ensemble_seed=vc.ENSEMBLE_SEED,
                             smoke=vc.SMOKE))
    done = set(rep["idx"])
    todo = [i for i in idx if i not in done]
    print(f"replicas: {len(done)} done, {len(todo)} to go | sweeps {SWEEPS} | "
          f"late ≥ {LATE} | n_or {N_OR} | seed {SEED} | device {vc.device}")
    late = [n for n in SWEEPS if n >= LATE]
    CF = casimir_fundamental(vc.NC)
    t_start = time.time()
    for k, i in enumerate(todo):
        t0 = (7 * i) % vc.LT  # spread the frozen slice over the time axis
        U4 = configs[i].to(vc.device)
        U4c = U4.to(torch.complex128)
        F0 = slice_force(U4c, t0)
        frozen = frozen_slice_mask(U4.shape[:-2], t0, device=vc.device)
        U = U4.clone()
        torch.manual_seed(1_000_003 * SEED + i)  # per chain: resume-invariant
        curve, Fl = [], torch.zeros_like(F0)
        for n in range(1, max(SWEEPS) + 1):
            U, _ = heatbath_overrelaxation_sweep(U, vc.G, vc.BETA, n_or=N_OR, xi=vc.XI,
                                                 frozen=frozen)
            if n in SWEEPS:
                F = slice_force(U.to(torch.complex128), t0)
                curve.append(float((F0 * F).sum()))
                if n >= LATE:
                    Fl += F / len(late)
        if not torch.equal(U[1:, t0], U4[1:, t0]):
            raise SystemExit("the frozen slice moved — the conditional chain is broken")
        U3 = U4c[1:, t0][None]  # (1, 3, *Λ, nc, nc)
        Fs = spatial_force(U3, vc.G, vc.BETA_S)[0]
        # Stein on this slice: ∇h·(F_s + F⁰) − 4 C_F h, E[·] = 0 (tests/test_vacuum.py).
        gh = spatial_force(U3, vc.G, float(vc.NC))[0]
        P = plaquette_tensor(U3, vc.G)
        h = float(P.diagonal(dim1=-2, dim2=-1).sum(-1).real.sum())
        rep["idx"].append(i)
        rep["t0"].append(t0)
        rep["U3"].append(U4[1:, t0].cpu())
        rep["F0"].append(F0.float().cpu())
        rep["Flate"].append(Fl.float().cpu())
        rep["curve"].append(curve)
        rep["FsFs"].append(float((Fs * Fs).sum()))
        rep["FsF0"].append(float((Fs * F0).sum()))
        rep["FsFl"].append(float((Fs * Fl).sum()))
        rep["stein"].append(float((gh * (Fs + F0)).sum()) - 4 * CF * h)
        if (k + 1) % 10 == 0 or k + 1 == len(todo):
            torch.save(rep, path)
            el = time.time() - t_start
            print(f"  {k + 1}/{len(todo)}  {el / (k + 1):.1f} s/config", flush=True)
    torch.save(rep, path)
    report(rep)


def report(rep):
    n = len(rep["idx"])
    curve = torch.tensor(rep["curve"], dtype=torch.float64)  # (n, len(SWEEPS))
    F0F0 = torch.tensor([float((f.double() ** 2).sum()) for f in rep["F0"]])
    S = torch.tensor([float((a.double() * b.double()).sum())
                      for a, b in zip(rep["F0"], rep["Flate"])])
    ncomp = rep["F0"][0].numel()
    print(f"\n{n} test configurations, {ncomp} force components per slice")
    print("V0 — the conditional autocorrelation c(n) = ⟨F⁰·F(n)⟩ / ⟨‖F⁰‖²⟩ "
          "(must plateau; the plateau is the predictable fraction):")
    for j, s in enumerate(rep["meta"]["sweeps"]):
        r, e = vc.jackknife_ratio(curve[:, j], F0F0, vc.JACK_BLOCK)
        print(f"   n = {s:3d}: {r:.4f} ± {e:.4f}")
    sw = rep["meta"]["sweeps"]
    lt = [j for j, s in enumerate(sw) if s >= rep["meta"]["late"]]
    if len(lt) >= 2:
        d, e = vc.jackknife_diff(curve[:, lt[0]], curve[:, lt[-1]], F0F0, vc.JACK_BLOCK)
        verdict = "plateau" if abs(d) < 2 * e + 1e-12 else "NOT A PLATEAU — raise the sweeps"
        print(f"   first − last late point: {d:+.4f} ± {e:.4f}  → {verdict}")
    sig, se = vc.jackknife_ratio(S, F0F0, vc.JACK_BLOCK)
    print(f"V1 — predictable fraction ‖E[F|U]‖²/⟨‖F‖²⟩ = {sig:.4f} ± {se:.4f}; "
          f"signal per component {float(S.mean()) / ncomp:.4g}, "
          f"noise per component {float((F0F0 - S).mean()) / ncomp:.4g}")
    st = torch.tensor(rep["stein"], dtype=torch.float64)
    z = float(st.mean() / (st.std() / math.sqrt(n))) if n > 1 else float("nan")
    print(f"Stein on the measured slices: mean {float(st.mean()):+.3f} (z = {z:+.2f}; "
          f"|z| > 4 means the force convention is broken on this ensemble)")


if __name__ == "__main__":
    main()
