"""Pre-flight and classical arms for the flow-free topology study (WP2).

Everything here is the *classical* half of `notes/flow_free_topology.md`: it
produces the targets a network will later be trained against, and the baselines
it has to beat. No network is built or trained in this file.

Five phases, selected with ``TOPO_PHASE`` (comma-separated, cached artifacts are
skipped):

  chain     One long chain per (β, L) stored at n_skip = 1, then τ_int of the
            **flowed topological charge** and of the plaquette. Topological
            modes are the slowest in the theory, so n_skip must be set from
            τ_int(Q) — τ_int(plaquette) will quietly under-report it, and
            topological freezing at β = 2.5 is a live risk that has to be
            measured rather than assumed.

  gate0     **The gate.** Flow a handful of configurations to t/a² = 16 and plot
            Q(t). The primary rung t/a² = 2 must sit on a plateau and Q must be
            near an integer. If it does not at β = 2.3 — plausible, the coarsest
            lattice being where dislocations survive longest — the primary rung
            moves and the fact is recorded rather than hidden. Also checks the
            integrator step size: t/a² = 2 at ε = 0.02 against ε = 0.005.

  ensemble  The production ensembles, cached under datasets/.

  targets   q_clov at t = 0 (the network's classical reference) and the five-rung
            ladder q_t(x), one flow trajectory per configuration, cached.

  arms      The classical baselines on the test split: the identity
            (q̂ = q_clov[U], raw and with a fitted multiplicative Z), and the
            **best linear filter** — the least-squares-optimal convolution on the
            Manhattan-R ball, both unconstrained and hypercubic-symmetrised.
            This is the most important control in the whole study: the gap
            between the best filter and the best network *is* the nonlinear
            content of the flow, and if a network does not clearly beat it the
            task is a convolution and the study stops.

Run (pre-flight, the two phases that must be read first):

    python scripts/measure_topology.py
    TOPO_SMOKE=1 python scripts/measure_topology.py          # seconds, L=4, CPU

Run (production, needs the GPU and a night):

    TOPO_PHASE=ensemble,targets,arms python scripts/measure_topology.py
"""

import itertools
import math
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import functools

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from gelt.flow import flow_trajectory, wilson_flow
from gelt.lattice import SU, plaquette_tensor, topological_charge_density
from gelt.sampler import (
    _re_tr,
    heatbath_overrelaxation_sweep,
    integrated_autocorrelation_time,
    mcmc_ensemble,
)

for _d in ("results/topology", "datasets"):
    os.makedirs(_d, exist_ok=True)


# ── Environment overrides (same idiom as train_glueball.py) ──────────────────
def _flag(name):
    return f"--{name.lower().replace('topo_', '').replace('_', '-')}"


def _env_int(name, default):
    for a in sys.argv[1:]:
        if a.startswith(_flag(name) + "="):
            return int(a.split("=", 1)[1])
    return int(os.environ.get(name, default))


def _env_float(name, default):
    for a in sys.argv[1:]:
        if a.startswith(_flag(name) + "="):
            return float(a.split("=", 1)[1])
    return float(os.environ.get(name, default))


def _env_str(name, default):
    for a in sys.argv[1:]:
        if a.startswith(_flag(name) + "="):
            return a.split("=", 1)[1]
    return os.environ.get(name, default)


def _env_bool(name, default=False):
    for a in sys.argv[1:]:
        if a == _flag(name):
            return True
        if a.startswith(_flag(name) + "="):
            return a.split("=", 1)[1] not in ("0", "false", "False", "")
    v = os.environ.get(name)
    return default if v is None else v not in ("0", "false", "False", "")


def _env_floats(name, default):
    for a in sys.argv[1:]:
        if a.startswith(_flag(name) + "="):
            return tuple(float(x) for x in a.split("=", 1)[1].split(","))
    v = os.environ.get(name)
    return default if not v else tuple(float(x) for x in v.split(","))


def _env_ints(name, default):
    for a in sys.argv[1:]:
        if a.startswith(_flag(name) + "="):
            return tuple(int(x) for x in a.split("=", 1)[1].split(","))
    v = os.environ.get(name)
    return default if not v else tuple(int(x) for x in v.split(","))


SMOKE = _env_bool("TOPO_SMOKE")

# ── Tunables ─────────────────────────────────────────────────────────────────
D = 4  # the Levi-Civita symbol needs exactly four directions
GROUP = SU(2)

# β and L are *paired* rows, chosen so the physical volume is roughly constant
# (≈(1.3 fm)⁴) while a halves — see notes/flow_free_topology.md §8. Both
# architectures are translation-equivariant, so training across the rows is
# legitimate. The scale itself must be re-derived in-repo from
# rectangular_wilson_loop before it enters any physics claim; it does not enter
# anything measured here.
BETAS = _env_floats("TOPO_BETAS", (2.3, 2.4, 2.5))
LS = _env_ints("TOPO_LS", (8, 12, 16))

# The five-rung target ladder, in **lattice units** t/a² — identical across β on
# purpose, so r_sm/a (and with it the receptive-field requirement) is held fixed
# and a mixed-β model's degradation is attributable to scale adaptivity rather
# than to a capacity wall at one β. r_sm/a = √(8 t/a²) = 2.0, 2.8, 4.0, 5.7, 8.0;
# the top rung sits at the Manhattan-8 boundary on purpose, as the capacity control.
# Moved up after the 2026-09-14 pre-flight (notes §12.5): at t/a² = 2 the charge
# shows no integer structure beyond the null at any β, and the renormalisation
# Z* has not converged there either. Structure sets in at t/a² = 3–4
# (r_sm/a ≈ 4.9–5.7) on every row that can host it. The pre-registered response
# to exactly this was "the primary rung moves, and the fact is recorded".
T_LADDER = _env_floats("TOPO_LADDER", (1.0, 2.0, 4.0, 6.0, 8.0))
T_PRIMARY = _env_float("TOPO_T_PRIMARY", 4.0)
FLOW_EPS = _env_float("TOPO_EPS", 0.02)

N_CONFIGS = _env_int("TOPO_N_CONFIGS", 1200)  # 800 train / 200 val / 200 test
N_THERM = _env_int("TOPO_N_THERM", 500)
# Measured by the `chain` phase 2026-09-14: τ_int(Q) = 0.55 / 0.64 / 2.76 at
# β = 2.3/2.4/2.5, so 2·τ_int ≤ 5.5 at the worst row. τ_int's own error there is
# ±1.2 (window 17, 400 samples), so 10 keeps ≳3·τ_int of margin — and halves the
# sampling cost against the provisional 20. **No topological freezing** was found
# at any row: 266–322 integer-sector changes in 400 sweeps.
N_SKIP = _env_int("TOPO_N_SKIP", 10)
N_OR = _env_int("TOPO_N_OR", 4)
N_CHAIN = _env_int("TOPO_N_CHAIN", 400)  # consecutive sweeps for τ_int
N_GATE = _env_int("TOPO_N_GATE", 16)  # configs carried through the Q(t) scan
T_GATE_MAX = _env_float("TOPO_T_GATE_MAX", 16.0)
FILTER_R = _env_int("TOPO_FILTER_R", 8)  # Manhattan radius of the linear arm
CHUNK = _env_int("TOPO_CHUNK", 16)  # configurations flowed at once
SPLITS = (2 / 3, 1 / 6, 1 / 6)  # train / val / test, contiguous and chain-ordered
Z_BAND = (0.6, 1.1)  # plausible range for the charge renormalisation Z(β, t)
PHASES = _env_str("TOPO_PHASE", "chain,gate0").split(",")

if SMOKE:
    # A few seconds on a CPU: every phase runs, no phase means anything.
    BETAS, LS = (2.4,), (4,)
    T_LADDER, T_PRIMARY, T_GATE_MAX = (0.5, 1.0), 1.0, 2.0
    N_CONFIGS, N_THERM, N_SKIP, N_CHAIN, N_GATE = 24, 20, 2, 40, 2
    FILTER_R, CHUNK = 2, 8
    PHASES = _env_str("TOPO_PHASE", "chain,gate0,ensemble,targets,arms").split(",")

DEVICE = torch.device(
    _env_str(
        "TOPO_DEVICE",
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
)
# SU(2) projection needs complex linalg, which MPS lacks — the same restriction
# su2_attention_correlator.py carries. The flow itself is linalg-free at nc = 2.
DTYPE = torch.complex64

torch.manual_seed(0)
np.random.seed(0)

SWEEP = functools.partial(heatbath_overrelaxation_sweep, n_or=N_OR)


# ── Cache paths ──────────────────────────────────────────────────────────────
def _tag(beta, L, n=None):
    t = f"b{beta:g}_L{L}"
    return t if n is None else f"{t}_n{n}"


def _ens_path(beta, L):
    return f"datasets/topo_ens_{_tag(beta, L, N_CONFIGS)}.pt"


def _targets_path(beta, L):
    return f"datasets/topo_targets_{_tag(beta, L, N_CONFIGS)}_eps{FLOW_EPS:g}.pt"


def _result_path(phase, beta, L):
    return f"results/topology/{phase}_{_tag(beta, L)}.pt"


# ── Chunked primitives ───────────────────────────────────────────────────────
def clover_density(U, chunk=None):
    """q_clov(x) for a batch of configurations, chunked over the batch."""
    out = []
    for i in range(0, U.shape[0], chunk or CHUNK):
        block = U[i : i + (chunk or CHUNK)].to(DEVICE)
        out.append(topological_charge_density(block, GROUP).cpu())
    return torch.cat(out)


def flowed_densities(U, times, chunk=None, progress=False, definitions=("clover",)):
    """Flowed charge density at each requested time, under each discretisation.

    One trajectory per configuration, snapshots taken along it: the cost is the
    longest rung, not the sum of the rungs — and not the sum over definitions
    either, which is why they are asked for together rather than by re-flowing.
    Returns ``{definition: {t: (B, *Λ)}}``.
    """
    out = {d: {t: [] for t in times} for d in definitions}
    step = chunk or CHUNK
    starts = range(0, U.shape[0], step)
    if progress:
        starts = tqdm(starts, desc=f"flow → {max(times):g}", unit="chunk")
    for i in starts:
        block = U[i : i + step].to(DEVICE)
        # progress=False on the inner call: one bar per chunk per ladder segment
        # turns a log into a wall of carriage returns.
        snaps = flow_trajectory(block, GROUP, times, eps=FLOW_EPS)
        for t, V in zip(times, snaps):
            for d in definitions:
                out[d][t].append(
                    topological_charge_density(V, GROUP, definition=d).cpu()
                )
    return {d: {t: torch.cat(v) for t, v in ts.items()} for d, ts in out.items()}


def mean_plaquette(U, chunk=None):
    out = []
    for i in range(0, U.shape[0], chunk or CHUNK):
        P = plaquette_tensor(U[i : i + (chunk or CHUNK)].to(DEVICE), GROUP)
        out.append((_re_tr(P) / GROUP.nc).flatten(1).mean(1).cpu())
    return torch.cat(out)


def _load_or_make_ensemble(beta, L, quiet=False):
    path = _ens_path(beta, L)
    if os.path.exists(path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if not quiet:
            print(f"  [cache] {path}  ({blob['U'].shape[0]} configs)")
        return blob["U"]
    gb = N_CONFIGS * D * (L**D) * GROUP.nc**2 * 8 / 1024**3
    print(
        f"  sampling {N_CONFIGS} configs  SU(2) L={L} D={D} β={beta} "
        f"(n_therm={N_THERM}, n_skip={N_SKIP}, HB+{N_OR}×OR) → {gb:.1f} GiB"
    )
    t0 = time.time()
    U, acc = mcmc_ensemble(
        L=L,
        D=D,
        gaugegroup=GROUP,
        beta=beta,
        n_configs=N_CONFIGS,
        n_therm=N_THERM,
        n_skip=N_SKIP,
        sweep_fn=SWEEP,
        dtype=DTYPE,
        device=DEVICE,
        progress=True,
    )
    U = U.cpu()
    torch.save(
        {
            "U": U,
            "meta": dict(
                beta=beta, L=L, D=D, n_configs=N_CONFIGS, n_therm=N_THERM,
                n_skip=N_SKIP, n_or=N_OR, acceptance=acc,
            ),
        },
        path,
    )
    print(f"  wrote {path}  ({time.time() - t0:.0f} s)")
    return U


# ── Phase: chain (τ_int) ─────────────────────────────────────────────────────
def phase_chain(beta, L):
    """τ_int of Q (flowed to the primary rung) and of the plaquette.

    Sampled at n_skip = 1 so the series is the Markov chain itself. The
    production n_skip should be ≳ 2·τ_int(Q); τ_int(plaquette) is reported
    alongside only to show how much slower the topology is.
    """
    path = _result_path("chain", beta, L)
    if os.path.exists(path):
        print(f"  [cache] {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    print(f"  chain of {N_CHAIN} sweeps at n_skip=1 …")
    chain, _ = mcmc_ensemble(
        L=L, D=D, gaugegroup=GROUP, beta=beta, n_configs=N_CHAIN,
        n_therm=N_THERM, n_skip=1, sweep_fn=SWEEP, dtype=DTYPE,
        device=DEVICE, progress=True,
    )
    chain = chain.cpu()
    plaq = mean_plaquette(chain).numpy()
    q = flowed_densities(chain, (T_PRIMARY,))["clover"][T_PRIMARY]
    Q = q.flatten(1).sum(1).numpy()

    out = {"beta": beta, "L": L, "Q": Q, "plaquette": plaq}
    for name, series in (("Q", Q), ("plaquette", plaq)):
        rho, tau, window = integrated_autocorrelation_time(series)
        out[f"tau_{name}"] = tau
        out[f"rho_{name}"] = rho
        out[f"window_{name}"] = window
        print(f"    τ_int({name:10s}) = {tau:6.2f}  (window {window})")
    print(
        f"    ⇒ n_skip ≳ {math.ceil(2 * out['tau_Q']):d} sweeps from Q; "
        f"{math.ceil(2 * out['tau_plaquette']):d} from the plaquette. "
        f"Production N_SKIP is {N_SKIP}."
    )
    if 2 * out["tau_Q"] > N_SKIP:
        print(
            "    !! N_SKIP is below 2·τ_int(Q): the ensemble is NOT decorrelated "
            "in the topological sector. Raise TOPO_N_SKIP before sampling."
        )
    # Sector history, the thing that shows freezing at a glance.
    print(
        f"    Q range [{Q.min():+.2f}, {Q.max():+.2f}], "
        f"{np.abs(np.diff(np.round(Q))).astype(bool).sum()} integer-sector "
        f"changes over {len(Q)} sweeps"
    )
    torch.save(out, path)
    return out


# ── Phase: gate0 (the Q(t) plateau) ──────────────────────────────────────────
def phase_gate0(beta, L):
    """Flow to t/a² = T_GATE_MAX and check that the primary rung is on a plateau."""
    path = _result_path("gate0", beta, L)
    if os.path.exists(path):
        print(f"  [cache] {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    print(f"  {N_GATE} configs, flowing to t/a²={T_GATE_MAX:g} …")
    U, _ = mcmc_ensemble(
        L=L, D=D, gaugegroup=GROUP, beta=beta, n_configs=N_GATE, n_therm=N_THERM,
        n_skip=N_SKIP, sweep_fn=SWEEP, dtype=DTYPE, device=DEVICE, progress=True,
    )
    U = U.cpu()

    # A grid dense enough to see the plateau's edges, with every ladder rung on it.
    grid = sorted(
        set(list(T_LADDER)) | {0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0,
                               12.0, T_GATE_MAX}
    )
    grid = [t for t in grid if t <= T_GATE_MAX]
    # Both discretisations off the one trajectory. They converge to the same
    # continuum charge as the field smooths, so the gap between them at large t
    # is a discretisation-error scale — which is what tells an honest Z < 1 apart
    # from a bug in the normalisation.
    dens = flowed_densities(
        U, tuple(grid), progress=True, definitions=("clover", "plaquette")
    )
    Qt = np.stack([dens["clover"][t].flatten(1).sum(1).numpy() for t in grid])
    Qt_plaq = np.stack([dens["plaquette"][t].flatten(1).sum(1).numpy() for t in grid])

    print(f"    {'t/a²':>6} {'r_sm/a':>7} {'⟨Q⟩':>9} {'rms(Q)':>8} {'⟨|Q−[Q]|⟩':>10}")
    for i, t in enumerate(grid):
        dev = np.abs(Qt[i] - np.round(Qt[i])).mean()
        print(
            f"    {t:6g} {math.sqrt(8 * t):7.2f} {Qt[i].mean():+9.4f} "
            f"{Qt[i].std():8.4f} {dev:10.4f}"
        )

    # The gate itself: at the primary rung, how far is Q from an integer, and how
    # much does it move between the neighbouring rungs?
    ip = grid.index(T_PRIMARY)
    dev_primary = float(np.abs(Qt[ip] - np.round(Qt[ip])).mean())
    lo, hi = max(ip - 1, 0), min(ip + 1, len(grid) - 1)
    drift = float(np.abs(Qt[hi] - Qt[lo]).mean())
    print(
        f"    gate: at t/a²={T_PRIMARY:g}  ⟨|Q−[Q]|⟩ = {dev_primary:.3f}, "
        f"|Q({grid[hi]:g})−Q({grid[lo]:g})| = {drift:.3f} per config"
    )
    # A lattice with no topology at all passes "Q is near an integer" trivially
    # (every Q is near zero), so the spread across configurations is part of the
    # gate: without it a too-small volume reads as a clean pass.
    spread = float(Qt[ip].std())
    # ⟨|Q−[Q]|⟩ = 0.25 for ANY broad distribution — uniform, Gaussian, anything
    # wide compared to the integer spacing. It is the null, not a threshold, and
    # a first version of this gate used it as one (calling 0.239 a PASS). The
    # criterion is distance from the null, and it is read after the Z-scan of the
    # `gatefit` phase, because a multiplicative renormalisation Z < 1 moves Q off
    # the integers without meaning the topology is unresolved.
    if spread < 0.1:
        verdict, why = "DEGENERATE", "  — no topological content; the volume is too small"
    elif dev_primary < 0.15 and drift < 0.5:
        verdict, why = "PASS", ""
    else:
        verdict, why = "LOOK", "  — run TOPO_PHASE=gatefit before moving the rung"
    print(f"    rms(Q) across configs at the primary rung: {spread:.3f}")
    print(f"    ⟨|Q−[Q]|⟩ null for a structureless distribution: 0.250")
    print(f"    gate verdict: {verdict}{why}")
    # The smoothing radius must fit inside the box: at r_sm ≳ L/2 the flow has
    # wrapped the torus and the rung means nothing. This is not a property of β,
    # so it is checked here rather than per-configuration.
    for t in T_LADDER:
        if math.sqrt(8 * t) > L / 2:
            print(
                f"    !! ladder rung t/a²={t:g} has r_sm/a = {math.sqrt(8 * t):.1f} "
                f"> L/2 = {L / 2:g}: the flow smooths over the whole lattice"
            )

    # Integrator check: the production step size against a 4× finer one.
    sub = U[: min(4, U.shape[0])].to(DEVICE)
    coarse = topological_charge_density(
        wilson_flow(sub, GROUP, T_PRIMARY, eps=FLOW_EPS), GROUP
    ).flatten(1).sum(1)
    fine = topological_charge_density(
        wilson_flow(sub, GROUP, T_PRIMARY, eps=FLOW_EPS / 4), GROUP
    ).flatten(1).sum(1)
    dq = (coarse - fine).abs().max().item()
    print(
        f"    step size: |Q(ε={FLOW_EPS:g}) − Q(ε={FLOW_EPS / 4:g})| ≤ {dq:.2e} "
        f"over {sub.shape[0]} configs"
    )

    out = {
        "beta": beta, "L": L, "grid": np.array(grid), "Qt": Qt,
        "Qt_plaq": Qt_plaq,
        "dev_primary": dev_primary, "drift": drift, "verdict": verdict,
        "spread": spread,
        "eps_residual": dq,
    }
    torch.save(out, path)
    _plot_gate0(out)
    return out


def _plot_gate0(res):
    grid, Qt = res["grid"], res["Qt"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for b in range(Qt.shape[1]):
        ax.plot(grid, Qt[:, b], lw=0.9, alpha=0.8)
    ax.axvline(T_PRIMARY, color="k", ls="--", lw=1, label=f"primary t/a²={T_PRIMARY:g}")
    for n in range(-4, 5):
        ax.axhline(n, color="0.85", lw=0.5, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("flow time $t/a^2$")
    ax.set_ylabel("$Q(t)$")
    ax.set_title(f"Gate 0: SU(2) β={res['beta']:g} L={res['L']} — {res['verdict']}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = f"results/topology/gate0_{_tag(res['beta'], res['L'])}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    wrote {path}")


def phase_gatefit(beta, L):
    """Offline re-analysis of a cached `gate0` run — seconds, no GPU, no flowing.

    `gate0` answers "is Q near an integer?" with ⟨|Q−[Q]|⟩, and that statistic
    has a null of **0.25** for any distribution broad compared to the integer
    spacing (uniform, Gaussian, anything). A reading of 0.24 is therefore not a
    near-miss, it is the absence of integer structure. What it cannot separate is
    the two reasons Q might sit off the integers:

      · the flow has not resolved the topology — dislocations survive, the rung
        is too early, and the target field is UV noise;
      · the lattice charge carries a multiplicative renormalisation,
        Q_latt ≃ Z(β,t)·Q with Z < 1 on a rough field, so an honest integer
        structure sits at spacing Z instead of 1.

    So this phase scans Z and reports where the integer structure is sharpest.
    A Z* near 1 with a residual still at the null means the first reading; a Z*
    below 1 with a residual well under the null means the second, and the rung is
    usable. A Z* that lands near 1/2 or 2 would mean neither — it would mean the
    normalisation of q_x is wrong, which is why the scan is deliberately wide.

    Also printed: the smoothing radius against the box (a rung with
    r_sm ≳ L/2 has wrapped the torus), the clover-vs-naive gap (a discretisation
    scale, and a bug's fingerprint if it does not shrink as the field smooths),
    and an SEM on every deviation, because 16 configurations is thin.
    """
    path = _result_path("gate0", beta, L)
    if not os.path.exists(path):
        print(f"  no gate0 dump at {path}; run TOPO_PHASE=gate0 first")
        return
    res = torch.load(path, map_location="cpu", weights_only=False)
    grid, Qt = res["grid"], res["Qt"]
    Qt_plaq = res.get("Qt_plaq")
    n = Qt.shape[1]
    zs = np.linspace(0.4, 2.5, 421)

    print(
        f"  {n} configs; the null for ⟨|Q−[Q]|⟩ is 0.250 (ANY broad "
        f"distribution), not a threshold"
    )
    head = (
        f"  {'t/a²':>6} {'r_sm/a':>7} {'r_sm/L':>7} {'rms Q':>7} "
        f"{'dev(Z=1)':>14} {'Z*':>6} {'dev(Z*)':>14} {'rms Q/Z*':>9}"
    )
    print(head + ("" if Qt_plaq is None else f" {'|clov−plaq|':>11}"))
    rows = {}
    for i, t in enumerate(grid):
        Q = Qt[i]
        dev1 = np.abs(Q - np.round(Q))
        # dev(Z) = ⟨|Q/Z − [Q/Z]|⟩. Large Z drives every Q toward 0, which is an
        # integer, so rms(Q/Z*) is printed beside it: a collapsed spread is how
        # that degenerate minimum announces itself.
        devs = np.array([np.abs(Q / z - np.round(Q / z)).mean() for z in zs])
        j = int(devs.argmin())
        zstar = float(zs[j])
        dstar = np.abs(Q / zstar - np.round(Q / zstar))
        line = (
            f"  {t:6g} {math.sqrt(8 * t):7.2f} {math.sqrt(8 * t) / L:7.2f} "
            f"{Q.std():7.3f} {dev1.mean():7.3f}±{dev1.std() / math.sqrt(n):<6.3f} "
            f"{zstar:6.2f} {dstar.mean():7.3f}±{dstar.std() / math.sqrt(n):<6.3f} "
            f"{(Q / zstar).std():9.3f}"
        )
        if Qt_plaq is not None:
            line += f" {np.abs(Q - Qt_plaq[i]).mean():11.3f}"
        if not Z_BAND[0] <= zstar <= Z_BAND[1]:
            line += "  * Z* out of band: aliasing, not a renormalisation"
        print(line)
        rows[float(t)] = dict(
            rms=float(Q.std()), dev1=float(dev1.mean()),
            sem1=float(dev1.std() / math.sqrt(n)), zstar=zstar,
            devstar=float(dstar.mean()),
            semstar=float(dstar.std() / math.sqrt(n)),
            rms_scaled=float((Q / zstar).std()),
        )

    # Z is a renormalisation factor: it lies in (0, 1] and rises toward 1 as the
    # field smooths. A minimiser that reports Z* far below that is not measuring
    # a renormalisation, it has found an aliasing solution — dividing by a small
    # Z inflates the spread until *some* alignment with the integers appears, and
    # rms(Q/Z*) gives it away (the pre-flight produced two such entries, β=2.4 at
    # t=1 with Z*=0.44 and β=2.5 at t=8 with Z*=0.46, both flagged usable by an
    # earlier version of this filter). So the band is part of the criterion, and
    # the wide scan is kept only as the diagnostic it was meant to be.
    usable = [
        t for t, r in rows.items()
        if r["devstar"] + r["semstar"] < 0.20
        and r["rms_scaled"] > 0.3
        and Z_BAND[0] <= r["zstar"] <= Z_BAND[1]
        and math.sqrt(8 * t) <= L / 2
    ]
    # Z at the smoothest rungs, where it should have converged: a measured
    # quantity, and a check on the charge's normalisation (a Z* near 1/2 or 2
    # would indict the definition of q_x, not the physics).
    z_conv = np.mean([rows[float(t)]["zstar"] for t in grid[-3:]])
    print(f"  Z at the three smoothest rungs: {z_conv:.2f}")
    print(
        f"\n  rungs with real integer structure and r_sm ≤ L/2: "
        f"{sorted(usable) if usable else 'NONE'}"
    )
    if not usable:
        print(
            "  ⇒ Gate 0's integrality half FAILS at this row: no flow time that "
            "fits inside the box shows integer structure beyond the null.\n"
            "    The density target q_t(x) is still well defined — it is the "
            "flowed reference, whatever its normalisation — but reading R4 as "
            "'reproduces the integer Q-histogram' is not available here, and\n"
            "    that has to be recorded rather than worked around."
        )
    torch.save({"beta": beta, "L": L, "rows": rows, "usable": sorted(usable)},
               _result_path("gatefit", beta, L))
    return rows


# ── Phase: ensemble / targets ────────────────────────────────────────────────
def phase_ensemble(beta, L):
    _load_or_make_ensemble(beta, L)


def phase_targets(beta, L):
    path = _targets_path(beta, L)
    if os.path.exists(path):
        print(f"  [cache] {path}")
        return
    U = _load_or_make_ensemble(beta, L)
    print(f"  clover density at t=0 and the ladder {list(T_LADDER)} …")
    t0 = time.time()
    q0 = clover_density(U)
    ladder = flowed_densities(U, T_LADDER, progress=True)["clover"]
    torch.save(
        {
            "q0": q0,
            "ladder": {float(t): v for t, v in ladder.items()},
            "meta": dict(
                beta=beta, L=L, D=D, n_configs=U.shape[0], ladder=list(T_LADDER),
                eps=FLOW_EPS, integrator="rk3", definition="clover",
            ),
        },
        path,
    )
    print(f"  wrote {path}  ({time.time() - t0:.0f} s)")


# ── Phase: arms (identity + best linear filter) ──────────────────────────────
def _l1_ball_sites(L, R):
    """Distinct lattice sites within Manhattan radius R, as index tuples.

    Offsets are reduced modulo L and de-duplicated: on an L = 8 lattice a
    Manhattan-8 ball wraps the torus several times over, and feeding the same
    site to the filter under two names makes the normal equations singular.
    Ordered by |Δ|₁ so the representative kept for a wrapped site is the
    shortest one.
    """
    rng = range(-R, R + 1)
    seen, offsets = set(), []
    for dx in sorted(
        (
            (a, b, c, d)
            for a in rng for b in rng for c in rng for d in rng
            if abs(a) + abs(b) + abs(c) + abs(d) <= R
        ),
        key=lambda t: sum(abs(v) for v in t),
    ):
        key = tuple(v % L for v in dx)
        if key not in seen:
            seen.add(key)
            offsets.append(key)
    return offsets


def _hypercubic_orbits(offsets, L):
    """Group offsets into orbits of the hypercubic group (permutations × signs).

    The isotropic filter has one weight per orbit. Signs are applied to the
    *signed* representative before wrapping, so the orbit of a wrapped site is
    computed on the torus, as it must be.
    """
    signed = [tuple((v + L // 2) % L - L // 2 for v in o) for o in offsets]
    index = {o: i for i, o in enumerate(offsets)}
    orbit_of = [-1] * len(offsets)
    n_orbits = 0
    for i, s in enumerate(signed):
        if orbit_of[i] >= 0:
            continue
        orbit_of[i] = n_orbits
        for perm in _PERMS4:
            for sign in _SIGNS4:
                img = tuple((sign[k] * s[perm[k]]) % L for k in range(4))
                j = index.get(img)
                if j is not None and orbit_of[j] < 0:
                    orbit_of[j] = n_orbits
        n_orbits += 1
    return np.array(orbit_of), n_orbits


_PERMS4 = list(itertools.permutations(range(4)))
_SIGNS4 = list(itertools.product((1, -1), repeat=4))


def _corr_maps(q, y):
    """Lattice correlations, exactly (periodic ⇒ FFT is not an approximation).

    Returns ``A(δ) = Σ_{b,x} q_b(x) q_b(x+δ)`` and ``C(δ) = Σ_{b,x} q_b(x+δ) y_b(x)``
    as full lattice fields. These are all the normal equations need: for the
    filter ``ŷ(x) = Σ_Δ w_Δ q(x+Δ)`` the Gram matrix is ``G_ij = A(Δ_j − Δ_i)``
    and the right-hand side is ``h_i = C(Δ_i)``, because translation invariance
    on a periodic lattice is exact.
    """
    axes = tuple(range(1, q.dim()))
    Fq = torch.fft.fftn(q.double(), dim=axes)
    Fy = torch.fft.fftn(y.double(), dim=axes)
    A = torch.fft.ifftn((Fq * Fq.conj()).sum(0), dim=tuple(range(q.dim() - 1))).real
    C = torch.fft.ifftn((Fq * Fy.conj()).sum(0), dim=tuple(range(q.dim() - 1))).real
    return A, C


def _fit_linear_filter(q, y, offsets, orbit_of=None, ridge=1e-8):
    """Least-squares-optimal filter on ``offsets`` (optionally orbit-averaged)."""
    A, C = _corr_maps(q, y)
    idx = torch.tensor(offsets, dtype=torch.long)  # (n_off, 4)
    n_off, L = idx.shape[0], A.shape[0]
    # G_ij = A(Δ_j − Δ_i), h_i = C(Δ_i), both read off the correlation fields.
    # Assembled in row blocks: a Manhattan-8 ball in 4D holds ~3.6k offsets, and
    # materialising the (n, n, 4) index difference would cost 0.4 GiB for what is
    # a 0.1 GiB matrix.
    G = torch.empty(n_off, n_off, dtype=A.dtype)
    flat = A.reshape(-1)
    for i in range(0, n_off, 256):
        d = (idx[None, i : i + 256, :] - idx[:, None, :]) % L  # (n, blk, 4)
        G[:, i : i + 256] = flat[
            ((d[..., 0] * L + d[..., 1]) * L + d[..., 2]) * L + d[..., 3]
        ]
    h = C[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]]
    if orbit_of is not None:
        M = torch.zeros(len(offsets), int(orbit_of.max()) + 1, dtype=G.dtype)
        M[torch.arange(len(offsets)), torch.as_tensor(orbit_of)] = 1.0
        G, h, basis = M.T @ G @ M, M.T @ h, M
    else:
        basis = None
    G = G + ridge * torch.diag(G).mean() * torch.eye(G.shape[0], dtype=G.dtype)
    theta = torch.linalg.solve(G, h)
    w = theta if basis is None else basis @ theta
    # cond() is an SVD; skip it once the system is big enough for that to cost
    # more than the fit itself. It is a diagnostic, not part of the answer.
    cond = float(torch.linalg.cond(G)) if G.shape[0] <= 1024 else float("nan")
    return w, cond


def _apply_filter(q, offsets, w):
    """``ŷ(x) = Σ_Δ w_Δ q(x+Δ)`` for a whole batch, via one FFT pair.

    A Manhattan-8 ball in 4D holds thousands of offsets, so the direct route
    (one roll per offset) is thousands of passes over the field; as a
    cross-correlation it is two transforms.
    """
    axes = tuple(range(1, q.dim()))
    kernel = torch.zeros(q.shape[1:], dtype=torch.float64)
    idx = torch.tensor(offsets, dtype=torch.long)
    kernel[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] = w.double()
    Fk = torch.fft.fftn(kernel, dim=tuple(range(kernel.dim())))
    return torch.fft.ifftn(
        torch.fft.fftn(q.double(), dim=axes) * Fk.conj(), dim=axes
    ).real


def _r2(pred, y):
    """Per-site R², the study's M1."""
    ss_res = ((pred - y) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def _charge_rms(pred, y):
    """M2: RMS error on the total charge Q reconstructed from the density."""
    axes = tuple(range(1, y.dim()))
    return float(((pred.sum(axes) - y.sum(axes)) ** 2).mean().sqrt())


def _naive_apply(q, offsets, w):
    """``ŷ(x) = Σ_Δ w_Δ q(x+Δ)`` by explicit rolls — the definition the FFT path
    implements, kept as the oracle rather than as the production route."""
    out = torch.zeros_like(q)
    for wd, dx in zip(w.tolist(), offsets):
        shifted = q
        for ax, d in enumerate(dx):
            if d:
                shifted = torch.roll(shifted, shifts=-d, dims=ax + 1)
        out = out + wd * shifted
    return out


def phase_selftest(beta=None, L=None):
    """Gate the linear arm: recover a known filter exactly, from synthetic data.

    The best-linear-filter arm decides whether the whole study is worth running,
    and its two halves — normal equations assembled from FFT correlations, and
    the filter applied as an FFT cross-correlation — are exactly the kind of
    convention that can be wrong by a sign, a conjugate or a shift while still
    producing plausible numbers. So both are pinned against the naive
    roll-by-roll definition on data where the answer is known: y is *generated*
    by a random filter, so the fit must recover that filter and reach R² = 1.
    """
    torch.manual_seed(0)
    Ls, B, R = 6, 8, 2
    q = torch.randn(B, Ls, Ls, Ls, Ls, dtype=torch.float64)
    offsets = _l1_ball_sites(Ls, R)
    w_true = torch.randn(len(offsets), dtype=torch.float64)
    y = _naive_apply(q, offsets, w_true)

    fft_pred = _apply_filter(q, offsets, w_true)
    err_apply = (fft_pred - y).abs().max().item()
    w_fit, cond = _fit_linear_filter(q, y, offsets, ridge=0.0)
    err_w = (w_fit - w_true).abs().max().item()
    r2 = _r2(_apply_filter(q, offsets, w_fit), y)

    print(f"  apply: max|FFT − naive rolls|        = {err_apply:.3e}")
    print(f"  fit:   max|ŵ − w_true|               = {err_w:.3e}  (cond {cond:.1e})")
    print(f"  fit:   R² on the generating filter   = {r2:.12f}")
    ok = err_apply < 1e-10 and err_w < 1e-8 and abs(r2 - 1) < 1e-12
    print(f"  selftest: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("linear-filter self-test failed; the arm is not trustworthy")

    # The orbit-averaged variant must be the *same* answer when the truth is
    # already isotropic, and never better than the free one on the training data.
    orbit_of, _ = _hypercubic_orbits(offsets, Ls)
    w_iso_true = torch.tensor(
        [w_true[orbit_of == orbit_of[i]].mean() for i in range(len(offsets))]
    )
    y_iso = _naive_apply(q, offsets, w_iso_true)
    w_iso, _ = _fit_linear_filter(q, y_iso, offsets, orbit_of=orbit_of, ridge=0.0)
    err_iso = (w_iso - w_iso_true).abs().max().item()
    print(f"  isotropic fit: max|ŵ − w_true|       = {err_iso:.3e}")
    if err_iso > 1e-8:
        raise SystemExit("isotropic linear-filter self-test failed")


def phase_arms(beta, L):
    print("  self-test of the linear-filter estimator:")
    phase_selftest()
    path = _targets_path(beta, L)
    if not os.path.exists(path):
        print(f"  no targets at {path}; run TOPO_PHASE=targets first")
        return
    blob = torch.load(path, map_location="cpu", weights_only=False)
    q0 = blob["q0"].double()
    y = blob["ladder"][float(T_PRIMARY)].double()
    n = q0.shape[0]
    n_tr = int(round(SPLITS[0] * n))
    n_va = int(round(SPLITS[1] * n))
    tr = slice(0, n_tr)
    te = slice(n_tr + n_va, n)
    print(
        f"  splits (contiguous, chain-ordered): {n_tr} train / {n_va} val / "
        f"{n - n_tr - n_va} test; target t/a²={T_PRIMARY:g}"
    )

    rows = {}
    # 1. Identity, raw — q̂ = q_clov[U]. The unflowed density is dominated by
    #    dislocations, so this is expected to be poor; it is the floor.
    rows["identity"] = (_r2(q0[te], y[te]), _charge_rms(q0[te], y[te]))
    # 2. Identity with the multiplicative renormalisation fitted on train
    #    (Q_latt ≃ Z(β)·Q): the honest version of the same arm.
    Z = float((q0[tr] * y[tr]).sum() / (q0[tr] ** 2).sum())
    rows[f"identity×Z (Z={Z:.3f})"] = (
        _r2(Z * q0[te], y[te]),
        _charge_rms(Z * q0[te], y[te]),
    )
    # 3/4. The best linear filter — the control that decides whether the task is
    #      a convolution. Fitted on train only, evaluated on test.
    offsets = _l1_ball_sites(L, FILTER_R)
    orbit_of, n_orbits = _hypercubic_orbits(offsets, L)
    print(f"  linear filter: {len(offsets)} distinct sites within Manhattan "
          f"{FILTER_R}, {n_orbits} hypercubic orbits")
    for name, orb in (("linear filter (isotropic)", orbit_of),
                      ("linear filter (free)", None)):
        t0 = time.time()
        w, cond = _fit_linear_filter(q0[tr], y[tr], offsets, orbit_of=orb)
        pred = _apply_filter(q0[te], offsets, w)
        rows[name] = (_r2(pred, y[te]), _charge_rms(pred, y[te]))
        print(f"    {name}: cond(G) = {cond:.2e}, fit {time.time() - t0:.1f} s")

    print(f"\n  {'arm':34s} {'R² (M1)':>10} {'RMS ΔQ (M2)':>12}")
    for name, (r2, rms) in rows.items():
        print(f"  {name:34s} {r2:10.4f} {rms:12.4f}")
    print(
        "\n  The best linear filter is the number a network must clearly beat: "
        "the gap between it and the best net is the nonlinear content of the flow."
    )
    out = _result_path("arms", beta, L)
    torch.save(
        {"beta": beta, "L": L, "t": T_PRIMARY, "rows": rows,
         "n_offsets": len(offsets), "n_orbits": n_orbits, "Z": Z},
        out,
    )
    print(f"  wrote {out}")


# ── Main ─────────────────────────────────────────────────────────────────────
_PHASES = {
    "selftest": phase_selftest,
    "chain": phase_chain,
    "gate0": phase_gate0,
    "gatefit": phase_gatefit,
    "ensemble": phase_ensemble,
    "targets": phase_targets,
    "arms": phase_arms,
}

if __name__ == "__main__":
    if len(BETAS) != len(LS):
        raise SystemExit(f"TOPO_BETAS and TOPO_LS must pair up: {BETAS} vs {LS}")
    unknown = [p for p in PHASES if p not in _PHASES]
    if unknown:
        raise SystemExit(f"unknown phase(s) {unknown}; known: {list(_PHASES)}")
    for beta, L in zip(BETAS, LS):
        if math.sqrt(8 * T_PRIMARY) > L / 2:
            print(
                f"!! β={beta:g} L={L} cannot host the primary rung: "
                f"r_sm/a = {math.sqrt(8 * T_PRIMARY):.2f} > L/2 = {L / 2:g}. The "
                f"flow wraps the torus there; this row needs a larger L or a "
                f"lower rung, and either choice has to be recorded."
            )
    print(
        f"device: {DEVICE}   phases: {PHASES}\n"
        f"rows: {[f'β={b:g} L={l}' for b, l in zip(BETAS, LS)]}   "
        f"ladder t/a² = {list(T_LADDER)} (primary {T_PRIMARY:g}), ε = {FLOW_EPS:g}"
        + ("   [SMOKE]" if SMOKE else "")
    )
    for phase in PHASES:
        for beta, L in zip(BETAS, LS):
            print(f"\n── {phase}  β={beta:g} L={L} ───────────────────────────")
            _PHASES[phase](beta, L)
    if PHASES == ["chain", "gate0"]:
        print(
            "\nPre-flight done. Read the gate verdict and τ_int(Q) above, set "
            "TOPO_N_SKIP, then:\n"
            "    TOPO_PHASE=ensemble,targets,arms python scripts/measure_topology.py"
        )
