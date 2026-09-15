"""The smearing baseline for the flow-free topology task.

`measure_topology.py`'s `arms` phase measured the best least-squares convolution
on `q_clov[U]` and got R² = 0.017 at β = 2.4, L = 12, t/a² = 4. That number is
**not** a ceiling on what a local model can do, and reading it as one would kill
the study for the wrong reason.

The linear arm is handed one scalar per site. `q_clov` is a particular quadratic
contraction of the field strength, so most of the information in the links is
already gone before the filter sees it, and no convolution of the scalar can get
it back. The flow does not work that way and neither do the networks: both smear
the *links* and contract afterwards. So the honest local baseline — the one that
answers "is `q_t(x)` determined by a bounded neighbourhood of the unflowed field
at all?" — is APE smearing of the links, followed by the same clover density.

That is this probe. It is also the classical practitioner's method, which makes
it the baseline a network has to beat for the result to mean anything; the
linear filter never was.

What it measures, per smearing level n and per ladder rung t:

  M1  per-site R² of `Z·q_clov[APE^n(U)]` against `q_t`, Z fitted on train and
      the R² evaluated on test — the same protocol and the same `_r2` as the
      `arms` phase, so the rows are directly comparable to `identity×Z` and to
      `linear filter (free)`.
  M2  RMS error on the reconstructed total charge.
  ρ_Q the correlation of the lattice-summed charges, which is where the τ=0
      signal was visible (+0.30) when the per-site correlation was not (+0.02).

Reading it:

  * some n reaches R² ~ 0.5-0.8   → the target IS reachable from a bounded
    neighbourhood. The study has a real gate: beat the smearing ladder.
  * the ladder tops out near the linear filter's 0.017 at every n → `q_t` at
    this rung is not recoverable from a local patch of the raw field by any
    local smoother, and neither network will do better for reasons that have
    nothing to do with attention. Stop, or drop the primary rung.

Smearing is **4D** here (`directions=range(D)`), i.e. cooling: there is no
transfer-matrix interpretation at stake and stripping UV fluctuation from q(x)
is the whole point. That is the case `ape_smear`'s docstring calls out as the
right one for topology, and it is not the spatial-only default the spectroscopy
code uses.

Cost: the ladder is walked **incrementally**, so the whole run costs
`max(ladder)` smearing steps, not their sum — about 10% of the flow that
produced the targets. Reads the cached ensemble and targets; makes no
configurations and runs no flow.

Run:

    python scripts/probe_smearing_baseline.py
    SMEAR_LADDER=1,2,4,8,16,32 SMEAR_ALPHA=0.6 python scripts/probe_smearing_baseline.py
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gelt.glueball import ape_smear
from gelt.lattice import SU, topological_charge_density


def _flag(name):
    return "--" + name.lower()


def _env_str(name, default):
    for a in sys.argv[1:]:
        if a.startswith(_flag(name) + "="):
            return a.split("=", 1)[1]
    return os.environ.get(name, default)


def _env_int(name, default):
    return int(_env_str(name, str(default)))


def _env_float(name, default):
    return float(_env_str(name, str(default)))


def _env_ints(name, default):
    v = _env_str(name, "")
    return default if not v else tuple(int(x) for x in v.split(","))


def _env_floats(name, default):
    v = _env_str(name, "")
    return default if not v else tuple(float(x) for x in v.split(","))


# ── Tunables ─────────────────────────────────────────────────────────────────
D = 4
GROUP = SU(2)
DTYPE = torch.complex64  # matches measure_topology.py's ensembles

# Row and cache identity — kept on the TOPO_* names so the paths line up with
# measure_topology.py's artifacts without a second convention to remember.
BETAS = _env_floats("TOPO_BETAS", (2.4,))
LS = _env_ints("TOPO_LS", (12,))
N_CONFIGS = _env_int("TOPO_N_CONFIGS", 1200)
FLOW_EPS = _env_float("TOPO_EPS", 0.02)

# The smearing ladder. Wide on purpose: the APE-step ↔ flow-time correspondence
# is only approximate, so the level that best matches a given t/a² is found by
# scanning rather than by converting.
LADDER = _env_ints("SMEAR_LADDER", (1, 2, 4, 8, 16, 24, 32, 48, 64))
ALPHA = _env_float("SMEAR_ALPHA", 0.5)
CHUNK = _env_int("SMEAR_CHUNK", 100)  # configs resident on the device at once

SPLITS = (2 / 3, 1 / 6, 1 / 6)  # identical to measure_topology.py's `arms`

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


def _tag(beta, L, n=None):
    t = f"b{beta:g}_L{L}"
    return t if n is None else f"{t}_n{n}"


def _ens_path(beta, L):
    return f"datasets/topo_ens_{_tag(beta, L, N_CONFIGS)}.pt"


def _targets_path(beta, L):
    return f"datasets/topo_targets_{_tag(beta, L, N_CONFIGS)}_eps{FLOW_EPS:g}.pt"


def _r2(pred, y):
    """Per-site R² — byte-for-byte the definition in measure_topology.py."""
    ss_res = ((pred - y) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def _charge_rms(pred, y):
    axes = tuple(range(1, y.dim()))
    return float(((pred.sum(axes) - y.sum(axes)) ** 2).mean().sqrt())


def _corr(a, b):
    if a.numel() < 2:  # a one-configuration split has no covariance
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def smeared_densities(U, ladder):
    """`q_clov[APE^n(U)]` for every n in `ladder`, one incremental walk.

    Returns ``{n: (B, *Λ) float64}`` on the CPU. The walk is segmented so the
    cost is ``max(ladder)`` smearing steps rather than their sum, and the links
    are chunked over the configuration axis so peak device memory is set by
    ``SMEAR_CHUNK`` and not by the ensemble size.
    """
    dirs = list(range(D))  # 4D cooling — see the module docstring
    out = {n: [] for n in ladder}
    n_chunks = (U.shape[0] + CHUNK - 1) // CHUNK
    for c in range(n_chunks):
        sl = slice(c * CHUNK, min((c + 1) * CHUNK, U.shape[0]))
        V = U[sl].to(DEVICE)
        done = 0
        for n in ladder:
            V = ape_smear(
                V, GROUP, alpha=ALPHA, n_steps=n - done, directions=dirs
            )
            done = n
            q = topological_charge_density(V, GROUP).double().cpu()
            out[n].append(q)
        print(f"    chunk {c + 1}/{n_chunks} done", flush=True)
    return {n: torch.cat(v, dim=0) for n, v in out.items()}


def probe(beta, L):
    ens, tgt = _ens_path(beta, L), _targets_path(beta, L)
    for p in (ens, tgt):
        if not os.path.exists(p):
            print(f"  missing {p} — run measure_topology.py first")
            return

    blob = torch.load(tgt, map_location="cpu", weights_only=False)
    ladder_t = {float(k): v.double() for k, v in blob["ladder"].items()}
    q0 = blob["q0"].double()

    U = torch.load(ens, map_location="cpu", weights_only=False)
    if isinstance(U, dict):  # the ensemble phase may store a blob
        U = U["U"] if "U" in U else next(iter(U.values()))
    U = U.to(DTYPE)

    n = q0.shape[0]
    n_tr = int(round(SPLITS[0] * n))
    n_va = int(round(SPLITS[1] * n))
    tr, te = slice(0, n_tr), slice(n_tr + n_va, n)
    print(
        f"  splits (contiguous, chain-ordered): {n_tr} train / {n_va} val / "
        f"{n - n_tr - n_va} test"
    )
    print(f"  4D APE cooling, α = {ALPHA}, ladder {list(LADDER)}, chunk {CHUNK}")

    t0 = time.time()
    qs = smeared_densities(U, LADDER)
    qs[0] = q0  # the unsmeared arm, so the table carries its own floor
    print(f"  smearing + densities: {time.time() - t0:.0f} s")

    rows = {}
    for t in sorted(ladder_t):
        y = ladder_t[t]
        print(f"\n  target t/a² = {t:g}   (rms Q = {y.sum(tuple(range(1, y.dim()))).std():.4f})")
        print(f"  {'APE n':>6} {'Z':>10} {'R² (M1)':>10} {'RMS ΔQ (M2)':>12} {'ρ_Q':>8}")
        for k in [0] + list(LADDER):
            x = qs[k]
            # Z is the least-squares rescale fitted on train alone; the flowed
            # charge is renormalised relative to any unflowed one, so comparing
            # without it would measure the normalisation and not the shape.
            Z = float((x[tr] * y[tr]).sum() / (x[tr] ** 2).sum())
            r2 = _r2(Z * x[te], y[te])
            rms = _charge_rms(Z * x[te], y[te])
            axes = tuple(range(1, y.dim()))
            rho = _corr(x[te].sum(axes), y[te].sum(axes))
            rows[(t, k)] = (Z, r2, rms, rho)
            print(f"  {k:6d} {Z:10.4f} {r2:10.4f} {rms:12.4f} {rho:8.4f}")
        best = max(LADDER, key=lambda k: rows[(t, k)][1])
        print(
            f"    best level n = {best}: R² = {rows[(t, best)][1]:.4f} "
            f"against the linear filter's 0.017 at t/a² = 4"
        )

    os.makedirs("results/topology", exist_ok=True)
    out = f"results/topology/smearbase_{_tag(beta, L, N_CONFIGS)}_a{ALPHA:g}.pt"
    torch.save(
        {"beta": beta, "L": L, "alpha": ALPHA, "ladder": list(LADDER),
         "splits": SPLITS, "rows": rows},
        out,
    )
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    if len(BETAS) != len(LS):
        raise SystemExit(f"TOPO_BETAS and TOPO_LS must pair up: {BETAS} vs {LS}")
    print(f"device: {DEVICE}")
    for beta, L in zip(BETAS, LS):
        print(f"\n── smearing baseline  β={beta:g} L={L} " + "─" * 20)
        probe(beta, L)
