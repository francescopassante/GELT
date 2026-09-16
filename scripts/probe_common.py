"""
=========================================================================
The M1 probe's shared definitions — ensemble, splits, arms, estimator.
=========================================================================

Design record and pre-registered readings: ``notes/m1_probe.md``.

This module exists for the same reason ``su2_attention_correlator.py`` imports
its estimator from ``z2_attention_correlator.py``: the arms of an A/B are only
an A/B for as long as everything except the block is *the same code*. Ensemble,
timeslice extraction, splits, target construction, standardisation, the R²
sufficient statistics and the jackknife all live here, so ``train_probe.py``
cannot drift from ``probe_preflight.py`` and neither can drift from
``probe_readings.py``.

Nothing here samples: the ensembles are ``train_glueball.py``'s cached ones,
addressed by the identical cache key.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gelt.blocks import GELT
from gelt.lattice import SU, build_transport_average, plaquette_tensor
from gelt.lcnn import LCNN, build_axis_transports
from gelt.probe_targets import BALL_RADIUS, TARGETS, action_density, build_targets


# ── Environment overrides ────────────────────────────────────────────────────
# Same two routes as train_glueball.py (env var or ``--name=value`` in argv),
# because env vars do not survive every container wrapper.
def _flag_for(name):
    return f"--{name.lower().replace('probe_', '').replace('_', '-')}="


def env_str(name, default):
    flag = _flag_for(name)
    for a in sys.argv[1:]:
        if a.startswith(flag):
            return a.split("=", 1)[1]
    return os.environ.get(name, default)


def env_int(name, default):
    return int(env_str(name, default))


def env_float(name, default):
    return float(env_str(name, default))


def env_flag(name, default):
    # Bare ``--null`` as well as ``--null=1``, matching train_glueball._env_flag.
    flag = _flag_for(name).rstrip("=")
    if flag in sys.argv[1:]:
        return True
    v = env_str(name, None)
    return default if v is None else v not in ("0", "false", "False", "")


# ── Ensemble — train_glueball.py's, by the same cache key ────────────────────
gaugegroup = SU(2)
NC = gaugegroup.nc
L = 12
D3 = 3  # the probe is per-timeslice 3D, exactly as the glueball operator is
BETA = 2.4
XI = 3.0
LT = 24
N_CONFIGS_CACHE = 2000  # the cached ensemble's size — part of the key
MODEL_DTYPE = torch.complex64

ENSEMBLE_SEED = env_int("PROBE_ENSEMBLE_SEED", 0)

# How much of that ensemble the probe uses. The supervision is per *site*, so
# 200 configs × 6 timeslices × 12³ sites is 1.5 M training examples for a
# ~15 k-parameter model — the bottleneck is wall-clock per epoch, not data.
# Timeslices are strided over Lt rather than taken contiguously: neighbouring
# timeslices of one configuration are the most correlated samples available,
# and the jackknife blocks over configurations anyway.
N_CONFIGS = env_int("PROBE_N_CONFIGS", 200)
N_SLICES = env_int("PROBE_N_SLICES", 6)

TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.1  # (TEST_FRACTION = 1 − TRAIN − VAL = 0.2)
JACK_BLOCK = env_int("PROBE_JACK_BLOCK", 5)  # blocked jackknife, in configs

R = 2  # GELT's per-layer L1-ball radius
LAYERS = 4  # both architectures
LCNN_K = 2  # the L-CNN's axis-aligned hop length
MLP_HIDDEN = 32


def cache_path(seed=None):
    """``train_glueball.py``'s ensemble cache path — the identical key."""
    seed = ENSEMBLE_SEED if seed is None else seed
    return (
        f"datasets/glueball_configs_L{L}_Lt{LT}_b{BETA}_xi{XI}_N{N_CONFIGS_CACHE}"
        + ("" if seed == 0 else f"_seed{seed}")
        + ".pt"
    )


def load_timeslices(seed=None, n_configs=None, n_slices=None, verbose=True):
    """Spatial 3D links, as ``(n_configs, n_slices, 3, L, L, L, nc, nc)``.

    Time is lattice axis 0 of the 4D configuration, so directions 1..3 at a
    fixed time are one 3D slice's spatial links — the same extraction
    ``train_glueball.config_inputs`` performs, with the configuration axis kept
    separate so the splits and the jackknife can block on it.

    Raises if the cache is absent: this probe deliberately samples nothing.
    """
    seed = ENSEMBLE_SEED if seed is None else seed
    n_configs = N_CONFIGS if n_configs is None else n_configs
    n_slices = N_SLICES if n_slices is None else n_slices
    path = cache_path(seed)
    if not os.path.exists(path):
        raise SystemExit(
            f"Ensemble cache {path} not found. The M1 probe reuses the glueball "
            f"ensembles and samples nothing itself — run "
            f"scripts/train_glueball.py (or measure_glueball.py) first, or point "
            f"PROBE_ENSEMBLE_SEED at a seed whose cache exists."
        )
    if verbose:
        print(f"Loading cached ensemble {path} …")
    configs = torch.load(path, map_location="cpu")[:n_configs].to(MODEL_DTYPE)
    # Strided timeslices: evenly spaced over the temporal extent.
    t_idx = torch.linspace(0, LT - 1, n_slices).round().long()
    U3 = configs[:, 1:, t_idx]  # (n, 3, n_slices, L, L, L, nc, nc)
    U3 = U3.movedim(2, 1).contiguous()  # (n, n_slices, 3, L,L,L, nc,nc)
    if verbose:
        print(f"  timeslices: {tuple(U3.shape)}  t = {t_idx.tolist()}")
    return U3


def build_all_targets(U3, names=TARGETS, verbose=True):
    """``{name: (n_configs, n_slices, *Λ)}`` plus ``f`` itself, on the CPU.

    Cheap enough to precompute for the whole probe set (one ball traversal per
    slice, no links involved beyond the plaquettes) and small enough to hold:
    one float32 scalar per site.
    """
    n, s = U3.shape[0], U3.shape[1]
    flat = U3.reshape(n * s, 3, L, L, L, NC, NC)
    # float64 from here down. The targets are *deterministic* functions of the
    # links — there is no irreducible noise and R² = 1 is attainable in
    # principle — so the only way a small dynamic range could hurt is
    # numerically, and 129 accumulated shells in float32 is exactly where that
    # would happen. Standardisation then puts every target on the same scale
    # for the fit.
    f = action_density(flat, gaugegroup).double()
    tgt = {k: v.reshape(n, s, L, L, L) for k, v in build_targets(f, BALL_RADIUS, names).items()}
    if verbose:
        for k, v in tgt.items():
            print(
                f"  {k}: mean {v.mean():+.5f}  std {v.std():.5f}  "
                f"range [{v.min():+.4f}, {v.max():+.4f}]"
            )
    return f.reshape(n, s, L, L, L), tgt


def splits(n_configs=None):
    """Contiguous, chain-ordered train/val/test config indices.

    Contiguous rather than shuffled because the ensemble is a Markov chain:
    interleaved splits would put nearly-identical configurations on both sides
    of the test boundary. Every timeslice of a configuration stays on the same
    side, for the same reason.
    """
    n = N_CONFIGS if n_configs is None else n_configs
    n_tr = int(round(TRAIN_FRACTION * n))
    n_va = int(round(VAL_FRACTION * n))
    idx = torch.arange(n)
    return idx[:n_tr], idx[n_tr : n_tr + n_va], idx[n_tr + n_va :]


def standardize(y, train_idx):
    """``(y − μ)/σ`` with μ, σ from the *train* configurations only.

    Returns ``(y_std, mu, sigma)``. Every target is standardised so the MSE
    and the R² are on one scale across T0/T1/T2 and the readout head never has
    to travel decades of output scale before the fit begins.
    """
    ytr = y[train_idx]
    mu = ytr.mean()
    sigma = ytr.std().clamp_min(1e-12)
    return (y - mu) / sigma, mu.item(), sigma.item()


# ── Inputs ───────────────────────────────────────────────────────────────────
def probe_inputs(U3_batch, arch, device):
    """``(W, T)`` for one batch of 3D slices, ``(b, 3, L,L,L, nc,nc)`` in.

    W is the *thin* spatial plaquette field — three channels, no smearing. The
    targets are built from exactly these plaquettes (``f`` is their normalised
    trace), so the information the task needs is present in the input and the
    probe measures how an architecture *reduces a neighbourhood*, not whether it
    can reconstruct one. Feeding the smeared ladder ``train_glueball.py`` uses
    would hand T0 — a convolution of f — most of the way to its answer.

    T is the architecture's own transport, which is the confound R-B removes and
    R-C carries: GELT's shortest-path-averaged L1-ball against the L-CNN's
    axis-aligned link products.
    """
    U = U3_batch.to(device)
    W = plaquette_tensor(U, gaugegroup)  # (b, 3, L,L,L, nc,nc)
    if arch == "lcnn":
        T = build_axis_transports(U, LCNN_K, gaugegroup)
    else:
        T = build_transport_average(U, R, gaugegroup)
    return W, T


# ── Arms ─────────────────────────────────────────────────────────────────────
# Fixed before the first run (notes/m1_probe.md §4). The widths are the ones
# that put every arm's real-DOF count within 15% of GELT's, except `frozen`,
# which is deliberately the *nested* ablation — strictly fewer parameters, so a
# tie there is the stronger statement. `frozen_matched` is the disambiguator.
ARMS = {
    "gelt": dict(arch="gelt", alpha_mode="softmax", d_model=16, d_qkv=6),
    "frozen": dict(arch="gelt", alpha_mode="frozen", d_model=16, d_qkv=6),
    "frozen_matched": dict(arch="gelt", alpha_mode="frozen", d_model=16, d_qkv=10),
    "lcnn": dict(arch="lcnn", c_hidden=6, normalize_shifts=False),
    "lcnn_norm": dict(arch="lcnn", c_hidden=6, normalize_shifts=True),
    # The two signed arms (notes/m1_probe.md §8). Identical geometry and
    # therefore identical parameter count to `gelt` — they are one nonlinearity
    # removed, so "matched-parameter" is not an argument that has to be made for
    # them, it is arithmetic.
    "signed": dict(arch="gelt", alpha_mode="signed", d_model=16, d_qkv=6),
    "signed_bounded": dict(
        arch="gelt", alpha_mode="signed_bounded", d_model=16, d_qkv=6
    ),
    "signed_l1": dict(arch="gelt", alpha_mode="signed_l1", d_model=16, d_qkv=6),
}
DOF_TOLERANCE = 0.15


def real_dofs(model):
    """Parameter count in real degrees of freedom (a complex weight is two)."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())


def build_arm(name, seed=0, grad_checkpoint=True):
    """The model for one arm. ``seed`` fixes the initialisation only.

    Every arm ends at a **zero-initialised readout**, so all of them start from
    the identical prediction ŷ ≡ 0 — which is the standardised training mean.
    An MSE gradient at ŷ ≡ 0 is nonzero (unlike the Rayleigh loss of
    ``train_glueball.py``, audit item 3), so this costs nothing and removes an
    output-scale difference between the two architectures' reference inits from
    the comparison.
    """
    if name not in ARMS:
        raise SystemExit(f"unknown arm {name!r}; expected one of {sorted(ARMS)}")
    spec = dict(ARMS[name])
    arch = spec.pop("arch")
    torch.manual_seed(seed)
    common = dict(
        gaugegroup=gaugegroup, L=L, D=D3, dtype=MODEL_DTYPE, mlp_hidden=MLP_HIDDEN,
        mlp_out=1, reduction="none", in_channels=3, grad_checkpoint=grad_checkpoint,
    )
    if arch == "lcnn":
        model = LCNN(K=LCNN_K, n_layers=LAYERS, gate="softplus", **spec, **common)
        with torch.no_grad():
            model.head_fc2.weight.zero_()
            model.head_fc2.bias.zero_()
    else:
        model = GELT(
            R=R, nhead=2, gemhsa_layers=LAYERS, mlp_zero_init=True, **spec, **common
        )
    return model, arch


def dof_table():
    """``{arm: (real_dofs, ratio to gelt)}`` — the matched-parameter claim."""
    ref = real_dofs(build_arm("gelt", grad_checkpoint=False)[0])
    return {
        k: (real_dofs(build_arm(k, grad_checkpoint=False)[0]),
            real_dofs(build_arm(k, grad_checkpoint=False)[0]) / ref)
        for k in ARMS
    }


# ── The R² estimator and its jackknife ───────────────────────────────────────
# Per configuration we keep only the six sums an R² needs. They are additive
# over configurations, so *any* subset's R² — every jackknife sample included —
# is exact from them, and a run's dump is kilobytes rather than the 33 MB its
# per-site predictions would be.
STAT_KEYS = ("n", "sy", "syy", "sp", "spp", "syp")


def accumulate_stats(y, p):
    """The six per-configuration sums for target ``y`` and prediction ``p``."""
    y = y.reshape(-1).double()
    p = p.reshape(-1).double()
    return torch.tensor(
        [y.numel(), y.sum(), (y * y).sum(), p.sum(), (p * p).sum(), (y * p).sum()],
        dtype=torch.float64,
    )


def r2_from_stats(stats):
    """``R² = 1 − SSE/SST`` from a ``(n_configs, 6)`` block of :data:`STAT_KEYS`.

    SST uses the mean of whatever subset is passed, so a jackknife sample is
    self-consistent: both the residual and the variance it is measured against
    are recomputed from the same configurations.
    """
    n, sy, syy, sp, spp, syp = stats.sum(dim=0).tolist()
    sst = syy - sy * sy / n
    sse = syy - 2 * syp + spp
    return 1.0 - sse / sst


def jackknife(stats_list, fn, block=None):
    """Blocked-jackknife ``(value, error)`` of ``fn`` over configurations.

    ``stats_list`` is one or more ``(n_configs, 6)`` blocks; ``fn`` receives the
    same number of (block-deleted) blocks and returns a scalar. Passing two
    blocks and ``fn = lambda a, b: r2(a) − r2(b)`` is the **correlated** ΔR²:
    both arms lose the same configurations in every sample, so the shared
    ensemble fluctuation cancels instead of being counted twice.
    """
    single = not isinstance(stats_list, (list, tuple))
    blocks = [stats_list] if single else list(stats_list)
    n = blocks[0].shape[0]
    block = JACK_BLOCK if block is None else block
    if n < 2 * block:
        raise ValueError(
            f"blocked jackknife needs at least two blocks: {n} configurations "
            f"at block size {block}. Raise PROBE_N_CONFIGS or lower "
            f"PROBE_JACK_BLOCK."
        )
    edges = list(range(0, n, block))
    samples = []
    for lo in edges:
        keep = torch.cat([torch.arange(0, lo), torch.arange(min(lo + block, n), n)])
        args = [b[keep] for b in blocks]
        samples.append(fn(*args) if not single else fn(args[0]))
    nb = len(samples)
    s = torch.tensor(samples, dtype=torch.float64)
    mean = s.mean().item()
    err = math.sqrt((nb - 1) / nb * ((s - s.mean()) ** 2).sum().item())
    full = fn(*blocks) if not single else fn(blocks[0])
    # Bias-corrected point estimate, error from the deleted-block spread.
    return full, err, mean
