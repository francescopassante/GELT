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

Nothing here samples: the ensembles are the cached ones of ``train_glueball.py``
(SU(2)) or ``train_z2_glueball.py`` (Z₂), addressed by the identical cache keys.

**``PROBE_GROUP`` selects the study**, and nothing else in this module moves:

======  ==================================  ========================================
 group   task                                what changes
======  ==================================  ========================================
 su2     the M1 mechanism assay               SU(2), nc = 2, complex64; 12³ spatial
         (``notes/m1_probe.md``)              timeslices of a 4D anisotropic
                                              configuration; targets T0/T1/T2.
 z2      the vortex-geometry candidate        Z₂, nc = 1, float32; a 48 × 24 × 24
         (``where_attention_can_win.md`` §9)  configuration **whole** — it is already
                                              3D, so one configuration is one sample
                                              and there is no timeslice extraction;
                                              targets V1/V2, plus a supervision mask.
======  ==================================  ========================================

``R``, ``LAYERS``, ``LCNN_K``, ``MLP_HIDDEN``, the splits, the standardisation,
the R² sufficient statistics and the jackknife are **shared by construction** —
that is the point of the switch being here rather than in a sibling module.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gelt import probe_targets, vortex_targets
from gelt.blocks import GELT
from gelt.lattice import SU, Z2, build_transport_average, plaquette_tensor
from gelt.lcnn import LCNN, build_axis_transports
from gelt.probe_targets import BALL_RADIUS, action_density


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


# ── The study switch ─────────────────────────────────────────────────────────
GROUP = env_str("PROBE_GROUP", "su2").lower()
if GROUP not in ("su2", "z2"):
    raise SystemExit(f"PROBE_GROUP must be 'su2' or 'z2' (got {GROUP!r})")
IS_Z2 = GROUP == "z2"

# ── Ensembles — the training scripts', by the same cache keys ────────────────
if IS_Z2:
    gaugegroup = Z2()
    L = 24  # the spatial extent; the configuration is LT × L × L
    LT = 48
    LATTICE = (LT, L, L)  # non-cubic, and fed whole
    BETA = env_float("PROBE_Z2_BETA", 0.7520)  # §9.6's primary coupling
    XI = 1.0  # the Z₂ ensembles are isotropic
    N_CONFIGS_CACHE = 2000
    MODEL_DTYPE = torch.float32  # Z₂ is real; nc = 1
else:
    gaugegroup = SU(2)
    L = 12
    LT = 24
    LATTICE = (L, L, L)  # one spatial timeslice of the 4D configuration
    BETA = 2.4
    XI = 3.0
    N_CONFIGS_CACHE = 2000  # the cached ensemble's size — part of the key
    MODEL_DTYPE = torch.complex64
NC = gaugegroup.nc
D3 = 3  # both studies are 3D: SU(2) per timeslice, Z₂ natively

ENSEMBLE_SEED = env_int("PROBE_ENSEMBLE_SEED", 0)

# How much of that ensemble the probe uses. The supervision is per *site*, so
# 200 configs × 6 timeslices × 12³ sites is 1.5 M training examples for a
# ~15 k-parameter model — the bottleneck is wall-clock per epoch, not data.
# Timeslices are strided over Lt rather than taken contiguously: neighbouring
# timeslices of one configuration are the most correlated samples available,
# and the jackknife blocks over configurations anyway.
# Z₂ uses fewer configurations because each one is 27 648 sites against the
# SU(2) study's 1 728 per timeslice — 2.7× the supervision per configuration
# even after the slice axis is counted.
N_CONFIGS = env_int("PROBE_N_CONFIGS", 100 if IS_Z2 else 200)
# A Z₂ configuration is *already* 3D, so one configuration is one sample and
# there is no slice axis to stride over. Keeping the axis at length 1 rather
# than removing it is deliberate: every shape downstream — the splits, the
# per-configuration statistics, train_probe.py's batching — stays the code it
# already was, and the switch cannot introduce a second layout to maintain.
N_SLICES = 1 if IS_Z2 else env_int("PROBE_N_SLICES", 6)

TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.1  # (TEST_FRACTION = 1 − TRAIN − VAL = 0.2)
JACK_BLOCK = env_int("PROBE_JACK_BLOCK", 5)  # blocked jackknife, in configs

# The L-Conv initialisation the Z₂ arms need. **Not a handicap and not a
# taste**: at nc = 1 the L-Act gate multiplies by its own argument instead of
# damping, so a stack that starts at unit scale *per layer* reaches 1e21 after
# four of them and inf at the production volume, while GELT's field stays at 1
# through the same depth (the M2 mechanism, visible at initialisation). 0.5 is
# the largest scale on the stable plateau and reproduces the output magnitude
# the SU(2) arms get for free — the ladder is in gelt/lcnn.py's LConv.
Z2_LCNN_CONV_INIT = env_float("PROBE_Z2_LCNN_CONV_INIT", 0.5)

R = 2  # GELT's per-layer L1-ball radius
LAYERS = 4  # both architectures
LCNN_K = 2  # the L-CNN's axis-aligned hop length
MLP_HIDDEN = 32


# Which targets this study supervises on, and where they come from.
TARGET_MODULE = vortex_targets if IS_Z2 else probe_targets
TARGETS = TARGET_MODULE.TARGETS


def cache_path(seed=None):
    """The training script's ensemble cache path — the identical key.

    Z₂ has no ensemble seed: ``train_z2_glueball.py`` keys its cache on β alone,
    so the coupling plays the role the seed plays for SU(2) and a second
    ensemble means a second β rather than a second chain.
    """
    if IS_Z2:
        return f"datasets/z2_configs_L{L}_Lt{LT}_b{BETA}_N{N_CONFIGS_CACHE}.pt"
    seed = ENSEMBLE_SEED if seed is None else seed
    return (
        f"datasets/glueball_configs_L{L}_Lt{LT}_b{BETA}_xi{XI}_N{N_CONFIGS_CACHE}"
        + ("" if seed == 0 else f"_seed{seed}")
        + ".pt"
    )


def load_samples(seed=None, n_configs=None, n_slices=None, verbose=True):
    """Spatial 3D links, as ``(n_configs, n_slices, 3, *Λ, nc, nc)``.

    **SU(2)**: time is lattice axis 0 of the 4D configuration, so directions
    1..3 at a fixed time are one 3D slice's spatial links — the same extraction
    ``train_glueball.config_inputs`` performs, with the configuration axis kept
    separate so the splits and the jackknife can block on it. Timeslices are
    strided over ``LT`` rather than taken contiguously: neighbouring timeslices
    of one configuration are the most correlated samples available.

    **Z₂**: the cached configuration *is* the 3D lattice, so it is returned
    whole with a length-1 slice axis. Nothing is extracted and nothing is
    thrown away — the vortex clusters V1 measures are global objects on this
    box, and a slice of it would cut them.

    Raises if the cache is absent: this probe deliberately samples nothing.
    """
    seed = ENSEMBLE_SEED if seed is None else seed
    n_configs = N_CONFIGS if n_configs is None else n_configs
    n_slices = N_SLICES if n_slices is None else n_slices
    path = cache_path(seed)
    if not os.path.exists(path):
        raise SystemExit(
            f"Ensemble cache {path} not found. The probe reuses the training "
            f"scripts' ensembles and samples nothing itself — run "
            + (f"scripts/train_z2_glueball.py {BETA} (or "
               f"scripts/z2_vortex_preflight.py, which needs the same cache) "
               f"first, or point PROBE_Z2_BETA at a coupling whose cache exists."
               if IS_Z2 else
               f"scripts/train_glueball.py (or measure_glueball.py) first, or "
               f"point PROBE_ENSEMBLE_SEED at a seed whose cache exists.")
        )
    if verbose:
        print(f"Loading cached ensemble {path} …")
    configs = torch.load(path, map_location="cpu")[:n_configs].to(MODEL_DTYPE)
    if IS_Z2:
        U3 = configs.unsqueeze(1)  # (n, 1, 3, Lt, L, L, 1, 1)
        if verbose:
            print(f"  configurations: {tuple(U3.shape)}  (fed whole, β = {BETA})")
        return U3
    # Strided timeslices: evenly spaced over the temporal extent.
    t_idx = torch.linspace(0, LT - 1, n_slices).round().long()
    U3 = configs[:, 1:, t_idx]  # (n, 3, n_slices, L, L, L, nc, nc)
    U3 = U3.movedim(2, 1).contiguous()  # (n, n_slices, 3, L,L,L, nc,nc)
    if verbose:
        print(f"  timeslices: {tuple(U3.shape)}  t = {t_idx.tolist()}")
    return U3


def build_all_targets(U3, names=None, verbose=True):
    """``{name: (n_configs, n_slices, *Λ)}`` plus the scalar field behind them.

    Cheap enough to precompute for the whole probe set and small enough to hold:
    one float scalar per site.

    Both studies build their targets from **one gauge-invariant scalar per
    plaquette** and differ only in how a neighbourhood of it is reduced — the
    action density for SU(2) (T0/T1/T2), the vortex indicator for Z₂ (V1/V2).
    The returned first element is that field, which is what the pre-flight's
    linear filter is built from.
    """
    names = TARGETS if names is None else names
    n, s_ = U3.shape[0], U3.shape[1]
    flat = U3.reshape(n * s_, 3, *LATTICE, NC, NC)
    if IS_Z2:
        # The indicator is a bool per (plane, site); the targets reduce it.
        f = vortex_targets.vortex_field(flat, gaugegroup)
        built = vortex_targets.build_targets(f, names=names)
    else:
        # float64 from here down. The targets are *deterministic* functions of
        # the links — there is no irreducible noise and R² = 1 is attainable in
        # principle — so the only way a small dynamic range could hurt is
        # numerically, and 129 accumulated shells in float32 is exactly where
        # that would happen. Standardisation then puts every target on the same
        # scale for the fit.
        f = action_density(flat, gaugegroup).double()
        built = probe_targets.build_targets(f, BALL_RADIUS, names)
    tgt = {k: v.reshape(n, s_, *LATTICE) for k, v in built.items()}
    if verbose:
        for k, v in tgt.items():
            print(
                f"  {k}: mean {v.double().mean():+.5f}  std {v.double().std():.5f}  "
                f"range [{v.min():+.4f}, {v.max():+.4f}]"
            )
    return f.reshape(n, s_, *f.shape[1:]), tgt


def build_mask(U3, verbose=True):
    """``(n_configs, n_slices, *Λ)`` bool, or ``None`` when the study has none.

    **Z₂ supervises only on sites that carry a vortex.** The other ~90% have
    V1 exactly 0 and a linear filter predicts them from the local plaquette
    count, so an unmasked loss spends most of its gradient on *"is there a
    vortex here"* — which is not the question and is not where an architecture
    can separate (`notes/where_attention_can_win.md` §9.4, measured: the
    all-sites R² reads 0.65 where the masked one reads 0.19).

    The SU(2) study has no mask: its targets are defined at every site.
    """
    if not IS_Z2:
        return None
    n, s_ = U3.shape[0], U3.shape[1]
    flat = U3.reshape(n * s_, 3, *LATTICE, NC, NC)
    v = vortex_targets.vortex_field(flat, gaugegroup)
    mask = v.any(dim=1).reshape(n, s_, *LATTICE)
    if verbose:
        print(f"  supervision mask: {mask.double().mean():.4f} of sites carry a vortex")
    return mask


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


def standardize(y, train_idx, mask=None):
    """``(y − μ)/σ`` with μ, σ from the *train* configurations only.

    Returns ``(y_std, mu, sigma)``. Every target is standardised so the MSE
    and the R² are on one scale across the study's targets and the readout head
    never has to travel decades of output scale before the fit begins.

    ``mask`` restricts the moments to the **supervised** sites. It has to: with
    ~90% of Z₂ sites carrying V1 = 0, moments over every site would put the
    trivial predictor somewhere other than 0 on the supervised subset, and
    ``train_probe.py``'s divergence and collapse thresholds — which are absolute
    because the target is standardised — would quietly stop meaning what they
    say.
    """
    ytr = y[train_idx]
    if mask is not None:
        ytr = ytr[mask[train_idx]]
    mu = ytr.mean()
    sigma = ytr.std().clamp_min(1e-12)
    return (y - mu) / sigma, mu.item(), sigma.item()


# ── Inputs ───────────────────────────────────────────────────────────────────
def probe_inputs(U3_batch, arch, device, transport="average"):
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

    ``transport`` selects which GELT transport is built and is ignored by the
    L-CNN, whose ``build_axis_transports`` has no such knob. It is a property of
    the *inputs*, not of the model — every GELT-family arm is the same network
    whatever it is set to, so the transport arms are matched in parameters by
    arithmetic rather than by argument (``notes/m1_probe.md`` §4.4).
    """
    U = U3_batch.to(device)
    W = plaquette_tensor(U, gaugegroup)  # (b, 3, L,L,L, nc,nc)
    if arch == "lcnn":
        T = build_axis_transports(U, LCNN_K, gaugegroup)
    else:
        T = build_transport_average(U, R, gaugegroup, mode=transport)
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
    # The transport arms (notes/m1_probe.md §4.4). `transport` is not a model
    # keyword — `build_arm` pops it and `probe_inputs` builds T with it — so
    # these are the `gelt` network to the byte, fed a different transport. The
    # confound R-C carries has two halves and this varies the first one alone:
    # path averaging. The *offset set* is untouched — a "single" arm still
    # reaches every L1-ball offset, diagonals included, along one canonical
    # path — so neither arm is "GELT with the L-CNN's transport".
    "gelt_single": dict(arch="gelt", alpha_mode="softmax", d_model=16, d_qkv=6,
                        transport="single"),
    "gelt_projected": dict(arch="gelt", alpha_mode="softmax", d_model=16,
                           d_qkv=6, transport="projected"),
    # The vortex candidate's 2 × 2 (notes/where_attention_can_win.md §9.3 and
    # §9.5). In Z₂ the two transports are not "better and worse" but "present
    # and absent": a path-averaged T is a hard vortex mask, T² = (1+P)/2 ∈
    # {0,1}, while a single-path T is ±1 and its adjoint action is the
    # identity. So {softmax, frozen} × {average, single} varies M1 and that
    # mask independently, and `frozen_single` is the cell the M1 probe's arm
    # table never needed. Nested in `frozen` exactly as `frozen` is in `gelt`.
    "frozen_single": dict(arch="gelt", alpha_mode="frozen", d_model=16,
                          d_qkv=6, transport="single"),
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
    if IS_Z2 and ARMS[name].get("transport") == "projected":
        # build_transport_average refuses mode="projected" for Z₂: projecting a
        # vanishing path average back onto the group needs the arbitrary 0 → +1
        # tie-break that is CLAUDE.md caveat 1's defect in another place. Fail
        # here, at construction, rather than three hours into a batch.
        raise SystemExit(
            f"arm {name!r} is not available with PROBE_GROUP=z2: a projected "
            f"transport is ill-defined there (see build_transport_average)."
        )
    spec = dict(ARMS[name])
    arch = spec.pop("arch")
    spec.pop("transport", None)  # an input property; see `arm_transport`
    torch.manual_seed(seed)
    common = dict(
        # LATTICE rather than L: the Z₂ box is 48 × 24 × 24 and is fed whole,
        # which is what gelt.blocks.lattice_extents exists for. in_channels is 3
        # either way — D(D−1)/2 spatial plaquettes at D = 3, for both groups.
        gaugegroup=gaugegroup, L=LATTICE, D=D3, dtype=MODEL_DTYPE,
        mlp_hidden=MLP_HIDDEN, mlp_out=1, reduction="none", in_channels=3,
        grad_checkpoint=grad_checkpoint,
    )
    if arch == "lcnn":
        if IS_Z2:
            spec.setdefault("conv_init_scale", Z2_LCNN_CONV_INIT)
        model = LCNN(K=LCNN_K, n_layers=LAYERS, gate="softplus", **spec, **common)
        with torch.no_grad():
            model.head_fc2.weight.zero_()
            model.head_fc2.bias.zero_()
    else:
        model = GELT(
            R=R, nhead=2, gemhsa_layers=LAYERS, mlp_zero_init=True, **spec, **common
        )
    return model, arch


def arm_transport(name):
    """Which ``build_transport_average`` mode this arm's inputs are built with.

    Split out of the model spec because it is data, not architecture: two arms
    that differ only here share every parameter, and `train_probe.py` has to
    reach it without constructing the model.
    """
    if name not in ARMS:
        raise SystemExit(f"unknown arm {name!r}; expected one of {sorted(ARMS)}")
    return ARMS[name].get("transport", "average")


def available_arms():
    """The arms this study can actually build — see :func:`build_arm`."""
    return tuple(
        k for k in ARMS
        if not (IS_Z2 and ARMS[k].get("transport") == "projected")
    )


def dof_table():
    """``{arm: (real_dofs, ratio to gelt)}`` — the matched-parameter claim."""
    ref = real_dofs(build_arm("gelt", grad_checkpoint=False)[0])
    return {
        k: (real_dofs(build_arm(k, grad_checkpoint=False)[0]),
            real_dofs(build_arm(k, grad_checkpoint=False)[0]) / ref)
        for k in available_arms()
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
