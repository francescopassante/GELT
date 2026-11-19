"""Shared ground of the vacuum-wavefunctional study (gelt/vacuum.py for the physics).

Everything the four entry points must agree on lives here, so they cannot drift:
the ensemble (train_glueball's cache, by the identical key), the splits, the
classical arms, the networks, the replica dump and the ε estimator.

    vacuum_replicas.py          conditional chains on the test split → signal,
                                noise floor and the per-config replicas (V100)
    vacuum_classical.py         force-matching least squares of every classical
                                arm, ε on the replicas (V100, ~30 min)
    train_vacuum.py             a network trained by force matching, residual
                                over a classical arm (V100, hours)
    vacuum_relative_weights.py  exact ψ₀² ratios along configuration ladders —
                                ground truth no model was trained on (V100)
    vacuum_readout.py           the pre-registered readings (offline, seconds)

The estimator. For a model whose prediction of E[F_temp | U] at slice U is g(U),
with F⁰ the force on the measured configuration and F̄ the mean of conditional
replicas decorrelated from it (same slice, the rest resampled),

    e(U) = ‖g‖² − g·(F⁰ + F̄) + F⁰·F̄       (unbiased for ‖g − m‖², m = E[F|U])
    S(U) = F⁰·F̄                            (unbiased for ‖m‖²)
    ε    = Σ e / Σ S                        (0 = the exact vacuum score, 1 = Φ ≡ 0)

so ε is an *absolute* error, with the irreducible noise of the temporal force
removed — not a loss whose floor is unknown. Blocked jackknife over configs.

Every variable is VAC_NAME in the environment or --vac-name=value in argv; an
unknown --vac-* flag is refused (scripts/probe_common.py's discipline).
VAC_SMOKE=1 swaps in a 8 × 4³ ensemble sampled on the spot: plumbing, never a result.
"""

import functools
import math
import os
import sys

import torch

_FLAGS = {
    "smoke", "device", "ensemble-seed", "replica-configs", "replica-sweeps", "late",
    "n-or-cond", "fit-configs", "ridge", "bb-r", "arch", "alpha-mode", "input-levels",
    "residual", "init-seed", "lr", "epochs", "patience", "warmup", "batch",
    "train-configs", "val-configs", "tag", "family", "rw-config", "rw-target-std",
    "rw-max-rungs", "rw-smax", "rw-meas", "rw-eq", "rw-nets", "rw-mode", "rw-seed",
}


def cfg(name, default):
    """VAC_FOO from --vac-foo=value in argv, else the environment, else default."""
    flag = "--" + name.lower().replace("_", "-") + "="
    for a in sys.argv[1:]:
        if a.startswith(flag):
            return a.split("=", 1)[1]
    return os.environ.get(name, default)


def flag(name, default=False):
    v = cfg(name, None)
    return default if v is None else v not in ("0", "false", "False", "")


def validate_argv(*also):
    known = _FLAGS | set(also)
    bad = [a for a in sys.argv[1:] if a.startswith("--vac-")
           and a[len("--vac-"):].split("=", 1)[0] not in known]
    other = [a for a in sys.argv[1:] if a.startswith("--") and not a.startswith("--vac-")]
    if bad or other:
        raise SystemExit(f"unknown flag(s) {bad + other}; known: "
                         f"{sorted('--vac-' + k for k in known)}")


def levels(s):
    return tuple(int(x) for x in str(s).split(",") if x != "")


SMOKE = flag("VAC_SMOKE")
ENSEMBLE_SEED = int(cfg("VAC_ENSEMBLE_SEED", 0))
# train_glueball reads its seed at import; the cache key must be the same file.
os.environ["GLUEBALL_ENSEMBLE_SEED"] = str(ENSEMBLE_SEED)
_argv, sys.argv = sys.argv, sys.argv[:1]  # train_glueball must not see our flags
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_glueball as tg  # noqa: E402

sys.argv = _argv

from gelt.blocks import GELT  # noqa: E402
from gelt.lattice import build_transport_average, plaquette_tensor, random_links  # noqa: E402
from gelt.lcnn import LCNN, build_axis_transports  # noqa: E402
from gelt.glueball import ape_smear  # noqa: E402
from gelt.sampler import heatbath_overrelaxation_sweep  # noqa: E402
from gelt.vacuum import (  # noqa: E402
    bilinear_basis,
    lie_grad,
    loop_basis,
)

G = tg.gaugegroup
NC = G.nc
L, LT = (4, 8) if SMOKE else (tg.L, tg.LT)
BETA, XI = tg.BETA, tg.XI
BETA_T, BETA_S = BETA * XI, BETA / XI
OUT = "results/vacuum" + ("/smoke" if SMOKE else "")
os.makedirs(OUT, exist_ok=True)
ENS = f"_ens{ENSEMBLE_SEED}"

_dev = cfg("VAC_DEVICE", "")
device = torch.device(_dev or ("cuda" if torch.cuda.is_available() else "cpu"))


def require_cuda(what):
    if device.type != "cuda" and not SMOKE:
        raise SystemExit(f"{what} runs on the V100 (no CUDA here). VAC_SMOKE=1 is "
                         "the plumbing check on any device.")


# ── Ensemble and splits ───────────────────────────────────────────────────────

def load_ensemble():
    """``(N, 4, Lt, L, L, L, 2, 2)``: train_glueball's cache, or the smoke chain."""
    if SMOKE:
        path = f"{OUT}/smoke_configs.pt"
        if not os.path.exists(path):
            torch.manual_seed(1234)
            sweep = functools.partial(heatbath_overrelaxation_sweep, n_or=1, xi=XI)
            U = random_links(L, 4, G, dtype=torch.float64, Lt=LT)
            for _ in range(30):
                U, _ = sweep(U, G, BETA)
            cfgs = []
            for _ in range(60):
                for _ in range(2):
                    U, _ = sweep(U, G, BETA)
                cfgs.append(U.clone())
            torch.save(torch.stack(cfgs).to(torch.complex64), path)
        return torch.load(path)
    if not os.path.exists(tg.CACHE):
        raise SystemExit(f"no ensemble at {tg.CACHE}: this study samples nothing — "
                         "it reads train_glueball's cache (run that first).")
    return torch.load(tg.CACHE)


def splits(N):
    """train_glueball's contiguous, chain-ordered split: index ranges."""
    n_train = int(round(tg.TRAIN_FRACTION * N))
    n_val = int(round(tg.VAL_FRACTION * N))
    return (range(0, n_train), range(n_train, n_train + n_val),
            range(n_train + n_val, N))


def evenly(rng, n):
    """``n`` indices spread evenly over a range (all of it if n ≥ len)."""
    rng = list(rng)
    if n >= len(rng):
        return rng
    return [rng[int(i * len(rng) / n)] for i in range(n)]


# ── Classical arms: one union basis, every arm a subset of it ─────────────────

SHAPES = ((1, 1), (1, 2), (2, 2), (2, 3), (3, 3))
LOOP_LEVELS = (0, 2, 4, 6)
BB_R = int(cfg("VAC_BB_R", 3))


def union_ops(U3):
    """``(B, K)``: every loop operator of every level, then the B·B shells."""
    a, _ = loop_basis(U3, G, SHAPES, LOOP_LEVELS, alpha=tg.SMEAR_ALPHA)
    b, _ = bilinear_basis(U3, G, BB_R)
    return torch.cat([a, b], dim=1)


UNION_LABELS = ([f"n{lv}·{r}x{t}" for lv in LOOP_LEVELS for r, t in SHAPES]
                + [f"BB·r{r}" for r in range(BB_R + 1)])
_BB = [l for l in UNION_LABELS if l.startswith("BB")]
ARMS = {
    # Φ = c·Σ Re Tr P: ψ₀² = e^{−β_eff S_s}, the rescaled-coupling form that
    # "dimensional reduction" predicts at long wavelength.
    "dr": ["n0·1x1"],
    "shapes": [f"n0·{r}x{t}" for r, t in SHAPES],
    "shapes_sm": [l for l in UNION_LABELS if not l.startswith("BB")],
    # Greensite–Olejník-type B·K·B with a free radial kernel, |Δ|₁ ≤ BB_R.
    "bb": _BB,
    "dr_bb": ["n0·1x1"] + _BB,
    "full": list(UNION_LABELS),
}
ARM_ORDER = ["dr", "shapes", "shapes_sm", "bb", "dr_bb", "full"]


def arm_index(arm):
    return [UNION_LABELS.index(l) for l in ARMS[arm]]


def classical_path():
    return f"{OUT}/classical{ENS}.pt"


def replica_path():
    return f"{OUT}/replicas{ENS}.pt"


def classical_phi(arm, coef):
    """Φ_arm(U3) = Σ_k c_k O_k(U3) as a function, for lie_grad."""
    idx = arm_index(arm)

    def phi(U3):
        ops = union_ops(U3)[:, idx]
        return ops @ coef.to(device=ops.device, dtype=ops.dtype)
    return phi


def classical_grad(arm, coef, U3):
    """∇Φ_arm at complex128 slices, no graph: ``(B, 3, *Λ, 3)`` float64."""
    if arm in (None, "", "none"):
        return torch.zeros(U3.shape[:-2] + (NC * NC - 1,), dtype=torch.float64,
                           device=U3.device)
    g, _ = lie_grad(classical_phi(arm, coef.to(U3.device)), U3.to(torch.complex128))
    return g


# ── Networks ──────────────────────────────────────────────────────────────────

def net_config():
    return dict(
        arch=str(cfg("VAC_ARCH", "gelt")).lower(),
        alpha_mode=str(cfg("VAC_ALPHA_MODE", "softmax")).lower(),
        input_levels=levels(cfg("VAC_INPUT_LEVELS", "0")),
        residual=str(cfg("VAC_RESIDUAL", "full")).lower(),
        init_seed=int(cfg("VAC_INIT_SEED", 0)),
    )


def net_stem(c):
    a = c["arch"] + ("" if c["arch"] != "gelt" or c["alpha_mode"] == "softmax"
                     else f"_{c['alpha_mode']}")
    lv = "-".join(str(x) for x in c["input_levels"])
    return (f"{OUT}/net_{a}_sm{lv}_res{c['residual']}_init{c['init_seed']}{ENS}"
            + str(cfg("VAC_TAG", "")))


def build_net(c):
    """The glueball study's architecture (train_glueball's hyperparameters), one
    scalar per site, readout zero-initialised so the run starts at Φ_net ≡ 0."""
    torch.manual_seed(c["init_seed"])
    n_in = 3 * len(c["input_levels"])
    if c["arch"] == "gelt":
        m = GELT(gaugegroup=G, L=L, D=3, R=tg.R, nhead=tg.NHEAD,
                 gemhsa_layers=tg.GEMHSA_LAYERS, d_qkv=tg.D_QKV, gate=tg.GATE,
                 dtype=tg.MODEL_DTYPE, mlp_hidden=tg.MLP_HIDDEN, mlp_out=1,
                 reduction="none", init_scale=tg.INIT_SCALE,
                 qk_init_scale=tg.QK_INIT_SCALE, mlp_zero_init=True,
                 d_model=tg.D_MODEL, alpha_mode=c["alpha_mode"],
                 grad_checkpoint=False, in_channels=n_in)
    elif c["arch"] == "lcnn":
        m = LCNN(gaugegroup=G, L=L, D=3, K=tg.LCNN_K, c_hidden=tg.LCNN_C_HIDDEN,
                 n_layers=tg.LCNN_LAYERS, dtype=tg.MODEL_DTYPE,
                 mlp_hidden=tg.MLP_HIDDEN, mlp_out=1, reduction="none",
                 gate=tg.GATE, in_channels=n_in, init_scale=tg.LCNN_INIT_SCALE,
                 grad_checkpoint=False)
        with torch.no_grad():
            m.head_fc2.weight.zero_()
            m.head_fc2.bias.zero_()
    else:
        raise SystemExit(f"VAC_ARCH must be gelt or lcnn, got {c['arch']!r}")
    return m.to(device)


def net_inputs(c, U3):
    """Plaquettes at the input APE levels (3D smearing of the slice) and the
    transport — all from ``U3``, so the Lie gradient sees through both."""
    Ws, Us, done = [], U3, 0
    for lv in sorted(c["input_levels"]):
        if lv > done:
            Us = ape_smear(Us, G, alpha=tg.SMEAR_ALPHA, n_steps=lv - done,
                           directions=range(3))
            done = lv
        Ws.append(plaquette_tensor(Us, G))
    W = torch.cat(Ws, dim=1)
    T = (build_transport_average(U3, tg.R, G) if c["arch"] == "gelt"
         else build_axis_transports(U3, tg.LCNN_K, G))
    return W, T


def phi_net(model, c, U3):
    W, T = net_inputs(c, U3)
    return model(W, T).flatten(1).sum(1)


def net_grad(model, c, U3, create_graph=False):
    """∇Φ_net at complex64 slices: ``(B, 3, *Λ, 3)`` float32."""
    g, _ = lie_grad(lambda U: phi_net(model, c, U), U3.to(tg.MODEL_DTYPE),
                    create_graph=create_graph)
    return g


def load_net(path):
    ck = torch.load(path, map_location=device, weights_only=False)
    c = ck["config"]
    m = build_net(c)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, c


# ── The ε estimator on the replica dump ───────────────────────────────────────

def per_config_error(g, rep, i):
    """(e_i, S_i) for a prediction ``g`` (float64, slice shape) at replica config i."""
    F0, Fl = rep["F0"][i].to(torch.float64), rep["Flate"][i].to(torch.float64)
    g = g.to(torch.float64).reshape(F0.shape)
    e = (g * g).sum() - (g * (F0 + Fl)).sum() + (F0 * Fl).sum()
    return float(e), float((F0 * Fl).sum())


def jackknife_ratio(num, den, block):
    """Σnum/Σden and its blocked-jackknife error. ``num``/``den`` per config."""
    num, den = torch.as_tensor(num, dtype=torch.float64), torch.as_tensor(den, dtype=torch.float64)
    n = num.shape[0]
    nb = max(2, n // block)
    edges = [round(k * n / nb) for k in range(nb + 1)]
    full = num.sum() / den.sum()
    samp = torch.stack([(num.sum() - num[a:b].sum()) / (den.sum() - den[a:b].sum())
                        for a, b in zip(edges[:-1], edges[1:])])
    err = math.sqrt((nb - 1) / nb * float(((samp - samp.mean()) ** 2).sum()))
    return float(full), err


def jackknife_diff(num_a, num_b, den, block):
    """(Σa − Σb)/Σden, paired over the same configs."""
    a = torch.as_tensor(num_a, dtype=torch.float64)
    b = torch.as_tensor(num_b, dtype=torch.float64)
    return jackknife_ratio(a - b, den, block)


JACK_BLOCK = 2 if SMOKE else tg.JACK_BLOCK
