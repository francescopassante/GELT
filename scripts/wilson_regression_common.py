"""Shared spine of the 1+1D Wilson-loop reproduction (PRL 128, 032003, Fig. 3).

Everything that must not drift between the data generator, the trainer and the
figure lives here: the beta ladder, the loop definitions, the architecture table
transcribed from the Supplemental Material, the dataset loader and the two MSE
conventions. The two attention scripts in this repo share one estimator by
import for exactly this reason; the same discipline applies to a comparison
whose headline numbers are four MSEs.

**Scope: the L-CNN half of Fig. 3, and nothing else.** The Letter's panels also
carry a baseline CNN, and this does not reproduce it --- that comparison is not
what the arm in this repo needs to stand on, and their baseline sweep is 2 680
models across 264 architectures and four activation functions. The four
``PAPER_MSE_LCNN`` values are the target; the CNN's are quoted in
``notes/wilson_regression_1p1d.md`` §1 for scale and are not computed here.

**The two MSEs.** The Letter's Fig. 3 plots one point per test configuration,
not one per lattice site: their ``LCNN.mse(global_average=True)`` averages the
per-site predictions over the lattice and compares against the lattice-averaged
label, because the baseline CNNs they plot alongside end in a global average
pool and can only produce one number per configuration. So the number to quote
against ``2.2e-11 / 2.1e-9 / 1.1e-8 / 1.4e-7`` is the **lattice-averaged** MSE.
``mse_site`` is reported alongside because it is the loss actually minimised and
because an architecture can be right on average and wrong site by site --- the
averaged MSE alone cannot tell those apart, and on an 8 x 8 lattice it is 64
times more forgiving.

**Architectures.** ``LCNN_ARCHS`` is SM Table V verbatim, in the table's own
``L-CB(k, n_in, n_out)`` notation, not paraphrased into a different
parametrisation, so a reader can check it against the PDF line by line. ``k`` is
*their* kernel-size convention (range ``[0, k-1]`` per axis, positive shifts
only); :class:`gelt.lcnn_reference.LCNNRef` takes ``K = k - 1``.
"""

import os
import sys

import torch

# ── Config plumbing (the probe_common discipline: no silently ignored flag) ──
_QUERIED = set()


def _flag_for(name):
    flag = f"--{name.lower().removeprefix('wr_').replace('_', '-')}="
    _QUERIED.add(flag)
    return flag


def cfg(name, default=None):
    """``--name=value``, else ``$NAME``, else ``default`` (type of ``default``)."""
    flag = _flag_for(name)
    raw = None
    for a in sys.argv[1:]:
        if a.startswith(flag):
            raw = a.split("=", 1)[1]
            break
    if raw is None:
        raw = os.environ.get(name)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw not in ("0", "false", "False", "")
    return raw


def validate_argv(*also):
    """Refuse an unrecognised ``--name=value``.

    Same reason as ``scripts/probe_common.validate_argv``: a misspelled flag
    that is ignored in silence makes a run report its defaults while its command
    line claims otherwise, and the command line is what ends up in the log.
    """
    known = set(_QUERIED) | {_flag_for(n) for n in also}
    unknown = [a for a in sys.argv[1:]
               if a.startswith("--") and "=" in a
               and not any(a.startswith(k) for k in known)]
    if unknown:
        raise SystemExit("unrecognised option(s): " + ", ".join(unknown)
                         + "\nknown: " + ", ".join(sorted(known)))


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ``WR_DATA_DIR`` exists so a smoke set (a short chain, a couple of couplings)
# can be written somewhere the production set is not, rather than over it.
DATA_DIR = cfg("WR_DATA_DIR", os.path.join(REPO, "datasets", "wilson1p1d"))
DUMP_DIR = cfg("WR_DUMP_DIR", os.path.join(REPO, "dumps"))
RESULT_DIR = os.path.join(REPO, "results", "wilson_regression")

# Eq. (12): W^(m x n)_{x,01}. Keys are the artifact names, values are (m, n).
LOOPS = {"W11": (1, 1), "W12": (1, 2), "W22": (2, 2), "W44": (4, 4)}

# Fig. 3's L-CNN MSEs (lattice-averaged), in the Letter's own printing. These
# are the pre-registered targets of the reproduction, not a fit.
PAPER_MSE_LCNN = {"W11": 2.2e-11, "W12": 2.1e-9, "W22": 1.1e-8, "W44": 1.4e-7}

# ── SM Table V: the L-CNN architectures in 1+1D ──────────────────────────────
# Each entry is the stack of L-CB(k, n_in, n_out) layers; Trace and the single
# per-site Linear(2 * n_out, 1) are implied and are added by the builder.
LCNN_ARCHS = {
    ("W11", "small"):  [(1, 1, 1)],
    ("W12", "small"):  [(2, 1, 2)],
    ("W12", "medium"): [(3, 1, 4)],
    ("W12", "large"):  [(4, 1, 8)],
    ("W22", "small"):  [(2, 1, 2), (2, 2, 2)],
    ("W22", "medium"): [(3, 1, 4), (3, 4, 4)],
    ("W22", "large"):  [(4, 1, 8), (4, 8, 8)],
    ("W44", "small"):  [(2, 1, 2), (2, 2, 2), (3, 2, 2), (3, 2, 2)],
    ("W44", "medium"): [(3, 1, 4), (3, 4, 4), (4, 4, 4), (4, 4, 4)],
    ("W44", "large"):  [(4, 1, 8), (4, 8, 8), (4, 8, 8), (4, 8, 8)],
}
# SM Table V's own N_param column. The builder checks against it, because a
# silently wider network is the one way this reproduction could "succeed"
# without reproducing anything.
LCNN_NPARAM = {
    ("W11", "small"): 12, ("W12", "small"): 35, ("W12", "medium"): 117,
    ("W12", "large"): 329, ("W22", "small"): 125, ("W22", "medium"): 1305,
    ("W22", "large"): 13521, ("W44", "small"): 465, ("W44", "medium"): 4833,
    ("W44", "large"): 39905,
}

# ── SM SS VI.A / Table V caption: training hyper-parameters ─────────────────
# "Models for W(1x1) and W(1x2) are trained for a maximum of 20 epochs with a
#  batch size of 50 and early stopping (patience value 5). The learning rate was
#  set to 3e-3. Models for W(2x2) and W(4x4) are trained for a maximum of 100
#  epochs using a batch size of 50, early stopping (patience value 25) and a
#  learning rate of 1e-3."  AdamW, zero weight decay, for all models.
TRAIN_HP = {
    "W11": dict(lr=3e-3, epochs=20, patience=5),
    "W12": dict(lr=3e-3, epochs=20, patience=5),
    "W22": dict(lr=1e-3, epochs=100, patience=25),
    "W44": dict(lr=1e-3, epochs=100, patience=25),
}
BATCH_SIZE = 50


def pick_device(requested=None):
    """cuda -> cpu, the repo's order minus MPS.

    MPS is skipped deliberately rather than by omission: the reference layers
    are complex throughout and ``torch.view_as_complex`` / complex ``einsum``
    on Metal are incomplete, so an MPS run fails deep inside vendored code with
    an error that says nothing about why.
    """
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def beta_ladder(b_min=0.1, b_max=6.0, n=11):
    """The coupling grid. See ``wilson_regression_data.py`` on why ``n = 11``."""
    return torch.linspace(float(b_min), float(b_max), int(n), dtype=torch.float64)


# ── Datasets ─────────────────────────────────────────────────────────────────

def load_split(split, L=8, target="W11", dtype=torch.complex64):
    """``(U, W, y, beta, meta)`` for one split, cast to the training dtype.

    Stored in complex128 / float64 (see the generator's docstring on why the
    label precision matters); the network runs in float32, so the cast happens
    here and in one place.
    """
    path = os.path.join(DATA_DIR, f"{split}_L{L}.pt")
    if not os.path.exists(path):
        raise SystemExit(
            f"missing dataset {path}. Generate it first:\n"
            f"    python scripts/wilson_regression_data.py"
        )
    d = torch.load(path, map_location="cpu", weights_only=False)
    if target not in d["labels"]:
        raise SystemExit(f"{path} has no label {target!r}; has "
                         f"{sorted(d['labels'])}")
    real = torch.float64 if dtype == torch.complex128 else torch.float32
    return (d["U"].to(dtype), d["W"].to(dtype),
            d["labels"][target].to(real), d["beta"].to(real), d["meta"])


def mse_pair(pred, true):
    """``(mse_site, mse_avg)`` for per-site tensors ``(B, *Lam)``.

    ``mse_avg`` is Fig. 3's convention -- both sides averaged over the lattice
    first. See the module docstring.
    """
    site = ((pred - true) ** 2).mean().item()
    dims = tuple(range(1, pred.ndim))
    avg = ((pred.mean(dim=dims) - true.mean(dim=dims)) ** 2).mean().item()
    return site, avg


# ── The model ────────────────────────────────────────────────────────────────

def lcnn_dof_count(layers, conv_impl="ref", D=2):
    """Parameter count of a Table V stack, per L-CB parametrisation.

    ``"exact"`` is the table's own number; ``"ref"`` is the vendored class,
    which carries one extra transported slot per layer (see
    :mod:`gelt.lcnn_exact`). Both include the trailing ``Linear(2 n_out, 1)``.
    """
    total = 0
    for k, n_in, n_out in layers:
        shifts = D * (k - 1)                    # positive shifts only
        slots = max(1, shifts) if conv_impl == "exact" else 1 + shifts
        total += n_out * (2 * n_in + 1) * (2 * n_in * slots + 1)
    return total + 2 * layers[-1][2] + 1


def build_lcnn(target, size, L, conv_impl="ref", init_w=1.0, check_nparam=True):
    """SM Table V's network, built through :class:`gelt.lcnn_reference.LCNNRef`.

    ``conv_impl="ref"`` is the vendored ``LConvBilin`` as published;
    ``"exact"`` is :mod:`gelt.lcnn_exact`, whose parameter count matches the
    table. ``check_nparam`` refuses a build whose count is neither -- which is
    what a wrong ``symmetric``, ``head_hidden`` or ``in_channels`` would
    silently look like.
    """
    from gelt.lattice import SU
    from gelt.lcnn_reference import LCNNRef

    key = (target, size)
    if key not in LCNN_ARCHS:
        raise SystemExit(f"no Table V architecture for {key}; have "
                         f"{sorted(LCNN_ARCHS)}")
    layers = LCNN_ARCHS[key]
    model = LCNNRef(
        SU(2), L=L, D=2,
        K=[k - 1 for k, _, _ in layers],       # their kernel_size is our K + 1
        c_hidden=[n_out for _, _, n_out in layers],
        n_layers=len(layers),
        in_channels=1,                          # 1+1D: one plaquette channel
        reduction="none",                       # per-site output, no lattice avg
        head_hidden=0,                          # their head: one Linear per site
        symmetric=False,                        # SM SS III: positive shifts only
        use_act=False,                          # no L-Act anywhere in the Letter
        init_w=init_w,
        conv_impl=conv_impl,
        dtype=torch.complex64,
    )
    n = sum(p.numel() for p in model.parameters())
    expected = lcnn_dof_count(layers, conv_impl)
    if check_nparam and n != expected:
        raise SystemExit(
            f"{key} built with {n} parameters, expected {expected} for "
            f"conv_impl={conv_impl!r} (Table V: {LCNN_NPARAM[key]})"
        )
    return model, n
