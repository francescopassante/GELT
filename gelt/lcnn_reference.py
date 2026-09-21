"""The authors' own L-CNN as a drop-in baseline arm.

``gelt/lcnn.py`` is *our* implementation of Favoni et al. (2012.12901). This
module is the other thing: the authors' layers, loaded verbatim out of the
vendored ``lge-cnn-master/`` tree, wrapped so they present exactly the I/O of
:class:`gelt.lcnn.LCNN` and can be selected as an arm wherever that one can.

Why it exists. Every GELT-vs-L-CNN number in this repo was produced against our
reimplementation, and `scripts/bench_lcnn_reference.py` has said in its own
docstring from the beginning that the two are **not the same function**: their
``LConvBilin`` carries one merged 3-index bilinear kernel
``weight[n_out, 2·n_in+1, 2·n_in·(1+Σ|range|)+1]`` over ``[1, W, W†]`` and its
transported copies, ours factors that through L-Conv's ``c_out``, i.e. a
low-rank member of the same family. A baseline a reviewer will ask about should
be the authors', not ours.

Three things the wrapper has to get right, each of which is a trap:

1. **Kernel-size convention.** Theirs is ``kernel_range = [-(k-1), k-1]`` per
   axis, so ``kernel_size=k`` reaches ``k-1`` steps; ours is ``K`` hops per axis
   in both orientations. **Their ``kernel_size`` is our ``K + 1``**, and at
   ``D = 3`` both then carry the same 13 transported terms
   (``1 + 2·D·K``). Passing their ``kernel_size=2`` for our ``K=2`` would halve
   the baseline's receptive field per layer — Manhattan 4 instead of 8 over four
   layers — which is precisely the silently unfair baseline this module exists
   to avoid. The constructor takes ``K`` in *our* convention and translates.
2. **Layout.** Theirs is ``[B, N^D, N_C, nc, nc, 2]`` — lattice flattened, real
   and imaginary split into a trailing axis, and the links packed into the same
   tensor as the leading ``D`` channels (``unpack_x``/``repack_x``). Ours is
   ``(B, C, *Λ, nc, nc)`` complex with the links passed separately. The two
   conversions here are views and one ``view_as_real`` wherever possible.
3. **Z₂ is real, their code is not.** ``nc = 1`` runs in this repo carry a real
   dtype; the reference layers are complex throughout (they split re/im by
   hand). Real inputs are promoted to complex on the way in and the imaginary
   part is dropped on the way out, so a Z₂ arm costs 2× the memory here. That is
   a property of their parametrisation, not a defect.

**There is no activation layer, by default, because there is none in the
paper.** PRL 128, 032003 Tables V–VI list every architecture they report, in
1+1D and 3+1D, and each is a bare stack of L-CB followed by ``Trace`` and a
single ``Linear``: no L-Act anywhere. The Letter introduces L-Act under
"Additional layers" as something that *can* be applied (Eq. 7, with
``g = ReLU(ReTr[W])`` as the suggested choice) and never uses it — "L-Bilins are
already nonlinear". ``LActPoly`` is not in the Letter at all; it belongs to
their later fixed-point-action code. So ``use_act`` is **False** by default and
``degree_range`` / ``use_relu`` only matter when it is switched on.

That default is not cosmetic. With ``LActPoly`` after every layer the glueball
stack's field at four layers went as ``init_w^198`` and there was no usable
initialisation scale at all (``notes/lcnn_reference_switch.md`` §7): a random
polynomial multiplier per layer compounds the degree growth that L-Bilin already
has.

The per-site head is shared with :class:`gelt.lcnn.LCNN` by default — it carries
no gauge structure, it reads traces which are already invariant, and matching it
is what keeps the comparison about the equivariant stack. ``head_hidden=0``
gives **their** head instead, the single per-site ``Linear`` of Tables V–VI with
no hidden layer and no ReLU. Worth having as more than a fidelity option: a
saturating ReLU on a hidden layer is what turned a small field into an exactly
constant output in the first glueball run.

Two further divergences from the paper, both deliberate and both recorded in
``notes/lcnn_reference_switch.md`` §8: their convolutions use **positive shifts
only** by default (SM §III), while this wrapper passes ``use_symmetric=True`` so
the receptive field matches GELT's; and their deeper models **grow the kernel
size with depth** (L-CB(2,·), L-CB(2,·), L-CB(3,·), L-CB(3,·)), while ``K`` here
is uniform.

See ``notes/m1_probe.md`` §4 and ``notes/lcnn_shootout.md`` for what is being
compared and why. ``lge-cnn-master/`` is MIT-licensed third-party code and is
not modified — everything here is a wrapper.
"""

import importlib.util
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_REF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "lge-cnn-master", "lge_cnn", "nn", "layers.py",
)

_ref_module = None


def reference_layers():
    """The authors' ``layers.py``, loaded by path and cached.

    By path rather than by import because ``lge_cnn/nn/__init__.py`` pulls in
    h5py, which this repo does not depend on; ``layers.py`` itself needs only
    torch and numpy. Same route ``scripts/bench_lcnn_reference.py`` takes.

    Note for anyone debugging a slow first step: several helpers in that file
    are decorated ``@torch.compile``. If dynamo misbehaves on a given box,
    ``TORCHDYNAMO_DISABLE=1`` turns it off without touching vendored code.
    """
    global _ref_module
    if _ref_module is None:
        if not os.path.exists(_REF_PATH):
            raise FileNotFoundError(
                f"vendored reference L-CNN not found at {_REF_PATH}. It is "
                f"tracked in this repo; a partial clone is the usual cause."
            )
        spec = importlib.util.spec_from_file_location("_lge_layers", _REF_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _undecorate_compiled(module)
        _ref_module = module
    return _ref_module


def _undecorate_compiled(module):
    """Strip ``@torch.compile`` from their helpers, in our copy of the module.

    ``layers.py`` decorates ``unpack_x`` (and friends) with ``torch.compile``.
    Under inductor that raised ``BackendCompilerFailed`` on the V100 — a
    FakeTensor error from a ``uint8`` clone — and took the profiler with it,
    while the plain training path happened to survive. Compiling a two-line
    slice buys nothing here and the failure mode is a whole phase, so it is
    unwrapped rather than worked around with a global ``TORCHDYNAMO_DISABLE``:
    that would also silence ``PROFILE_COMPILE``, which is about *our* code.

    Surgical and reversible — each wrapper carries the original callable, and
    only this module's names are rebound. Vendored source is untouched.
    """
    for name in dir(module):
        fn = getattr(module, name, None)
        orig = getattr(fn, "_torchdynamo_orig_callable", None)
        if orig is not None:
            setattr(module, name, orig)


# ── Layout conversion ────────────────────────────────────────────────────────
# Ours  : (B, C, *Λ, nc, nc)          complex (or real, for Z₂)
# Theirs: (B, X, C, nc, nc, 2)        X = prod(Λ), real/imag split
#
# Both conversions are index gymnastics only. ``view_as_real``/``view_as_complex``
# need the complex axis last and the tensor contiguous, which is why the
# ``.contiguous()`` calls are where they are and not optional.

def to_ref_layout(x):
    """``(B, C, *Λ, nc, nc)`` → ``(B, X, C, nc, nc, 2)``."""
    B, C = x.shape[0], x.shape[1]
    nc = x.shape[-1]
    if not x.is_complex():
        x = x.to(torch.complex128 if x.dtype == torch.float64 else torch.complex64)
    x = x.movedim(1, -3)                      # (B, *Λ, C, nc, nc)
    x = x.reshape(B, -1, C, nc, nc).contiguous()
    return torch.view_as_real(x)


def from_ref_layout(x, lattice, complex_out=True):
    """``(B, X, C, nc, nc, 2)`` → ``(B, C, *Λ, nc, nc)``."""
    x = torch.view_as_complex(x.contiguous())
    B, _, C, nc, _ = x.shape
    x = x.reshape(B, *lattice, C, nc, nc)
    x = x.movedim(-3, 1)
    return x if complex_out else x.real


class LCNNRef(nn.Module):
    """Favoni et al.'s L-CNN, their code, our I/O.

    ``forward(W, U)`` mirrors :meth:`gelt.lcnn.LCNN.forward` in shape and in
    meaning, with one difference that is forced by their design: the second
    argument is the **raw links** ``(B, D, *Λ, nc, nc)``, not the precomputed
    axis transports ``(B, D, K, *Λ, nc, nc)``. ``LConvBilin`` transports its own
    field one step at a time inside the layer, so there is nothing to
    precompute; ``scripts/probe_common.probe_inputs`` returns the links for this
    arch for that reason.

    Arguments follow :class:`gelt.lcnn.LCNN` wherever the two overlap. ``K`` is
    in *our* convention (hops per axis per layer, both orientations) — see the
    module docstring on why that is not their ``kernel_size``.

    ``init_w`` is their init-scale knob and is the analogue of our
    ``conv_init_scale``: the same stack-vs-layer scaling problem exists here
    (an L-CB stack is a matrix polynomial of degree ``2^n_layers``), so it must
    be gated at the production geometry before any training run — their
    parametrisation is not ours and a value measured for ours does not carry
    over. See ``scripts/z2_init_gate.py``.
    """

    def __init__(
        self,
        gaugegroup,
        L,
        D: int,
        K: int,
        c_hidden,                 # int, or a per-layer sequence
        n_layers: int,
        dtype: torch.dtype = torch.complex64,
        mlp_hidden: int = 32,
        mlp_out: int = 1,
        reduction: str = "sum",
        in_channels: int | None = None,
        init_scale: float = 1.0,
        grad_checkpoint: bool = False,
        init_w: float = 1.0,
        use_act: bool = False,
        degree_range: int = 2,
        use_relu: bool = True,
        head_hidden: int | None = None,
        dilation: int = 1,
        use_unit_elements: bool = True,
        symmetric: bool = True,
    ):
        super().__init__()
        if reduction not in ("sum", "mean", "none"):
            raise ValueError(
                f"reduction must be 'sum', 'mean', or 'none', got {reduction!r}"
            )
        ref = reference_layers()
        self.reduction = reduction
        self.K = K
        self.D = D
        self.gaugegroup = gaugegroup
        self.grad_checkpoint = grad_checkpoint

        # Per-axis extents. An int is the cubic shorthand, as everywhere else in
        # this repo (the Z₂ box is 48 × 24 × 24).
        lattice = tuple(L) if isinstance(L, (tuple, list)) else (int(L),) * D
        if len(lattice) != D:
            raise ValueError(f"L={L!r} does not describe {D} axes")
        self.lattice = lattice
        self.dims = list(lattice)

        nc = 1 if gaugegroup.nc is None else gaugegroup.nc
        self.nc = nc
        c_in_plaq = D * (D - 1) // 2 if in_channels is None else in_channels
        self.c_in_plaq = c_in_plaq

        # ``c_hidden`` may be a per-layer sequence. That is their own idiom —
        # their models take a ``conv_ch`` list — and here it is what makes a
        # matched-parameter width reachable at all: the first layer costs
        # (2·c_in+1)·(2·c_in(1+2DK)+1) per output channel and the rest cost far
        # less, so a single uniform integer steps over the tolerance band in one
        # hop. Widening only the first layer is the fine adjustment.
        if isinstance(c_hidden, (tuple, list)):
            hidden = [int(c) for c in c_hidden]
            if len(hidden) != n_layers:
                raise ValueError(
                    f"c_hidden has {len(hidden)} entries for {n_layers} layers"
                )
        else:
            hidden = [int(c_hidden)] * n_layers
        self.hidden = hidden
        self.c_hidden = hidden[-1]      # what the head reads

        # Their kernel_size, in their convention. See the module docstring.
        self.kernel_size_ref = K + 1

        widths = [c_in_plaq] + hidden
        self.convs = nn.ModuleList(
            [
                ref.LConvBilin(
                    dims=self.dims,
                    kernel_size=self.kernel_size_ref,
                    dilation=dilation,
                    n_in=widths[i],
                    n_out=widths[i + 1],
                    nc=nc,
                    init_w=init_w,
                    use_unit_elements=use_unit_elements,
                    use_symmetric=symmetric,
                )
                for i in range(n_layers)
            ]
        )
        self.use_act = use_act
        self.acts = nn.ModuleList(
            [
                ref.LActPoly(
                    dims=self.dims,
                    n_in=widths[i + 1],
                    nc=nc,
                    init_w=init_w,
                    degree_range=degree_range,
                    use_relu=use_relu,
                )
                if use_act else nn.Identity()
                for i in range(n_layers)
            ]
        )

        # Their weights are real parameters cast to complex at use, so the real
        # dtype of this model is the real part of ours.
        real_dtype = (
            torch.float64
            if dtype in (torch.complex128, torch.float64)
            else torch.float32
        )
        self.real_dtype = real_dtype
        self.to(real_dtype)
        self._fix_reference_dtypes(real_dtype)

        # Head: shared with gelt.lcnn.LCNN, byte for byte. LTrace gives the
        # complex trace per channel, i.e. 2·c_hidden reals per site.
        # head_hidden=0 is the paper's head: one Linear per site, no hidden
        # layer, no ReLU (Tables V-VI). None keeps gelt.lcnn.LCNN's, which is
        # what the matched-parameter widths were chosen against.
        hidden = mlp_hidden if head_hidden is None else head_hidden
        self.head_hidden = hidden
        if hidden:
            self.head_fc1 = nn.Linear(2 * self.c_hidden, hidden).to(real_dtype)
            self.head_fc2 = nn.Linear(hidden, mlp_out).to(real_dtype)
        else:
            self.head_fc1 = None
            self.head_fc2 = nn.Linear(2 * self.c_hidden, mlp_out).to(real_dtype)
        if init_scale != 1.0:
            with torch.no_grad():
                self.head_fc2.weight.mul_(init_scale)
                self.head_fc2.bias.mul_(init_scale)

    def _fix_reference_dtypes(self, real_dtype):
        """``nn.Module.to`` does not reach their ``unit_matrix``.

        ``LConvBilin.unit_matrix`` is a plain attribute, not a buffer or a
        parameter, so a ``.to(float64)`` on the module leaves it float32 and the
        ``repack_x`` that concatenates it to the field raises on dtype. Same for
        ``LActPoly``, which allocates one and never uses it.
        """
        for m in list(self.convs) + list(self.acts):
            if not hasattr(m, "unit_matrix"):
                continue          # nn.Identity, when use_act is False
            m.unit_matrix = m.unit_matrix.to(real_dtype)
            m.unit_matrix_re = m.unit_matrix_re.to(real_dtype)
            m.unit_matrix_im = m.unit_matrix_im.to(real_dtype)
            if hasattr(m, "unit_tensors"):
                m.unit_tensors = {}

    def _clear_unit_caches(self, device):
        """Their ``get_unit`` cache is keyed on shape alone, not on device.

        Harmless while a model stays put; wrong the first forward after a
        ``.to(device)``, which would hand a CPU tensor to a CUDA ``cat``. The
        cached value is an ``expand`` of a (nc, nc, 2) tensor — a view, not a
        copy — so dropping it every forward costs nothing measurable.
        """
        for m in self.convs:
            if getattr(m, "unit_tensors", None):
                any_dev = next(iter(m.unit_tensors.values())).device
                if any_dev != device:
                    m.unit_tensors = {}

    def forward(self, W: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """W : ``(B, C_in_plaq, *Λ, nc, nc)`` — plaquettes.
        U : ``(B, D, *Λ, nc, nc)`` — **raw links**, not axis transports.
        """
        # ndim, not just shape[1]: the axis-transport tensor gelt.lcnn.LCNN
        # takes has the *same* D on axis 1 and one extra axis for the hop
        # count, so a shape check alone lets it through and it then flattens
        # into the channel axis with plausible-looking numbers.
        if U.ndim != W.ndim or U.shape[1] != self.D:
            raise ValueError(
                f"LCNNRef.forward expects raw links (B, D, *Λ, nc, nc); got a "
                f"second argument of shape {tuple(U.shape)}. The axis-transport "
                f"tensor gelt.lcnn.LCNN takes is not what LConvBilin wants — it "
                f"transports one step at a time internally."
            )
        self._clear_unit_caches(W.device)
        u = to_ref_layout(U)
        w = to_ref_layout(W)
        ref = reference_layers()
        x = ref.repack_x(u, w)

        for conv, act in zip(self.convs, self.acts):
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():  # noqa: E501
                # Same trade as gelt.lcnn.LCNN: recompute in backward rather
                # than keep the layer's (n_out, w_in, t_w) intermediate, which
                # is this block's memory wall exactly as the channel-pair outer
                # product is ours.
                x = checkpoint(conv, x, use_reentrant=False)
            else:
                x = conv(x)
            x = act(x)

        _, w = ref.unpack_x(x, len(self.dims))
        tr = torch.einsum("bxwiic -> bxwc", w)          # (B, X, C, 2)
        B = tr.shape[0]
        trace = tr.reshape(B, *self.lattice, 2 * self.c_hidden)

        h = F.relu(self.head_fc1(trace)) if self.head_fc1 is not None else trace
        site_out = self.head_fc2(h).squeeze(-1)         # (B, *Λ)

        if self.reduction == "none":
            return site_out
        spatial_dims = tuple(range(1, site_out.ndim))
        if self.reduction == "sum":
            return site_out.sum(dim=spatial_dims)
        return site_out.mean(dim=spatial_dims)


def reference_dof_count(D: int, K: int, c_in: int, c_hidden: int, n_layers: int,
                        degree_range: int = 2, use_relu: bool = True,
                        mlp_hidden: int = 32, mlp_out: int = 1,
                        use_unit_elements: bool = True,
                        use_act: bool = False, head_hidden=None) -> int:
    """Real parameter count, from their shapes, without building the model.

    Their kernel is quadratic in the input width — ``(2·c_in+1)·(2·c_in·(1+2DK)+1)``
    per output channel — so the *first* layer dominates whenever the task feeds
    many input channels (the glueball stack's 12 smeared plaquette channels put
    7825 parameters per output channel into layer 1 alone at D=3, K=2). That is
    why a matched-parameter width against this parametrisation is much narrower
    than against ours, and why it has to be recomputed per task rather than
    carried over. ``scripts/z2_dof_table.py`` is the authority; this exists so a
    width can be chosen before a GPU is involved.
    """
    n_terms = 1 + 2 * D * K                      # their 1 + Σ(|a|+|b|), K = k-1
    total = 0
    hidden = (
        [int(c) for c in c_hidden]
        if isinstance(c_hidden, (tuple, list))
        else [int(c_hidden)] * n_layers
    )
    widths = [c_in] + hidden
    for i in range(n_layers):
        w_in = 2 * widths[i] + (1 if use_unit_elements else 0)
        t_w = 2 * widths[i] * n_terms + (1 if use_unit_elements else 0)
        total += widths[i + 1] * w_in * t_w
        if use_act:
            total += widths[i + 1] * (degree_range + (1 if use_relu else 0))
    hh = mlp_hidden if head_hidden is None else head_hidden
    if hh:
        total += 2 * hidden[-1] * hh + hh                    # head_fc1
        total += hh * mlp_out + mlp_out                      # head_fc2
    else:
        total += 2 * hidden[-1] * mlp_out + mlp_out          # the paper's head
    return total
