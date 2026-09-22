"""The L-CB parametrisation the *Letter* reports, where the vendored code drifted.

``gelt/lcnn_reference.py`` runs Favoni et al.'s ``LConvBilin`` verbatim out of
``lge-cnn-master/``. That is their code, but it is not quite the network whose
parameter counts PRL 128, 032003 Table V prints, and the difference is one
transported slot per layer.

**What differs.** Their ``LConvBilin.forward`` seeds the transported list with
the field itself, ``t_w = [w]``, and then appends the ``D·(k−1)`` shifted copies
(positive shifts only, SM §III). So the bilinear form is taken over
``1 + D(k−1)`` slots and the kernel is ``[n_out, 2n_in+1, 2n_in(1+D(k−1))+1]``.
Table V's counts are reproduced exactly — all ten architectures, four kernel
sizes, five widths — by ``max(1, D(k−1))`` slots instead, i.e. by the
transported copies **alone**, with the local field kept only when there are none
(``k = 1``, the ``W^{1×1}`` architecture, where the list would otherwise be
empty). ``tests/test_lcnn_exact.py`` pins that against the printed table.

Concretely, at ``D = 2``:

====================  ==========  ============  ==========
architecture          Table V     vendored      this module
====================  ==========  ============  ==========
``W^{1×1}`` small             12            12           12
``W^{1×2}`` small             35            47           35
``W^{2×2}`` large         13 521        15 745       13 521
``W^{4×4}`` large         39 905        46 481       39 905
====================  ==========  ============  ==========

**Why it is not cosmetic.** The extra slot is the local ``W`` on the *right* of
the bilinear, so the vendored layer can form ``W_i · W_j`` at one site while the
Letter's can only form ``W_i · (transported W_j)``. The unit element already
supplies ``W · 𝟙``, so what the drift adds is the same-site square — a strictly
larger function class at strictly more parameters. Every comparison in this repo
that quotes "the authors' own L-CNN at N parameters" is therefore quoting a
slightly wider model than the paper's, and a reproduction of Fig. 3 that wants to
say "their architecture, their parameter count" has to use this one.

Which is right is not decidable from the published source — the repository has
moved on since the Letter — so neither is treated as the correction of the other:
``lcnn_ref`` is the code as published, ``lcnn_exact`` is the architecture as
reported, and ``scripts/wilson_regression.py`` runs either.

Nothing here modifies ``lge-cnn-master/``. The class below is a subclass of
theirs that changes the weight shape and the slot list and inherits everything
else, including the transport, the conjugate augmentation and the unit elements.
"""

import numpy as np
import torch

from gelt.lcnn_reference import reference_layers

_EXACT_CLS = None


def exact_conv_class():
    """``LConvBilinExact``, built on first use.

    Lazily, because the base class only exists once ``reference_layers()`` has
    executed the vendored ``layers.py`` by path — there is no import-time name
    to subclass.
    """
    global _EXACT_CLS
    if _EXACT_CLS is not None:
        return _EXACT_CLS

    ref = reference_layers()

    class LConvBilinExact(ref.LConvBilin):
        """Their L-CB with the Letter's slot set (see the module docstring)."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.n_slots = self._slot_count()
            self._rebuild_weight()

        def _slot_count(self) -> int:
            """``max(1, Σ_axes (|a| + |b|))`` — transported copies only."""
            shifts = sum(abs(a) + abs(b) for a, b in self.kernel_range)
            return max(1, shifts)

        def _rebuild_weight(self):
            """Re-allocate ``self.weight`` at the Letter's shape.

            Their ``__init__`` has already sized it for ``1 + Σ|range|`` slots;
            everything about the initialisation below (the fan-based variance,
            ``init_w``, the zeroed residual and bias rows) is copied from theirs
            so that only the shape differs. Re-allocating rather than slicing
            keeps the two parametrisations independent: a slice would alias
            their buffer and silently inherit a future change to their init.
            """
            w_in_size = 2 * self.n_in + (1 if self.use_unit_elements else 0)
            t_w_size = 2 * self.n_in * self.n_slots + (
                1 if self.use_unit_elements else 0
            )
            variance = 1.0 / (w_in_size * t_w_size)
            weight = torch.empty(self.n_out, w_in_size, t_w_size)
            torch.nn.init.normal_(weight, std=self.init_w * np.sqrt(variance))
            if self.use_unit_elements:
                # Residual and bias terms start at zero, exactly as theirs do.
                torch.nn.init.constant_(weight[:, :, -1], 0.0)
                torch.nn.init.constant_(weight[:, -1, :], 0.0)
            self.weight = torch.nn.Parameter(data=weight, requires_grad=True)

        def forward(self, x):
            """``bilin_implementation == 2``, over the Letter's slot set.

            Their ``forward`` dispatches over five equivalent contraction
            orders; this reproduces the module default (their comment: "the
            good one") and asserts that the default has not moved, because a
            silent switch there would mean this layer stopped matching the one
            it is the counterpart of.
            """
            ref_mod = reference_layers()
            if ref_mod.bilin_implementation != 2:
                raise RuntimeError(
                    "vendored layers.py changed bilin_implementation to "
                    f"{ref_mod.bilin_implementation}; LConvBilinExact implements "
                    "only the '2' contraction order."
                )
            u, w = ref_mod.unpack_x(x, len(self.dims))

            # Transported terms only — no leading local copy. The fallback to
            # [w] fires exactly when the kernel has no shifts at all (k = 1),
            # which is the W^{1x1} architecture of Table V.
            t_w = []
            for axis in range(len(self.dims)):
                for i, o in zip([0, 1], [-1, +1]):
                    w_transport = w
                    for _d in range(abs(self.kernel_range[axis][i])):
                        for _step in range(self.dilation):
                            w_transport = ref_mod.transport(
                                u, w_transport, axis=axis, orientation=o,
                                dims=self.dims,
                            )
                        t_w.append(w_transport)
            if not t_w:
                t_w = [w]
            t_w = torch.cat(t_w, dim=2)

            # enlarge tensors by complex conjugates
            w_c, t_w_c = ref_mod.cconj(w), ref_mod.cconj(t_w)
            w = ref_mod.repack_x(w, w_c)
            t_w = ref_mod.repack_x(t_w, t_w_c)

            # enlarge tensors by unit matrices (adds bias and residual term)
            if self.use_unit_elements:
                unit_shape = list(w.shape)
                unit_shape[2] = 1
                unit_tensor = self.get_unit(unit_shape, device=w.device)
                w = ref_mod.repack_x(w, unit_tensor)
                t_w = ref_mod.repack_x(t_w, unit_tensor)

            wc, twc = torch.view_as_complex(w), torch.view_as_complex(t_w)
            weight_c = self.weight.type(wc.dtype)
            tmp = torch.einsum("uvw, bxwjk -> bxuvjk", weight_c, twc)
            wc = torch.einsum("bxvij, bxuvjk -> bxuik", wc, tmp)

            return ref_mod.repack_x(u, torch.view_as_real(wc))

    _EXACT_CLS = LConvBilinExact
    return _EXACT_CLS


def paper_dof_count(layers, D: int = 2, use_unit_elements: bool = True) -> int:
    """Table V's ``N_param`` from an architecture spec, without building it.

    ``layers`` is the table's own notation, ``[(k, n_in, n_out), ...]`` for a
    stack of ``L-CB(k, n_in, n_out)``, and the count includes the ``Trace`` (free)
    and the single per-site ``Linear(2·n_out, 1)`` with bias that every
    architecture in Tables V–VI ends with.
    """
    total = 0
    for k, n_in, n_out in layers:
        slots = max(1, D * (k - 1))
        w_in = 2 * n_in + (1 if use_unit_elements else 0)
        t_w = 2 * n_in * slots + (1 if use_unit_elements else 0)
        total += n_out * w_in * t_w
    return total + 2 * layers[-1][2] + 1
