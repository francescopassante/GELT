"""The Letter's L-CB parametrisation, and the 1+1D reproduction's premises.

``tests/test_lcnn_reference.py`` pins the wrapper around the vendored
``LConvBilin``. This file pins the *other* L-CB — :mod:`gelt.lcnn_exact`, whose
parameter counts are the ones PRL 128, 032003 Table V prints — and the
architecture tables of ``scripts/wilson_regression_common.py`` that the Fig. 3
reproduction is built from.

What is pinned, and why each would otherwise be a silent failure:

* **Table V, all ten architectures.** A reproduction whose networks are wider
  than the paper's would "succeed" without reproducing anything. The count is
  checked against the printed table, not against a formula this repo also wrote.
* **The one structural difference.** ``exact`` and ``ref`` are shown to be the
  *same function* once the slots the vendored class adds are zeroed — which is
  the claim :mod:`gelt.lcnn_exact` makes in prose, turned into a 1e-12
  comparison. If the slot ordering in their ``t_w`` ever changes, this fails.
* **Gauge invariance** of the per-site readout at SU(2), complex128 — if this
  goes, the arm is not an L-CNN.
* **Translational equivariance and volume transfer.** The Letter trains at
  8 x 8 and tests up to 64 x 64 without retraining; that is a property of the
  architecture, so it is testable with random weights and no training at all.
* **The training hyper-parameters**, transcribed from SM SS VI.A — a drift
  there is a silently different experiment, not a failed one.
* **W^(1x1) lies in the span of the smallest architecture**, constructively:
  the 12-parameter L-CB(1,1,1) + Trace + Linear(2,1) reproduces
  ``Re Tr P / N_c`` exactly at a hand-set parameter value. The Letter calls that
  task "trivial"; if it were not exactly representable here, the 2.2e-11 target
  would be unreachable for a reason that has nothing to do with training.
"""

import os
import sys

import pytest
import torch

from gelt import SU, link_gauge_transformation, plaquette_tensor, random_links
from gelt.lattice import rectangular_wilson_loop
from gelt.lcnn_exact import paper_dof_count
from gelt.lcnn_reference import LCNNRef

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "scripts")
)

import wilson_regression_common as wrc  # noqa: E402

GROUP = SU(2)


def _build(target, size, L=8, conv_impl="exact", dtype=torch.complex128):
    layers = wrc.LCNN_ARCHS[(target, size)]
    return LCNNRef(
        GROUP, L=L, D=2,
        K=[k - 1 for k, _, _ in layers],
        c_hidden=[c for _, _, c in layers],
        n_layers=len(layers),
        in_channels=1, reduction="none", head_hidden=0,
        symmetric=False, use_act=False, conv_impl=conv_impl, dtype=dtype,
    )


@pytest.mark.parametrize("key", sorted(wrc.LCNN_ARCHS))
def test_exact_matches_table_v(key):
    """Every Table V architecture, at the table's own N_param."""
    model = _build(*key)
    n = sum(p.numel() for p in model.parameters())
    assert n == wrc.LCNN_NPARAM[key], (key, n, wrc.LCNN_NPARAM[key])
    assert paper_dof_count(wrc.LCNN_ARCHS[key]) == wrc.LCNN_NPARAM[key]


@pytest.mark.parametrize("key", sorted(wrc.LCNN_ARCHS))
def test_vendored_exceeds_table_by_one_slot(key):
    """The published class is wider, by exactly the predicted amount.

    Not a complaint about their code — it is the arithmetic that justifies
    having two implementations at all, so it is pinned rather than asserted.
    """
    model = _build(*key, conv_impl="ref")
    n = sum(p.numel() for p in model.parameters())
    assert n == wrc.lcnn_dof_count(wrc.LCNN_ARCHS[key], "ref")
    assert n >= wrc.LCNN_NPARAM[key]
    if any(k > 1 for k, _, _ in wrc.LCNN_ARCHS[key]):
        assert n > wrc.LCNN_NPARAM[key]


def test_exact_is_ref_with_the_extra_slots_zeroed():
    """The two L-CBs are one function, up to the slots the vendored one adds.

    Their ``t_w`` is ``[w, transported…]`` and then the same list conjugated and
    a unit element; the exact one drops the two ``w`` blocks. Copying the
    surviving columns across and zeroing the dropped ones must reproduce the
    output to round-off — which tests the *ordering* of that axis, not just its
    length.
    """
    torch.manual_seed(0)
    L, k, n_in, n_out, D = 6, 3, 2, 3, 2
    ref_model = _build("W22", "medium", L=L, conv_impl="ref")
    exact_model = _build("W22", "medium", L=L, conv_impl="exact")

    for ref_c, ex_c in zip(ref_model.convs, exact_model.convs):
        n_in = ref_c.n_in
        shifts = sum(abs(a) + abs(b) for a, b in ref_c.kernel_range)
        w = ref_c.weight.detach().clone()
        w.zero_()
        # exact block layout: [transported | conj transported | unit]
        # ref   block layout: [w, transported | w*, conj transported | unit]
        n_t = n_in * shifts
        src = ex_c.weight.detach()
        w[:, :, n_in:n_in + n_t] = src[:, :, :n_t]
        off = n_in * (1 + shifts)
        w[:, :, off + n_in:off + n_in + n_t] = src[:, :, n_t:2 * n_t]
        w[:, :, -1] = src[:, :, -1]
        ref_c.weight.data.copy_(w)
    ref_model.head_fc2.load_state_dict(exact_model.head_fc2.state_dict())

    U = random_links(L, D, GROUP, dtype=torch.float64, N=2)
    W = plaquette_tensor(U, GROUP)
    a = ref_model(W, U)
    b = exact_model(W, U)
    assert torch.allclose(a, b, atol=1e-12), (a - b).abs().max().item()


@pytest.mark.parametrize("conv_impl", ["ref", "exact"])
@pytest.mark.parametrize("key", [("W11", "small"), ("W12", "medium"),
                                 ("W44", "small")])
def test_gauge_invariance_of_per_site_readout(key, conv_impl):
    torch.manual_seed(1)
    L, D = 6, 2
    model = _build(*key, L=L, conv_impl=conv_impl)
    # A zero-initialised head would make any field look invariant.
    torch.nn.init.normal_(model.head_fc2.weight, std=0.5)
    torch.nn.init.normal_(model.head_fc2.bias, std=0.5)

    U = random_links(L, D, GROUP, dtype=torch.float64, N=2)
    omega = GROUP.random((L, L), dtype=torch.float64)
    Ug = torch.stack([link_gauge_transformation(u, omega, GROUP) for u in U])

    out = model(plaquette_tensor(U, GROUP), U)
    out_g = model(plaquette_tensor(Ug, GROUP), Ug)
    assert torch.allclose(out, out_g, atol=1e-10), (out - out_g).abs().max()


def test_translation_equivariance_and_volume_transfer():
    """Per-site output shifts with the input, and survives a change of volume.

    The volume check is the Letter's own generalisation claim reduced to
    arithmetic: a configuration tiled 2 x 2 into a 16 x 16 lattice is a legal
    gauge configuration there, and a translationally equivariant network with a
    per-site head must return the tiled output. No training involved.
    """
    torch.manual_seed(2)
    L, D = 8, 2
    model = _build("W22", "small", L=L)
    torch.nn.init.normal_(model.head_fc2.weight, std=0.5)
    for c in model.convs:
        torch.nn.init.normal_(c.weight, std=0.3)

    U = random_links(L, D, GROUP, dtype=torch.float64, N=1)
    out = model(plaquette_tensor(U, GROUP), U)

    shifted = torch.roll(U, shifts=(2, -3), dims=(2, 3))
    out_shift = model(plaquette_tensor(shifted, GROUP), shifted)
    assert torch.allclose(torch.roll(out, shifts=(2, -3), dims=(1, 2)),
                          out_shift, atol=1e-10)

    tiled = U.repeat(1, 1, 2, 2, 1, 1)
    model.update_dims(2 * L)
    out_big = model(plaquette_tensor(tiled, GROUP), tiled)
    assert out_big.shape == (1, 2 * L, 2 * L)
    assert torch.allclose(out_big, out.repeat(1, 2, 2), atol=1e-10)
    model.update_dims(L)
    assert torch.allclose(model(plaquette_tensor(U, GROUP), U), out, atol=1e-12)


def test_w11_is_exactly_representable_by_the_12_parameter_net():
    """``W^(1x1) = Re Tr P / N_c`` is in the span of Table V's smallest network.

    Constructive: the unit element on the local side and the field on the
    transported side give ``1 · W = W``, the Trace makes that
    ``(Re Tr W, Im Tr W)``, and a Linear with weights ``(1/N_c, 0)`` and no bias
    is the label. Nothing is fitted.
    """
    L, D = 6, 2
    model = _build("W11", "small", L=L)
    with torch.no_grad():
        for c in model.convs:
            c.weight.zero_()
            c.weight[0, -1, 0] = 1.0          # (unit) x (local W)
        model.head_fc2.weight.zero_()
        model.head_fc2.weight[0, 0] = 1.0 / GROUP.nc   # Re Tr W / nc
        model.head_fc2.bias.zero_()

    U = random_links(L, D, GROUP, dtype=torch.float64, N=3)
    label = rectangular_wilson_loop(U, GROUP, R=1, T=1, mu=0, nu=1)
    out = model(plaquette_tensor(U, GROUP), U)
    assert torch.allclose(out, label, atol=1e-12), (out - label).abs().max()


def test_training_hyperparameters_are_the_letters():
    """SM SS VI.A, transcribed. A drift here is a silently different experiment."""
    assert wrc.BATCH_SIZE == 50
    for t in ("W11", "W12"):
        assert wrc.TRAIN_HP[t] == dict(lr=3e-3, epochs=20, patience=5)
    for t in ("W22", "W44"):
        assert wrc.TRAIN_HP[t] == dict(lr=1e-3, epochs=100, patience=25)
