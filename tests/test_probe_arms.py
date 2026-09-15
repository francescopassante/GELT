"""The M1 probe's arms are matched — ``notes/m1_probe.md`` §2.

The arm table in that note is the experiment's premise: a ΔR² is a statement
about *mechanism* only for as long as the budgets agree, so the widths that make
them agree are a test, not a comment. Same reasoning as
``tests/test_lcnn.py``'s parameter match against the trained GELT nets.

The registry lives in ``scripts/probe_common.py`` rather than in ``gelt/``
because it is experiment configuration, not library — hence the path insert,
which is the same route the scripts use to import each other.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
)

import probe_common as pc  # noqa: E402


def _dofs(name):
    return pc.real_dofs(pc.build_arm(name, grad_checkpoint=False)[0])


@pytest.mark.parametrize("arm", ["frozen_matched", "lcnn", "lcnn_norm"])
def test_arms_are_within_tolerance_of_gelt(arm):
    ratio = _dofs(arm) / _dofs("gelt")
    assert abs(ratio - 1.0) <= pc.DOF_TOLERANCE, (
        f"{arm}/gelt real-DOF ratio {ratio:.3f} at {pc.ARMS[arm]} — no longer "
        f"matched-parameter, so a ΔR² against it would confound capacity"
    )


def test_frozen_is_the_nested_ablation():
    """``frozen`` is deliberately *smaller*: deleting the score path deletes
    weights. A tie against it is the stronger statement — GELT has both the
    extra parameters and the mechanism — which is why it is not widened, and why
    ``frozen_matched`` exists separately."""
    assert _dofs("frozen") < _dofs("gelt")
    assert _dofs("frozen") < _dofs("frozen_matched")


def test_only_the_block_differs_between_gelt_and_frozen():
    """Same layer count, same d_model, same readout — the ablation changes
    ``alpha_mode`` and the width that follows from deleting Q_s/K, nothing
    else."""
    a, b = pc.ARMS["gelt"], pc.ARMS["frozen"]
    assert a["arch"] == b["arch"] == "gelt"
    assert a["d_model"] == b["d_model"]
    assert a["d_qkv"] == b["d_qkv"]
    assert (a["alpha_mode"], b["alpha_mode"]) == ("softmax", "frozen")

    gelt, _ = pc.build_arm("gelt", grad_checkpoint=False)
    frozen, _ = pc.build_arm("frozen", grad_checkpoint=False)
    assert len(gelt.gemhsa_models) == len(frozen.gemhsa_models) == pc.LAYERS
    for g, f in zip(gelt.gemhsa_models, frozen.gemhsa_models):
        assert g.n_proj == 4 and hasattr(g, "rope_freq")
        assert f.n_proj == 2 and not hasattr(f, "rope_freq")
        assert f.alpha_logits.shape == (g.H, g.n_offsets)


def test_every_arm_has_a_zero_initialised_readout():
    """All five arms start at ŷ ≡ 0, the standardised training mean, so no arm
    begins with an output-scale advantage. MSE has a nonzero gradient there —
    unlike the Rayleigh loss of ``train_glueball.py`` (audit item 3), where zero
    init means training never starts."""
    import torch

    for name in pc.ARMS:
        model, arch = pc.build_arm(name, grad_checkpoint=False)
        last = model.head_fc2 if arch == "lcnn" else model.mlp.fc2
        assert torch.count_nonzero(last.weight) == 0, name
        assert torch.count_nonzero(last.bias) == 0, name
