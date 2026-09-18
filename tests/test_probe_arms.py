"""The M1 probe's arms are matched — ``notes/m1_probe.md`` §2.

The arm table in that note is the experiment's premise: a ΔR² is a statement
about *mechanism* only for as long as the budgets agree, so the widths that make
them agree are a test, not a comment. Same reasoning as
``tests/test_lcnn.py``'s parameter match against the trained GELT nets.

The registry lives in ``scripts/probe_common.py`` rather than in ``gelt/``
because it is experiment configuration, not library — hence the path insert,
which is the same route the scripts use to import each other.
"""

import importlib
import os
import sys

import pytest
import torch

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

    for name in pc.available_arms():
        model, arch = pc.build_arm(name, grad_checkpoint=False)
        last = model.head_fc2 if arch == "lcnn" else model.mlp.fc2
        assert torch.count_nonzero(last.weight) == 0, name
        assert torch.count_nonzero(last.bias) == 0, name


# ── PROBE_GROUP=z2 ──────────────────────────────────────────────────────────
# The switch of ``notes/where_attention_can_win.md`` §9.5. What matters is that
# it moves the *study* and not the comparison: the arms have to stay matched,
# and the SU(2) study has to be untouched by its existence.

@pytest.fixture
def z2():
    """``probe_common`` re-imported with PROBE_GROUP=z2, then restored.

    The module reads its configuration once at import, which is what makes the
    arms un-driftable; testing both studies in one process therefore means
    reloading it. The fixture puts it back, so import order cannot leak a group
    into the tests above.
    """
    before = os.environ.get("PROBE_GROUP")
    os.environ["PROBE_GROUP"] = "z2"
    try:
        yield importlib.reload(pc)
    finally:
        if before is None:
            os.environ.pop("PROBE_GROUP", None)
        else:
            os.environ["PROBE_GROUP"] = before
        importlib.reload(pc)


def test_z2_switch_moves_the_study_and_not_the_geometry(z2):
    assert z2.IS_Z2 and z2.NC == 1 and z2.MODEL_DTYPE is torch.float32
    assert z2.LATTICE == (48, 24, 24)  # non-cubic, and fed whole
    assert z2.N_SLICES == 1  # a Z₂ configuration is already 3D
    assert z2.TARGETS == ("V1", "V2")
    assert z2.cache_path().startswith("datasets/z2_configs_")
    # the shared geometry is exactly the SU(2) study's
    assert (z2.R, z2.LAYERS, z2.LCNN_K, z2.MLP_HIDDEN) == (2, 4, 2, 32)


@pytest.mark.parametrize("arm", ["frozen_matched", "lcnn", "lcnn_norm"])
def test_z2_arms_are_still_matched_to_gelt(arm, z2):
    ref = z2.real_dofs(z2.build_arm("gelt", grad_checkpoint=False)[0])
    got = z2.real_dofs(z2.build_arm(arm, grad_checkpoint=False)[0])
    assert abs(got / ref - 1.0) <= z2.DOF_TOLERANCE, (
        f"{arm}/gelt real-DOF ratio {got / ref:.3f} under PROBE_GROUP=z2"
    )


def test_z2_refuses_the_projected_transport(z2):
    """``build_transport_average`` is ill-defined for Z₂ at mode='projected' —
    it needs the 0 → +1 tie-break that is CLAUDE.md caveat 1's defect. The arm
    must fail at construction, not three hours into a batch."""
    assert "gelt_projected" not in z2.available_arms()
    with pytest.raises(SystemExit, match="not available with PROBE_GROUP=z2"):
        z2.build_arm("gelt_projected")


def test_frozen_single_is_the_fourth_cell(z2):
    """The 2 × 2 of §9.3: {softmax, frozen} × {average, single}. In Z₂ the
    transport axis is present/absent rather than better/worse, so all four cells
    have to exist and only two things may vary across them."""
    cells = {
        ("softmax", "average"): "gelt",
        ("frozen", "average"): "frozen",
        ("softmax", "single"): "gelt_single",
        ("frozen", "single"): "frozen_single",
    }
    for (alpha, transport), name in cells.items():
        assert name in z2.available_arms()
        assert z2.ARMS[name]["alpha_mode"] == alpha
        assert z2.arm_transport(name) == transport
        assert z2.ARMS[name]["d_model"] == 16 and z2.ARMS[name]["d_qkv"] == 6
    # the two transport variants of one alpha_mode are the same network
    assert (z2.real_dofs(z2.build_arm("gelt", grad_checkpoint=False)[0])
            == z2.real_dofs(z2.build_arm("gelt_single", grad_checkpoint=False)[0]))
    assert (z2.real_dofs(z2.build_arm("frozen", grad_checkpoint=False)[0])
            == z2.real_dofs(z2.build_arm("frozen_single", grad_checkpoint=False)[0]))


def test_the_su2_study_survives_the_round_trip():
    """Switching to Z₂ and back must leave the SU(2) arms exactly as they were —
    the M1 probe's 132 completed runs stay comparable to anything run later.

    Done explicitly rather than through the fixture, because the assertion is
    about what the *restore* leaves behind and a fixture tears down after the
    test body has run.
    """
    before = os.environ.get("PROBE_GROUP")
    try:
        os.environ["PROBE_GROUP"] = "z2"
        importlib.reload(pc)
        assert pc.IS_Z2
        os.environ.pop("PROBE_GROUP")
        mod = importlib.reload(pc)
    finally:
        if before is not None:
            os.environ["PROBE_GROUP"] = before
        importlib.reload(pc)
    assert not mod.IS_Z2
    assert mod.TARGETS == ("T0", "T1", "T2")
    assert mod.N_SLICES == 6 and mod.LATTICE == (12, 12, 12)
    assert mod.real_dofs(mod.build_arm("gelt", grad_checkpoint=False)[0]) == 15405
