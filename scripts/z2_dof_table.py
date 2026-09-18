"""The matched-parameter table under whichever ``PROBE_GROUP`` is selected.

Split out of ``probe_batch.sh`` because an inline heredoc inside a batch phase
is exactly the kind of thing that silently stops running when the surrounding
quoting changes — and this table is the first thing part 0 / part z-gate prints,
i.e. the one a reader trusts to say the arms are still matched.

    python scripts/z2_dof_table.py --group=z2
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import probe_common as pc  # noqa: E402


def main():
    print(f"group {pc.GROUP}  lattice {pc.LATTICE}  targets {pc.TARGETS}"
          + (f"  β {pc.BETA}  conv_init_scale {pc.Z2_LCNN_CONV_INIT}"
             if pc.IS_Z2 else f"  ensemble seed {pc.ENSEMBLE_SEED}"))
    print(f"{'arm':16s} {'real DOFs':>9s}   ratio to gelt")
    bad = []
    for k, (d, r) in pc.dof_table().items():
        nested = k.startswith("frozen") and "matched" not in k
        if nested:
            note = "nested ablation, fewer by design"
        elif abs(r - 1) <= pc.DOF_TOLERANCE:
            note = "OK"
        else:
            note = "** OUT OF TOLERANCE **"
            bad.append(k)
        print(f"{k:16s} {d:9d}   {r:5.3f}   {note}")
    if bad:
        print(f"\n** {bad} are no longer matched-parameter: a ΔR² against them "
              f"would confound capacity with mechanism.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
