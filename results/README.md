# results/ — generated artifacts, grouped by study

Everything here is **produced by a script**, never hand-edited. The directory is
gitignored: figures, checkpoints and result dumps are outputs, and the ones that
matter are reproduced in `glueball_report/` and `attention_report/`.

Ensembles are *not* here — cached link configurations live in `datasets/`
(`*_configs_*.pt`), keyed by lattice parameters, because they are inputs shared
across studies rather than results of any one of them.

| directory | study | produced by |
|---|---|---|
| `sampler/` | sampler and lattice validation | `validate_sampler_z2.py`, `validate_sampler_su2.py`, `validate_anisotropy.py` |
| `glueball/` | 0⁺⁺ glueball spectroscopy | `measure_glueball.py`, `train_glueball.py`, `fit_glueball_overlap.py`, `check_glueball_autocorrelation.py`, `operator_decomposition.py` |
| `attention/` | attention as a measurement | `visualize_glueball_attention.py`, `topology_attention.py`, `z2_attention_correlator.py`, `z2_beta_scan.py`, `beta_scan.py`, `check_cooling.py`, `train_z2_glueball.py` |
| `wilson_regression/` | per-site Wilson-loop regression | `train_gelt.py`, `train_cnn.py`, `train_lcnn.py`, `compare_transport_modes.py` |
| `dual/` | exact ξ from the dual Ising model | `dual_ground_truth.py` |
| `fair_fight/` | is the classical comparator a straw man? | `z2_fair_fight.py`, `su2_fair_fight.py` |

## The fair fights (2026-09-07/08)

`fair_fight/` holds the audit that re-states two headlines, so anything read out
of `glueball/` or `attention/` should be checked against it first:

- `su2_fair_fight.{pt,png}` — §6.2's ΔA₀ is +0.077 ± 0.022 against the published
  single-shape basis and +0.013 ± 0.029 against a shape-extended one at matched
  inputs. The overlap claim does not survive; "one learned operator equals 21
  classical ones" does.
- `z2_fair_fight.{pt,png}` — the strengthened classical arm is as accurate as the
  trained attention field against the dual truth, so "the only arm consistent
  with the exact answer" is withdrawn.
- `*_obars_*.pt` — per-configuration Ō series for every arm, kept so any later
  correlated comparison runs offline instead of costing a GPU pass.

Numbers and their limits: `notes/audit_2026-09-06.md` §6.

## Report figures

Both LaTeX reports pull figures from here via

```latex
\graphicspath{{../}{../results/}{../results/sampler/}{../results/glueball/}{../results/attention/}{../results/wilson_regression/}}
```

so `\includegraphics{glueball_validation}` keeps working with a bare filename.
**If you move a figure between these directories, the reports still build** —
but if you rename one, fix the `\includegraphics` call too.

## Naming

`RUN_TAG` suffixes (`_ens1`, `_R6`, …) mark non-default seeds or radii so a
replication never overwrites the original run's artifacts. Files marked
`_SUPERSEDED_*` are kept only so a number quoted in an old note can still be
traced back; do not use them for anything.
