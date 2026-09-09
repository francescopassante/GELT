# results/ — generated artifacts, grouped by study

Everything here is **produced by a script**, never hand-edited. The directory is
gitignored: figures, checkpoints and result dumps are outputs. The two files
that must survive a clone (the test-split Ō dumps consumed by the offline
audits) live in `dumps/` instead, which is tracked.

Ensembles are *not* here — cached link configurations live in `datasets/`
(`*_configs_*.pt`), keyed by lattice parameters, because they are inputs shared
across studies rather than results of any one of them.

| directory | study | produced by |
|---|---|---|
| `sampler/` | sampler and lattice validation | `validate_sampler_z2.py`, `validate_sampler_su2.py`, `validate_anisotropy.py` |
| `wilson_regression/` | per-site Wilson-loop regression (main.tex §"Validation and tests") | `train_gelt.py`, `train_cnn.py`, `train_lcnn.py` |
| `glueball/` | 0⁺⁺ glueball spectroscopy (main.tex §"SU(2) Glueball spectroscopy") | `measure_glueball.py`, `train_glueball.py`, `fit_glueball_overlap.py`, `check_glueball_autocorrelation.py`, `operator_decomposition.py` |
| `attention/` | attention as a lattice operator (main.tex §"Attention as a physical field") | `z2_beta_scan.py`, `train_z2_glueball.py`, `z2_attention_correlator.py`, `su2_attention_correlator.py` |
| `fair_fight/` | is the classical comparator a straw man? | `su2_fair_fight.py` |

## What main.tex is read off

- `attention/z2_attention_correlator_diag_R6.pt` — **Table `tab:train_rnd_gevp`**
  (R = 6, 1200 unseen configs per ensemble, four β). Regenerate the figure from
  it offline, no GPU: `ZAC_REPLOT=<that file> python scripts/z2_attention_correlator.py`.
- `attention/su2_attention_correlator_table.tex` — **Table `tab:su2_train_rnd_gevp`**,
  paste-ready.
- `glueball/*_test_obars.pt` (also in `dumps/`) — the per-configuration Ō(t) the
  cosh fits and every offline audit run on. **Table `tab:overlap`** / ΔA₀.
- `fair_fight/su2_fair_fight.pt` — ΔA₀ against every strengthened classical arm.
  Read it together with `notes/audit_2026-09-06.md` §6.4/§6.5: the headline
  survives against `deep`, and `full`/`shapes_sm` are numerically unusable.

## Two artifacts whose producing script is gone

Kept because something still reads or cites them, after the 2026-09-09 cleanup
removed the studies that wrote them (history at `cfa0a7e`):

- `attention/beta_scan.pt` — SU(2) classical masses on independent ensembles.
  `su2_attention_correlator.py` prints a cross-check against it and skips if it
  is absent.
- `attention/z2_attention_readout.pt`, `attention/glueball_attention*.png` — cited
  by `notes/attention_as_operator.md` §1 and included by `reports/glueball/`.

## Naming

`RUN_TAG` suffixes (`_ens1`, `_R6`, …) mark non-default seeds or radii so a
replication never overwrites the original run's artifacts.

## Report figures

The reports under `reports/` pull figures from here via

```latex
\graphicspath{{../../}{../../results/}{../../results/sampler/}{../../results/glueball/}{../../results/attention/}{../../results/wilson_regression/}}
```

so `\includegraphics{glueball_validation}` keeps working with a bare filename.
**If you move a figure between these directories, the reports still build** —
but if you rename one, fix the `\includegraphics` call too.
