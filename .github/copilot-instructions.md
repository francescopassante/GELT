# Copilot instructions for `lattice-gauge-theory`

## Build, test, lint, and run commands

This repository currently does **not** define a formal build system, lint configuration, or automated test suite (no `pyproject.toml`, `requirements*.txt`, `pytest`/`tox` config, or `Makefile` checked in).

Use the script entry points that exist in the codebase:

```bash
python main.py            # L-scan training driver (runs training across multiple L values)
python model.py           # prints torchsummary for the baseline CNN
python visualize.py       # plots a seeded random 2D Z2 lattice
python L_scan_plots.py    # regenerates saved baseline plots
```

Single-test command: **not available yet** (no test suite is present in the repository).

## High-level architecture (current state)

The implemented pipeline is a **baseline CNN regression workflow** for lattice gauge configurations:

1. `lattice.py` defines gauge-group primitives (`GaugeGroup`, `Z2`) and pure tensor operations for link sampling, plaquette construction, Wilson action, and ML channel flattening.
2. `data.py` builds datasets from lattice tensors (`build_link_datasets`, `build_plaquette_datasets`), computes labels via `action`, and returns train/val/test splits.
3. `model.py` contains `LatticeCNN`, a circular-padded 2D ConvNet baseline (reference model only; not gauge-equivariant).
4. `train.py` wires data + model into training/evaluation (`full_pipeline`) with early stopping and checkpoint loading, returning a `TrainResult` with MSE, label variance, and R².
5. `main.py` runs the L-scan experiment loop across lattice sizes and saves per-L checkpoints.
6. `L_scan_plots.py` visualizes previously saved baseline metrics and normalizes by analytic label variance.

Design documents (`gauge_invariant_NN_review.md`, `architecture.md`, `roadmap.md`) describe the **target** gauge-equivariant G-GAT architecture and phased research plan; this architecture is not yet implemented in code.

## Key conventions specific to this repo

- The codebase uses **pure tensor operations** (the old OO lattice wrappers are intentionally removed).
- Canonical tensor layouts:
  - Links: `(D, L, ..., L, nc, nc)`
  - Plaquettes: `(D(D-1)/2, L, ..., L, nc, nc)` with `(mu, nu)` and `mu < nu` in lexicographic order.
- Keep color axes `(nc, nc)` **even for Z2 (`nc=1`)**; avoid shape-special-casing for abelian debug runs.
- Use explicit matrix algebra everywhere: `A @ B` for products and `group.dagger(...)` for inverses/conjugate-transpose.
- Periodic boundary conditions are implemented with `torch.roll`; do not replace with manual modulo indexing.
- Wilson action convention is fixed: `S = beta * sum_p(1 - Re Tr(P_p)/nc)`.
- Reproducibility is seed-driven (`torch.Generator` and explicit `seed` threading through data/training entry points).
- `LatticeCNN` is intentionally **2D-only** (`Conv2d`, raises for `D != 2`) and serves as a non-equivariant baseline.
- Current datasets are Haar-random link configurations; a Metropolis/heat-bath MC data path is still a pending milestone.
