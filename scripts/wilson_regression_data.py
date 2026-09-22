"""Datasets for the 1+1D Wilson-loop regression of PRL 128, 032003 (Fig. 3).

What this builds: pure SU(2) gauge configurations on an 8 x 8 two-dimensional
lattice at a ladder of couplings, and for each of them the four per-site labels
the Letter regresses,

    W^(m x n)_{x,01} = Re Tr[ U^(m x n)_{x,01} ] / N_c ,   (m,n) in
    {(1,1), (1,2), (2,2), (4,4)},

exactly Eq. (12) of the Letter / Eq. (9) of the Supplemental Material. Time is
lattice axis 0 and 1+1D has a single plane, so ``mu, nu = 0, 1`` is the only
choice and the network's ``W`` input is the one plaquette channel
``D(D-1)/2 = 1``.

**The Monte Carlo is theirs** (SM SS I): Metropolis with the update
``U' = V U``, ``V = exp(i sum_a T^a X^a)``, ``X^a = A eta^a`` with ``eta^a``
standard normal and amplitude ``A = 0.5``, ten hits per link per sweep, and
``N_warmup = 2 x 10^3`` sweeps from a random start. The one thing done
differently is deliberate and is a strengthening, not a shortcut: they run *one*
chain per coupling and save a configuration every ``N_obs = 10^2`` sweeps,
whereas this runs one **independent chain per configuration**, all in a single
batched sweep (:func:`gelt.sampler.metropolis_sweep_multichain`). The samples
are then decorrelated by construction rather than by a skip interval, and the
whole ladder costs ``N_warmup`` sweeps instead of ``N_warmup + N_obs x N_cfg``.
``WR_CHAINS`` caps the batch when memory is the binding constraint, and the
shortfall is made up the paper's way — extra snapshots ``WR_SKIP`` sweeps apart
down each chain.

The correctness gate is not a convention check: 2D SU(2) is exactly solvable and
``<Re Tr P>/N_c = I_2(beta)/I_1(beta)``, which
``tests/test_sampler.py::test_multichain_metropolis_mean_plaquette_matches_exact_2d``
pins for this sampler. Read that before trusting any MSE produced downstream.

**The beta ladder.** SM Table I says ``beta in {0.1, ..., 6.0}`` and SM SS II says
the step is ``(beta_max - beta_min)/N_beta`` with ``N_beta = 10``. Those two
readings differ by one: ten steps between the stated endpoints is *eleven*
couplings. This takes the table's endpoints literally --- 11 values,
0.1, 0.69, ..., 6.0 --- so ``WR_N_BETA=11`` is the default and the per-beta counts
(910 / 91 / 91) put the totals at 10010 / 1001 / 1001 against their
10^4 / 10^3 / 10^3. ``WR_N_BETA=10`` drops the top coupling and makes the totals
exact; it is a knob because nothing here can decide which they ran.

Artifacts land in ``datasets/wilson1p1d/`` as ``{split}_L{L}.pt``. Links are
stored in complex128 and labels in float64 **regardless of the training dtype**:
the Letter's best MSE is 2.2e-11, i.e. errors of ~5e-6 on a quantity of order
one, and a 16-link W^(4x4) loop accumulated in complex64 is good to ~1e-6. The
label must not be the noise floor of the thing being measured. Training casts
down at load time.

Env overrides (each also accepted as ``--name=value`` on the command line):
``WR_L``, ``WR_N_BETA``, ``WR_BETA_MIN``, ``WR_BETA_MAX``, ``WR_N_TRAIN``,
``WR_N_VAL``, ``WR_N_TEST`` (per coupling), ``WR_WARMUP``, ``WR_HITS``,
``WR_AMPLITUDE``, ``WR_CHAINS``, ``WR_SKIP``, ``WR_SEED``, ``WR_DEVICE``,
``WR_TEST_SIZES`` (extra test-only lattice sizes, e.g. ``16,32,64`` --- the
Letter's generalisation check, since an L-CNN transfers to any volume without
retraining), ``WR_N_TEST_LARGE`` (total configurations per extra size).
"""

import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gelt.lattice import SU, plaquette_tensor, random_links, rectangular_wilson_loop
from gelt.sampler import _su2_proposal_paper, metropolis_sweep_multichain
from wilson_regression_common import (
    DATA_DIR,
    LOOPS,
    beta_ladder,
    cfg,
    pick_device,
    validate_argv,
)

GROUP = SU(2)
D = 2


def sample_configs(L, betas_per_config, warmup, n_hits, amplitude, device,
                   dtype=torch.complex128, progress_every=200):
    """Thermalised links, one independent chain per configuration.

    ``betas_per_config`` is a ``(B,)`` tensor: each chain carries its own
    coupling, so the whole ladder thermalises in one set of kernels.
    """
    B = betas_per_config.numel()
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    U = random_links(L, D, GROUP, dtype=real_dtype, N=B).to(device)
    betas = betas_per_config.to(device=device, dtype=real_dtype)

    t0 = time.time()
    acc = None
    for sweep in range(warmup):
        U, acc = metropolis_sweep_multichain(
            U, GROUP, betas, propose_fn=_su2_proposal_paper,
            n_hits=n_hits, epsilon=amplitude,
        )
        if progress_every and (sweep + 1) % progress_every == 0:
            el = time.time() - t0
            print(f"    sweep {sweep + 1}/{warmup}  "
                  f"{(sweep + 1) / el:.1f} sweeps/s  "
                  f"acc {acc.mean().item():.3f} "
                  f"[{acc.min().item():.3f}, {acc.max().item():.3f}]",
                  flush=True)
    return U, betas, acc


def advance(U, betas, n_sweeps, n_hits, amplitude):
    for _ in range(n_sweeps):
        U, _ = metropolis_sweep_multichain(
            U, GROUP, betas, propose_fn=_su2_proposal_paper,
            n_hits=n_hits, epsilon=amplitude,
        )
    return U


def labels_for(U):
    """The four per-site traced loops of Eq. (12), as a dict of ``(B, *Lam)``."""
    return {
        name: rectangular_wilson_loop(U, GROUP, R=m, T=n, mu=0, nu=1)
        for name, (m, n) in LOOPS.items()
    }


def collect(L, n_per_beta_total, betas_grid, warmup, n_hits, amplitude,
            max_chains, skip, device):
    """``(U, beta)`` for ``n_per_beta_total`` configurations at every coupling.

    Chains are capped at ``max_chains`` per coupling; whatever the cap leaves
    short is collected the paper's way, as further snapshots ``skip`` sweeps
    apart down the same chains. The two routes are interleaved so that the
    returned block is ordered ``(coupling, draw)`` -- the split downstream slices
    it per coupling, so train / val / test are balanced in beta by construction.
    """
    n_chains = min(max_chains, n_per_beta_total)
    n_draws = math.ceil(n_per_beta_total / n_chains)
    n_beta = betas_grid.numel()

    betas_per_config = betas_grid.repeat_interleave(n_chains)
    print(f"  {n_beta} couplings x {n_chains} chains "
          f"({n_beta * n_chains} configurations in flight), "
          f"{n_draws} draw(s) per chain, warmup {warmup} sweeps")
    U, betas, acc = sample_configs(
        L, betas_per_config, warmup, n_hits, amplitude, device
    )
    print(f"  acceptance per coupling: "
          + ", ".join(f"{b:.2f}:{a:.3f}" for b, a in
                      zip(betas_grid.tolist(),
                          acc.view(n_beta, n_chains).mean(dim=1).tolist())))

    draws = [U.view(n_beta, n_chains, *U.shape[1:]).cpu()]
    for d in range(1, n_draws):
        print(f"  extra draw {d}/{n_draws - 1}: {skip} decorrelation sweeps")
        U = advance(U, betas, skip, n_hits, amplitude)
        draws.append(U.view(n_beta, n_chains, *U.shape[1:]).cpu())

    # (n_beta, n_draws * n_chains, ...) → trim to the requested count
    out = torch.cat(draws, dim=1)[:, :n_per_beta_total]
    beta_out = betas_grid.view(n_beta, 1).expand(n_beta, n_per_beta_total)
    return out.reshape(-1, *out.shape[2:]), beta_out.reshape(-1)


def save_split(path, U, beta, meta):
    W = plaquette_tensor(U, GROUP)                 # (B, 1, L, L, 2, 2)
    payload = {
        "U": U.to(torch.complex128),
        "W": W.to(torch.complex128),
        "beta": beta.to(torch.float64),
        "labels": {k: v.to(torch.float64) for k, v in labels_for(U).items()},
        "meta": meta,
    }
    torch.save(payload, path)
    n = U.shape[0]
    means = {k: f"{v.mean().item():+.4f}" for k, v in payload["labels"].items()}
    print(f"  wrote {path}  N={n}  <W>={means}")


def main():
    validate_argv()
    L = int(cfg("WR_L", 8))
    n_beta = int(cfg("WR_N_BETA", 11))
    b_min = float(cfg("WR_BETA_MIN", 0.1))
    b_max = float(cfg("WR_BETA_MAX", 6.0))
    n_train = int(cfg("WR_N_TRAIN", 910))
    n_val = int(cfg("WR_N_VAL", 91))
    n_test = int(cfg("WR_N_TEST", 91))
    warmup = int(cfg("WR_WARMUP", 2000))
    n_hits = int(cfg("WR_HITS", 10))
    amplitude = float(cfg("WR_AMPLITUDE", 0.5))
    max_chains = int(cfg("WR_CHAINS", 4096))
    skip = int(cfg("WR_SKIP", 100))
    seed = int(cfg("WR_SEED", 0))
    device = pick_device(cfg("WR_DEVICE", None))
    extra_sizes = [int(s) for s in str(cfg("WR_TEST_SIZES", "")).split(",") if s]
    n_test_large = int(cfg("WR_N_TEST_LARGE", 1000))

    os.makedirs(DATA_DIR, exist_ok=True)
    torch.manual_seed(seed)

    betas = beta_ladder(b_min, b_max, n_beta)
    meta = dict(L=L, D=D, nc=2, betas=betas.tolist(), warmup=warmup,
                n_hits=n_hits, amplitude=amplitude, skip=skip, seed=seed,
                sampler="metropolis_multichain/paper_proposal", loops=LOOPS)

    print(f"device {device}   betas {[round(b, 4) for b in betas.tolist()]}")
    total = n_train + n_val + n_test
    print(f"L={L}: {total} configurations per coupling "
          f"({n_train} train / {n_val} val / {n_test} test), "
          f"{total * n_beta} in total")

    U, beta = collect(L, total, betas, warmup, n_hits, amplitude,
                      max_chains, skip, device)
    # Per coupling the block is contiguous, so the slices below keep every
    # split balanced across beta -- the paper's datasets are too (SM Table I).
    U = U.view(n_beta, total, *U.shape[1:])
    beta = beta.view(n_beta, total)
    cuts = {"train": slice(0, n_train),
            "val": slice(n_train, n_train + n_val),
            "test": slice(n_train + n_val, total)}
    for name, sl in cuts.items():
        save_split(os.path.join(DATA_DIR, f"{name}_L{L}.pt"),
                   U[:, sl].reshape(-1, *U.shape[2:]),
                   beta[:, sl].reshape(-1),
                   dict(meta, split=name))

    # The Letter also tests the *same trained model* on larger lattices, which
    # an L-CNN can do without retraining (translational equivariance + a per-site
    # head). Those sets are test-only.
    for Lx in extra_sizes:
        per_beta = max(1, n_test_large // n_beta)
        print(f"L={Lx}: test-only, {per_beta} configurations per coupling")
        Ux, bx = collect(Lx, per_beta, betas, warmup, n_hits, amplitude,
                         max_chains, skip, device)
        save_split(os.path.join(DATA_DIR, f"test_L{Lx}.pt"), Ux, bx,
                   dict(meta, L=Lx, split="test"))


if __name__ == "__main__":
    main()
