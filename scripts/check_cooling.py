"""Pre-flight for the topological-localization study: how much cooling?

``n_cool`` is the one free parameter of the ground-truth field, and it is not
free to guess. Too little and q(x) is still ultraviolet noise — the total
charge Q sits nowhere near an integer and "topologically rich region" means
nothing. Too much and cooling starts annihilating instanton–anti-instanton
pairs and drives Q to zero, destroying the very structure the study wants to
correlate against. The usable window is the plateau in between.

This script measures that window on a real thermalised SU(2) ensemble: it
cools in unit steps and tracks, at each step, the Wilson action (which must
descend monotonically), the mean distance of Q from the nearest integer (which
must fall), and the individual Q trajectories (which should flatten into
integer plateaus, then decay as pairs annihilate).

Read the printed table, pick n_cool in the plateau, and use it for the
localization study.

Run:
    python scripts/check_cooling.py

Writes ``cooling_validation.png``.
"""

import functools
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from gelt.lattice import SU, action, topological_charge_density
from gelt.sampler import heatbath_overrelaxation_sweep, mcmc_ensemble
from gelt.topology import cool

# Output artifacts are grouped by study under results/; create the dirs the
# first time this runs in a fresh clone (they hold generated files only).
for _d in ("results/sampler", "results/glueball", "results/attention",
          "results/wilson_regression", "datasets"):
    os.makedirs(_d, exist_ok=True)



# ── Tunables ──────────────────────────────────────────────────────────────────
L = 8  # isotropic 4D lattice: 8^4 = 4096 sites (topology needs room for a lump)
D = 4  # topological charge is defined only in D=4
BETA = 2.4  # SU(2) scaling window; the debug default beta=1 has no topology
N_CONFIGS = 16  # enough Q trajectories to see the plateau; this is a diagnostic
N_THERM = 200
N_SKIP = 5
N_OR = 4
MAX_COOL = 40  # cool this far to see BOTH the plateau and the eventual decay
COOL_ALPHA = 0.5

CACHE = f"datasets/topo_configs_L{L}_D{D}_b{BETA}_N{N_CONFIGS}.pt"

gaugegroup = SU(2)
# cuda → cpu: the group projection inside cooling is a complex SVD, which MPS
# cannot do (same reason beta_scan.py skips it).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensemble():
    if os.path.exists(CACHE):
        print(f"Loading cached {CACHE}")
        return torch.load(CACHE)
    print(f"Sampling N={N_CONFIGS} at L={L}^4, β={BETA} …")
    sweep = functools.partial(heatbath_overrelaxation_sweep, n_or=N_OR)
    configs, acc = mcmc_ensemble(
        L=L, D=D, gaugegroup=gaugegroup, beta=BETA, n_configs=N_CONFIGS,
        n_therm=N_THERM, n_skip=N_SKIP, sweep_fn=sweep, progress=True,
    )
    print(f"  acceptance = {acc:.2f}")
    os.makedirs("datasets", exist_ok=True)
    torch.save(configs, CACHE)
    print(f"  cached → {CACHE}")
    return configs


def main():
    print(f"device: {device}")
    U = ensemble().to(torch.complex128).to(device)

    # Cool in unit steps, recording the diagnostics after each one.
    steps, actions, devs, Qs = [], [], [], []
    for n in range(MAX_COOL + 1):
        if n > 0:
            U = cool(U, gaugegroup, n_steps=1, alpha=COOL_ALPHA)
        Q = topological_charge_density(U, gaugegroup).flatten(start_dim=1).sum(dim=1)
        S = action(U, gaugegroup, beta=1.0).mean()
        steps.append(n)
        actions.append(S.item())
        devs.append((Q - Q.round()).abs().mean().item())
        Qs.append(Q.cpu().numpy())
        if n % 5 == 0 or n == MAX_COOL:
            print(f"  n_cool={n:3d}   S/plaq={S.item():10.2f}   "
                  f"⟨|Q−round Q|⟩={devs[-1]:.4f}   ⟨|Q|⟩={np.abs(Qs[-1]).mean():.3f}")

    Qs = np.array(Qs)  # (MAX_COOL+1, N_CONFIGS)
    best = int(np.argmin(devs))
    print(f"\nFlattest Q at n_cool = {best} (⟨|Q−round Q|⟩ = {devs[best]:.4f}).")
    print("Pick n_cool in the plateau — where the deviation has fallen but ⟨|Q|⟩")
    print("has not yet decayed towards zero (pair annihilation).")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(steps, actions, "o-")
    ax[0].set_xlabel("cooling steps")
    ax[0].set_ylabel("Wilson action (β=1)")
    ax[0].set_title("Action descends monotonically\n(the defining property)")
    ax[0].set_yscale("log")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(steps, devs, "o-")
    ax[1].axvline(best, ls=":", color="red", label=f"min at {best}")
    ax[1].set_xlabel("cooling steps")
    ax[1].set_ylabel(r"$\langle |Q - \mathrm{round}\,Q| \rangle$")
    ax[1].set_title("Q migrates towards integers\n(q(x) becomes a topological field)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    for c in range(Qs.shape[1]):
        ax[2].plot(steps, Qs[:, c], alpha=0.7, lw=1)
    ax[2].axhline(0, color="k", lw=0.5)
    for k in range(-3, 4):
        ax[2].axhline(k, color="gray", ls=":", lw=0.5)
    ax[2].set_xlabel("cooling steps")
    ax[2].set_ylabel("Q per configuration")
    ax[2].set_title("Q trajectories: integer plateaus,\nthen pair annihilation")
    ax[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"Cooling pre-flight — SU(2), {L}⁴, β={BETA}, N={N_CONFIGS}", fontsize=13
    )
    fig.tight_layout()
    fig.savefig("results/attention/cooling_validation.png", dpi=130, bbox_inches="tight")
    print("Saved results/attention/cooling_validation.png")


if __name__ == "__main__":
    main()
