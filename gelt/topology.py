"""Cooled topological charge density — the ground truth for the localization study.

``lattice.topological_charge_density`` computes the naive plaquette charge
density ``q(x)`` directly from the thin links. On a thermalised ensemble that
field is useless as ground truth: it is dominated by ultraviolet fluctuation,
its total ``Q = Σ_x q(x)`` sits nowhere near an integer, and "topologically
rich region" is not a statement one can make about it. The standard remedy is
to remove the short-distance noise first, by cooling.

Cooling is APE smearing over *all four* directions: each link is replaced by
the projected weighted sum of itself and its staples, iterated. Because it
lowers the action monotonically towards the nearest classical solution, the
UV fluctuation is stripped while the instanton content — which is a local
minimum of the action, not a fluctuation about one — survives. The standard
diagnostic is that ``Q`` migrates towards integers as cooling proceeds; that
is what ``tests/test_topology.py`` asserts.

Unlike the spatial smearing of ``gelt.glueball``, cooling deliberately touches
the time links. That destroys the transfer-matrix interpretation and must
never be used for spectroscopy — but no temporal interpretation is at stake
when the object of interest is a static snapshot of the topological landscape.
"""

from typing import Tuple

import torch

from .glueball import ape_smear
from .lattice import GaugeGroup, topological_charge_density


def cool(
    U: torch.Tensor,
    gaugegroup: GaugeGroup,
    n_steps: int = 10,
    alpha: float = 0.5,
    progress: bool = False,
) -> torch.Tensor:
    """Cool an ensemble: APE smearing over every direction, ``n_steps`` times.

    Parameters
    ----------
    U : ``(B, D, *Λ, nc, nc)`` batched links.
    n_steps : cooling sweeps. The charge plateaus after ~10–20 on a small
        lattice; too many eventually annihilate instanton–anti-instanton pairs
        and drive ``Q`` to zero, so this is a tunable to scan, not to maximise.
    alpha : staple weight, as in ``ape_smear``.

    Returns
    -------
    Cooled links of the same shape.
    """
    return ape_smear(
        U,
        gaugegroup,
        alpha=alpha,
        n_steps=n_steps,
        progress=progress,
        directions=range(U.shape[1]),
    )


def cooled_charge_density(
    U: torch.Tensor,
    gaugegroup: GaugeGroup,
    n_steps: int = 10,
    alpha: float = 0.5,
    progress: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cool, then measure ``q(x)`` — the localization study's ground-truth field.

    Returns
    -------
    ``(q, Q)`` where ``q`` is ``(B, *Λ)`` per-site charge density and ``Q`` is
    ``(B,)`` the total charge per configuration. ``Q`` close to integers is the
    signal that cooling has done its job; report it alongside any localization
    correlation, since a non-integer ``Q`` means the field being correlated
    against is still noise.
    """
    Uc = cool(U, gaugegroup, n_steps=n_steps, alpha=alpha, progress=progress)
    q = topological_charge_density(Uc, gaugegroup)
    return q, q.flatten(start_dim=1).sum(dim=1)
