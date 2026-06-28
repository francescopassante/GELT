# Glueball spectroscopy — GELT as a learned variational operator

A plan for moving GELT from per-configuration regression toward **glueball
spectroscopy**, with GELT trained to **maximize overlap with the glueball
ground state**. The central claim: this is the cleanest, most principled use
of the architecture — the loss value at convergence *is* the physics answer
(the glueball mass), the trained network *is* the optimal interpolating
operator, and the attention map becomes a measurement of what spatial loop
structure that operator uses. It lands directly on the thesis spine in
`notes/explainability.md` ("the attention map is a measurement"), applied to a
real, hard observable instead of a toy target.

## Status & findings (updated 2026-06-28)

**The classical baseline (§6.1) is built and was run; a clean ground-state
plateau is *not yet* resolved with a single operator.** What exists:

- `gelt/glueball.py`: operator (`glueball_operator`), spatial APE smearing
  (`ape_smear`), connected vacuum-subtracted correlator (`connected_correlator`),
  `effective_mass`, and `jackknife_effective_mass` — plus the **multi-level GEVP**
  added 2026-06-28: `smearing_operator_basis`, `connected_correlator_matrix`,
  `gevp_eigenvalues` (robust eigh-whitening with an eigenvalue floor, not
  Cholesky — low statistics can make C(t₀) indefinite), `gevp_effective_mass`,
  `jackknife_gevp_effective_mass`. All covered by `tests/test_glueball.py`
  (gauge covariance/invariance, synthetic mass recovery, two-state GEVP recovery).
- The SU(2) heat-bath + overrelaxation sampler and the `τ_int` diagnostic
  (`check_glueball_autocorrelation.py`, smeared-operator τ_int ≈ 2 → `n_skip = 5`).
- `scripts/measure_glueball.py`: four-panel figure, now with the GEVP ground
  state overlaid; caches the sampled ensemble under `datasets/` so the (cheap)
  GEVP analysis can be re-tuned without re-running the (expensive) sampling.

**The run (`L=12 D=4 β=2.4 N=2000`, heat-bath+OR, acc 1.00):** code checks pass
(synthetic mass recovered, ⟨O⟩ rises with smearing). But the *physics* is
**marginal** — the single smeared operator's `m_eff(Δ)` shows only a weak signal
around **m·a ≈ 0.8 at Δ = 1–2**, then slides into noise (negative `m_eff`,
huge bars) by Δ ≈ 3. `C(Δ)` hits its O(1) noise floor by Δ ≈ 3, so there is no
late-Δ signal to plateau on. This is the textbook 0⁺⁺ problem (§7): the glueball
is heavy, the signal dies in ~3 slices, and more statistics barely helps
(Lepage). **The lever is operator overlap, not N** — hence the GEVP basis.

**Next step:** re-run `measure_glueball.py` (it will load the cached ensemble)
and check whether the multi-level GEVP ground state plateaus earlier/lower than
the single smeared operator. If it gives a credible `m_G`, that is the
ground-truth number to validate GELT against (§6.2). If still marginal, widen
the basis (more/larger smearing levels, R×T loop shapes), try a smaller β / finer
lattice where m·a is smaller, or add stout-smeared loops. Only then train GELT.

## 0. Where we are vs. what spectroscopy needs

Everything built so far is **per-configuration regression toward a known
scalar function** — action, `Q`, Wilson loops, with a label `y` to fit.
Glueball spectroscopy is a different object: there is **no label**. You
measure a zero-momentum-projected temporal correlator and extract a mass from
its exponential decay. The lightest channel is the scalar **0⁺⁺**, built from
spatial Wilson loops.

The gap is three things the codebase does not yet have:

1. **A real 3+1D ensemble with a trustworthy time axis.** 4D SU(N) targets and
   a Metropolis sampler exist, but spectroscopy needs many well-decorrelated
   configs at a coupling where a mass is resolvable. Metropolis critical
   slowing fights this — heat-bath + overrelaxation (`notes/sampling.md`) is
   the prerequisite long pole.
2. **Connected correlator measurement + plateau fitting** — does not exist (the
   only `C(t)` in the repo is the plaquette *autocorrelation* in
   `validate_sampler.py`, a sampler diagnostic, not a physics correlator).
3. **Operator construction & smearing** — glueball signals are notoriously
   noisy; no cooling/smearing exists yet (also flagged in `fable_audit.md`).

## 1. The variational principle (why this is doable and clean)

A glueball operator is a gauge-invariant scalar field `O(x, t)`. Project to
zero momentum by summing over the **spatial** slice at fixed time:

```
Ō(t) = Σ_{x⃗} O(x⃗, t)        (sum over spatial axes only — keep the time axis)
```

Form the **vacuum-subtracted** connected correlator, averaged over time
origins `t₀` for statistics (time-translation invariance):

```
C(Δ) = ⟨ Ō(t₀+Δ) Ō(t₀) ⟩ − ⟨Ō⟩²  =  Σ_{n>0} |⟨0|Ō|n⟩|² e^{−m_n Δ}
```

The effective mass `m_eff(Δ) = log[C(Δ)/C(Δ+1)] → m_G` as `Δ→∞`, and
`m_eff(Δ) ≥ m_G` for all `Δ` (variational upper bound under reflection
positivity). The punchline:

```
R = C(1) / C(0) = ⟨Ō T̂ Ō⟩_c / ⟨Ō Ō⟩_c
```

is a **Rayleigh quotient of the transfer matrix** `T̂ = e^{−Ĥ}` in the state
`Ō|0⟩` (vacuum removed). Over *all* operators its maximum is `e^{−m_G}`, the
largest non-vacuum eigenvalue — the lightest glueball. The maximizer is the
optimal-overlap operator. Hence the loss:

> **Loss = −C(1)/C(0)** (minimize), and at the optimum `m_G = −log R`.

The loss value *is* the mass; the trained network *is* the operator.
Unsupervised, variational (a rigorous upper bound on `m_G`), and the answer
falls out of the converged loss.

Two structural gifts:

- The quotient is **scale-invariant** (`O → λO` cancels), so the output need
  not be normalized.
- Because the scalar 0⁺⁺ shares the vacuum's quantum numbers, after
  subtraction the lightest surviving state *is* the 0⁺⁺ glueball — so for the
  ground state the cubic-group / J^PC projection can be **deferred**. It is
  only needed for excited states or other channels.

## 2. Architecture change (minimal)

GELT already emits the right object. With `reduction="none"`,
`GELT.forward` returns a per-site gauge-invariant scalar field
`O(x)` of shape `(B, *Λ)` (`blocks_rope.py:668-670`) — an operator density.
No new head is needed. Instantiate `GELT(..., reduction="none")` and do the
zero-momentum projection in the training loop.

## 3. Training loop (Rayleigh loss)

```python
# batch U: B configs of a 4D ensemble; axis order (B, D, *Λ, nc, nc),
#          Nt = Λ[time axis], periodic in time.
O    = model(W, T)                     # (B, *Λ)  per-site invariant field
Obar = O.sum(dim=spatial_axes_only)    # (B, Nt)  zero-momentum proj per timeslice

# vacuum subtraction + time-origin averaging, estimated over the batch
mu   = Obar.mean()                     # ⟨Ō⟩ — 0⁺⁺ has a NONZERO VEV; must subtract
d    = Obar - mu
C0   = (d * d).mean()                              # connected variance
C1   = (d.roll(-1, dims=1) * d).mean()             # one-step, summed over all t₀
R    = C1 / (C0 + eps)
loss = -R                                          # ⇔ minimize m_eff(0→1)
```

At convergence `m_G ≈ −log(R)`. Cross-check by computing `m_eff(Δ) =
log[C(Δ)/C(Δ+1)]` at larger `Δ` and confirming it plateaus to the same value.

## 4. Pitfalls — where it is actually hard

1. **Ratio-estimator bias.** `C(1)/C(0)` is nonlinear in batch-estimated
   means → systematically biased gradient on small batches. Use large
   batches, accumulate, and validate the final number with a **jackknife**
   over an independent ensemble. This is the single biggest engineering risk.
2. **Vacuum subtraction is essential and noisy.** Unlike `Q` (zero VEV), the
   0⁺⁺ operator has nonzero `⟨Ō⟩`; without subtraction you measure the vacuum,
   not the glueball — and `⟨Ō⟩` is itself a noisy batch estimate inside the
   loss.
3. **Constant-operator collapse.** If `O(x)` becomes config-independent,
   `C(0)→0` and `R` diverges (`0/0`). Guard with `eps`; monitor `C(0)`; add a
   variance floor / penalty if it drifts toward zero.
4. **Ensemble requirement.** Needs well-thermalized, decorrelated 4D **SU(2)**
   configs at a coupling where a mass is resolvable. Metropolis critical
   slowing hurts; heat-bath + overrelaxation is the prerequisite long pole.
   Glueball SNR is the field's classic hard problem — but the `R = C(1)/C(0)`
   anchor lives at **small** Δ where SNR is *best*, which is exactly why this
   formulation is the tractable one.

## 5. Validation (the proof, and the thesis result)

Measure the *same* `m_eff(Δ)` curve for a **plain plaquette operator** (and an
APE-smeared loop) on the same ensemble, classically. Two things must hold, and
together they are the proof:

- GELT's plateau value **agrees** with the classical asymptotic `m_G`
  (meaningful because the variational bound makes it an upper bound — agreement
  is not luck).
- GELT's `m_eff(Δ)` plateaus **earlier / lower at small Δ** than the
  plaquette's — i.e. it found a better operator (higher ground-state overlap).
  That earlier plateau *is* the win, and the L-CNN baseline can be made to
  compete on the same quantity (matched-parameter shootout).

Thesis payoff: the trained operator's **attention map shows what spatial loop
structure the optimal glueball operator attends to** — "attention as
measurement" on a real observable.

## 6. Deliverables, in order

1. **✅ Classical 0⁺⁺ correlator + `m_eff` extraction** — *built* (reuses
   `rectangular_wilson_loop`): timeslice-summed scalar operator, spatial APE
   smearing, connected vacuum-subtracted `C(Δ)`, `m_eff(Δ)` + jackknife on an
   SU(2) heat-bath ensemble. **Outcome (see Status block):** the single-operator
   `m_eff` does *not* yet plateau cleanly (weak m·a ≈ 0.8, drowns by Δ ≈ 3).
1b. **✅ Multi-level smearing GEVP (classical)** — *built* 2026-06-28: a
   variational basis of operators at several APE levels (`smearing_operator_basis`),
   correlator matrix (`connected_correlator_matrix`), robust GEVP solver
   (`gevp_eigenvalues`), per-state `m_eff` + jackknife. This is the standard fix
   (Morningstar–Peardon) for the §6.1 overlap problem; the immediate task is to
   **read off a credible ground-state `m_G` from the GEVP** to anchor everything
   below. (Distinct from deliverable 3, which is the *learned* GEVP inside GELT.)
2. **`train_glueball.py`** — `GELT(reduction="none")` + Rayleigh loss +
   jackknife eval; compare GELT vs. plaquette vs. L-CNN `m_eff` curves.
   *Gated on getting a trustworthy classical `m_G` from deliverable 1b.*
3. **(extension) Multi-operator GEVP inside GELT** — network emits a *vector* of
   operators → generalized eigenproblem → excited states / other J^PC channels
   (needs the cubic-group projection deferred in §1). The learned analogue of
   deliverable 1b's hand-built smearing basis.

## 7. Smearing — the crucial enabler, and what it means for GELT

Smearing is **not optional** for the classical baseline; it is the single
technique that makes glueball spectroscopy work at all, and §5's passing
mention of an "APE-smeared loop" undersold it. The reason is signal-to-noise.

**Why thin-link operators fail.** The connected correlator's signal decays as
`C(Δ) ~ e^{−m_G Δ}`, but its statistical variance is set by vacuum fluctuations
and is roughly **Δ-independent**, so the relative error grows like
`e^{+m_G Δ}/√N` (the Lepage argument; this is the §4 SNR pitfall sharpened).
The 0⁺⁺ is heavy (`m_G·a` is order 1 on typical lattices), so the signal is
gone within a couple of time slices. You *must* read the `m_eff(Δ)` plateau at
**small Δ** — there is no late-Δ signal to wait for. A thin plaquette operator
overlaps poorly onto the ground state (dominated by UV fluctuation, couples to
high-lying states and lattice artifacts), so its `m_eff(Δ)` starts far too high
and descends only slowly — past the point where the signal has drowned. You
never reach the plateau.

**What smearing does.** Spatial smearing (APE or stout: replace each spatial
link by a projected sum of itself and its staples, iterated; reuse
`staple_sum`) builds **spatially extended** operators whose size matches the
physical glueball wavefunction. That raises ground-state overlap so the plateau
appears at small Δ where SNR is still alive. The modern glueball spectrum
(Morningstar–Peardon and successors) rests on a *variational basis* of loops at
several smearing/blocking levels solved by GEVP. Smear **spatial links only** —
never in time, or you corrupt the transfer-matrix / spectral interpretation in
§1.

**So is it needed for GELT?** The nuance — and it is a thesis selling point:

- **GELT is, in part, a *learned* smearing.** Its L1-ball transport-averaging
  plus stacked attention is a gauge-covariant, multi-scale, content-dependent
  smearing, and the variational loss (§1) optimizes overlap *directly* —
  exactly what hand-tuned smearing approximates. Honest framing: "the network
  learns its own glueball operator (its own smearing) instead of us tuning APE
  steps by hand." A cleaner story than bolting smearing on.
- **But not a free lunch — receptive field is the GELT analogue of smearing
  level.** GELT can only build an operator as extended as `R` (L1-ball radius)
  × depth allows. If `R`×depth is smaller than the physical glueball size, no
  training reaches the plateau — the same failure as an under-smeared classical
  operator. The receptive-field budget *is* the "how much smearing" knob, and
  it is the same memory gate as offset-chunked attention in `fable_audit.md`.
- **Pre-smeared input is a sensible warm start.** GELT's `W` and `T` are built
  from **thin**, UV-noisy links. Feeding it stout-smeared links instead
  (smearing is gauge-covariant preprocessing; stout is differentiable, so it
  can even be folded in as trainable layers, CASK-style — see
  `papers_review.md`) hands the network a cleaner, better-conditioned input so
  it need not learn UV-smoothing from scratch. De-risks the optimization
  without touching the variational principle.
- **Multi-scale basis ↔ multi-operator GEVP.** The classical "several smearing
  levels" basis maps onto the §6 deliverable-3 GELT extension: emit a *vector*
  of operators at different effective sizes (different layers/heads) and solve
  the GEVP — the learned analogue of a multi-smearing-level variational basis.

**Smearing ≠ cooling.** Distinct from the topological-charge cooling in
`fable_audit.md`: cooling is many sweeps that *flow the config toward classical
solutions* to expose topology (it changes the physics). Operator smearing is a
few APE/stout steps tuned purely to maximize ground-state *overlap* (it leaves
the ensemble alone). Both are built from staples, but used for different ends —
keep them separate.

## 8. Prerequisite long poles (independent of the network)

- **Heat-bath + overrelaxation SU(2) sampler** (`notes/sampling.md`): without
  it there is no usable 4D ensemble. Biggest single cost.
- **Spatial APE/stout smearing** (reusing `staple_sum`): the crucial enabler
  for the classical baseline and operator overlap — see §7 for the full role
  (including why GELT only partly replaces it); also a `fable_audit.md` item.

## 9. Verdict

Doable and well-posed: the math is clean, the architecture already emits the
right object, and the loss is a one-screen change. The honest costs are
exactly two — the **heat-bath sampler** (no ensemble without it) and the
**ratio-estimator bias/noise** of the loss. Neither is a showstopper; both are
real work. Build the classical baseline (§6.1) before involving the network.
