# Explainability — why GELT over a CNN

## The core argument

A CNN can match GELT's *receptive field* by adding depth or kernel size — that's
a parameter tradeoff, not a real difference. The one thing a CNN **cannot** do at
any budget: its weights are constants, the same for every configuration. GELT's
attention `α_{x→n}(U)` is computed *from the configuration*, so it's a
content-dependent, per-site aggregation map.

So the thesis spine isn't "transformer beats CNN on MSE" (a coin-flip on local
targets). It's: **the attention map is a measurement.** It exposes physical
structure the network discovers without supervision — and no convolutional
weight can provide that.

Crucially, this only works because the attention score `Re Tr[Q†K̃]` is
**gauge-invariant**: whatever structure you read off is physical, not a gauge
artifact. A vanilla transformer's attention on raw links would be gauge-dependent
noise. Equivariance is what makes the attention interpretable at all.

The honest catch (Jain & Wallace, "Attention is not Explanation"): a big weight
doesn't prove the output depends on it. The fix is *intervention* (ablate, see
if the prediction moves). And physics gives what NLP can't: **ground truth** —
you know where the instantons are, you know ξ(β). So you can *validate* attention
against a known field. Gauge theory is a better testbed for honest attention-
interpretability than language.

## The interesting results

1. **Emergent correlation length.** Train on a ξ-coupled target; show the
   attention decay length `ℓ_att(β)` tracks the MC correlation length `ξ(β)` —
   an unsupervised discovery of a physical scale. Strong version: does `ℓ_att`
   peak at `β_c` the way ξ does? ("The network's gaze critically slows down.")

2. **Spatial localization.** Reduce attention to a per-site saliency and show it
   concentrates on topological lumps — overlay on `q(x)`. The lattice analogue of
   "the ViT attends to the object," but validated against the true `q(x)`.

3. **Head specialization / layer-wise coarsening.** Different heads = different
   physical roles (UV smoother vs IR / topology head). Attention range growing
   with depth = a learned RG-like coarsening (vs a CNN's, which grows trivially
   by construction).

Caveat tying 1–2 together: interpretability only appears on targets that *couple*
to the relevant scale. On a purely local observable the attention collapses to
nearest-neighbor and there's nothing to see — same coin as "local tasks tie on
MSE." You harvest the maps on the tasks where attention is forced to spread.

## How to actually do it (brief)

- **Extract α** per block from a forward pass (add a hook in `blocks_rope.py`).
- **ℓ_att study:** measure the decay of `|α|` vs offset. R must exceed ξ or it
  gets clipped — but ξ in lattice units is only ~1–4, so R≤5 suffices. Do it in
  3D Z₂ (ξ tunable via β, large R cheap); confirm in 4D SU(2) at R≤4. Cheap
  proxy if R is too slow: sweep R=1..4, find where MSE saturates, correlate that
  with ξ(β).
- **Localization:** reduce α to a per-site scalar `S(n) = Σ_{l,h,x} α^{(l,h)}_{x→n}`
  ("incoming attention mass"), correlate `S(n)` with cooled `|q(n)|` (Spearman,
  vs a shuffled-q null). For end-to-end attribution use attention rollout
  `Π_l(ᾱ^{(l)}+I)`; for rigor, **ablate** attention onto high-q sites and check
  the predicted Q drops by the removed charge.

## Abstract-ready framing

> Equivariant convolution and equivariant attention are near-equivalent
> *predictors* on local observables, by design. They differ in one structural
> respect — attention's aggregation is a gauge-invariant, configuration-resolved
> function of the field — and we exploit that as a *measurement*: the attention
> map exposes physical structure (correlation length, topological lumps, RG-like
> coarsening) that the network discovers without supervision, validated against
> known lattice ground truth. No convolutional weight can provide this at any
> parameter budget.
