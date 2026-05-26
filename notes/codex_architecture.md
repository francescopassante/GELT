# Architecture diagnosis and proposed direction

This note records the current diagnosis of the GELT architectures after the
Wilson-loop proxy experiments:

- the legacy fused `GEMHSA` learns `1x1`, `1x2`, `2x2`, `1x3`, `2x3`, but not
  `3x3` even with enough nominal depth;
- the split `GEAttention + GEFFN` architecture can fail earlier, e.g. already
  at `2x3`;
- the actual thesis goal is not rectangular Wilson-loop regression itself, but
  a gauge-invariant/equivariant network that can learn physically interesting
  quantities in `U(1)`, `SU(2)`, and `SU(3)`.

The Wilson-loop tasks should therefore be interpreted as a controlled diagnostic
for whether the architecture can build non-local gauge-invariant information
from local gauge-covariant fields. They are not the final physics benchmark, but
they probe a mechanism that many physical observables need: transport, local
composition, and invariant readout.

## 1. What the Wilson-loop proxy is testing

A rectangular Wilson loop is a clean target because it has a known algebraic
structure. In two-dimensional `Z2`, an `m x n` loop is exactly the product of
the `m n` plaquettes inside the rectangle. In non-abelian groups, the exact
construction is ordered in link space rather than a scalar plaquette product,
but the learning problem still tests the same architectural abilities:

1. move gauge-covariant local features between sites by parallel transport;
2. keep all features anchored at a common site so they transform under the same
   local gauge matrix;
3. multiply/combine partial transported objects without breaking gauge
   covariance;
4. reduce them through traces or other class functions to gauge invariants.

These are not special only to Wilson loops. Local action density, extended
operators, smeared observables, glueball-like correlators, Polyakov-loop
features, topological-charge proxies, and other physical quantities all depend
on some combination of transported local field content and gauge-invariant
composition. Wilson loops are a deliberately sharp test of whether the network
has learned that algebra.

However, Wilson-loop regression is also harsher than many final observables.
Many physical quantities are sums, averages, correlations, or smooth local
density functionals. They may not require exact reconstruction of one
corner-anchored `3x3` loop. So failure on `3x3` does not prove the current model
is useless for every physical observable. It does show that the model has not
yet learned a scalable, reliable loop-building mechanism.

## 2. Legacy fused GEMHSA: expressive but hard to optimize

The legacy block computes attention and bilinear multiplication in one fused
value path:

```text
V_weighted(x) = sum_y alpha(x,y) T_xy V(y) T_xy^dagger
out(x)        = Q(x)^dagger V_weighted(x)
```

In the code, this is the value path of `GEMHSA.attend`: attention first produces
`V_weighted`, then the block multiplies by `Q_dag`.

This has a real expressivity advantage: one block can both route a neighboring
covariant feature to `x` and multiply it with a local covariant feature. That is
why it can learn loops up to `2x3`. It has a path to compositional loop growth.

The problem is optimization. Each layer has to learn four coupled decisions at
once:

- which offset the softmax should attend to;
- which channel projection should become `Q`;
- which channel projection should become `V`;
- how `w_mix` should preserve the partial loop for later layers.

For small loops this is still trainable. For `3x3`, the target has plaquette
degree 9 in `Z2`, so it needs at least four multiplicative composition stages
under a loop-doubling picture. At that depth, the gradient signal that tells an
early layer which offset and channel will be useful three layers later is weak
and highly entangled. The architecture can plausibly represent the target, but
SGD has to discover a deep compositional circuit through a strongly coupled
routing/multiplication landscape.

This is the main diagnosis for the `3x3` wall: not a simple receptive-field
failure and not necessarily a theorem-level expressivity failure, but a poor
scaling of the optimization problem.

## 3. Split architecture: cleaner operators, but a new routing bottleneck

The split architecture separates the fused operation into:

```text
GEAttention:
    W_attn(x) = sum_y alpha(x,y) T_xy V(y) T_xy^dagger

GEFFN:
    W_ffn(x) = Q(x)^dagger V(x)
```

This is conceptually closer to a standard transformer: attention routes, the
FFN performs local nonlinear processing. It also matches the gauge-equivariant
logic of L-CNN more cleanly: a transport-like operation plus a local bilinear
operation.

But the current implementation also removes the most useful part of the fused
block: transported multiplication in the same step. `GEFFN` is purely local. It
can only multiply objects already colocated at the same anchor site. Therefore
each useful non-local product now requires a choreography:

1. attention must move the right partial object to site `x`;
2. the residual stream must keep that object in a usable channel;
3. the local FFN must multiply it with another compatible local object.

This can be harder than the fused block if attention is too narrow. The current
attention path has several bottlenecks:

- with few heads, there are few independent offset choices per layer;
- all value channels inside one head share the same attention distribution;
- softmax attention is positive and normalized, so it forms convex mixtures
  rather than arbitrary signed/algebraic transport combinations;
- the current offset set excludes the zero offset, so explicit self-routing is
  missing and the model relies on the residual stream for "do not move";
- when a later FFN multiplies mixtures, it creates many wrong cross-terms that
  the model must learn to cancel.

This explains why the split model can fail earlier than the fused model. It
solves one coupling problem but introduces a stronger colocation problem. It is
not enough to say "attention then FFN"; for gauge physics, the attention layer
must provide enough independent transported feature lanes for the FFN to
multiply.

## 4. Are the current architectures fundamentally flawed?

For exact scalable Wilson-loop construction, yes: in their current form both
architectures are not reliable scaling mechanisms.

The fused block is too coupled. The split block is too bottlenecked. Both rely
too heavily on the optimizer discovering an implicit algebraic circuit from
random initialization.

For general physical quantities, the answer is more nuanced.

The current architectures may still learn physical observables well if the
observable is dominated by local density, short-range correlations, broad
averages, or smooth functions of plaquette-level content. Examples include
simple Wilson action regression, local energy/action density, or observables
whose relevant signal is already visible in small loops and short transported
features. In these regimes, exact construction of a large corner-anchored loop
is not necessary, and the inductive bias may be sufficient.

But if the final physical target depends on extended coherent gauge structure,
large loops, long-range correlations, tunneling/topology-sensitive features, or
multi-scale non-local operators, the current failures are a warning. A network
that cannot reliably learn `3x3` rectangular loops in a controlled proxy is
unlikely to scale robustly to harder non-local physics in `U(1)`, `SU(2)`, or
`SU(3)` without stronger architectural support.

Therefore the Wilson-loop failure should not be interpreted as "the thesis idea
is invalid." It should be interpreted as "the architecture needs a more explicit
gauge-covariant feature algebra."

## 5. Proposed architecture: routed equivariant operator algebra

The better direction is to keep the transformer flavor, but make the loop- and
operator-building primitives explicit. The architecture should expose reliable
operations for:

1. transporting covariant features to a common anchor;
2. storing several independently transported features per site;
3. multiplying local covariant features;
4. reading out gauge invariants through traces and learned scalar heads.

Concretely, use a block with three branches:

```text
Input: W(x), covariant matrix-valued channels at each site

Branch A: algebraic transport mixing
    for each offset r in an L1 ball:
        W_r(x) = T_{x,r} W(x+r) T_{x,r}^dagger
    mix the transported channels with learned real/complex coefficients

Branch B: soft attention routing
    same transported values, but weighted by gauge-invariant attention scores
    this keeps the transformer/adaptive-routing component

Branch C: local bilinear/multilinear FFN
    build Q_i(x)^dagger V_j(x), and optionally Q_i^dagger V_j V_k
    from the current colocated channel bank

Residual update:
    W <- W + alpha_A A(W) + alpha_B B(W) + alpha_C C(W)
```

Important details:

- Include the zero offset. The model needs an explicit identity transport path.
- Use many independent routed lanes. Heads should correspond to independent
  offset/channel routes, not only to a wider value dimension sharing one
  softmax.
- Keep softmax attention, but do not make it the only transport mechanism.
  Physics operators often need signed/algebraic sums, not only convex
  averaging.
- Separate routing and multiplication, but allow enough transported channels to
  be present before multiplication.
- Consider local multilinear FFNs, e.g. `Q^dagger V V2`, for faster growth of
  operator degree when needed.
- Consider constructive or near-constructive initialization for the algebraic
  transport branch: start with basis-like offset routes rather than asking
  attention to discover all routes from scratch.

This is not abandoning the transformer idea. It is replacing vanilla NLP-style
token mixing with a gauge-covariant token algebra. Attention remains the
adaptive part, but exact transport/multiplication primitives carry the physical
inductive bias.

## 6. Why this is fundamentally better

The proposed direction is better than the fused `GEMHSA` because routing and
multiplication no longer have to be learned as a single entangled decision. A
transport branch can learn "which neighboring covariant features should be
available at this site" while the bilinear branch learns "which local features
should be multiplied." The gradients become more local in function space.

It is better than the current split architecture because colocation is no
longer limited to a small number of softmax mixtures. The model gets an
explicit bank of transported features over offsets/channels, including self.
The FFN can multiply actual transported objects rather than mixtures that
contain many unwanted cross-terms.

It is better for physics because it mirrors the algebra of lattice gauge
observables:

- gauge covariance is preserved at every hidden layer;
- gauge invariance is produced only by class-function readouts such as traces;
- non-locality is built by parallel transport, not by gauge-breaking flattening;
- composition is done by matrix products at a common anchor, matching how
  Wilson lines, loops, and extended operators are formed.

The expected scaling behavior is therefore different. The current models ask
SGD to invent a hidden symbolic construction. The proposed model gives SGD a
basis of legal gauge-covariant constructions and asks it to select and combine
them.

## 7. Experimental checks

To validate the diagnosis, run these ablations before committing to a large
rewrite:

1. Add the zero offset to the attention/transport table and test whether the
   split model improves on `2x3`.
2. Increase `nhead` substantially at fixed or moderately increased `d_qkv`.
   If the issue is independent routing lanes, heads should matter more than
   only value width.
3. Add a parallel non-softmax transport-mixing branch and compare against pure
   softmax attention.
4. Test patterns with more routing before multiplication, e.g. `AAFAAFAF`, and
   more multiplication after a transport bank, e.g. `AFFAFF`.
5. Build a constructive unit test on `Z2`: with fixed/manual parameters, verify
   whether the architecture can exactly build `2x3` or `3x3` from plaquettes.
   If a hand construction is impossible or extremely brittle, training will not
   reliably find it.
6. Evaluate physics tasks separately from Wilson-loop proxies:
   - local Wilson action or action density;
   - small-loop averages;
   - correlation functions at increasing separations;
   - topology-sensitive or tunneling-sensitive observables where available.

The final decision should not be based only on rectangular loops. But larger
Wilson loops are a valuable stress test: they reveal whether the network has
learned a scalable gauge-covariant operator algebra, which is exactly the kind
of machinery needed for non-local physics.
