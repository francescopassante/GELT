# GELT — diagnosis of the 3×3 Wilson-loop wall and a transformer-faithful fix

Working notes following the empirical observation that, on per-site Wilson-loop
regression with the current `GEMHSA` block, the model trains cleanly on
`(1×1), (1×2), (2×2), (1×3), (2×3)` but plateaus on `(3×3)` across every
hyperparameter combination tried (`L, β, R, layers, heads, d_qkv, d_model,
α_init, lr, batch size, init_scale, gate`, etc.).

Constraint on the fix: must stay attention-based / transformer-like (that's
the thesis's central novelty over L-CNN). Single-path transports
(`build_transport_average(mode="single")`) are already in use, so `T` is
unitary and `T·T† = I` exactly.

## 1. What is *not* the problem

- **Receptive field.** The 3×3 target spans `|Δx|_1 ≤ 4` from corner `x`. With
  `R·L ≥ 4` the residual stream at `x` can already see every plaquette in the
  loop after one forward pass. Tried `R ∈ {1, 2, 3, 4}` and `L ∈ {2, 3, 4, 6}`;
  the wall is independent of receptive field.
- **Pure expressivity.** By the L-CNN universality argument (architecture.html
  §8), the block class can represent any function of Wilson loops up to size
  `2^L`. `2^4 = 16 ≥ 9`, so 4 layers is enough in principle.
- **Path-averaged dumbbell pollution.** This is the obvious failure mode for
  `mode="average"` transport: `T·T† ≠ I` in non-abelian groups, the bilinear
  `Q†·T·V·T†` carries non-cancelling tails of length `2|Δx|_1`, and longer
  loops accumulate the pollution. **Ruled out:** we're on `mode="single"`,
  so `T·T† = I` exactly and each bilinear is a clean rectangular Wilson-loop
  primitive.
- **Optimisation hygiene.** Adam clipping, target standardisation, ReZero
  identity-at-init, MLP zero-init, untied per-offset biases — all in place.

## 2. Diagnosis: the block fuses two qualitatively different operators

Rewrite the GEMHSA value path with `T·T† = I`. By cyclic trace identity the
output matrix at `x` per head is

```
W_out(x) = Q†(x) · ( Σ_y α(x→y) · T_xy · V(y) · T†_xy )
        = Q†(x) · V_weighted(x)
```

A single block does *two* qualitatively different jobs in one step:

1. **Routing.** The softmax over offsets builds `V_weighted(x)`, an
   attention-weighted aggregate of neighbour V's transported to `x`. This is
   purely *additive* (a convex combination) and gauge-equivariant.
2. **L-Bilin.** The left-multiplication `Q†(x) · V_weighted(x)` is the
   multiplicative step that doubles loop length per block. This is purely
   *multiplicative* and local.

In an L-CNN these are two separate operators: L-Conv extends loops by a
single-link parallel transport (linear in plaquette degree per layer),
L-Bilin multiplies loops at the same site (degree-doubling). In a standard
transformer they are also two separate operators: multi-head attention
aggregates token features, the feed-forward block provides per-token
nonlinearity. **GEMHSA collapses them into one forward pass.**

The cost: every layer's gradient has to simultaneously decide

- which offset to focus the softmax on,
- how to project `Q` (which determines what gets multiplied on the left),
- how to project `V` (which determines what gets aggregated *and* multiplied
  on the right via the same vector),
- and how `w_mix` reassembles heads into channels.

These four decisions are coupled. The "right" attention pattern only makes
sense given the "right" channel mix, and vice versa.

## 3. Why 3×3 is the depth where the coupling bites

Plaquette degree of an `m×n` Wilson loop in 2D Z₂ (abelian Stokes; analogous
counting for SU(N) by L-CNN's argument):

| Target | Plaquette degree `d` | Min `L` such that `2^L ≥ d` |
|--------|----------------------|-----------------------------|
| 1×1    | 1                    | 0                           |
| 1×2    | 2                    | 1                           |
| 1×3    | 3                    | 2                           |
| 2×2    | 4                    | 2                           |
| 2×3    | 6                    | 3                           |
| **3×3**| **9**                | **4**                       |

3×3 is the first target that requires `L ≥ 4` layers of multiplicative
composition. By layer 4 the loss-to-routing gradient path traverses *three*
prior fused attention/bilinear steps, and the signal that says "head `h` at
layer `ℓ` should concentrate on `Δx = ê_μ` because that's what lets layer
`ℓ+1` multiply against the right channel" is washed out by the coupling.
This is roughly the same phenomenon you see in regular transformers when
the FFN-after-attention is replaced by an attention-with-multiplication-
baked-in — composition stops past a small depth.

The block represents 3×3 in principle. The *optimisation landscape* at
depth 4, with routing and multiplication fused, is dominated by the coupled-
decision problem rather than by the target's signal.

## 4. Proposed change: split into Attention + Gauge-Equivariant FFN

Replace each GEMHSA layer with two sub-blocks, matching the standard
transformer template:

### 4.1 Sub-block A — pure attention (no L-Bilin)

```
Q_a, K_a, V_a = on-site projections of augment(W)
score(x,y,h) = Re Tr[ Q_a†(x) · T_xy · K_a(y) · T†_xy ] / √(d·n_c) + b_h(Δx)
α            = softmax(score) over offsets in the L1-ball
W ← W + α_attn · channel_mix( Σ_y α(x→y) · T_xy · V_a(y) · T†_xy )
                                                       ^ NO Q† factor here
```

Identical to a standard MHA layer in spirit (softmax-weighted sum of
values), only with gauge-equivariant transport wrapping the value. This is
the *routing* primitive. The score remains a gauge-invariant Frobenius
product, so the softmax still sees the right inductive bias for loop
alignment.

### 4.2 Sub-block B — gauge-equivariant L-Bilin "FFN" (no attention)

```
Q_b, V_b = on-site channel projections of augment(W)
W'(x)    = Q_b†(x) · V_b(x)                    # purely local, NO transport
W ← W + α_bilin · channel_mix(W')
```

This is the *multiplicative* primitive. No transport, no neighbours, no
softmax — at every site, it multiplies two adjoint projections of the
residual stream. Each multiplication is a single uncoupled objective for the
gradient: "pick channels of `Q_b` and `V_b` whose product is closest to the
next sub-loop we need." This is exactly Favoni et al.'s L-Bilin (the same
operator L-CNN uses for loop doubling), dropped into the FFN slot of the
transformer block.

### 4.3 The "transformer layer"

```
W ← W + AttentionBlock(W, T)
W ← W + FFNBlock(W)
```

This is *more* transformer-faithful than the current GEMHSA, not less.
Standard transformers have always had attention and FFN as separate
sub-blocks; the current GELT design's fusion is what's unusual.

## 5. Why this should unstick 3×3 specifically

- **Decoupled gradients.** Attention only optimises routing (which `Δx`
  aligns `Q_a` and `K_a`); L-Bilin only optimises which channels to
  multiply. Each sub-block has one objective per gradient step.
- **Cheap depth in the multiplicative path.** Loop doubling is now done by
  very cheap L-Bilin sub-blocks (one local matmul per site per channel).
  Stacking 4–6 of them is essentially free, so reaching plaquette degree 9
  is no longer at the edge of the optimisable regime. The architecture also
  supports unequal counts — e.g. 2 attention blocks + 4 L-Bilin blocks —
  directly addressing the "I need one more multiplication than I have
  attention layers" problem.
- **Identity-at-init still works.** Each sub-block gets its own ReZero `α`;
  warm-start `α_bilin > α_attn` so the L-Bilin path engages first (it's the
  part that actually creates new loop content), and let attention
  concentrate on top of an already non-trivial bilinear feature space.
- **Universality argument is unchanged.** L-CNN's inductive proof becomes
  the *exact* factorisation when you read the new "layer" as
  L-Conv-like + L-Bilin. The attention block is strictly more expressive
  than L-Conv (it's a softmax-weighted sum over an entire L1-ball of
  transports, not a fixed axis-aligned step). So this is a strict superset
  of L-CNN's expressivity, with strictly less optimisation difficulty than
  fused GEMHSA at depth.

## 6. Gauge equivariance of the split block

Attention sub-block, output at `x`:
`Σ_y α · T_xy · V_a(y) · T†_xy`. Under site-local Ω, `α` is invariant (it's
the softmax of an invariant Frobenius product), `T_xy` transforms as
`Ω_x · T_xy · Ω†_y`, and `V_a(y)` as `Ω_y · V_a(y) · Ω†_y`. The product
`T_xy · V_a(y) · T†_xy` becomes `Ω_x · T · V · T† · Ω†_x` — covariant at `x`.
Sum and channel mix preserve covariance. ✓

L-Bilin sub-block, output at `x`:
`Q_b†(x) · V_b(x)`. Under site-local Ω, `Q_b(x) → Ω_x · Q_b(x) · Ω†_x`, so
`Q_b†(x) → Ω_x · Q_b†(x) · Ω†_x`, and the product is
`Ω_x · Q_b†(x) · Ω†_x · Ω_x · V_b(x) · Ω†_x = Ω_x · (Q_b† · V_b) · Ω†_x`.
Covariant at `x`. ✓

The score in the attention block is gauge-invariant by the same Frobenius
argument as the original (architecture.html §3.4).

## 7. Minimal-edit diagnostic: untie the ReZero scalar inside `attend`

Before committing to the full split, a 5-minute change tests whether the
*scaling* coupling alone is enough. Add two per-path learnable scalars
inside `GEMHSA.attend`:

```python
# attend(...) returns, with current behaviour reproduced exactly by
# (alpha_attn=0, alpha_bilin=1):
bilin = torch.matmul(Q_dag, V_weighted)
return self.alpha_attn * V_weighted + self.alpha_bilin * bilin
```

with `self.alpha_attn = nn.Parameter(...)` and
`self.alpha_bilin = nn.Parameter(...)` exposed through `__init__`. Then
warm-start `alpha_attn_init = 0.3` (or similar) while keeping
`alpha_bilin_init = 1.0`. This forces a transformer-style pure-attention
contribution into the value path from step 0, without touching the bilinear.

This is already implemented in `gelt/blocks.py` — defaults reproduce the
old block exactly, and the existing equivariance tests pass.

- **If 3×3 unsticks with `alpha_attn ≈ 0.3`:** the coupling diagnosis is
  right, the full split (§4) is the principled next step.
- **If it doesn't:** the diagnosis is incomplete; revisit §1.

## 8. Complementary moves (optional, all preserve transformer structure)

1. **Deeper value path inside the bilinear sub-block:** chain two local
   matmuls `Q_b† · V_b · V_c` per block. This gives loop-*tripling* per
   bilinear block; 3×3 becomes reachable in `⌈log_3 9⌉ = 2` multiplicative
   steps instead of 4. Same gauge-equivariance argument (every left
   multiplication is by a covariant-at-`x` factor).
2. **Polynomial features on the trace head.** Currently
   `MLP(Re Tr W, Im Tr W)`. Add explicit products `tr_i · tr_j` (or
   low-degree Gram features) before the MLP. In the abelian Z₂ case this
   closes the last factor of degree at the readout, instead of forcing it
   through the residual stream. In SU(N) it provides a useful approximation
   to the multilinear structure of larger Wilson loops.
3. **Decoupled ReZero schedule.** Once the split block is in place, warm
   the bilinear `α` faster than the attention `α` (e.g. cosine ramp with
   different end points). Attention should concentrate *after* the bilinear
   has populated the channel space with non-trivial loop content.
4. **Wider `d_model` paired with shallower stacks.** With the split block,
   each layer is cheap, so widening (more channels per L-Bilin) is
   often cheaper than going deeper. More channel slots → more room for the
   binary-tree decomposition of the 9-plaquette product to lay itself out.

## 9. Build order

In order, smallest-edit first:

1. **Step 7 is done** (`gelt/blocks.py:GEMHSA` now has `alpha_attn` and
   `alpha_bilin` parameters; defaults reproduce the original block).
   Validate the diagnosis by sweeping `alpha_attn_init ∈ {0, 0.1, 0.3, 0.5}`
   on the 3×3 target. Expected behaviour: a non-zero `alpha_attn_init`
   moves the wall.
2. If step 1 helps, implement §4: a new `GEAttention` module (sub-block A)
   and a new `GEFFN` module (sub-block B), and a `GELT` variant that
   alternates them (or supports an arbitrary interleaving pattern). Keep
   the existing `GEMHSA` around for ablations.
3. Re-run the §7 + §7.2 stress test from architecture.html on the new
   blocks (drift at machine ε in float64). Then redo the
   `(1×1)…(3×3)` Wilson-loop sweep; expected result is monotone scaling
   without a wall at 3×3.
4. Continue to `notes/roadmap.md` Phase 3 (SU(2) with the new architecture
   replacing the fused GEMHSA in the matched-parameter shootout).

The original `architecture.html` should be updated to reflect the split
block once §4 lands, with the fused-GEMHSA version preserved as the §12
"variants" entry (legacy / ablation baseline).
