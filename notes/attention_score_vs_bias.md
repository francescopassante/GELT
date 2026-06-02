# Why the score bias does all the work — diagnosis and proposed fixes

Analysis of the GEMHSA attention mechanism (`gelt/blocks.py`), prompted by the
empirical finding:

- **Bias only** (drop the `Q†K̃` term, keep the per-(head, offset) bias in the
  score) → **complete convergence**.
- **`Q†K̃` only** (drop the bias, keep the content score — the variant currently
  live, bias commented out at `blocks.py:273`) → **stalls** at train/val ≈ 0.6
  on 1×2 loops, ≈ 0.8 on 2×2, and the val loss *grows*.

This document explains why, and proposes attention mechanisms that give the
data-dependent path an honest job.

---

## 1. What the two variants actually compute

Logit for offset `n` at site `x`, head `h`:

```
s_n(x) = b_h[n]  +  λ · c_n(x)
c_n(x) = (1/√(Nc·d)) · Re Σ_a Tr[ Q†_a(x) · T_n(x) K_a(x+Δx_n) T†_n(x) ]
```

- `b_h[n]` is **constant in `x` and across configurations**. It is, almost
  exactly, a learnable **L-Conv kernel weight per transport offset**. `softmax(b)`
  gives a fixed mixing of the transported values; the value path
  `Q†·Σ α_n Ṽ_n` then multiplies them. So **bias-only GEMHSA is L-Conv (fixed
  offset weights) + L-Bilin (multiplicative value)** — the L-CNN primitive that
  provably builds the loop. That is why it converges.
- `c_n(x)` is the **two-loop correlator** (architecture.html §3.4): gauge-invariant,
  but it **fluctuates with the configuration and the site**.

So the real question is: *for this target, does the optimal routing need to be
configuration-dependent?* It does not — and that single fact explains every
number observed.

---

## 2. Why `Q†K̃`-only stalls (and val grows)

Decompose any offset-selectivity the logit can produce into a **config-mean** part
and a **config-fluctuation** part:

```
s_n(x) = E_cfg[s_n]   +   (s_n − E_cfg[s_n])
         └ mean routing ┘   └ per-config wiggle ┘
```

The target is a **fixed-geometry Wilson loop**. The Bayes-optimal routing —
*which* offsets to multiply to extend the loop — is **the same recipe for every
configuration**. All per-configuration information lives in the *values* `Ṽ`,
never in the routing. So the optimal logit has **zero** config-fluctuation part.

Compare the two parametrizations against that optimum:

- **Bias supplies a pure config-mean term and nothing else.** Its gradient
  `∂L/∂b_h[n]` accumulates *coherently* over the whole batch (one scalar per
  offset, same sign every config) → high gradient SNR, low variance, learns the
  geometric kernel cleanly.
- **`c_n(x)` cannot produce a constant** (it is `Re Tr Q†K̃`, which fluctuates
  unless Q/K collapse). Anything it can contribute to the *mean* routing, the
  bias can contribute too — noiselessly. But `c_n` unavoidably drags along a
  config-fluctuation term, and for a fixed target **every bit of that fluctuation
  is error**. So the content path is **weakly dominated** by the bias: same
  reachable mean, strictly more variance.

That domination is exactly the observed symptoms:

1. **Stall.** With the bias removed, the only handle on the *mean* routing is the
   config-mean of `c_n`, which must be extracted from a high-variance channel.
   The coherent (sign-consistent-across-batch) part of `∂L/∂w^Q, ∂L/∂w^K` is
   tiny; what remains correlates with per-config `Ṽ` fluctuations. Q/K random-walk
   and never acquire a stable geometric preference → the value path gets a noisy
   offset selection → it plateaus where the residual/on-site channel + MLP can
   carry it (R²≈0.4 on 1×2, ≈0.2 on 2×2).
2. **Val grows.** The only way to push train loss down through a config-fluctuating
   router is to *fit the fluctuations* — overfit the noise in `Q†K̃`. Train drifts
   down, val climbs. Classic signature of a router latching onto a non-generalizing
   signal.
3. **The 1×2 case is the sharpest tell.** A 1×2 rectangle needs the routing to
   *break the axis symmetry* (extend more along one axis). The bias does this
   directly once untied across the point-group orbit (`blocks.py:119–123`). The
   content term can break that symmetry only through a contrived asymmetry between
   the `P` and `P†` augmented channels (the §3.10 non-point-group-equivariance
   loophole) — a far harder optimization, and it still pays variance for a
   preference the bias gives for free. On rectangles content is not just
   dominated, it is fighting uphill.

**Second, mechanical aggravator (code-specific):** Q is shared between score and
value (§3.6; `blocks.py:264` and `:284`). With a content-only score, `∂L/∂Q`
carries *both* "be a good correlator probe" and "be a good left loop-factor." The
noise-only score gradient then actively corrupts the Q the value path needs. The
bias is a separate parameter, so bias-only routing leaves Q free to be a clean
loop factor — an independent reason bias is clean and content is not.

---

## 3. The reframe (the real conclusion)

**Content-based routing is the wrong tool for a fixed-geometry observable — by
construction, not by a bug.** A fixed loop is geometrically trivial: one static
recipe works for every configuration, and a learned per-offset bias is the
*minimal exact parametrization of a static recipe*. (Slide 14 of the presentation
brief says this informally; §2 above is the rigorous version.)

Two genuinely different goals, wanting different fixes:

- **(A) Just predict the fixed loop well** → keep the bias; "attention" here is
  correctly a *learned static convolution over transport paths*. Don't fight it.
- **(B) Make the attention mechanism earn its place** (the thesis headline:
  attention-range ↔ correlation-length, cross-β generalization, a real physical
  observable) → give the *data-dependent* path a job where the optimal routing
  genuinely varies with the configuration / coupling. The `Q†K̃` correlator is
  data-dependent but *signal-free for routing* on a fixed target, so it will never
  earn its place there no matter how it is tuned.

---

## 4. Proposals, ranked

### Proposal 0 — bias + λ·content, not bias XOR content (do this now)

The experiment removed the bias entirely. The transformer lesson is you need
*both* relative-position (bias) and content; pure content with no positional prior
fails on any task needing fixed geometry. Uncomment the bias (`blocks.py:273`;
`b_h` already exists) and gate the content with a *learnable scalar* `λ_h`
initialized at ~0:

```
s_n(x) = b_h[n] + λ_h · c_n(x)
```

At init geometry dominates (recovers the converging model); the optimizer switches
content on only where it helps and can drive `λ→0` where it doesn't. Safe baseline,
makes the story honest ("attention = geometric prior + learnable content
correction"), cannot be worse than bias-only. **Note this is a correctness fix —
the variant currently live is the *worse* one, content-only with the bias commented
out.**

### Proposal 1 — route on local invariants; make the range/temperature data-dependent

The right data-dependent quantity in LGT is not the `Q†K̃` two-loop correlator —
it's *local gauge invariants*: action density `Re Tr P(x)/Nc`,
`|1 − Re Tr P(x)|`, or a learned invariant scalar field `m(x)`. Use them to set the
softmax range:

```
s_n(x) = b_h[Δx_n] − |Δx_n|₁ / ξ_h(x),    ξ_h(x) = softplus(W_ξ · invariants(x) + c)
```

Directly implements "attention reaches far in smooth/cold regions, localizes in
rough/hot regions" — the attention-range-vs-correlation-length result the roadmap
calls central. Now the data-dependence carries real physical content, so its
gradient is systematic rather than noise; on a fixed-loop target it degrades
gracefully to `ξ≈const` (≈ bias), so it cannot underperform the current model.

### Proposal 2 — condition routing on β; train multi-β (cleanest way to force data-dependence)

A single static bias provably *cannot* be simultaneously optimal across β (the
relevant loop scale changes with the coupling). Make the bias β-conditioned,
`b_h(Δx, β) = MLP_h([Δx-features, β])`, and train on several β at once. The model
is then *forced* to use an input to modulate routing — the "multi-target/multi-β"
lever in Slide 15. Blocker: the `CLAUDE.md` caveat "datasets do not store β";
fixing that (roadmap Phase 1.5) is the prerequisite. Paired with a real physical
target (χ_t, string tension, an order parameter near a transition), this is the
actual answer to "an attention that learns something."

### Proposal 3 — decouple the value query `Q'` from the score `Q`

Add the §12 "decoupled output query": a separate projection `Q'` for the value path
so the score's Q can specialize as a correlator probe without degrading the loop's
left factor. Removes the Q-reuse conflict from §2. Cheap (+1 projection, ~+33%
projection params).

### Proposal 4 — reconsider the softmax simplex

Building a loop is *additive over paths/offsets* (combine contributions), but
softmax forces winner-take-all competition (`Σα=1`). L-Conv uses free,
unnormalized weights. A learnable-temperature softmax, or the `tanh(s)`
non-normalized variant the spec already sanctions (§3.5), lets the value path
accumulate over offsets instead of trading them off — more faithful to
loop-doubling and possibly better on the harder loops. Keep softmax only where
routing is genuinely a *selection*.

**Recommended order:** Proposal 0 immediately (correctness fix). Then, for the
thesis novelty, Proposal 2 + 1 together (a target/regime that *requires*
data-dependent routing, plus a routing signal with real physical content), with
Proposal 3 as support.

---

## 5. One experiment to confirm the diagnosis before building

`report_attention_state` (in `scripts/train_gelt_diagnosis.py`) already logs
everything needed. Run the **`Q†K̃`-only** config and check two predictions of the
"noise-not-signal" hypothesis:

1. **Batch-mean `α̅[n]` ≈ uniform / orbit-degenerate** while **per-config `α`
   fluctuates strongly** → routing has no systematic geometric preference, only
   config noise.
2. The **coherent part of `|∇w_Q|, |∇w_K|`** (gradient sign consistent across the
   batch) ≈ 0 even though the per-batch norm is nonzero.

Sharper falsification: **freeze Q,K at init (no grad) in the content-only model**
and train everything else. If val tracks the trainable-Q,K run, the score channel
carries no learnable routing signal — confirming the bias is doing the work for a
structural reason, not a tuning one.

---

## 6. Deeper reflection — can `Q†K̃` ever be a good pattern? (bilateral filtering, RG, a better score)

The §2 result ("content is dominated by bias") is specific to *fixed-geometry*
targets. The deeper question is what `Q†K̃` physically computes, and whether there
is a regime or a different score where it is the right tool. There is.

### 6.1 What `Re Tr[Q†K̃]` is, physically

`Q(x)` is a small covariant loop at `x`; `K̃_n(x) = T_n K(x+Δx_n) T†_n` is the
neighbor's loop transported into `x`'s frame. So

```
c_n(x) = Re Tr[ Q†(x) · K̃_n(x) ]
```

is a lattice **glueball / scalar correlator** `⟨O(x) O(x+Δx)⟩` of local
field-strength operators. Its offset-dependence decays like `exp(−|Δx|/ξ)` with the
correlation length `ξ`. Content attention is therefore a **coherence detector whose
natural length scale is `ξ`.**

Rewrite it as a distance:

```
|Q − K̃_n|²_F = |Q|² + |K̃_n|² − 2 Re Tr[Q†K̃_n]
⇒ softmax_n( λ·Re Tr[Q†K̃_n] ) ∝ exp(−|Q−K̃_n|²/2σ²) · exp(+|K̃_n|²/2σ²)
                                  └── bilateral range kernel ──┘   └ magnitude bias ┘
```

So the GEMHSA block — `Q†K̃` score + softmax-over-offsets + value path — **already is
a gauge-equivariant bilateral filter / non-local-means (adaptive smearing)**: it
averages each site's covariant content with neighbors, weighted by spatial proximity
(the bias `b[n]` = *domain* kernel) and post-transport field similarity (`Q†K̃` =
*range* kernel). CASK (2501.16955), cited in arch §3.4 for the Frobenius score, is
doing this same smearing-style attention — not loop-building.

### 6.2 Can `Q†K̃` be good? Yes — for a different job than the Wilson loop

- **Wrong tool for a fixed loop, unfixably.** §2 is target-agnostic: for *any*
  fixed-geometry observable the optimal routing is config-independent, so the bias is
  the exact minimal parametrization and every content score (dot-product, L2,
  value-product, …) is weakly dominated. The failure is a **task mismatch**, not an
  empty mechanism.
- **Right tool for coherence/scale-adaptive tasks**, where optimal routing is
  field-dependent:
  - **Adaptive smearing / RG flow.** Stout/HYP smearing and gradient flow are
    field-aware (edge-preserving: smooth the vacuum, keep topological lumps). A
    gauge-equivariant bilateral filter is a *learnable adaptive smear* — which a
    fixed-weight L-Conv (bias-only) structurally cannot be. Stacked GEMHSA blocks =
    a learnable gauge-covariant RG/flow, content score setting the local smoothing
    rate.
  - **Correlation-length result, for free.** Because `c_n(x)` is a glueball
    correlator `~exp(−|Δx|/ξ)`, its offset-selectivity stretches as `ξ` grows, so the
    effective attention range `Σ_n α_n |Δx_n|` **tracks `ξ(β)` dynamically** — a fixed
    bias gives a constant range. This is the roadmap's headline "attention-range ↔
    correlation-length," and it *requires* content attention. (Bounded by per-block
    radius `R` × depth, so need `R·n_layers ≳ ξ` in the critical window.)

One-sentence reason content earns its place here and not on the loop: **the Wilson
loop's optimal receptive field is fixed; a smearing/critical observable's optimal
receptive field is field- and coupling-dependent, and content attention buys exactly
a data-dependent receptive field.**

### 6.3 A more promising pattern: negative-distance (L2 / bilateral) attention

If the job is adaptive smoothing, dot-product `Q†K̃` is the *approximate* bilateral
kernel — it drops the `|K̃_n|²` key-norm term. Restore it:

```
s_n(x) = b_h[n] − |Q(x) − K̃_n(x)|²_F / (2 σ_h(x)²)
```

Still exactly gauge-invariant (`Q−K̃` is covariant at `x`; `Tr[A†A]` invariant).
Three advantages over dot-product:

1. **Removes the magnitude bias.** Dot-product reward grows with `|K̃_n|` (the
   neighbor's action/field-strength magnitude) → it is drawn toward rough, high-action
   neighbors, the noisiest places to transport through. The `−|K̃_n|²` term penalizes
   exactly that → L2 smooths *through coherent, low-action regions* (edge-preserving).
2. **Monotone in similarity** (a true range kernel in `[0,1]` peaked at `Q≈K̃`),
   not a signed correlator that can route on anti-alignment.
3. **Natural home for a data-dependent temperature** `σ_h(x) = softplus(W_σ ·
   invariants(x))`: similarity picks the direction, `σ(x)` sets the local range (=
   local correlation length). Unifies Proposals 1 and 0; degrades to ≈ bias as
   `σ → ∞`. Strictly more expressive than an isotropic range-from-invariants head.

Physical fork (not a tuning choice — depends on the observable):
- **Dot-product → attends to salient/high-action structure** → defect/topology
  localization (find the instanton/vortex).
- **Negative-distance → attends to coherent/low-action background** → smoothing/RG/
  denoising.

Let different heads carry different signs and let the data choose.

### 6.4 Patterns considered and rejected as weaker

- **Score on the value product `Re Tr[Q†Ṽ_n]`** (route toward offsets that build a
  large-trace loop): more loop-aligned than `Q†K̃`, but still config-independent-optimal
  on a fixed loop → still dominated by bias (§2). Doesn't escape.
- **Pure range-from-invariants `s_n = b[n] − |Δx_n|/ξ(x)`** (no neighbor content):
  interpretable for the ξ-plot but can only set an isotropic radius — can't preserve a
  specific edge/direction. It is the cheap special case of L2 with the similarity term
  dropped → use it as the ablation *baseline against* L2, not the final mechanism.

### 6.5 Bottom line

`Q†K̃` is a gauge-covariant **coherence kernel / bilateral-filter range term** — the
machinery for adaptive smearing and field-dependent receptive fields. It cannot beat a
learned bias on a fixed Wilson loop because that task has no field-dependent routing to
exploit (and no score can). To make it earn its place: (i) change the target to one
whose optimal receptive field is field/β-dependent — cooled/flowed topological charge,
or a near-critical observable across a β-window; and (ii) switch the score from
dot-product to **negative-distance (L2) with a data-dependent temperature**, the
correct bilateral kernel, which makes the attention range track `ξ` by construction.

Decisive experiments:
- **Cross-β / near-critical** (3D Z₂ Phase 1, or SU(2) over a β-window): plot effective
  attention range `Σ α_n |Δx_n|` vs β against `ξ(β)` — flat for bias-only, tracking for
  content. Cross-β generalization should also favor content.
- **Adaptive smearing → cooled/flowed topological charge from the raw config**: a target
  whose optimal routing is field-dependent smoothing; content (esp. L2) should beat
  bias-only. The cleanest "attention earns its place."
