  Audit: GELT code + notes/explainability.md
  
  TL;DR

  The architecture is sound and the gauge-equivariance machinery is well built and well tested — for one of the two block variants. The main problems are not in the math but in the engineering around it: the
  trained variant (blocks_rope) is the untested one, two parameters are dead code (one of which the train script believes is load-bearing), and the RoPE axis assignment silently leaves half the lattice axes
  positionally unencoded in your actual 4D training config. The explainability program is genuinely feasible and is the strongest part of the thesis framing, but two of its three studies hit a memory wall
  in the current "materialize all offsets" attention implementation, and one proposed technique (attention rollout) is not valid for your multiplicative value path.

  ---
  1. Architecture

  What's good. The core design holds up well: adjoint transport T·K·T† with the score Re Tr[Q†K̃] is gauge-invariant by construction, the bilinear value path Q_v†·Ṽ preserves the L-CNN loop-doubling argument,
  decoupled σ_QK/σ_V init scales fix a real softmax-collapse problem (the comments documenting why are excellent), the fused QKV and the transport trick folding (H, d) into the matmul column are both
  correct and smart, and ChannelLift with identity-extend init is the right way to decouple width from D(D-1)/2.

  Real issues, in priority order:

  1. self.alpha is dead in both variants, and train_gelt.py thinks it matters. Both blocks_rope.py:471 and blocks_bias.py:411 return W + W_act; the ReZero line is commented out. Yet train_gelt.py:206 sets
  alpha_init=0.5 with a long comment claiming "α=0.5 puts the multiplicative path at ~half the residual stream (α=0.05 left it at ~4%…)". That comment describes a model that no longer exists — α receives no
  gradient, the diagnostic print at train_gelt.py:105 will show +0.500 forever, and any conclusions you drew from "α warm-start fixed the stall" need rechecking, because what actually changed is the residual
  went from W + α(W_act − W) to W + W_act (a much bigger change than α=0→0.5). Either delete the parameter or restore the ReZero branch; don't leave it ambiguous in a thesis codebase.
  2. b_h is dead in blocks_bias (score = score # + bias... at blocks_bias.py:288), so the "bias" variant currently has no positional information at all — its docstring (step 3) is wrong. This is what makes
  the test fail: the dead parameters have grad=None. The failing test is doing its job — listen to it.
  3. RoPE axis coverage is broken in your actual training config. pair_axis = [p % D for p in range(n_pairs)] with d_qkv=4 gives n_pairs=2, so in D=4 only axes 0 and 1 carry a rotation. Offsets ±ê₂, ±ê₃ get
  the identity rotation — positionally indistinguishable from the self-offset and from each other. You removed the orbit-tied bias precisely because the model needed axis selectivity; this config silently
  reintroduces isotropy along half the axes. Fix: require d_qkv ≥ 2D (raise in __init__ if n_pairs < D), and in train_gelt.py use d_qkv=8 for D=4.
  4. The trained variant is the untested one. tests/test_blocks.py, gelt/__init__.py, and check_gelt_invariance.py all use blocks_bias; train_gelt.py trains blocks_rope. The RoPE rotation is real and
  channel-diagonal so equivariance should hold, but "should" is exactly what the §7 stress test exists to replace. Cheapest fix: parametrize the equivariance tests over both modules (@pytest.mark.parametrize
  over the imported module).
  5. The two files are ~90% duplicated and have already drifted (docstrings, the dead bias). Merge into one blocks.py with pos_encoding ∈ {"rope", "bias", "none"}. For the thesis this also gives you a clean
  ablation axis instead of two diverging forks.
  6. No normalization in the residual stream. The softplus L-Act gate g(Tr W_mix)·W_mix is quadratic in the sublayer output, and the init-scale gymnastics in the comments (init_scale=10, qk_init_scale,
  σ_V=0.02…) are symptoms of fighting scale drift by hand. A gauge-equivariant RMSNorm is cheap and legal: divide each channel by sqrt(mean_x ‖W‖²_F) — the Frobenius norm is invariant under W → ΩWΩ†, so
  equivariance is untouched. This would likely make the 4-layer stack much less init-sensitive and is a nice small contribution to write up.

  Minor: the Trace readout for complex models feeds imag channels that are zero for real groups (harmless), and topological_charge_density loops over all 24 permutations when antisymmetry reduces it to 3
  distinct products × 8 (it runs once per dataset, so only worth fixing if it shows up in profiling).

  2. Efficiency

  The hot path is mostly well done (fused QKV, fused transport, α-weighting before the Q_v† matmul). Remaining issues, ordered by impact:

  1. Diagnostics force GPU syncs every layer, every step. The with torch.no_grad() blocks in attend and forward call .item() ~10 times per layer per forward — each is a host-device sync. On CUDA this can
  easily dominate step time for small models. They also keep _last_score/_last_alpha alive (extra memory). Gate them behind a self.collect_diagnostics flag that the diagnosis script flips on; default off in
  training.
  2. Memory scaling is the real ceiling: every offset is materialized. K[nb_indexer] builds (B, H, d, n_off, *Λ, nc, nc), then the KV cat doubles it, then the transport and the score's broadcast product
  (Q_e.conj() * K_tilde) each allocate the same size again. At R=1 this is fine. At the R the explainability program needs (see below), n_off is 230–320 and these tensors hit tens of GB. The fix that unlocks
  the science: chunk over offsets — loop over offset blocks, accumulate the softmax online (running max + running denominator, flash-attention-style) and the α-weighted V sum. The score product specifically
  can also be a fused reduction (einsum/reshaped matmul over d·nc²) instead of broadcast-multiply-then-sum, which removes one full-size intermediate even unchunked.
  3. The identity offset is concatenated per layer, per batch. Each GEMHSA.forward does torch.cat([identity_T, T]) (and the same for T_dag) — a full copy of the transport tensor for every layer in the stack.
  Hoist the cat into GELT.forward next to where T_dag is already computed once.
  4. Precomputed T in the dataset won't scale. At D=4, L=4, R=1 it's ~65 MB — fine. At the explainability scales (4D, L=8, R=4: 320 offsets × 4096 sites × nc² complex64 ≈ 42 MB per config, ~42 GB for N=1000)
  it's untenable. build_transport_average is fast; compute T on the GPU per batch from the link configs instead of storing it in the TensorDataset. This also fixes the pin_memory cost of shipping huge T
  tensors every step.
  5. Small stuff: metropolis_sweep computes proposals and ΔS for all sites but uses one parity (2× waste, fine at this scale, and the checkerboard logic itself is correct — I verified same-parity μ-links
  never share a plaquette and the staple reuse across n_hits is valid); the pre-training shape printout in train_gelt.py:279 runs a full forward on CPU.

  3. Is explainability.md feasible for GELT?

  The framing is genuinely strong, and the central claim survives scrutiny: the score Re Tr[Q†K̃] is gauge-invariant, so attention maps are physical observables in a way a CNN kernel can never be, and
  _last_alpha already gives you extraction for free (no hook needed — the note's "add a hook in blocks_rope.py" is already done, just gate it). Per study:

  1. ℓ_att(β) vs ξ(β) — feasible, with one architecture-dependent confound the note misses. In the RoPE variant, α conflates a configuration-independent positional prior (the learned RoPE rotations) with the
  content-dependent part. The configuration-averaged α profile is exactly the kind of static kernel a CNN bias has — so the "no convolutional weight can provide this" claim rests entirely on the
  config-dependent residual. You should measure Var_configs(α) per offset, or subtract the per-offset mean over configurations, and show the content part tracks ξ. This makes the result sharper, not weaker —
  but if you skip it, a referee will catch it. Practically: 3D Z₂ at R=5 is 230 offsets, nc=1 — trains today with reduced batch; 4D SU(2) at R=4 is 320 offsets and needs the offset-chunked attention from §2
  (at L=8, B=8 you're at several GB per intermediate ×autograd; at the L≳12 you'd want for ξ studies it's currently impossible). The R-sweep MSE-saturation proxy is a good fallback and works with the code
  as-is. One more gap: "ℓ_att(β) tracks ξ(β)" needs either per-β models (fine, weaker) or one β-conditioned model — and datasets don't store β yet (already flagged in CLAUDE.md).

  2. Localization on topological lumps — feasible, one missing prerequisite. S(n) = Σ α incoming-mass is well-defined from _last_alpha, and reduction="none" per-site supervision makes the ablation test
  ("predicted Q drops by removed charge") clean to implement — attend needs a small change to accept an α-mask, that's it. The missing piece: the note says "cooled |q(n)|", and there is no cooling/smearing
  in the codebase. The naive plaquette q_x you currently target is UV-noisy and non-integer; without ~10–50 cooling sweeps (or APE/stout smearing — a short, easy addition reusing staple_sum) the "lumps"
  you'd correlate against are largely noise. Also heed your own caveat: at R=1 on a per-site quadratic target, attention will collapse to self — this study needs the total-charge or ξ-coupled target.

  3. Head specialization / layer-wise coarsening — trivially feasible, but the current train config has nhead=1, so nothing to specialize yet.

  The one method to drop: attention rollout. Π_l(ᾱ^(l)+I) assumes the value path is linear in the attended tokens. Yours is deliberately not: the output is Q_v†·(Σ α Ṽ) (bilinear) followed by a nonlinear
  trace gate. Rollout would be a heuristic with no validity argument — and you don't need it, because your ablation/intervention test is strictly more rigorous and is the part Jain & Wallace would actually
  accept. I'd cut rollout from the plan rather than defend it.

  4. Concrete to-do, in order

  1. Fix the failing test by resolving the dead parameters (alpha, b_h) — decide, don't comment out.
  2. Merge blocks_bias/blocks_rope into one module with a pos_encoding switch; parametrize the equivariance tests over all variants.
  3. Enforce n_pairs ≥ D for RoPE and set d_qkv=8 in the 4D train config.
  4. Gate the .item() diagnostics behind a flag.
  5. Hoist the identity-offset cat to GELT.forward; switch the score to a fused reduction.
  6. Implement offset-chunked attention + on-the-fly transport — this is the gate on the whole explainability program at physical R.
  7. Add cooling/smearing (prerequisite for study 2) and β in the dataset (prerequisite for the strong version of study 1).
  8. Update CLAUDE.md — it still references gelt/blocks.py, gelt/train.py, L_scan.py, lr_scan.py, visualize.py, timer.py, none of which exist anymore, and doesn't mention lcnn.py or the two-variant split.

  One framing note to end on: the explainability document's instinct — "the attention map is a measurement, validated against known ground truth" — is the right thesis spine, and it's more defensible than
  the MSE-shootout framing. The architecture work needed to support it (items 3, 6, 7) is bounded and worth prioritizing over further init-scale tuning.