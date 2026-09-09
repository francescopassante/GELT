# Performance audit — the glueball training step (2026-09-08)

**What this is.** A record of why one `scripts/train_glueball.py` optimizer step
cost 7.77 s on a 32 GB V100 for a ~5k-parameter model, what was changed, what was
measured, and what is left. It is the source of truth for performance work on the
GELT hot path; read it before optimising anything in `gelt/blocks.py`,
`gelt/glueball.py`'s smearing, or `SU.project`.

Everything in §3 is **exactly equivalent** to what it replaced — same map, same
number up to floating-point rounding — and each item is pinned by a test. §5 is
the backlog and is *not* implemented. §7 is the honest list of what these numbers
do not establish.

---

## 1. The measurement that frames everything

From `logs/replication_ens1.log` (2026-07-04, 32 GB V100, `BATCH_CONFIGS = 6`):

| quantity | value |
|---|---|
| training step | **7.77 s** |
| epoch | **1942 s** (234 steps + a ~124 s val pass) |
| epochs run | 25 of 60 (early stop, `PATIENCE = 10`) |
| ensemble sampling | 47 min (cached afterwards) |
| phase wall time | ~15 h 05 m |

Now the roofline. A `TorchDispatchMode` probe that tallies the bytes of every
materialising ATen call (view/metadata ops excluded by name) says one GEMHSA layer
**materialises 43.7 GiB per forward** at the production shape — `B = 144` slices
(6 configs × 24 timeslices) of 12³, `R = 2` (`n_off = 25` including the on-site
slot), `d_model = 16`, `H = 2`, `d_qkv = 6`, `nc = 2`, complex64. Four layers, and
with gradient checkpointing roughly three passes (forward + recompute + backward):

    4 × 3 × 43.7 GiB ≈ 524 GiB  →  0.63 s at the V100's 900 GB/s

So the step was running at **~8% of the bandwidth roofline**. The problem was
never FLOPs; it was structural overhead — redundant materialisations, tiny-matrix
BLAS calls, an atomic scatter-add, GPU syncs, and one stage nobody had profiled.

---

## 2. Where the bytes went

Per layer per forward, at `B = 6` (scale by ×24 for production). Only
materialising ops are listed; the count after each is the number of calls.

| op | before | after |
|---|---|---|
| `mul` | 384.8 MiB ×8 | 301.7 MiB ×7 |
| `bmm` | 383.5 MiB ×3 | 383.5 MiB ×3 |
| `clone` | 244.6 MiB ×9 | 248.4 MiB ×10 |
| `cat` | 216.1 MiB ×4 | 26.3 MiB ×3 |
| `abs` + `pow` | 216.4 MiB ×16 | — |
| `index` | 189.8 MiB ×2 | 189.8 MiB ×1 |
| `stack` | 94.9 MiB ×1 | 3.8 MiB ×1 |
| `sum` | 9.0 MiB ×3 | 78.2 MiB ×7 |
| `add` / `sub` | 100.0 MiB ×3 | 11.0 MiB ×2 |
| **total** | **1.821 GiB, 76 non-view calls** | **1.240 GiB, 44 non-view calls** |

Scaled to production: **43.7 → 29.8 GiB per layer per forward**, and the ~10
`.item()` calls per block (each a GPU sync, each running again inside every
checkpoint recompute) went to zero.

Three items in the "before" column were not doing any work the physics needs:
`abs`+`pow` are the diagnostic norms; the 216 MiB of `cat` is 199 MiB of
concatenating K and V *after* gathering them separately; `stack` + four of the
eight `mul` + `sub` + `add` are the RoPE rotation of the transported keys.

---

## 3. What was fixed

### 3.1 `SU(2).project` in closed form — the stage nobody had profiled

`train_glueball.config_inputs` builds `INPUT_SMEAR_LEVELS = (0, 2, 4, 6)`, i.e.
**six APE smearing iterations per optimizer step**. The old `ape_smear` looped
`for b in range(B): for mu in dirs:` and called `SU.project` on each
`(Lt, L, L, L, nc, nc)` slab. Counting: 6 iterations × 6 configs × 3 spatial
directions = **108 `project` calls**, each on `Lt·L³ = 41 472` matrices, i.e.

    ~4.5 M batched 2×2 complex SVDs  +  ~4.5 M LU determinants  per training step

`torch.linalg.svd` on a batch of 2×2 matrices goes to `gesvdjBatched`, and
`torch.linalg.det` to `getrfBatched` — both absurd for a 2×2, and both on the
critical path of every step.

**The algebra.** `H = M†M` is 2×2 Hermitian PSD, so `tr H` and `det H` are real.
With `s = √det H` and `t = √(tr H + 2s)`:

    √H = (H + s·𝟙)/t ,   det(H + s·𝟙) = s·t²   ⇒   √H⁻¹ = adj(H + s·𝟙)/(s·t)

and `polar(M) = M · √H⁻¹`. Only `s` and `t` need a square root, both real; the
determinant that follows is `Q₀₀Q₁₁ − Q₀₁Q₁₀`. This is *the same map* the SVD
route computes, so nothing about `project`'s definition changes — only the route.

`H` is formed in **float64**, because squaring `M` squares its condition number.
Measured on 41 472 Haar-Gaussian matrices (the worst case, far from the group):
complex64 throughout gives a unitarity residual of 1.17e-3 — unacceptable —
while the float64 route gives **1.21e-7**, *better* than the SVD path's 5.96e-7.
MPS has no float64 and stays in complex64; that is a downgrade only for
far-from-group input, and MPS could not run the SVD path at all.

**Agreement on the input that actually reaches it.** 248 832 real APE matrices
built from `datasets/glueball_viz_configs_L12_Lt24_b2.4_xi3.0_N16_seed7.pt`
(same L, Lt, β, ξ as the training ensemble):

| α | \|closed form − SVD\| | \|`_project_su2` − SVD\| |
|---|---|---|
| 0.5 (production) | 3.76e-7 | 3.82e-7 |
| 0.7 | 4.58e-7 | 4.76e-7 |
| 1.0 (what `topology.cool` feeds it) | 4.44e-7 | 4.44e-7 |

All at complex64 rounding. Timing on those same matrices, CPU: `SU.project`
237.19 ms → **10.46 ms**, i.e. **23×**.

A note for the record: `gelt.sampler._project_su2` (orthogonal projection onto
`R₊·SU(2)`, already in the codebase for overrelaxation) is *another* 2.7× faster
and is the true nearest-SU(2) in Frobenius norm — the polar/det^(1/nc) map is only
approximately nearest. On far-from-group input the two differ by O(1) (measured
1.41 on Haar-Gaussians), on the smearing path by 4e-7. It was **not** adopted,
because the closed-form polar preserves `SU.project`'s definition for every input
and the remaining factor is worth less than that guarantee.

Tests: `test_su2_project_matches_svd_polar`,
`test_su2_project_agrees_with_svd_on_near_group_input`,
`test_su3_project_still_uses_the_general_route` in `tests/test_lattice.py`.

### 3.2 `ape_smear` vectorised over the configuration batch

`staple_sum` gained `batched=True`: every op in it is already elementwise over a
leading configuration axis, so this only shifts the direction indexing and the
roll axes by one. The Python loop went from `n_steps × B × len(dirs)` iterations
to `n_steps × len(dirs)`, i.e. **108 → 18 per training step**.

The batch is sliced by a `chunk_bytes` budget (default 256 MiB) because
`staple_sum` holds ~10 temporaries of one slice each, and the *eval* path smears a
400-config batch in one call — unchunked that is several GiB of transients where
the old per-config loop had none. Chunking is bit-exact: smearing never couples
two configurations.

A/B of the whole ladder on 6 real configs, CPU: **5.852 s → 0.790 s = 7.4× per
training step**, with outputs agreeing to 4.49e-7 / 4.53e-7 / 4.92e-7 at levels
2 / 4 / 6 (i.e. complex64 rounding after six smearing steps).

The MCMC sweeps deliberately keep the unbatched `staple_sum`: a Markov chain is
one sequential configuration, there is nothing to batch.

Tests: `test_ape_smear_chunking_is_bit_exact`,
`test_batched_staple_sum_matches_per_config` in `tests/test_glueball.py`; the
existing `test_ape_smear_gauge_covariant` still gates covariance.

### 3.3 The introspection stashes are opt-in

`GEMHSA` stashed `_last_score`, `_last_alpha` and ten `_last_*_norm` scalars on
every forward, unconditionally. Two of those scalars are
`K_tilde.abs().pow(2).mean().sqrt().item()` and the same for `V_tilde` — full
reductions over *offset-expanded* tensors, materialising `abs` and `pow`
intermediates: **~12% of the block's memory traffic** (216 MiB of the 1.821 GiB
above), for a number nothing in the repo reads. The `.item()` calls are ~10 GPU
syncs per block per forward, and with gradient checkpointing all of it runs twice.

Now: `store_attention` (the `_last_score`/`_last_alpha` stash) and `diagnostics`
(the scalars), both `False` by default, both settable across a stack with
`GELT.set_introspection(...)`. It was applied symmetrically to the (since
retired) `blocks_bias` so the two variants stayed mergeable; the 2026-09-09
cleanup merged them by deleting that one, and `gelt/blocks.py` is what is left.

Only `_last_alpha` has consumers, and all of them switch it on explicitly:
`z2_attention_correlator.py`, `su2_attention_correlator.py` (three more —
`z2_attention_readout.py`, `topology_attention.py`,
`visualize_glueball_attention.py` — were retired in the cleanup). `train_z2_glueball.py`'s `attention_stats` turns it
on around the readout and off again, so its training steps do not pay for it.
`_last_score` and the scalars have no consumers at all — the
`scripts/train_gelt_diagnosis.py` the docstring points at does not exist.

Test: `test_introspection_is_off_by_default`.

### 3.4 The GEMHSA hot path

Four changes, all algebraic identities.

**(a) One gather for K and V.** They are adjacent in the fused QKV output and the
`(4, H)` strides merge, so `QKV[:, 1:3].reshape(B, 2H, d, …)` is a *view*.
`attend` now takes that view, gathers once and transports once, and splits K̃/Ṽ
off afterwards as views. The concatenation the size of the whole offset-expanded
neighbourhood (199 MiB per layer per forward) is gone, as is one of the two
`index` calls.

**(b) The Δx = 0 transport slot is prepended once.** It used to be two
`torch.cat` calls over the full `(B, n_off, *Λ, nc, nc)` table *inside every
layer's forward*, and again inside every checkpoint recompute — 16 copies of it
per step at 4 layers. `GELT.attn` now does it once for the stack
(`GEMHSA.prepend_self_offset`), and a layer handed a table that already has the
slot does not add a second one.

**(c) RoPE folded into the query** (`GEMHSA.rope_score`). Writing the planar
rotation of channel-pair `p` out and using the linearity of the Frobenius product
`⟨A, B⟩ = Σᵢⱼ conj(Aᵢⱼ)Bᵢⱼ`:

    Σ_s ⟨Q_{p,s}, rope(K̃)_{p,s}⟩
      = cos(θ_p·Δx) [⟨Q_{p,0}, K̃_{p,0}⟩ + ⟨Q_{p,1}, K̃_{p,1}⟩]
      + sin(θ_p·Δx) [⟨Q_{p,1}, K̃_{p,0}⟩ − ⟨Q_{p,0}, K̃_{p,1}⟩]

Both brackets are Frobenius products against the **unrotated** K̃ — the first with
Q, the second with Q's pair-swapped copy `Q'_{p,0} = Q_{p,1}`,
`Q'_{p,1} = −Q_{p,0}`. Q carries no offset axis, so swapping it touches
`n_offsets` times less data than rotating K̃. `apply_rope` is kept in the module as
the reference the tests check against. Accounting at `B = 6`: `apply_rope` plus
the old score cost 569 MiB (a 94.9 MiB clone, four 47.4 MiB muls, a `sub`, an
`add`, a 94.9 MiB `stack`, and the 94.9 MiB score product); the two contractions
cost 237 MiB.

**(d) The neighbour gather's backward, written as a gather.** This was the
headline. Left to autograd, `KV[nbr_idx]`'s backward is
`index_put_(accumulate=True)`: an atomic scatter-add in which every input element
receives `n_offsets = 25` separate contributions. Measured standalone on CPU at
`(24, 4, 6, 12³, 2, 2)` complex64 (30.4 MiB gathered 25× = 759.4 MiB):

| route | forward | backward | total |
|---|---|---|---|
| `X[nbr_idx]` (autograd) | 75.2 ms | **4775.4 ms (63.5× the forward)** | 4850.6 ms |
| `roll` per offset + `stack` | 368.4 ms | 3029.2 ms | 3397.6 ms |
| `X[nbr_idx]`, gather backward | 92.5 ms | 2616.0 ms | **2708.5 ms (1.79×)** |

The forward is a pure *translation* per offset, so the gradient is itself a
gather:

    dL/dX (x) = Σᵢ gᵢ(x − Δxᵢ)

and indexing the offset axis alongside the lattice axes expresses the whole sum as
one advanced-index read plus one reduction — no atomics, no Python loop over
offsets. `_OffsetGather` keeps autograd's forward and supplies exactly that
backward, with `_nbr_idx_inv` (the forward index with the offsets negated) as a
buffer next to `_nbr_idx`. The gradient is re-gathered in slices of the offset
axis (`_GRAD_GATHER_BUDGET = 512 MiB`) because the backward runs while the much
larger recomputed forward activations of the same layer are still alive;
chunking only changes the summation order.

**Net effect of (a)–(d).** One GEMHSA layer, forward + backward, same weights and
inputs, CPU, 12³, `R = 2`, `C = 16`, `H = 2`, `d_qkv = 6`, complex64, against the
committed version imported side by side:

| batch | before | after | speedup |
|---|---|---|---|
| B = 4 | fwd 147.7 / bwd 427.5 / 575.2 ms | fwd 110.1 / bwd 196.0 / 306.1 ms | **1.88×** |
| B = 8 | fwd 440.2 / bwd 751.6 / 1191.8 ms | fwd 228.9 / bwd 321.7 / 550.6 ms | **2.16×** |

Outputs identical to 1.19e-7 on `|out| ~ 3.6`. The ratio grows with batch (more
of the cost sits in the big tensors rather than fixed overhead), so at the
production `B = 144` it should be at least 2.16×.

Tests: `tests/test_blocks.py` — a new file, because the trained variant had
none (CLAUDE.md caveat 1). It contains gauge equivariance for SU(2) on both gates
and for Z₂; the whole optimised block against a **naive oracle** written out in
the test (two gathers, a concatenation, `apply_rope` on K̃, the plain Frobenius
score), matching on outputs, `_last_score`, `_last_alpha` *and* input gradients to
1e-12 for SU(2)/SU(3)/Z₂; `_OffsetGather`'s backward against autograd's, including
a forced-chunking path; the assertion that `_nbr_idx_inv` really is `_nbr_idx`
with the offsets negated (a mismatch would corrupt the gradient at every
mixed-sign offset and no shape check would catch it); the Δx = 0 prepend
refusing to double-apply; the introspection defaults; and **gradients identical
with and without gradient checkpointing**, since the custom Function runs inside
the replayed region and the prepend now runs outside it.

Test count: **111 → 129**.

---

## 4. What to measure next, before writing any more code

`scripts/profile_glueball_step.py` was rewritten. The old version built its inputs
inline — one thin `plaquette_tensor` level and **no APE ladder** — so it
understated the input stage by the whole smearing cost. That is where CLAUDE.md's
"transport is only 1.9%" came from: true of the transport, measured against a
denominator missing a stage. It is also why §3.1 went unnoticed for two months.

The rewritten script:

* times the real pipeline — `tg.config_inputs` → forward → `rayleigh_loss` →
  backward → clip + step — so it cannot drift from what training runs;
* splits `config_inputs` into smearing / plaquettes / transport (timed separately
  rather than instrumented in place, so `config_inputs` stays the single
  definition; a large gap between the sum and the whole means the reconstruction
  has drifted);
* micro-benchmarks each attention stage **forward and backward separately** —
  the number that matters, per §3.4(d);
* projects s/epoch and h/run, and prints the ens1 run's 7.77 s/step as the
  reference, so the comparison is against a real measurement rather than a
  rebuilt baseline;
* `PROFILE_DIAGNOSTICS=1` and `PROFILE_LEGACY_SMEAR=1` each A/B one change
  against its predecessor inside the same run; `PROFILE_MICRO=0` skips the
  micro-bench; `PROFILE_COMPILE=1` probes `torch.compile`.

```bash
PROFILE_DIAGNOSTICS=1 PROFILE_LEGACY_SMEAR=1 python scripts/profile_glueball_step.py
```

**No total speedup is claimed here.** The two halves of §3 are 7.4× (smearing
stage) and ~2.2× (attention block); how they weight into the step is a property of
the V100 that only this script can answer.

---

## 5. Future work, ranked

### 5.1 Adjoint (real SO(3)) representation of the transport — the biggest one left

The two `bmm` calls in `GEMHSA.transport` are 383 MiB per layer per forward with
`bwd/fwd ≈ 2.2`, and they are `(nc, nc) @ (nc, H·d·nc)` products over millions of
2×2 matrices — a shape cuBLAS handles badly.

For `nc = 2`, write any 2×2 complex matrix in the basis `{𝟙, σ¹, σ², σ³}`:
`M = m₀𝟙 + mₐσᵃ` with complex coefficients `m₀ = Tr(M)/2`, `mₐ = Tr(σᵃM)/2`.
Under `M → T M T†` with `T ∈ SU(2)`:

* `m₀` is **invariant** — a quarter of the data never moves;
* `mₐ → R^{ab}(T) m_b` with `R^{ab}(T) = ½ Tr[σᵃ T σᵇ T†]` a **real** SO(3)
  matrix.

So the transport becomes a real 3×3 matvec on a complex 3-vector: 9 real×complex
MACs per channel (~36 flops) against two complex 2×2 matmuls (16 complex MACs,
~128 flops) — **~3.5× fewer flops**, no permutes, no BLAS, and a fused elementwise
kernel instead of a batched GEMM over 2×2 operands. `R` depends on neither head
nor channel nor layer, so it is built **once per step** and shared by all four
layers (`B·n_off·|Λ|·9` floats ≈ 224 MB at production).

Downstream:

* the score becomes `Tr[Q†K̃] = 2(q̄₀k₀ + Σₐ q̄ₐ(Rk)ₐ)` — same cost as the current
  Frobenius product;
* the **value bilinear stays a 2×2 matmul**: a quaternion product needs 16
  multiplies against a matrix product's 8, and the bilinear is per-site (no offset
  axis) so it is cheap either way. Convert back to matrix form after the
  α-weighted sum, which is also per-site.

Generalises to SU(N) with `M = m₀𝟙/√N + m_a T^a` and `Ad(T)_{ab} = 2Tr[T^a T T^b T†]`
real in SO(N²−1). For N = 3: 64 real MACs (~128 flops) against 2·27 complex MACs
(~432 flops), still ~3.4×.

This is a change of colour basis, so it is exactly equivalent — but it touches
every index in the block. **Gate it on the oracle test and the equivariance tests
in `tests/test_blocks.py`**, which is exactly what they were written for.

### 5.2 Q/K/V memory layout — kill the transport's `permute` + `clone`

`GEMHSA.transport` permutes `(B, H, d, n, *Λ, nc, nc)` to put the colour row index
before `(H, d)`, then `reshape`s — which copies 189 MiB per layer per forward (the
bulk of the `clone` row in §2), with terrible stride locality.

If the QKV projection instead *produced* `(B, *Λ, nc_row, H, d, nc_col)`, the
offset gather would yield `(B, n, *Λ, nc, H, d, nc)` which **views** as
`(B, n, *Λ, nc, H·d·nc)` — exactly what the left matmul wants — and the result
views back for the right matmul with no copy either. The projection mixes only the
channel axis, so it commutes with any arrangement of `(Λ, nc, nc)`; the only new
copy is a permute of the *small* `W_aug` (263 MB at production, vs 4.78 GB saved).

Same matmuls, same order, exactly equivalent. Medium effort. Interacts with 5.1 —
if 5.1 lands, redo this on top of the coefficient layout rather than before.

### 5.3 Unmaterialised score and value contractions

`(Q.unsqueeze(3).conj() * K_tilde).sum(…)` and `(alpha_b * V_tilde).sum(dim=3)`
materialise the elementwise product before reducing it: ~190 MiB per layer per
forward that exists only to be summed away.

Eager PyTorch has no fusion for multiply-then-reduce, and the contraction is over
only 8–48 elements, so `einsum`/`tensordot`/`bmm` would batch millions of tiny
GEMMs (the 5.1 pathology) and, worse, may permute the 2.39 GB K̃ to do it. The two
real options are `torch.compile` (see 5.6) or offset chunking (5.4) so the product
stays cache-resident. Do not "fix" this with `einsum` without measuring.

### 5.4 Offset chunking — this is memory, not time

Processing the offset axis in chunks (scores first, softmax, then a second chunked
pass accumulating `α·Ṽ`, FlashAttention-style) would cut peak activation memory by
roughly the chunk count: stored per layer becomes Q/K/V/Q_v + α + output ≈ 530 MB
instead of ~26 GB.

**It does not reduce FLOPs or bytes moved**, and the recompute is the same as
today's gradient checkpointing. What it buys is headroom: physical `R` for the
explainability program (architecture backlog item 5) and a larger
`BATCH_CONFIGS`. It is easy to mis-sell this as a speedup — the only time it
would return is second-order, through better occupancy at larger batch, and
larger batch changes the loss (§6.3).

### 5.5 Schedule and statistics — the only remaining large wall-clock lever

The ens1 run's val minimum sat near epoch 15; `PATIENCE = 10` **epochs** carried
it to 25. At the old speed that tail was ~5.4 h.

* **Step-based validation.** Validate every ~¼ epoch on a val subset and count
  patience in validations instead of epochs: ~1.3–1.5× wall clock, and a better
  checkpoint (the intra-epoch val minimum is currently invisible). Not
  implemented, because it changes model selection and therefore needs A₀
  re-validated.
* **Cost-neutral variance reduction.** A batch of 12 configs × 12 contiguous
  timeslices costs the same as 6 × 24 but puts twice as many independent
  configurations into the batch VEV and `C(0)` estimates, which is where the
  ratio-estimator bias lives. Pair count is comparable (12·21 = 252 vs 6·48 = 288).
  **Trap:** `rayleigh_loss` forms `C(Δ)` with `d.roll(-dt, dims=1)`, which is
  periodic in time — valid over a full `Lt`, invalid over a sub-window, where
  slice 11 → slice 0 is not a Δ = 1 pair. The roll must go before the window does.
* **Val-input caching.** `held_out_obar` recomputes `config_inputs` for the
  *fixed* val set every epoch (124 s of the old 1942 s). W + T for 200 configs is
  ~9.5 GB, cacheable in host RAM. Marginal, and less attractive now that the
  smearing is 7.4× cheaper — listed so nobody re-derives it.

### 5.6 `torch.compile`

Wired into the profiler behind `PROFILE_COMPILE=1`, not adopted. Two known
obstacles: Inductor's complex64 support is patchy (elementwise usually decomposes
via `view_as_real`, reductions less reliably), and **`_OffsetGather` is a custom
autograd Function, so it graph-breaks**. If the probe looks promising, that
Function needs `torch.library` / `allow_in_graph` treatment first.

### 5.7 Small and known-small

Recorded so they are not re-derived: `augment`'s `cat` to `C' = 2C+1` channels is
263 MB per layer (~0.6% of the step) and could exploit `T·𝟙·T† = 𝟙` and
`T·W†·T† = (T·W·T†)†`; `build_transport_average` is ~12% of `config_inputs`, which
is itself a small slice once the smearing is fixed.

---

## 6. Rejected, with reasons

### 6.1 Caching W and T per configuration

They are deterministic functions of the configuration, recomputed every epoch —
25× redundant. But W is 15.9 MB and T is 31.8 MB per configuration in complex64,
so 2000 configurations is **~95 GB**. fp16 halves it and changes the network's
inputs. And after §3.1–3.2 the input stage is not what dominates. Rejected on
size, not on principle.

### 6.2 Lower precision

The V100 has no bf16. fp16's range is dangerous given this codebase's documented
history — the first `(0,2,4,6)` run reached `C(0) ~ 1e73` before `SCALE_REG`
existed — and Ō is a sum over 1728 sites feeding a correlator whose VEV
subtraction already requires the float64 cast in `rayleigh_loss`. It would buy
bandwidth that the structural fixes deliver anyway, at the price of re-validating
A₀. Rejected.

### 6.3 Larger batch

Cost is linear in batch: no FLOPs saved. And the loss estimates the VEV and `C(0)`
*from the batch*, so batch size is a physics knob, not a performance knob —
`BATCH_CONFIGS` must not move without re-validating. The memory headroom from 5.4
is better spent on 5.5's cost-neutral restructuring.

### 6.4 Real-valued projections (the quaternion subalgebra)

Plaquettes are group elements, and the real span of SU(2) is closed under
multiplication, so with **real** projection weights the entire block could run in
real float32 arithmetic — ~4× on the arithmetic and 2× on the traffic. But that
restricts the hypothesis class from complex quaternions to real ones: it is a
**model change**, not an optimisation, and no trained checkpoint would transfer.
Out of scope here; noted as an architecture question.

### 6.5 `_project_su2` on the smearing path

See §3.1: 2.7× beyond the closed-form polar, measurably identical on the smearing
path, but a genuinely different map on far-from-group input. Not worth giving up
`SU.project`'s definition for.

---

## 7. What these numbers do not establish

1. **Every timing here is CPU.** The ratios will differ on the V100, and in both
   directions. Batched tiny SVD and atomic scatter-add are *relatively worse* on
   a GPU than on a CPU, so §3.1 and §3.4(d) should grow; the CPU's serial
   advanced-indexing exaggerates the gather's absolute numbers. In particular the
   **63.5× backward/forward ratio must not be quoted as a GPU number** — it is
   CPU, and the argument for the change is exactness plus less traffic, not that
   figure.
2. **The real thing has not been run.** What is established: 129 tests pass, the
   optimised block matches a naive oracle to machine precision on outputs and
   gradients, and a toy-size end-to-end run of `train_glueball.py` completes
   through sampling, training, eval, GEVP, dump and plot.
3. **Bit-reproducibility against earlier runs is gone.** The smearing changed at
   the 4e-7 level, so re-running `GLUEBALL_EVAL_ONLY=1` on an existing checkpoint
   will not reproduce the old Ō arrays bit-for-bit. Every published number is a
   jackknifed mean with an error ≥ 1e-2 — five orders above — but exact
   reproduction of the dumps needs the pre-2026-09-08 commit.
4. `_OffsetGather` sums the `n_offsets` gradient contributions in a different
   order from autograd's atomics: ~1e-6 relative differences in float32 (exact in
   float64, which is what the test asserts). Immaterial for stochastic training,
   but it is a real difference.
5. **The two block variants have drifted further apart.** Only `blocks`'s
   `attend` was optimised (`blocks_bias` got the introspection switches, for
   symmetry). That makes architecture backlog item 2 — merge them behind a
   `pos_encoding` switch — *more* pressing, not less. It is the honest cost of
   this change.
6. `chunk_bytes = 256 MiB` and `_GRAD_GATHER_BUDGET = 512 MiB` are judgement
   calls, not measurements.
7. `ape_smear`'s progress bar changed granularity — it now advances per
   `(step, direction, chunk)` rather than per `(step, config)`, so its `total`
   differs from earlier logs. Cosmetic, but visible.
8. One casualty: `results/glueball/glueball_gelt.png` was overwritten by a
   toy-size smoke run (the checkpoint path was redirected, the figure path was
   not). It is gitignored, so there is no copy. Nothing depends on it — the
   reports use the renamed archives `glueball_gelt_GELT_beats_GEVP.png` and
   `glueball_gelt_from_raw_plaquettes.png`, both intact, and both
   `_test_obars.pt` dumps are intact. Regenerating it needs
   `GLUEBALL_EVAL_ONLY=1` on the V100, and even then the loss-history panel is
   unrecoverable because `train_hist` / `val_hist` are never persisted.
