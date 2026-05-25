# GELT — performance review of `gelt/blocks.py`

Target hardware: V100 / A100. Same architecture and math, only
implementation-level changes. Hot path is `_attend` (gather → transport →
score → softmax → value) and the per-block QKV projection.

## High-leverage

**1. `torch.compile(GELT)`** — biggest single win, ~zero invasiveness.
The forward is a long chain of small reshape / permute / elementwise /
matmul ops on complex tensors; the inductor backend will fuse the
pointwise+reduce passes (notably the score `mul+sum` and the
alpha-weighted sum) and elide several of the explicit `.contiguous()`
copies. Pin shapes with `dynamic=False`. Try it before any code change
to set a baseline.

**2. Fused QKV projection.** `blocks.py:326–328` issues three separate
`(H·d, C̃) @ (B, C̃, N)` matmuls. Standard transformer fix: stack into a
single weight `w_QKV: (3, H, d, C̃)`, do one big matmul, then
`.chunk(3, dim=1)`. On A100 this is roughly a 2–3× speedup of that step
at small `C̃` because cuBLAS amortizes launch + reduces the m-dim tile
waste.

**3. Compute `T_dag` once at the `GELT` level, not per layer.**
`blocks.py:255` runs `self.gaugegroup.dagger(T)` inside every
`GEMHSA.forward`. `T` is shared across all blocks. Pass `T_dag` (and the
dtype-cast `T`) into `GEMHSA.forward` from `GELT.attn`, do the dagger +
cast once. With 3–4 layers this is a free saving on bandwidth-bound ops
over a `(B, n_off, *Λ, nc, nc)` tensor.

**4. Score: replace mul+sum with `einsum`.** `blocks.py:261`:

```python
score = (Q_e.conj() * K_tilde).sum(dim=(2, -2, -1)).real
```

materializes the full `(B, H, d, n_off, *Λ, nc, nc)` complex product —
at R=2, D=3, B=32, L=8, nc=3 that's ~1 GB. Switch to

```python
score = torch.einsum('bhd...ij,bhd...ij->bh...', Q.conj().unsqueeze(3), K_tilde).real
```

(or its `opt_einsum` equivalent). PyTorch's einsum dispatches this as a
fused mul+reduce without materializing the product, saving memory and
one HBM round-trip. Order the contraction so `d, i, j` are the inner
loop.

## Medium

**5. Enable TF32 on A100 (no effect on V100).**

```python
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
```

Applies to the underlying float32 GEMMs that back complex64 matmul,
including `_transport_folded`. Verify gauge-invariance drift stays
acceptable in the stress test (`notes/architecture.html` §7) before
keeping it on.

**6. Drop the trailing `.contiguous()` in `_transport_folded`**
(`blocks.py:222`). The downstream consumers in `_attend` are an
elementwise product (line 261) and a final `Q_dag @ V_weighted`
(line 287) — neither requires contiguity in the leading axes. The
permute returns a view; let inductor / cuBLAS handle the layout. If a
downstream op chokes, add `.contiguous()` back at exactly that call site.

**7. Hoist constants out of the forward.** Trivial but cumulative:

- Cache the `identity` in `_augment` as a registered buffer of shape
  `(1, 1, *Λ, nc, nc)` (or `(1, 1, 1, …, nc, nc)` and expand) instead of
  `torch.eye` + `.expand` every step.
- `torch.is_complex(self.b_h)` (line 271) and the `bias.view(...)` shape
  can be resolved at `__init__`.
- The dtype-cast `if W.dtype != w_dtype` block (lines 306–309) can be
  hoisted to `GELT.forward`.

**8. Avoid the channel-mix `einsum`** (`blocks.py:334`):

```python
W_mix = torch.einsum("iha,bha...->bi...", self.w_mix, out)
```

This is a plain matmul `(C, H·d) @ (B, H·d, *Λ·nc·nc)`. Reshape and
call `torch.matmul` directly — slightly lower launch overhead than
einsum and avoids einsum's planner cost in eager mode.

## Lower priority / more invasive

**9. CUDA graphs for training.** Static shapes per epoch (constant B,
L, D, R) → `torch.cuda.make_graphed_callables` on the model. Removes
per-step Python + launch overhead; meaningful at small L where each
kernel is short. Pair with #1.

**10. Memory-efficient attention over offsets.** The `K[nb_indexer]` /
`V[nb_indexer]` gather at `blocks.py:246–249` blows up `K, V` by
`n_off` (24× at R=2, D=3). For larger lattices / batches that becomes
the dominant working set. Chunk over offsets: process k offsets at a
time, accumulate `max` / `logsumexp` / value-running-sum incrementally
(memory-efficient attention pattern). Significant refactor, only worth
it if hitting OOM at scale.

**11. `Q_dag` via `.adjoint()` view** at line 286: `Q.adjoint()`
returns a view PyTorch knows is a conjugate transpose; cuBLAS can
sometimes consume it without a copy. Worth a microbenchmark.

## Recommended order

If picking three: **#1 (`torch.compile`), #2 (fused QKV), and #3
(share `T_dag` across layers)**. They're independent, additive, and
total maybe 30–40 lines of changes.
