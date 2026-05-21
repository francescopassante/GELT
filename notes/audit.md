# GELT audit — the 2×2 Wilson-loop stall

**Symptom.** With the 1×1 rectangular Wilson loop as a per-site target
(`reduction="none"`, μν=01, L=8, D=3, Haar-Z₂), GELT trains to val loss
≈ 0 in a handful of epochs. Switching the target to the 2×2 Wilson loop —
same script, same hyperparameters, same architecture — the model freezes
at train ≈ val ≈ 1.0 in standardized units (= Var(y)). Every loop-up
attempt (LR ↑, depth ↑, head count ↑, `alpha_init=0.5`, `init_scale=10`,
`mlp_zero_init=False`) has left the loss within ~10⁻³ of 1.0.

This note pins down *why* and proposes the smallest architectural changes
likely to break it.

## 1. The two targets are not the same problem

For Z₂ (nc=1) the per-site readout of an `R×T` Wilson loop is

```
W(R,T,x) = Re Tr [ ∏ links around the R×T rectangle at x ] / nc
        = (product of links on the boundary)        (real ±1)
```

`R=T=1` reduces to the plaquette `P_{μν}(x)`. **`P_{μν}` is literally one
of the input channels of the GELT** (the dataset feeds the plaquette
tensor as `W^{(0)}`). So the 1×1 target is a tautological pass-through:
the residual stream `W^{(0)} → W^{(K)}` already carries `P(x)` to the
trace head; the MLP only has to learn the identity on the right channel
and the entire equivariant machinery is bypassed.

The 2×2 target is fundamentally different. In Z₂ (abelian, U=U†),

```
W_{2×2}(x) = P(x) · P(x+ê_μ) · P(x+ê_ν) · P(x+ê_μ+ê_ν)        (in Z₂)
           = product of the 8 links on the 2×2 boundary
```

so a successful model must compose a 4-fold product of plaquettes at the
corners of a 2×2 box. The per-site MLP only sees the trace channels at
site x, so this 4-product has to be inside `W^{(K)}(x)` at every x.

## 2. The target is orthogonal to the input at every site

Pooling across all sites and configs (Haar-Z₂, N=400 × 6³ ≈ 86 k samples):

```
y(x) ≈ a + Σ_c β_c P_c(x)            least squares
var(y)        = 1.0000
var(ŷ_local)  = 6 × 10⁻⁶              ← noise floor
var(resid)    = 1.0000
betas         ≈ [-1e-3, -1e-3, 0e-3, -2e-3]
```

Combinatorially: each 2×2 boundary loop uses 8 distinct boundary links,
and any single plaquette at x uses 4 links, three of which are interior
to the 2×2 box and therefore cancelled out of the 8-link product.
`E[P_c(x) · W_{2×2}(x)] = 0` exactly for Haar links.

**Consequence.** Any *site-local* predictor of `P(x)` has

```
MSE  =  Var(y)  +  Var(ŷ_local)  −  2·Cov(ŷ_local, y)
    ≥  Var(y) = 1.
```

The constant predictor `ŷ = 0` is therefore the global optimum *over the
class of site-local readouts of P(x)* — and that is exactly what the
optimizer drives the model to.

## 3. Empirical trajectory — the model converges to constant-zero

Running `scripts/audit_gelt.py` (L=6, R=2, 3 layers, nhead=4, d_qkv=16,
α_init=0.5, init_scale=10, mlp_zero_init=False, LR=1e-2):

```
INIT   var(out)=0.0120   var(local)=0.0012   var(multi)=0.0108   (90% multi-site!)
ep 1   var(out)=0.0015   loc/tot=0.34
ep 5   var(out)=0.0002
ep 20  var(out)=0.0012   α=[+0.39, +0.41, +0.40]   ‖fc2‖̄=0.055
```

* Total output variance drops two orders of magnitude after one epoch and
  stays there.
* `α` is *dropping* from 0.5 → ~0.39 in every layer — ReZero is choosing
  to attenuate the GEMHSA path, not amplify it.
* `‖fc2‖̄` drops from 0.07 → 0.055 — the MLP is being pushed toward the
  zero predictor.
* At init the multi-site fraction of output variance is **already 90 %**;
  the optimizer doesn't grow it, it shrinks the *whole* readout.

The model never gets stuck at a local minimum in the multi-site
composition direction — it never explores that direction at all because
the loss gradient steers it through the all-zero solution first.

## 4. Why optimization collapses to zero

After K equivariant layers the trace head feeds the MLP

```
T_x  :=  [ Re Tr W^{(K)}_c(x),  Im Tr W^{(K)}_c(x) ]_{c=0..C-1}
```

Decompose `W^{(K)}(x) = L(P(x)) + M(P-near-x)` into a site-local linear
piece `L` of order O(1) — set by the residual stream `W + α(W_act − W)`
which always carries the unmodified `P(x)` through — and a multi-site
piece `M` of order O(α^K · σ² · n_off) ≈ 10⁻² at init. The readout is

```
ŷ(x) = MLP( T_x ) = W_fc1·L(P(x))  +  W_fc1·M(P-near-x)  +  …
```

Both terms ride on the *same* downstream weights (`w_mix`, `fc2`, `α`).
There is no direction in weight space that scales down the L term
without scaling down the M term in the same proportion. The MSE
gradient, seeing E[L(P(x))·y] = 0, says "shrink the readout"; the
optimizer shrinks fc2, w_mix, and α together; M dies along with L.

This is the architectural failure mode. It is not a bad init, not a
learning-rate problem, not a capacity problem; it is the residual
stream pinning the readout to a feature that the loss tells the
optimizer to remove.

## 5. Why 1×1 doesn't show this failure

For the 1×1 target, `y(x) = Re P_{μν}(x)/nc`, and `Cov(P_c(x), y(x)) = 1`
for `c = (μ,ν)`. The MSE gradient *wants* fc2 to read out the
linear-in-P(x) component. ‖fc2‖ grows, α can stay near zero (or drift
either way), and the equivariant blocks are inert at convergence. The
1×1 run validates nothing about the GEMHSA's multi-site composition
ability — it is consistent with `α = 0` and randomly initialized
Q/K/V/mix.

## 6. Sub-leading observations (not the stall, but related)

* **`W_aug = [I, W, W†]` is degenerate on Z₂** — `W† = W` for real 1×1
  matrices, so 3 of the 7 augmented channels are exact copies of another
  3. Wastes capacity in the Q/K/V projections but doesn't cause the
  stall.
* **Transport averaging zeroes out at -1 plaquettes for diagonal Δx.**
  `T_{(1,1)}(x) = (U_μ(x)U_ν(x+μ̂) + U_ν(x)U_μ(x+ν̂))/2`. In Z₂ the two
  shortest paths' ratio is exactly `P_{μν}(x)`, so `T_{(1,1)}(x) ∈
  {±1, 0}` and `|T_{(1,1)}(x)|² ∈ {0, 1}`. Diagonal transports therefore
  *gate* on whether the 1×1 plaquette at x is +1 — they carry less
  information than the axis-aligned `T_{±ê_μ}`. Mitigated by the
  axis-aligned offsets in the L1-ball; not the root cause here.
* **At init `score` has std ≈ 0.04** even with `init_scale=10`. Softmax
  over 24 offsets (R=2, D=3) is essentially uniform. The orbit-tied
  bias `b_h` is zero-init and doesn't receive a strong gradient because
  the path it would amplify (multi-site composition) is being shrunk by
  the MLP. Bumping `init_scale` further pushes scores into the regime
  where softmax saturates on whichever offset's score happened to be
  largest at init, also bad.

## 7. Proposed fixes (smallest → most invasive)

### A. Strip the residual on the *last* GEMHSA layer

Two-line change inside `GEMHSA.forward`, gated by a `final_block` flag
set on the last block in the `GELT.__init__` stack:

```python
return self.alpha * W_act     # final block
return W + self.alpha * (W_act - W)   # otherwise
```

The trace head then reads only the gated/mixed path. The linear-in-P(x)
coefficient becomes one weight setting among many in `w_V`/`w_mix` and
can be driven to 0 by gradient descent independently of the multi-site
contributions. ReZero identity-at-init is preserved on layers 1..K-1.

Smallest risk: gauge equivariance is preserved (the gated/mixed path is
already equivariant), the rest of the stack is unchanged, and the change
is trivially reversible.

### B. Zero-init the `w_V` columns that mix `W` and `W†`

`W_aug = [I, W, W†]` has C̃ = 2C+1 columns. The "I" column gives a
per-(h, d) bias; the W and W† columns give the linear-in-input path.
Zero-initing the W and W† columns of `w_V` (only) makes V(x) constant
per slot at init, so the bilinear `Q†·V` is *purely* from Q's
W-dependence — and the entire value-path contribution is purely
multi-site (bilinear in inputs at x and x+Δx). The Q and K projections
keep their random init so attention can still steer.

This eliminates the linear-in-P(x) channel into the readout without
removing the residual stream. Combine optionally with (A) for belt and
braces.

### C. Two-stream readout

Keep the residual stream for training stability but read the trace head
off only the gated path. Define `W_out_for_residual = W + α(W_act − W)`
(unchanged) and `W_out_for_readout = W_act − W` (the *gated delta* only),
then send `W_out_for_readout` to the Trace block at the final layer.
Equivalent to (A) up to a per-layer scale, but easier to reason about.

### D. Add an explicit "loop-doubling" input channel

Pre-multiply pairs of plaquettes in the dataset builder: stack
`P_{μν}(x)`, `P_{μν}(x)·P_{μν}(x+ê_μ)`, etc., as additional input
channels. This is the L-CNN loop-doubling recipe applied at the data
layer rather than the architecture layer. Brings the model closer to the
"input is the answer" regime that made 1×1 trivial, at the cost of
making the architectural claim ("equivariant blocks compose products
from raw plaquettes") less interesting.

Use (D) only as a fallback if (A) and (B) both fail.

## 8. Suggested experimental order

1. Re-run on the V100 with the current script for ground truth on the
   stall (user is doing this).
2. Apply fix **(A)** — `final_block` flag stripping the residual at the
   last layer. Pass criterion: val loss drops noticeably below 1.0
   within 50 epochs at α_init=0.5, init_scale=1 (back to default).
3. If (A) is partial, layer **(B)** on top — zero-init `w_V`'s W and W†
   columns.
4. Sanity check after each fix: 1×1 target still trains to val ≈ 0, the
   gauge-equivariance unit test still passes at machine ε.
5. Only then consider (D).

## 9. What this means for the thesis claim

The pre-fix architecture is *expressive enough* (the value path
`α · Q†·Ṽ` produces bilinear-in-P contributions per layer, and 3 layers
can reach degree-8 polynomials in P with the L1-ball receptive field).
The failure is in optimization, specifically in the readout coupling.
After the fix, the equivariant blocks have to actively compose
products for any non-trivial target — which is precisely the regime the
thesis wants to evaluate. The fix sharpens the comparison against the
CNN baseline (which already reaches val ≈ 0.6 / R² ≈ 0.4 on the same
2×2 target via the global FC head) rather than weakening it.
