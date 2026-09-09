# What did the network find? — the learned operator decomposed against the classical span

Written 2026-09-06, **after** the run. Every other design record in `notes/` was
written before its measurement; this one was not, and §7 prices what that costs
and what was done to mitigate it. The study is a pure re-analysis of dumps that
already existed (`results/glueball/*_test_obars.pt`), so there was no ensemble to
pre-register against and no GPU time at stake — but the fit window, the metric
and the statistic could all in principle have been chosen after seeing the
answer, and that has to be said out loud rather than hidden behind a clean table.

Script: `scripts/operator_decomposition.py` (offline, CPU, seconds).
Figure/dump: `results/glueball/operator_decomposition.{png,pt}`.

---

## 1. The question §6.2 leaves open

`notes/glueball_spectroscopy.md` §6.2 and the paper's Table `tab:overlap` report

> the trained GELT operator carries more ground-state weight than the optimal
> combination of the classical Morningstar–Peardon basis:
> ΔA₀ = +0.078 ± 0.022 (3.6σ) over two independently sampled ensembles, at a
> consistent mass.

That is a statement that the network **wins**. It is not a statement about
**what it found**, and a referee who grants the win asks the next question
immediately:

> Is the advantage new operator content, or is it the same content in a linear
> combination the GEVP failed to locate?

Those are different results. The first says the classical variational space has
a ceiling and the network is above it. The second says the classical space was
fine and the *estimator* was noisy — still worth reporting, but a statement
about statistics, not about physics, and it would be fixed by more
configurations rather than by a network.

Nothing measured so far distinguishes them. The one hint on the record is §6.2's
"the enlarged five-operator GEVP collapses onto pure GELT", which is suggestive
but not quantitative: a GEVP collapsing onto one member says that member is best,
not that it lies outside the others' span.

## 2. The question has an exact answer, not an estimate

Operators acting on the vacuum are vectors in a Hilbert space, and the
equal-time connected correlator **is** the inner product on that space:

    ⟨O_a, O_b⟩ ≡ C_ab(0) = ⟨ d_a(t) d_b(t) ⟩,     d = Ō − ⟨Ō⟩,

which is exactly what `connected_correlator_matrix` already computes at Δ = 0.
So "the part of the learned operator the classical basis cannot express" is a
**literal orthogonal projection**, not a metaphor and not a fit:

    O_GELT = P + r,     P = proj_{span{Ō_i}} O_GELT,     r ⟂ span{Ō_i}.

Three things follow, and they are what make the decomposition worth doing:

1. **`P` is the strongest possible classical opponent for this comparison.** It
   is the best approximation of GELT *inside the span* by construction — better
   than the GEVP vector, better than APE×max, better than any classical operator
   selected for any other reason. So `ΔA₀(GELT − P) > 0` is a **lower bound** on
   the advantage over the whole span, and it cannot be attacked as "you compared
   against a badly chosen classical operator" (cf. the fair-fight audit,
   which attacks exactly that in the Z₂ study — see §6).

2. **The difference is far better conditioned than the published one.** The
   published ΔA₀ compares GELT to the classical GEVP ground vector: a *different*
   operator, correlated with GELT only through sharing configurations. `P` is
   GELT's own projection, so the two share every fluctuation living in the span
   and the jackknifed difference has much of that variance cancel.

3. **The same statistic applied to the classical operators themselves gives the
   missing scale.** "12.9% outside a four-dimensional span" means nothing on its
   own — *any* operator outside a 4D subspace has a large orthogonal component.
   The scale is the basis's own increments: how much new content does each rung
   of the smearing ladder add to the span of the rungs below it? That is free,
   it uses no new data, and it says whether the ladder is still growing or has
   saturated.

## 3. Design

- **Data.** The `(B, Nt) = (400, 24)` test-split Ō arrays already dumped by
  `train_glueball.py` for Run 5 and for the independent seed-1 replication:
  `gelt_obar` and the four-member `Obar_basis` (APE levels 0, 2, 4, 6), on
  configurations no checkpoint saw.
- **Conventions, inherited deliberately.** Fit window Δ ∈ [2, 7], `m_range`
  (0.05, 1.5), delete-block jackknife with the dump's own `jack_block = 10`,
  diagonal σ_Δ fixed from the full sample — i.e. identical to
  `fit_glueball_overlap.py`, so "same protocol as Table 4" is a fact about the
  code path rather than a claim.
- **The gate.** The script recomputes the published GELT-vs-GEVP comparison
  from its own code path and prints it beside the published values. It must
  reproduce them before anything else is read. It does (§4).
- **Amplitudes at a common fixed mass.** Z is linear in the operator, so
  `Z_G = Z_P + Z_r` holds exactly — but only if all three are read against the
  *same* exponential. Letting each piece fit its own m destroys the additivity
  that makes the split interpretable, so amplitudes come from a least-squares
  fit of `C(Δ) ≈ A·cosh_ref(Δ)` at the GELT reference mass. `Z_r/Z_G` is then
  read off the **cross**-correlator, `C_Gr(Δ) → Z_G Z_r e^{−mΔ}` against
  `C_GG(Δ) → Z_G² e^{−mΔ}`, which carries its own sign and needs no square root.
- **Contact-term test.** The whole decomposition is repeated under the metrics
  `C_ab(τ)` for τ = 1, 2, which weight the light states more heavily. A contact
  term — UV content living in C(0) and gone by Δ = 2, which is exactly what
  the (retired) rotational-symmetry study found the E-irrep contamination to be — must
  **shrink** as τ grows. Anything that grows is not a contact term.

## 4. Results (2026-09-06)

### The gate passes exactly

| | Run 5 | ens1 |
|---|---|---|
| m·a_t GELT | 0.3324 ± 0.0268 | 0.3745 ± 0.0308 |
| A₀ GELT | 0.9029 ± 0.0468 | 1.0134 ± 0.0617 |
| A₀ classical GEVP | 0.8374 ± 0.0555 | 0.9249 ± 0.0714 |
| ΔA₀ GELT − GEVP | **+0.0656 ± 0.0306 (2.1σ)** | **+0.0885 ± 0.0302 (2.9σ)** |

Identical to `fit_glueball_overlap.py` and to `su2_fair_fight.py`'s dump-only
mode, to the last digit.

One trap found while building the gate, recorded because it wastes an hour:
`fit_glueball_overlap.py` derives the GEVP arm's σ_Δ weights from the
**full-sample** projection sliced by the jackknife mask, while re-solving v₀
per sample inside the fit itself. Deriving σ from the per-sample projection
instead — which looks more principled — moves ΔA₀ from +0.066 to +0.050 and
makes the gate look failed when nothing has changed.

### The decomposition

| | Run 5 | ens1 |
|---|---|---|
| norm² fraction of O_GELT outside the classical span | **0.1289 ± 0.0041** (31.7σ) | **0.1169 ± 0.0030** (38.4σ) |
| `Z_r/Z_G` — ground-state amplitude carried by `r` | 0.1313 ± 0.0129 (10.1σ) | 0.1392 ± 0.0093 (14.9σ) |
| A₀ of `P` (best classical approximation of GELT) | 0.8295 ± 0.0509 | 0.9337 ± 0.0708 |
| A₀ of `r` alone | 0.4252 ± 0.1145 (3.7σ) | 0.3721 ± 0.0783 (4.8σ) |
| **ΔA₀ (GELT − P), correlated** | **+0.0734 ± 0.0255 (2.9σ)** | **+0.0797 ± 0.0294 (2.7σ)** |
| Δm (GELT − P) | −0.0058 ± 0.0126 (0.5σ) | −0.0209 ± 0.0128 (1.6σ) |

Combined over the two independent ensembles: **ΔA₀ = +0.0761 ± 0.0192 (4.0σ)**,
at a mass consistent with unchanged.

Two things to note about `A₀(P) = 0.830 ± 0.051`. First, it agrees with the
published GEVP arm's `0.837 ± 0.056` — the least-squares projection of GELT and
the variationally optimal classical operator land in the same place. That is a
non-trivial internal consistency check: it says the classical span really does
have a ceiling near A₀ ≈ 0.83, and that the ceiling is a property of the span
rather than an artifact of how the GEVP picks its vector. Second, it is what
makes `ΔA₀(GELT − P)` the honest headline: dropping `r` from GELT costs the
entire advantage.

### The mechanism: constructive interference, not a better operator

`r` on its own is a *poor* operator — A₀ = 0.43 against GELT's 0.90. It does not
win by being good; it wins by adding coherently. With `Z_G = Z_P + Z_r` and
`Z_r/Z_G ≈ 0.131`, removing `r` costs

    1 − (1 − 0.131)² = 24.5%  of the ground-state INTENSITY

while removing only **12.9%** of the norm². The ground-state fraction
`A₀ = A/C(0)` therefore falls, and that is the whole effect. (The two numbers are
of the right size to account for the observed 0.903 → 0.830; they are not
expected to match to three digits, because each arm's A₀ is quoted from its own
cosh fit rather than from the fixed-mass amplitudes.)

This is worth stating carefully in the paper, because the naive reading — "the
network found a better operator hiding outside the basis" — is wrong. What it
found is a *direction* that is mediocre by itself and valuable in superposition.
That is precisely what a variational method is supposed to exploit and precisely
what a four-member basis cannot reach.

### The scale: the classical ladder is a converged geometric series

The same statistic, applied to the classical operators themselves — new content
of rung *k* relative to the span of everything below it:

| addition | Run 5 | ens1 |
|---|---|---|
| APE×2 outside span{×0} | 0.6891 ± 0.0098 | 0.7049 ± 0.0084 |
| APE×4 outside span{×0, ×2} | 0.1971 ± 0.0047 | 0.1957 ± 0.0043 |
| APE×6 outside span{×0, ×2, ×4} | 0.0486 ± 0.0014 | 0.0500 ± 0.0012 |
| **GELT outside span{the whole basis}** | **0.1289 ± 0.0041** | **0.1169 ± 0.0030** |

Each rung adds ≈ 3.5–4× less than the one before, and the ratios replicate
across two independently sampled ensembles to the third decimal. Extrapolating
the decay, the *entire remaining tail* of the ladder — rungs 8, 12, 16, … — is
worth ≈ 2% of new content. The learned operator contributes 13%.

This is the sentence the study exists to license:

> The classical smearing ladder is a geometric series that has essentially
> converged; the learned operator is not in its limit, and the part of it that
> is not is what carries the ground-state advantage.

(The extrapolation is a heuristic, not a theorem: each fraction is normalised to
its own operator's norm, so summing them is dimensionally loose. It is quoted as
an order of magnitude and nothing turns on the second digit.)

### It is not a contact term

The out-of-span fraction under the `C(τ)` metric:

| metric | Run 5 | ens1 |
|---|---|---|
| C(0) | 0.1289 ± 0.0041 | 0.1169 ± 0.0030 |
| C(1) | 0.1563 ± 0.0062 | 0.1345 ± 0.0049 |
| C(2) | 0.1974 ± 0.0226 | 0.1810 ± 0.0576 |

It **grows** — the out-of-span content is *more* prominent in the light-state
sector than in the full operator. That is the opposite of the E-irrep
contamination of the retired rotational-symmetry study, which lived in C(0), left ξ
untouched (2.27 → 2.26 etc.) and was gone by Δ = 2. The right panel of the
figure shows the same thing directly: ρ(Δ) for `r` is flat from Δ ≈ 3, i.e. `r`
couples to the same ground state rather than dying with the UV.

### Window stability

`ΔA₀(GELT − P)`, C(0) metric, both ensembles:

| window | Run 5 | ens1 |
|---|---|---|
| [1, 6] | +0.0542 ± 0.0127 (4.3σ) | +0.0461 ± 0.0104 (4.4σ) |
| **[2, 7]** (quoted) | **+0.0734 ± 0.0255 (2.9σ)** | **+0.0797 ± 0.0294 (2.7σ)** |
| [2, 8] | +0.0726 ± 0.0263 (2.8σ) | +0.0923 ± 0.0306 (3.0σ) |
| [3, 8] | +0.0869 ± 0.0419 (2.1σ) | +0.1510 ± 0.0554 (2.7σ) |

Positive in every window on both ensembles, 2.1σ–4.4σ. The quoted window is the
inherited one, not the best one — [1, 6] would give a larger significance and is
*not* quoted, because Δ = 1 is where the classical arm's residual contamination
sits and `fit_glueball_overlap.py` excludes it for that reason.

## 5. Two controls this does NOT have

Both are stated up front because the numbers above must be read with them
attached, and neither can be run from a dump.

1. **A random-init GELT.** If an untrained network is *also* ≈ 13% outside the
   span, then being outside is **architectural** — the L1-ball transport reaches
   offsets and path-averages that no smeared plaquette contains — rather than
   learned. The norm fraction would then be a fact about the architecture and the
   *learned* claim would rest entirely on `Z_r/Z_G` and `ΔA₀`, which is a weaker
   but still real statement. This is one GPU eval pass with no training
   (`GLUEBALL_EVAL_ONLY` against an untrained checkpoint) and it is the single
   highest-value follow-up in the repo. **Pre-registered here, before running
   it:** the honest expectation is that a random net *is* substantially outside
   the span (the transport guarantees it) but carries far less of `Z`, so the
   discriminating statistic is `Z_r/Z_G`, not the norm fraction. If the random
   net matches on both, the study reduces to a statement about the architecture
   and must be reported that way.

2. **A stronger classical span.** This decomposes against the *published*
   four-level basis. `scripts/su2_fair_fight.py` already builds the strengthened
   ones (`deep` = more smearing levels, `shapes` = cubic-symmetrised R×T loops,
   `full` = both). The decomposition belongs there too, against `full`.

## 6. A prediction the fair fight will test

The two studies attack the same question from opposite sides — the fair fight
asks "was the opponent strong enough?", the decomposition asks "is the win
outside the opponent's reach?" — so they constrain each other, and the
decomposition makes a falsifiable prediction about the fair fight's outcome:

> The `deep` arm (more smearing levels) **cannot** close a 13% gap, because more
> smearing is exactly the direction the ladder has already converged along —
> rungs 8, 12, 16 are worth ≈ 1.3%, 0.35%, 0.1% of new content by the geometric
> fit above. Only the `shapes` arm, which adds genuinely new loop geometry
> rather than more radius, can move `A₀` materially.

If `deep` closes the gap anyway, this note's reading is wrong and the norm-based
scale argument must be withdrawn: new *norm* content and new *useful* content
would have come apart, and only the latter matters.

## 7. Known limitations, stated up front

- **Not pre-registered.** The measurement was run before this note was written.
  The mitigations are that the window and metric were *inherited* rather than
  chosen (§3), that both are scanned rather than reported at one setting (§4),
  and that the two ensembles were analysed with identical code and agree. None
  of that is as good as pre-registration, and it should not be described as if
  it were.
- **One theory, one lattice family, one architecture.** SU(2), 12³×24, β = 2.4,
  ξ = 3, R = 2, the Run-5 hyperparameters. The statement is about *this* learned
  operator against *this* classical basis.
- **The classical basis is the published one**, with the two controls of §5
  outstanding.
- **`A₀ > 1` on ens1** (1.013 ± 0.062). That is a fit artifact, not a
  probability violation — A₀ is a ratio of a fitted amplitude to a measured
  C(0) and can exceed 1 within errors when the operator is nearly pure. It is
  inherited from the published Table and is not made worse here, but the paper
  should say so once rather than leave a reader to notice it.
- **Δ = 0 is where the metric lives, and Δ = 0 is the noisiest place on the
  correlator.** The C(τ) scan (§4) is the mitigation and it points the right way,
  but the quoted 12.9% is a C(0) number and inherits C(0)'s UV sensitivity.

## 8. What this licenses, and what it does not

- **Do** report: the learned operator has a component outside the span of the
  entire classical basis; it is 12.9% / 11.7% of the norm² on two independent
  ensembles; it carries 13–14% of the ground-state amplitude; removing it costs
  the whole A₀ advantage (ΔA₀ = +0.076 ± 0.019, 4.0σ combined) at unchanged mass;
  and the classical ladder's own increments (0.69 → 0.20 → 0.05) show the basis
  is saturating while the network's contribution is not on that curve.
- **Do** report the mechanism honestly: `r` is a *poor* operator alone
  (A₀ = 0.43) that wins by constructive interference, not a better operator
  hiding outside the basis.
- **Do not** yet claim the out-of-span content is *learned* — §5 control 1 is
  what separates learned from architectural, and it has not been run.
- **Do not** quote `ΔA₀(GELT − P)` as replacing the published
  `ΔA₀(GELT − GEVP)`. They answer different questions and the paper should carry
  both: the GEVP difference is the comparison against the standard method, the
  P difference is the localisation of where the advantage lives.
- **Do not** transport the number to Z₂ without re-measuring. The Z₂ classical
  basis is a different and, as of 2026-09-06, a **broken** object — see
  `notes/audit_2026-09-06.md` §2.
