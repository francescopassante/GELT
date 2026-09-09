# GELT — piani futuri

I tre documenti di brainstorming che stavano in `plans1.md`, `plans2.md` e
`plans3.md` (scritti fra il 2026-09-06 e il 2026-09-07), uniti qui senza
riscrivere il contenuto: sono **proposte**, non risultati, e nessuna di esse è
implementata nel repo. Si sovrappongono di proposito — le tre liste sono state
generate da punti di partenza diversi, e le idee che compaiono in tutte e tre
(flow generativi equivarianti, spettro J^PC completo, smearing appreso,
fermioni/Dirac) sono quelle su cui converge ogni lettura del codice.

| parte | origine | taglio |
|---|---|---|
| [A](#parte-a--10-direzioni-possibili-per-gelt) | `plans1.md` | 10 direzioni lette dal codice (`lattice.py`, i sampler, `glueball.py`, i blocchi), ordinate per *tipo di scommessa*. In italiano. |
| [B](#parte-b--scientific-horizons-for-gelt-10-high-impact-research-plans) | `plans2.md` | 10 traiettorie scientifiche a lungo raggio, con matrice impatto/novità/fattibilità e formulazione matematica per ciascuna. La più lunga. |
| [C](#parte-c--gelt--10-future-routes-spark-plans) | `plans3.md` | 10 rotte brevi, ognuna con il primo esperimento concreto e il criterio di successo. Un paragrafo l'una. |

---

## Parte A — 10 direzioni possibili per GELT

Generate guardando il codice (`lattice.py` + `build_transport_average`, i sampler
esatti Z₂ heat-bath / SU(2) heat-bath+OR con anisotropia, `glueball.py` con GEVP,
cosh fit e jackknife, `ising.py` per la dualità, `lcnn.py` come baseline, i due
blocchi GEMHSA) e non le note esistenti. Ordinate per *tipo di scommessa*, non per
ordine di preferenza.

---

### 1. Base variazionale appresa per l'intero spettro di glueball (non un solo operatore)

- **Domanda**: una rete equivariante può produrre *k* operatori simultanei, ortogonali nella metrica `C_ab(0)`, uno per irrep del gruppo cubico (A₁⁺⁺, E⁺⁺, T₂⁺⁺, e i settori P/C dispari), e battere una base costruita a mano di ~100 loop?
- **Perché conta**: lo spettro eccitato è il problema costoso vero della spettroscopia di glueball; oggi si compra con basi enormi di loop smeared. Un generatore di basi appreso sposta il costo dall'uomo alla GPU.
- **Meccanismo**: loss = quoziente di Rayleigh a blocchi (traccia del GEVP proiettato) con vincolo di ortogonalità soft, `GELT(reduction="none")` con *k* teste di output; proiezione esplicita sulle 24 rotazioni del reticolo *dentro* il forward, non a posteriori.
- **Rischio**: medio. Il pezzo delicato è che il GEVP appreso non collassi su *k* copie dello stato fondamentale — serve un penalty su `C(0)` off-diagonale.
- **Esito falsificabile**: A₀ del primo eccitato in E⁺⁺ contro la base classica smeared, su configurazioni condivise, con jackknife correlato.

### 2. Sampler generativo equivariante: attention come coupling layer di un normalizing flow

- **Domanda**: il trasporto L1-ball con score gauge-invariante può fare da condizionatore *non locale* in un flow gauge-equivariante, dove i flow esistenti usano solo convoluzioni/plaquette locali?
- **Perché conta**: è la linea più visibile dell'ML-per-LGT (Kanwar–Albergo–Shanahan, Boyda per SU(N)) e il suo collo di bottiglia dichiarato è lo scaling in volume e la mode collapse. Un condizionatore ad ampio raggio è esattamente l'ingrediente mancante.
- **Meccanismo**: coupling attivo/passivo sui link mascherati; la parte attiva scala le plaquette con parametri prodotti dall'attention sulla parte congelata; Jacobiano diagonale per costruzione. Training reverse-KL sull'azione di Wilson.
- **Rischio**: **alto**. Ma il repo ha ground truth esatta gratis: Z₂ 2D/3D con heat-bath esatto e ⟨P⟩ analitico, SU(2) 2D con `I₂/I₁`. Si può fallire in modo pulito e informativo.
- **Esito falsificabile**: ESS del flow vs volume, a β fisso, contro il heat-bath. Se ESS non degrada come i flow convoluzionali, è un paper a sé.

### 3. Smearing appreso: sostituire APE/HYP/gradient flow con una mappa covariante differenziabile

- **Domanda**: la stessa architettura, usata come mappa link→link invece che come osservabile, produce uno smearing con raggio *tunabile*, esattamente covariante e differenziabile?
- **Perché conta**: lo smearing è ovunque (spettroscopia, topologia, azioni migliorate) ed è tutto a mano. Nagai–Tomiya hanno fatto la versione locale a cammino fisso; qui la novità è il pesaggio *appreso e dipendente dal contenuto* su tutti i cammini minimi.
- **Meccanismo**: `U'_μ(x) = project[(1−α)U + Σ_paths w(attention) · staple]`; test di covarianza già esistenti in `test_glueball.py` riusabili verbatim.
- **Rischio**: **basso**. È il progetto con il miglior rapporto risultato/rischio della lista.
- **Bonus strategico**: se la mappa è resa invertibile con Jacobiano calcolabile, diventa una *field transformation* dentro HMC — e il piano 3 si fonde col piano 2 dal lato "trivializing map" di Lüscher.

### 4. Fermioni: multigrid / precondizionatore appreso per l'operatore di Dirac

- **Domanda**: il blocco con trasporto L1-ball funziona da prolongatore/operatore di griglia grossa per un solver di Dirac?
- **Perché conta**: nella QCD reale l'inversione di Dirac domina il costo. Il multigrid gauge-equivariante (Lehner–Wettig) è l'area più "adottabile" dell'intero campo — un risultato qui viene letto dai lattice people, non solo dagli ML people.
- **Meccanismo**: modello di Schwinger 2D (U(1) + staggered) come testbed, poi SU(2) 3D; loss = riduzione del residuo dopo *n* iterazioni, oppure overlap con i vettori quasi-nulli.
- **Rischio**: medio-alto — richiede di aggiungere fermioni al repo (assenti oggi) e U(1) al registry dei sampler (che è letteralmente una entry in `_PROPOSAL_FN`).
- **Perché è credibile qui**: il pezzo difficile — trasporto parallelo corretto e testato per covarianza — esiste già.

### 5. Control variates equivarianti: ridurre la varianza di *ogni* misura

- **Domanda**: si può addestrare una rete equivariante `f` tale che `⟨∇·f⟩ = 0` esattamente (identità di Schwinger–Dyson sul gruppo), e usarla come control variate a media nulla per Wilson loop, ⟨P⟩, correlatori?
- **Perché conta**: è un guadagno *puro* di statistica, gratis, esatto, senza bias. Si applica retroattivamente a tutte le misure già fatte. Parente stretto delle contour deformations di Detmold–Kanwar–Lawrence–Wagman per SU(N), ma con un ansatz equivariante molto più espressivo.
- **Meccanismo**: loss = varianza empirica di `O − f`; il vincolo a media nulla è garantito dalla struttura, non addestrato.
- **Rischio**: **basso**, e il criterio di successo è un numero solo (fattore di riduzione della varianza).
- **Nota**: è il piano che rende meglio se il tempo GPU è poco, perché si allena su ensemble già in `datasets/`.

### 6. Congelamento topologico: proposte collettive apprese

- **Domanda**: una rete equivariante può proporre aggiornamenti collettivi che cambiano *Q* con accettanza non trascurabile dove l'HMC locale si congela?
- **Perché conta**: il topological freezing è il problema numero uno del reticolo fine. Qualunque progresso reale è pubblicabile su PRL.
- **Meccanismo**: U(1) compatto in 2D (dove il congelamento è drammatico, la topologia è esatta e il volume è minuscolo) come testbed onesto; accept/reject Metropolis esatto sopra la proposta appresa, quindi **zero bias per costruzione** anche se la rete è pessima.
- **Rischio**: alto sull'esito, nullo sulla correttezza. È il tipo di scommessa giusta: fallire dà comunque una misura di accettanza pubblicabile come negativa.

### 7. Il limite continuo di un osservabile appreso

- **Domanda**: un operatore appreso a un passo reticolare ha un limite continuo? La sua "scala" (raggio efficace, contenuto di loop) converge in unità fisiche quando `a → 0` a fisica fissa?
- **Perché conta**: nessuno lo chiede mai, e senza risposta ogni osservabile ML è un artefatto di un reticolo. È la domanda che trasforma una tesi di ML in una tesi di fisica.
- **Meccanismo**: 3–4 valori di β lungo una linea di scala fissa (r₀ o la massa gap stessa), stesso volume fisico, ri-addestramento indipendente a ogni β; si confronta la quantità rinormalizzata, non il numero grezzo.
- **Rischio**: medio, costo GPU alto ma prevedibile (l'anisotropia e il fit sono già scritti).
- **Esito falsificabile**: se il raggio efficace in unità fisiche *non* converge, la conclusione è netta e va detta.

### 8. Confinamento: potenziale statico e il tubo di flusso letto dall'attention

- **Domanda**: addestrando su correlatori di Polyakov / loop di Wilson R×T, l'attention si organizza lungo il tubo di flusso tra le due cariche statiche?
- **Perché conta**: dà una *figura* — la mappa di attention che disegna la stringa — con ground truth quantitativa (tensione di stringa σ, termine di Lüscher `−π/12R`, profilo trasverso del tubo, allargamento logaritmico).
- **Meccanismo**: SU(2) 3D dove σ è ben misurabile; sorgente statica fissata, si guarda `_last_alpha` condizionata sulla separazione R.
- **Rischio**: **basso-medio**. La parte di misura classica (σ dai loop R×T) è già supportata da `rectangular_wilson_loop`.
- **Perché è forte**: è interpretabilità con un predittore quantitativo, non con un aneddoto visivo.

### 9. RG reale sul reticolo: blocking equivariante appreso (neural MCRG)

- **Domanda**: si può imparare una trasformazione di blocking gauge-covariante che preserva la fisica a lunga distanza, e leggere da essa il flusso delle costanti di accoppiamento e gli esponenti critici?
- **Perché conta**: l'MCRG è classico ma laborioso; una versione appresa dà accesso alla traiettoria rinormalizzata e a ν, e in Z₂ 3D la dualità Ising fornisce esponenti *esatti* come verifica.
- **Meccanismo**: blocking 2→1 con l'attention come kernel; criterio = matching dei loop a lunga distanza tra reticolo bloccato e reticolo fine a β'.
- **Variante ad alto rischio/alta novità**: chiedere alla rete di scoprire un parametro d'ordine *non locale* (di disordine, tipo 't Hooft/monopolo) partendo solo da input locali gauge-invarianti. Un risultato positivo sarebbe sorprendente; uno negativo è comunque un teorema empirico interessante sui limiti dell'architettura.
- **Rischio**: medio-alto, ma la ground truth esatta via dualità lo rende difendibile.

### 10. Teoria: che algebra di loop genera l'attention a cammini mediati?

- **Domanda**: qual è esattamente la classe di funzionali di Wilson-loop rappresentabili da un GELT di profondità *N*, raggio *R*, larghezza *C*? Il path-averaged transport aumenta o *riduce* il set raggiungibile rispetto al cammino singolo dell'L-CNN?
- **Perché conta**: L-CNN ha un teorema di universalità; l'attention bilineare eredita l'argomento di loop-doubling, ma la media sui cammini è una modifica non banale — potrebbe *distruggere* la separazione di alcuni loop (proiettando su un sottospazio simmetrico) e va dimostrato, non assunto.
- **Meccanismo**: analisi carta-e-penna + esperimento minuscolo: costruire coppie di configurazioni distinte con gli stessi loop fino a lunghezza `ℓ` e verificare la separabilità; regredire l'operatore appreso su una base completa di loop per leggerne il contenuto.
- **Rischio**: **bassissimo** in compute, alto in fatica intellettuale.
- **Perché conta strategicamente**: è l'unico piano che *deve* essere fatto comunque, perché ogni altro risultato poggia sull'assunzione che l'architettura sia almeno espressiva quanto la baseline.

---

### Raccomandazione

Se dovessi scegliere tre e non dieci:

| | Piano | Impatto | Successo | Costo GPU |
|---|---|---|---|---|
| **Fondamenta** | **10** (teoria dell'espressività) | medio | quasi certo | ~0 |
| **Rendimento sicuro** | **3** (smearing appreso) | alto | alto | basso |
| **Scommessa** | **2** o **4** (flow / multigrid) | molto alto | ~30–40% | alto |

Il piano **10** va fatto in ogni caso e in parallelo: costa quasi nulla e mette in
sicurezza tutto il resto. Il **3** è la scelta migliore se serve un risultato certo,
perché riusa i test di covarianza esistenti e sbocca naturalmente nel **2**. Tra le
due scommesse, il **4** (fermioni/multigrid) ha il pubblico più ampio nella comunità
del reticolo, il **2** quello più ampio nella comunità ML — la scelta dipende da dove
vuoi essere letto.

Il piano **5** (control variates) è la sorpresa della lista: rischio basso, novità
reale, e gira sugli ensemble già in `datasets/` senza campionare niente di nuovo.

---

## Parte B — Scientific Horizons for GELT: 10 High-Impact Research Plans

This document establishes ten major, high-impact scientific trajectories originating from the **Gauge-Equivariant Lattice Transformer (GELT)** framework. Rather than focusing on local codebase tasks or incremental technical caveats, these proposals evaluate the paradigm shift introduced by GELT—namely:
1. **Exact, non-perturbative gauge equivariance and invariance** by construction;
2. **Matrix-bilinear value paths** ($\alpha \cdot Q^\dagger \cdot \tilde{V}$) enabling multiplicative gauge-loop scaling;
3. **All-shortest-path parallel transport averaging** across $L^1$ Manhattan balls via dynamic programming;
4. **Variational spectral optimization** via physics objectives (Rayleigh quotient saturation under transfer-matrix discipline);
5. **The attention map as a physical lattice observable** whose connected correlators possess a well-defined spectral mass and correlation length.

---

### Executive Comparison Matrix

| # | Scientific Trajectory | Primary Physics Domain | Scientific Impact (1–10) | Novelty (1–10) | Feasibility / Success Probability (1–10) | Computational Bottleneck |
|---|----------------------|------------------------|--------------------------|----------------|------------------------------------------|---------------------------|
| **1** | Complete $J^{PC}$ Glueball Spectrum & Hybrids | Hadron Spectroscopy | 9.0 | 9.0 | 9.0 | Receptive field memory at $R \ge 3$ |
| **2** | Scaling to $SU(3)$ & Physical Continuum Limit | Precision Lattice QCD | 9.5 | 8.5 | 8.5 | Memory footprint of $3\times 3$ transport tensors |
| **3** | Dynamical Quark Matter & Hadron Spectroscopy | Nuclear & Particle Physics | 10.0 | 9.5 | 7.5 | Fermion inversion / Dirac matrix contractions |
| **4** | Flux-Tube Geometry & Static Potential $V(r)$ | Confinement Physics | 8.5 | 8.5 | 9.0 | Statistical noise at large source separation $r$ |
| **5** | Continuous Normalizing Flows on Lie Manifolds | Algorithmic LGT & GenAI | 9.5 | 9.5 | 7.0 | Stiff ODE integration / trace Jacobian computation |
| **6** | Thermal Phase Transitions & Deconfinement | Thermal Field Theory | 8.5 | 8.5 | 9.0 | High aspect ratio spatial lattices ($N_s \gg N_t$) |
| **7** | Flow-Free Topological Charge Operators | Vacuum Topology & Strong CP | 9.0 | 9.0 | 8.0 | Discretization artifacts of non-smooth link fields |
| **8** | Neural Quantum States for Hamiltonian LGT | Quantum Simulation & VMC | 9.5 | 10.0 | 7.5 | Stochastic Reconfiguration sampling variance |
| **9** | Curved Spacetimes & Simplicial Gauge Fields | Quantum Gravity & Holography | 9.0 | 10.0 | 6.5 | Non-uniform graph parallel transport DP |
| **10** | Non-Perturbative Wilsonian Renormalization Flow | Renormalization & Field Theory | 9.5 | 9.5 | 7.5 | Expressivity of real-space block transformations |

---

### Plan 1: Complete $J^{PC}$ Glueball Spectrum and Exotic Hybrids via $O_h \times \mathcal{P} \times \mathcal{C}$ Projective Attention

#### 1. Physics Motivation & The Open Problem
For over three decades, non-perturbative glueball spectroscopy in pure Yang-Mills theory has relied on the Morningstar–Peardon variational basis of hand-crafted, spatially smeared closed Wilson loops. While the scalar ground state $0^{++}$ is relatively accessible, the rest of the glueball spectrum—notably tensor states ($2^{++}$), pseudoscalars ($0^{-+}$), vector states ($1^{+-}$), and especially **exotic hybrid glueballs** ($1^{-+}, 0^{+-}, 2^{+-}$ whose quantum numbers forbid mixing with conventional quark-antiquark mesons)—suffers from severe signal-to-noise degradation. Classical multi-level Generalized Eigenvalue Problems (GEVP) struggle with high excited-state contamination and severe basis truncation errors in these noisy channels.

#### 2. Core Hypothesis & GELT-Specific Mechanism
GELT's attention scores and value paths can be decomposed directly into the irreducible representations (irreps) of the octahedral point group $O_h$ (cubic symmetry group with parity) and charge conjugation $\mathcal{C}$. Because parallel transport is performed over the full $L^1$-ball rather than along single axial paths, GELT can explore the complete space of closed Wilsonian loops with non-trivial spatial geometries (e.g. folded staples, non-planar loops) in a single block. By formulating a multi-state trace Rayleigh quotient loss, GELT can simultaneously learn an orthogonal set of optimal variational operators for an entire spin-parity multiplex.

#### 3. Mathematical & Architectural Formulation
- **Cubic Group Projection:** The spatial point group on a hypercubic timeslice is the octahedral group $O$ (24 elements) with parity $\mathcal{P}$, forming $O_h = O \times \mathbb{Z}_2$ (48 elements). The five single-valued irreps of $O$ are:
  $$\Gamma \in \{A_1 \text{ (scalar, } J=0\text{)}, A_2, E \text{ (doublet, } J=2\text{)}, T_1 \text{ (triplet, } J=1\text{)}, T_2 \text{ (triplet, } J=2\text{)}\}$$
- **Projective Attention Layers:** Let $\mathcal{R} \in O_h$ denote a lattice rotation/inversion. Under spatial rotation, the displacement vector $\Delta \vec{x}$ transforms as $\mathcal{R} \Delta \vec{x}$, and the gauge links transform accordingly. For an irrep $\Gamma$ of dimension $d_\Gamma$ with character $\chi^{(\Gamma)}$, the symmetry projection operator is:
  $$\mathcal{P}^{(\Gamma)} = \frac{d_\Gamma}{|O_h|} \sum_{\mathcal{R} \in O_h} \chi^{(\Gamma)}(\mathcal{R})^* \mathcal{T}_{\mathcal{R}}$$
  Instead of post-hoc projection after readout, we embed $\mathcal{P}^{(\Gamma)}$ into the attention heads:
  $$\alpha_{x \to y}^{(\Gamma)} = \frac{d_\Gamma}{|O_h|} \sum_{\mathcal{R} \in O_h} \chi^{(\Gamma)}(\mathcal{R})^* \operatorname{Softmax}_y \left( \frac{1}{\sqrt{d_{k}}} \operatorname{Re}\Tr\left[ Q(x)^\dagger \tilde{K}(\mathcal{R}(y-x) + x) \right] \right)$$
- **Charge Conjugation $\mathcal{C}$:** Under charge conjugation, $U_\mu(x) \to U_\mu^*(x)$. Operators are split into even ($\mathcal{C}=+1$) and odd ($\mathcal{C}=-1$) channels via anti-symmetrization:
  $$\mathcal{O}^{(\mathcal{C}=\pm)}(U) = \frac{1}{2} \left[ \mathcal{O}(U) \pm \mathcal{O}(U^*) \right]$$
- **Multi-State Variational Objective (Trace Rayleigh-Ritz Loss):** For $K$ target states in a specific $J^{PC}$ channel, define a vector of operators $\vec{\mathcal{O}}(t) = (\mathcal{O}_1(t), \dots, \mathcal{O}_K(t))^T$. The loss minimizes the sum of energy levels by optimizing the generalized eigenvalues:
  $$\mathcal{L}_{\text{multi}} = -\Tr\left[ \left( \mathbb{E}[ \vec{\mathcal{O}}(0) \vec{\mathcal{O}}(0)^T ] + \epsilon \mathbb{I} \right)^{-1} \mathbb{E}[ \vec{\mathcal{O}}(1) \vec{\mathcal{O}}(1)^T ] \right] + \lambda_{\text{ortho}} \sum_{i \neq j} \left( \frac{C_{ij}(0)}{\sqrt{C_{ii}(0) C_{jj}(0)}} \right)^2$$

#### 4. Implementation & Computational Roadmap
1. **Milestone 1 (Symmetry Module):** Implement the discrete $O_h$ action on the DP transport tensor $T_{\Delta x}(x)$ and construct exact character projectors for $A_1^{++}, E^{++}, T_2^{++}, A_1^{-+}, T_1^{+-}$.
2. **Milestone 2 (Synthetic Multiplet Benchmark):** Validate on synthetic multi-exponential correlators with degenerate states and known orthogonal energy splittings.
3. **Milestone 3 ($SU(2)$ Spectrum Production):** Train on the existing anisotropic $12^3 \times 24$, $\beta=2.4$, $\xi=3.0$ ensemble to extract the $0^{++}, 2^{++}$, and $0^{-+}$ masses simultaneously. Compare ground-state overlap $A_0$ against standard 16-operator GEVP.
4. **Milestone 4 (Exotic Channel Search):** Probe the $1^{-+}$ hybrid channel, which has zero mixing with quark states in QCD, to establish whether GELT can extract a plateau where classical operators fail.

#### 5. Scientific Impact & Novelty
- **First deep learning architecture** to compute the full non-scalar glueball spectrum with exact space-time group-theoretic irrep guarantees.
- Delivers a variational benchmark that can supersede Morningstar–Peardon for exotic hybrid glueball states.

#### 6. Feasibility & Risk Analysis
- **Primary Risk:** Channel mixing caused by finite lattice discretization (cubic breaking of continuous rotational symmetry $SO(3) \to O_h$).
- **Mitigation:** In $O_h$, spin-2 decomposes into $E \oplus T_2$. Mass degeneracy between the extracted $E$ and $T_2$ states serves as an exact, built-in non-perturbative check of continuum restoration.

---

### Plan 2: Scaling to Pure $SU(3)$ Yang-Mills and the Physical Continuum Limit

#### 1. Physics Motivation & The Open Problem
The definitive benchmark of modern strong-interaction physics is $SU(3)$ quantum chromodynamics. While $SU(2)$ provides a clean theoretical sandbox, only $SU(3)$ maps directly to physical hadron and glueball states observed in experimental facilities (e.g. BESIII, GlueX, FAIR, Belle II). Crucially, any claim that neural network variational operators outperform classical GEVP must be proven to survive the **continuum limit** ($a \to 0$) and the **infinite-volume limit** ($L \to \infty$). Without a continuum extrapolation, an operator advantage might merely exploit unphysical ultraviolet lattice discretization artifacts.

#### 2. Core Hypothesis & GELT-Specific Mechanism
The tensor architecture of GELT is already formulated in coordinate-free matrix notation ($A @ B$, `group.dagger(A)`), meaning the color dimension $N_c=3$ is mathematically native. By combining a parallelized Cabibbo–Marinari $SU(3)$ pseudo-heatbath sampler with an offset-chunked attention kernel, GELT can be scaled to fine lattice spacings ($a \le 0.08\text{ fm}$) across multiple $\beta$ values to achieve the first machine-learned continuum extrapolation of the pure gauge glueball mass ratio $m_{0^{++}} / \sqrt{\sigma}$.

#### 3. Mathematical & Architectural Formulation
- **$SU(3)$ Group Primitives:**
  - Group elements $U \in SU(3)$ parameterized as complex $3\times 3$ matrices with $\det(U)=1$ and $U^\dagger U = \mathbb{I}$.
  - Color trace: $\Tr[U] = \sum_{a=1}^3 U_{aa}$. Action: $S = \beta \sum_{\square} (1 - \frac{1}{3} \operatorname{Re}\Tr P_\square)$.
  - Group projection $\Pi_{SU(3)}(V)$: SVD-based polar decomposition $V = W \Sigma Z^\dagger \implies U = W Z^\dagger$, followed by phase adjustment $\det(U)^{-1/3}$ to fix the special unitary determinant.
- **Memory-Chunked Attention Engine:** For $SU(3)$, the transport tensor $T_{\Delta x}(x)$ carries shape $(B, N_{\text{offsets}}, L_x, L_y, L_z, 3, 3)$. For $R=2$ in 3D, $N_{\text{offsets}} = 32$; for $R=3$, $N_{\text{offsets}} = 122$.
  To avoid the $O(N_{\text{offsets}} \cdot V)$ memory wall, we reformulate the attention computation into tiled offset chunks:
  $$\alpha_{x \to x+\Delta x} = \frac{\exp\left( \frac{1}{\sqrt{d_k}} \operatorname{Re}\Tr\left[ Q(x)^\dagger T_{\Delta x}(x) K(x+\Delta x) T_{\Delta x}^\dagger(x) \right] \right)}{\sum_{\Delta x' \in B_R} \exp\left( \frac{1}{\sqrt{d_k}} \operatorname{Re}\Tr\left[ Q(x)^\dagger T_{\Delta x'}(x) K(x+\Delta x') T_{\Delta x'}^\dagger(x) \right] \right)}$$
  The denominator is accumulated using online running max/sum reductions (FlashAttention-style) without storing the full $(B, N_{\text{offsets}}, V)$ tensor in high-bandwidth memory.
- **Continuum Extrapolation Formulation:** Ensembles are generated across four lattice couplings $\beta \in \{5.85, 6.0, 6.2, 6.4\}$ at fixed physical volume $L \cdot a \approx 1.5\text{ fm}$. The physical scale is set by the static quark potential Sommer parameter $r_0 \approx 0.5\text{ fm}$ or string tension $\sqrt{\sigma} \approx 440\text{ MeV}$. The continuum limit is extracted via Symanzik effective theory:
  $$m_{0^{++}}(\beta) \cdot r_0 = (m_{0^{++}} \cdot r_0)_{\text{cont}} + c_1 \left(\frac{a(\beta)}{r_0}\right)^2 + \mathcal{O}(a^4)$$

#### 4. Implementation & Computational Roadmap
1. **Milestone 1 ($SU(3)$ Sampler & Primitives):** Implement Cabibbo–Marinari 3-subgroup $SU(2)$ decomposition with Kennedy–Pendleton heatbath and overrelaxation in PyTorch/C++ CUDA.
2. **Milestone 2 (Chunked Attention Kernel):** Implement custom PyTorch autograd function performing streamed $L^1$-ball transport matmuls to reduce peak VRAM by $4\times$.
3. **Milestone 3 (Multi-$\beta$ Ensemble Generation):** Sample four anisotropic ensembles:
   - $\beta=5.85, 12^3 \times 36, \xi=3.0$ ($a_s \approx 0.12\text{ fm}$)
   - $\beta=6.00, 16^3 \times 48, \xi=3.0$ ($a_s \approx 0.09\text{ fm}$)
   - $\beta=6.20, 20^3 \times 60, \xi=3.0$ ($a_s \approx 0.068\text{ fm}$)
   - $\beta=6.40, 24^3 \times 72, \xi=3.0$ ($a_s \approx 0.051\text{ fm}$)
4. **Milestone 4 (Spectroscopy & Extrapolation):** Train GELT on each ensemble using the single-timeslice Rayleigh loss, perform jackknife cosh fits, and extrapolate $(m_{0^{++}}/\sqrt{\sigma})_{a \to 0}$.

#### 5. Scientific Impact & Novelty
- Directly proves whether neural network variational operators retain superior ground-state purity in physical $SU(3)$ QCD in the continuum limit.
- Provides the lattice community with a validated, open-source equivariant transformer capable of replacing legacy smearing suites.

#### 6. Feasibility & Risk Analysis
- **Primary Risk:** Exponential degradation of signal-to-noise at fine lattice spacings due to gauge fluctuations.
- **Mitigation:** The anisotropic formulation ($\xi=3.0$) keeps temporal spacing $a_t = a_s / 3$ ultra-fine, preserving correlator resolution over many timeslices before noise dominates.

---

### Plan 3: Inclusion of Dynamical and Valence Fermions: Gauge-Covariant Hadron Attention

#### 1. Physics Motivation & The Open Problem
The fundamental limitation of pure gauge theories is the absence of dynamical quark matter. All real-world hadrons (pions, protons, tetraquarks, glueball-meson hybrids) are bound states containing fundamental valence fermions $\psi(x)$ transforming under the fundamental representation of $SU(3)$. In standard lattice QCD, hadron spectroscopy relies on Gaussian quark smearing (Jacobi smearing) or distillation (LapH eigenvectors of the 3D gauge-covariant Laplacian) to suppress excited states. Constructing interpolating operators for multiquark systems (e.g. tetraquarks $X(3872)$, pentaquarks $P_c$) is notoriously difficult because point-to-point quark contractions cannot adequately capture the complex spatial correlations and color rearrangements of tetraquark wavefunctions.

#### 2. Core Hypothesis & GELT-Specific Mechanism
GELT's shortest-path parallel transport operator $T_{\Delta x}(x)$ acts directly on fundamental color vectors: $\psi(x) \to \Omega(x) \psi(x)$ and $T_{x \leftarrow y} \to \Omega(x) T_{x \leftarrow y} \Omega^\dagger(y)$, so that $\tilde{\psi}(y \to x) \equiv T_{x \leftarrow y} \psi(y)$ transforms identically to $\psi(x)$ under local gauge transformations. By constructing gauge-covariant attention between quark fields at different lattice sites, GELT can learn the optimal non-local spatial wavefunctions for mesons ($\bar{q} q$), baryons ($q q q$), and multiquark systems without arbitrary spatial trial wavefunctions.

#### 3. Mathematical & Architectural Formulation
- **Fundamental Representation Transport:** The transport average $T_{\Delta x}(x)$ is computed in the fundamental representation:
  $$\tilde{\psi}(y \to x) = T_{x \leftarrow y}[U] \, \psi(y) \quad \implies \quad \tilde{\psi}(y \to x) \xrightarrow{\Omega} \Omega(x) \, \tilde{\psi}(y \to x)$$
- **Gauge-Covariant Meson Attention Operator:** Let $\bar{\psi}(x)$ and $\psi(y)$ be anti-quark and quark fields with Dirac spinor indices $\alpha, \beta \in \{1,2,3,4\}$ and color indices $a, b \in \{1, \dots, N_c\}$. For a target Dirac channel $\Gamma \in \{\gamma_5 \text{ (pseudoscalar)}, \gamma_\mu \text{ (vector)}, \mathbb{I} \text{ (scalar)}\}$, the learned interpolating operator is:
  $$\mathcal{O}_{\text{meson}}(x) = \sum_{y \in B_R(x)} \alpha_{x \to y}[U] \left( \bar{\psi}_a(x) \, \Gamma \, \left[ T_{x \leftarrow y}[U] \psi(y) \right]_a \right)$$
  where $\alpha_{x \to y}[U]$ is a gauge-invariant attention score derived from local Wilson loops and gauge links:
  $$\alpha_{x \to y}[U] = \operatorname{Softmax}_y \left( \operatorname{Re}\Tr\left[ Q(x)^\dagger T_{x \leftarrow y} K(y) T_{x \leftarrow y}^\dagger \right] \right)$$
- **Integration with Distillation / LapH Framework:** In distillation, quark fields are projected onto the low-lying eigenmodes $v_i(x)$ of the gauge-covariant 3D Laplacian: $-\Delta[U] v_i = \lambda_i v_i$. GELT's attention map can act as a generalized distillation profile:
  $$\Phi_{ij}(t) = \sum_{x, y} v_i^\dagger(x) \, \alpha_{x \to y}[U] \, T_{x \leftarrow y}[U] \, v_j(y)$$
  replacing the rigid modal truncation of standard distillation with an adaptive, configuration-dependent attention weighting.
- **Variational Training via Meson Correlator:** Train on the effective pion or rho mass using the Rayleigh loss:
  $$\mathcal{L}_{\text{meson}} = -\frac{C_\Gamma(t=1)}{C_\Gamma(t=0)}, \quad C_\Gamma(t) = \sum_{\vec{x}} \langle \mathcal{O}_{\text{meson}}(\vec{x}, t) \, \mathcal{O}_{\text{meson}}^\dagger(\vec{0}, 0) \rangle$$

#### 4. Implementation & Computational Roadmap
1. **Milestone 1 (Valence Quarks on 2D Gross–Neveu / Schwinger Model):** Implement fundamental transport on 1+1D $U(1)$ and $SU(2)$ Schwinger models to benchmark gauge-covariant quark bilinear attention.
2. **Milestone 2 (Wilson/Clover Inverter Interface):** Interface GELT with standard Dirac inverters (CG/BiCGStab for Clover or Staggered fermions) to compute quark propagators $D^{-1}[U](x, y)$.
3. **Milestone 3 (Light Meson Spectroscopy in 3+1D):** Measure pion ($\pi$) and rho ($\rho$) ground-state overlap $A_0$ on unquenched $16^3 \times 32$ configurations, comparing directly against Jacobi-smeared sinks.
4. **Milestone 4 (Tetraquark Operator Construction):** Construct a non-local 4-quark attention operator for the $T_{cc}^+$ or $X(3872)$ channel, demonstrating suppression of two-meson scattering threshold contamination.

#### 5. Scientific Impact & Novelty
- First gauge-equivariant attention architecture operating directly on fundamental fermionic matter fields.
- Solves the longstanding problem of designing configuration-adaptive wavefunctions for multiquark exotic states.

#### 6. Feasibility & Risk Analysis
- **Primary Risk:** Fermion propagator computation is computationally heavy ($O(V)$ inversions for all-to-all propagators).
- **Mitigation:** Combine GELT strictly with distillation/LapH spaces (typically $N_{\text{modes}} \approx 64$), where all-to-all contractions scale as $O(N_{\text{modes}})$, bypassing point-to-point inversion walls.

---

### Plan 4: Microscopic Structure of the Confinement Flux Tube & Static Quark Potential $V(r)$

#### 1. Physics Motivation & The Open Problem
A defining feature of non-Abelian gauge theories is color confinement: the chromoelectric field between a static quark and antiquark collapses into a narrow, string-like flux tube, giving rise to a linear potential $V(r) = \sigma r - \frac{\pi}{12 r} + \text{const}$. While the asymptotic string tension $\sigma$ is well known, the microscopic internal structure of the flux tube—its transverse width profile $w^2(r)$, energy-density distribution, chromoelectric vs. chromomagnetic balance, and the Luscher roughening transition—suffers from poor signal-to-noise ratios at large distances ($r \ge 1.0\text{ fm}$). Standard Wilson loop and Polyakov loop correlators decay exponentially, causing classical operators to drown in UV gauge noise.

#### 2. Core Hypothesis & GELT-Specific Mechanism
GELT's attention map is a gauge-invariant physical observable capable of learning non-local correlations. When conditioned on the presence of two static Polyakov lines separated by distance $r$, GELT can construct the optimal variational state for the string flux tube. Furthermore, by reading out the attention field $\alpha_{x \to y}$ in the region between the static sources, one can extract the transverse energy profile and string fluctuations directly from the network's latent attention distribution.

#### 3. Mathematical & Architectural Formulation
- **Static Quark Potential from Polyakov Loop Correlators:** Let $P(\vec{x}) = \Tr\left[ \prod_{t=0}^{N_t-1} U_0(\vec{x}, t) \right]$ be the Polyakov loop at spatial position $\vec{x}$. The potential is extracted from:
  $$C_{P}(r) = \langle P(\vec{0}) \, P^\dagger(\vec{r}) \rangle \sim \exp\left( - \frac{V(r)}{T} \right) = \exp\left( - V(r) N_t a_t \right)$$
- **Variational String Ground State via Spatial GELT:** Instead of measuring bare Polyakov loops, define an open string operator on spatial slice $t$:
  $$\mathcal{S}_{r}(t) = \Tr\left[ P(\vec{0}, t) \, \mathcal{W}_{0 \to \vec{r}}[U] \, P^\dagger(\vec{r}, t) \, \mathcal{W}_{\vec{r} \to \vec{0}}[U] \right]$$
  where $\mathcal{W}_{0 \to \vec{r}}[U]$ is an attention-weighted open transport line learned by GELT:
  $$\mathcal{W}_{0 \to \vec{r}}[U] = \sum_{\text{paths } p} \alpha_p[U] \prod_{e \in p} U_e$$
- **Microscopic Chromoelectric / Chromomagnetic Energy Density:** The probe observable measuring flux-tube structure at displacement $\vec{x}_\perp$ from the string axis is:
  $$\rho(\vec{x}_\perp; r) = \frac{\langle P(\vec{0}) P^\dagger(\vec{r}) \, \mathcal{O}_{\text{plaq}}(\vec{x}_\perp) \rangle}{\langle P(\vec{0}) P^\dagger(\vec{r}) \rangle} - \langle \mathcal{O}_{\text{plaq}} \rangle$$
- **Effective String Theory (EST) Luscher Widening Test:** Effective string theory predicts logarithmic widening of the flux tube squared width $w^2(r)$ as a function of separation $r$:
  $$w^2(r) = w_0^2 + \frac{1}{2\pi \sigma} \ln\left(\frac{r}{r_0}\right)$$
  GELT's attention profile $\ell_{\text{att}}(x_\perp)$ will be fitted directly against this logarithmic prediction to test the conformal validity of the Nambu–Goto string description.

#### 4. Implementation & Computational Roadmap
1. **Milestone 1 (Polyakov Pipeline):** Implement Polyakov loop correlators with zero-momentum and multi-level spatial gauge link smearing on $SU(2)$ and $SU(3)$ lattices.
2. **Milestone 2 (GELT String Operator Training):** Train GELT to maximize ground-state overlap on the static potential $V(r)$ for separations $r/a \in [2, 10]$.
3. **Milestone 3 (Transverse Energy Density Mapping):** Reconstruct the 3D chromoelectric field distribution $\vec{E}^2(\vec{x}_\perp, x_\parallel)$ around the flux tube.
4. **Milestone 4 (Logarithmic Widening Extraction):** Test the EST logarithmic broadening law $w^2(r) \sim \ln r$ to precision level $<3\%$, checking whether GELT's attention map tracks the quantum fluctuations of the Luscher string.

#### 5. Scientific Impact & Novelty
- Provides the cleanest non-perturbative measurement of flux tube geometry and string roughening in non-Abelian gauge theory.
- Demonstrates how attention-based operators can suppress exponential noise in long-distance Wilson loop observables.

#### 6. Feasibility & Risk Analysis
- **Primary Risk:** Polyakov loop correlators have non-zero perimeter-law divergences requiring careful multiplicative renormalization.
- **Mitigation:** Use ratio schemes $\rho(\vec{x}_\perp; r)$ where UV divergences cancel exactly between numerator and denominator.

---

### Plan 5: Gauge-Equivariant Continuous Normalizing Flows (CNFs) via Attention-Driven Lie Algebra Drift Fields

#### 1. Physics Motivation & The Open Problem
Markov Chain Monte Carlo (MCMC) methods in lattice field theory face two catastrophic scaling walls:
1. **Critical Slowing Down:** Near the continuum limit ($a \to 0$), the autocorrelation time diverges as $\tau_{\text{int}} \sim a^{-z}$ with dynamical exponent $z \approx 2$.
2. **Topological Freezing:** The energy barrier between distinct topological sectors scales as $\Delta S \sim a^{-2}$, locking traditional local update algorithms (Metropolis, heatbath, HMC) into a single topological sector for millions of sweeps.
Normalizing flows offer an exact, independent-sample generative solution via Boltzmann reweighting $w(U) = e^{-S(U)} / p_\theta(U)$. However, existing discrete coupling architectures (e.g. real NVP, L-CNN flows) suffer from poor expressivity and low acceptance rates at physical lattice volumes because they cannot capture multi-scale, non-local Wilson loop correlations.

#### 2. Core Hypothesis & GELT-Specific Mechanism
Continuous Normalizing Flows (CNFs) formulated as Neural Ordinary Differential Equations on the Lie group manifold $SU(N_c)^{\otimes N_{\text{links}}}$ can overcome these bottlenecks. GELT's exact gauge-equivariant matrix-bilinear value path naturally outputs Lie-algebra-valued vector fields $v_\mu(x; U) \in \mathfrak{su}(N_c)$. Because GELT integrates shortest-path parallel transport over the full $L^1$-ball, its drift field generates non-local, multi-loop updates that cross topological barriers in continuous flow time.

#### 3. Mathematical & Architectural Formulation
- **Lie-Manifold Neural ODE:** For each link $U_\mu(x, t) \in SU(N_c)$, the flow evolves along continuous time $t \in [0, 1]$ according to the right-invariant vector field:
  $$\frac{d U_\mu(x, t)}{dt} = i \, v_\mu(x, t; U) \, U_\mu(x, t), \quad v_\mu(x, t; U) \in \mathfrak{su}(N_c)$$
  where $\mathfrak{su}(N_c)$ is the Lie algebra of traceless Hermitian $N_c \times N_c$ matrices: $v = \sum_{a=1}^{N_c^2-1} v^a \lambda_a$.
- **GELT Adjoint Drift Generator:** GELT takes the current gauge configuration $U(t)$ and produces an adjoint tensor $W_\mu(x) \in \mathbb{C}^{N_c \times N_c}$. The Lie algebra generator is extracted via traceless anti-hermitian projection:
  $$v_\mu(x, t; U) = \frac{1}{2i} \left[ W_\mu(x) - W_\mu(x)^\dagger - \frac{\mathbb{I}}{N_c} \Tr\left( W_\mu(x) - W_\mu(x)^\dagger \right) \right]$$
  Under a local gauge transformation $U_\mu(x) \to \Omega(x) U_\mu(x) \Omega^\dagger(x+\mu)$, the vector field transforms covariantly:
  $$v_\mu(x) \to \Omega(x) \, v_\mu(x) \, \Omega^\dagger(x)$$
  guaranteeing exact gauge equivariance along the entire trajectory.
- **Log-Determinant Jacobian via Hutchinson Trace Estimator:** The change in probability density along the flow trajectory satisfies the continuous continuity equation:
  $$\log p_1(U(1)) = \log p_0(U(0)) - \int_0^1 dt \sum_{x, \mu} \operatorname{div}_{U_\mu(x)} \left( i v_\mu(x, t) U_\mu(x, t) \right)$$
  The high-dimensional divergence is computed using unbiased Hutchinson stochastic trace estimators:
  $$\Tr\left[ \frac{\partial v}{\partial U} \right] = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, \mathbb{I})} \left[ \epsilon^\dagger \nabla_U (v^\dagger \epsilon) \right]$$
- **Loss Function:** Train using reverse Kullback–Leibler divergence with the target lattice Wilson action:
  $$\mathcal{L}_{\text{flow}} = \mathbb{E}_{U(0) \sim \text{Haar}} \left[ \log p_1(U(1)) + S(U(1)) \right]$$

#### 4. Implementation & Computational Roadmap
1. **Milestone 1 (Geometric ODE Integrator):** Implement a symplectic or matrix-exponential Lie-group ODE integrator (Runge–Kutta on $SU(2)$) ensuring link unitarity is preserved to machine precision ($10^{-15}$) along the trajectory.
2. **Milestone 2 (2D $U(1)$ and $SU(2)$ Topological Barrier Crossing):** Demonstrate that the flow transitions smoothly between distinct topological charge sectors ($Q = 0 \leftrightarrow \pm 1$) on $L=8, 16$ lattices.
3. **Milestone 3 (4D $SU(2)$ Scaling):** Train GELT-CNF on $8^4$ and $12^4$ lattices, evaluating the Metropolis-Hastings acceptance rate $P_{\text{acc}} = \min(1, e^{-\Delta S - \Delta \log p})$ against standard HMC.
4. **Milestone 4 (Critical Slowing Down Benchmark):** Measure the integrated autocorrelation time $\tau_{\text{int}}$ of the topological susceptibility $\chi_t$ across four decreasing lattice spacings, verifying elimination of critical slowing down.

#### 5. Scientific Impact & Novelty
- First continuous normalizing flow on a non-Abelian Lie group powered by an attention mechanism.
- Provides a direct computational route to bypass topological freezing in lattice field theory.

#### 6. Feasibility & Risk Analysis
- **Primary Risk:** Scaling the Hutchinson divergence computation to large 4D lattice volumes ($V = 16^4$).
- **Mitigation:** Use stochastic Hutchinson estimators with Rademacher noise and evaluate divergence only during training backpropagation; during inference, configurations are unweighted without computing the Jacobian on the forward pass.

---

### Plan 6: Finite-Temperature Deconfinement and Screening Thermodynamics via Attention Observables

#### 1. Physics Motivation & The Open Problem
At high temperatures, strongly interacting matter undergoes a fundamental phase transition from a confining hadronic phase to a deconfined Quark-Gluon Plasma (QGP). In pure gauge theory, this corresponds to the spontaneous breaking of the global center symmetry $\mathbb{Z}_{N_c}$, characterized by the Polyakov loop expectation value $\langle |P| \rangle$. Characterizing the phase transition—locating the critical temperature $T_c$, extracting universal critical exponents ($\nu, \beta, \gamma$), and determining the electric and magnetic screening masses ($m_E, m_M$) in the QGP—requires extensive spatial volume scaling ($N_s \gg N_t$). Standard thermodynamic observables (such as Polyakov loop susceptibilities or plaquette peaks) suffer from severe finite-volume rounding and renormalization ambiguities.

#### 2. Core Hypothesis & GELT-Specific Mechanism
In the existing GELT results, it was discovered that the connected correlator of an attention reduction $A(x) = f(\alpha_{x \to \cdot})$ behaves as a physical, gauge-invariant scalar lattice operator with a clean correlation length $\xi_A$. Across a thermodynamic phase transition, the attention field's correlation length $\xi_A(T)$ must diverge at $T_c$ with the universal critical exponent of the 3D Ising / 3D Potts universality class. Furthermore, because attention is non-local and configuration-adaptive, its spatial correlators can measure electric and magnetic screening lengths with superior ground-state overlap compared to standard spatial Wilson loops.

#### 3. Mathematical & Architectural Formulation
- **Finite-Temperature Geometry:** Anisotropic lattice $N_s^3 \times N_t$ with $N_s \gg N_t$. The temperature is $T = \frac{1}{a_t N_t} = \frac{\xi}{a_s N_t}$.
- **Center Symmetry Breaking & Attention Order Parameter:** Center transformations multiply all temporal links on a fixed time slice by a center element $z \in \mathbb{Z}_{N_c}$:
  $$U_0(\vec{x}, t_0) \to z \, U_0(\vec{x}, t_0), \quad z = e^{i 2\pi k / N_c} \mathbb{I}$$
  While local plaquettes are invariant, the Polyakov loop transforms as $P(\vec{x}) \to z P(\vec{x})$.
  We construct a center-sensitive attention operator by coupling spatial transport to temporal holonomies:
  $$\mathcal{A}_{\text{center}}(\vec{x}) = \sum_{\vec{y} \in B_R(\vec{x})} \alpha_{\vec{x} \to \vec{y}}[U] \, \operatorname{Re}\Tr\left[ P(\vec{x}) \, T_{\vec{x} \leftarrow \vec{y}} \, P^\dagger(\vec{y}) \, T_{\vec{x} \leftarrow \vec{y}}^\dagger \right]$$
- **Finite-Size Scaling of the Attention Correlation Length:** Near $T_c$, the attention correlation length $\xi_A(T, N_s)$ obeys the universal finite-size scaling ansatz:
  $$\frac{\xi_A(T, N_s)}{N_s} = \mathcal{F}\left( (T - T_c) N_s^{1/\nu} \right)$$
  Plotting $\xi_A / N_s$ across various spatial volumes $N_s \in \{16, 24, 32, 48\}$ produces a universal crossing point that yields $T_c$ and the critical exponent $\nu$ without arbitrary fit parameters.
- **Electric and Magnetic Screening Masses:** Spatial correlators in the deconfined phase ($T > T_c$) decay with screening masses:
  $$C_{\text{screen}}(z) = \sum_{x, y, t} \langle \mathcal{O}_{\text{screen}}(x, y, z, t) \, \mathcal{O}_{\text{screen}}(0) \rangle \sim e^{- m_{\text{screen}} z}$$
  where $\mathcal{O}_E = \operatorname{Re}\Tr P$ measures the electric screening mass $m_E \sim g T$, and $\mathcal{O}_M = \operatorname{Im}\Tr P$ or spatial loops measure the non-perturbative magnetic screening mass $m_M \sim g^2 T$.

#### 4. Implementation & Computational Roadmap
1. **Milestone 1 (Temperature Scan Pipeline):** Set up anisotropic ensembles at fixed $N_t = 4, 6$ and $N_s \in \{16, 24, 32\}$ scanning $\beta$ across the deconfinement transition for $SU(2)$ (second order) and $SU(3)$ (first order).
2. **Milestone 2 (Finite-Size Scaling Extraction):** Measure the attention correlator $\xi_A(T, N_s)$, determine the crossing point, and fit the critical exponent $\nu$ (comparing against the 3D Ising value $\nu \approx 0.630$ for $SU(2)$).
3. **Milestone 3 (Screening Mass Extraction):** Train GELT to optimize spatial screening correlators $C_{\text{screen}}(z)$ at $T = 1.5 T_c$ and $T = 2.0 T_c$.
4. **Milestone 4 (First-Order Interface Tension in $SU(3)$):** Apply the attention operator to detect coexistence of ordered and disordered phases at $T_c$ in $SU(3)$, extracting the surface tension $\sigma_{od}$ of the order-disorder interface.

#### 5. Scientific Impact & Novelty
- Establishes transformer attention as a rigorous thermodynamic observable for critical phenomena and phase boundary detection.
- Delivers high-precision measurements of non-perturbative magnetic screening masses, which are notoriously noisy in classical lattice thermodynamics.

#### 6. Feasibility & Risk Analysis
- **Primary Risk:** Severe spatial autocorrelation times near the second-order critical point (critical slowing down).
- **Mitigation:** Use cluster algorithms (for $Z_2$) or combined heatbath + overrelaxation (with $N_{\text{or}} = 10$) to keep autocorrelation times manageable across the transition.

---

### Plan 7: Flow-Free Topological Invariants & Non-Perturbative $\theta$-Vacuum Dynamics

#### 1. Physics Motivation & The Open Problem
The topological structure of non-Abelian gauge fields—characterized by the integer Pontryagin index $Q = \frac{1}{32\pi^2} \int d^4x \, \epsilon_{\mu\nu\rho\sigma} \Tr[F_{\mu\nu} F_{\rho\sigma}] \in \mathbb{Z}$—underpins crucial phenomena such as the axial $U(1)_A$ anomaly, the mass of the $\eta'$ meson (Witten–Veneziano formula), and the Strong CP problem. On a discrete lattice, the naive topological charge $Q_{\text{naive}} = \sum_x q(x)$ is not an integer and suffers from large multiplicative and additive renormalizations ($Z_Q < 1$) driven by short-distance UV fluctuations. Existing methods (cooling, gradient/Wilson flow, stout smearing) suppress UV noise by diffusing the gauge field over a flow time $t$. However, this diffusion inevitably distorts or annihilates small instanton-antiinstanton pairs, obscuring the true microscopic topological structure.

#### 2. Core Hypothesis & GELT-Specific Mechanism
GELT can act as an exact, volume-preserving, flow-free topological filter. Because GELT's attention layers can compute anti-symmetric, parity-odd matrix bilinears with multi-hop parallel transport, it can learn to isolate the invariant geometric curvature of the gauge field without introducing diffusive continuous flow. This enables the construction of a renormalized, integer-peaked topological charge operator that operates directly on uncooled gauge configurations.

#### 3. Mathematical & Architectural Formulation
- **Lattice Topological Charge Density:** On the lattice, the field strength tensor $F_{\mu\nu}(x)$ is represented by the clover leaf operator $C_{\mu\nu}(x)$:
  $$F_{\mu\nu}(x) = \frac{1}{8i} \left( Q_{\mu\nu}(x) - Q_{\mu\nu}^\dagger(x) \right), \quad Q_{\mu\nu} = \text{sum of four plaquettes in }(\mu, \nu)\text{ plane}$$
  The naive topological charge density is:
  $$q_{\text{naive}}(x) = \frac{1}{32\pi^2} \epsilon_{\mu\nu\rho\sigma} \operatorname{Re}\Tr\left[ F_{\mu\nu}(x) F_{\rho\sigma}(x) \right]$$
- **GELT Parity-Odd Topological Operator:** We design a specialized parity-odd GELT head. Under spatial reflection $\mathcal{P}$, $q(x)$ must flip sign: $\mathcal{P} q(x) = -q(-x)$.
  The attention layer couples orthogonal planes $(\mu, \nu)$ and $(\rho, \sigma)$ via an epsilon tensor contraction:
  $$q_{\text{GELT}}(x) = \sum_{\mu < \nu < \rho < \sigma} \epsilon_{\mu\nu\rho\sigma} \operatorname{Re}\Tr\left[ W_{\mu\nu}(x)^\dagger \, \tilde{W}_{\rho\sigma}(x) \right]$$
  where $W_{\mu\nu}(x)$ is the gauge-equivariant clover feature transformed through GELT's shortest-path transport blocks.
- **Self-Supervised Geometric Training Objective:** To avoid relying on destructive cooling as a ground truth, GELT is trained via an exact mathematical constraint: **integer quantization of total charge** and **invariance under infinitesimal gauge deformations**:
  $$\mathcal{L}_{\text{topo}} = \mathbb{E}\left[ \left( \sum_x q_{\text{GELT}}(x) - \operatorname{round}\left(\sum_x q_{\text{GELT}}(x)\right) \right)^2 \right] + \lambda_{\text{flow}} \mathbb{E}\left[ \left\Vert \frac{\partial Q_{\text{GELT}}}{\partial \text{cooling step}} \right\Vert^2 \right]$$
- **Extraction of Topological Susceptibility and $\theta$-Dependence:**
  $$\chi_t = \frac{\langle Q^2 \rangle}{V_{\text{spacetime}}}, \quad E(\theta) = \frac{1}{2} \chi_t \theta^2 \left( 1 + c_2 \theta^2 + \mathcal{O}(\theta^4) \right)$$
  The kurtosis $c_2 = \frac{\langle Q^4 \rangle - 3 \langle Q^2 \rangle^2}{12 \langle Q^2 \rangle}$ measures the non-Gaussianity of vacuum topological fluctuations.

#### 4. Implementation & Computational Roadmap
1. **Milestone 1 (Parity-Odd Architecture):** Implement the $\epsilon_{\mu\nu\rho\sigma}$ tensor contraction head in `gelt/blocks_rope.py` and verify exact anti-invariance under parity $\mathcal{P}$ and gauge invariance under $\Omega(x)$.
2. **Milestone 2 (Instanton Benchmark):** Evaluate on synthetic multi-instanton configurations with analytically known topological charges ($Q = 0, \pm 1, \pm 2$).
3. **Milestone 3 (Comparison with Wilson Flow):** Compare $Q_{\text{GELT}}$ against standard Wilson flow ($t_0$ scale) on $16^4$ $SU(3)$ configurations across $\beta \in [5.9, 6.2]$, demonstrating zero renormalization factor ($Z_Q = 1.00 \pm 0.01$).
4. **Milestone 4 (Kurtosis & Strong CP Dynamics):** Extract the higher-order topological susceptibility coefficient $c_2$ with reduced variance.

#### 5. Scientific Impact & Novelty
- Eliminates the need for arbitrary flow-time cutoffs or destructive cooling in topological charge measurements.
- Preserves short-distance instanton-antiinstanton pairs that are normally annihilated by diffusive smearing.

#### 6. Feasibility & Risk Analysis
- **Primary Risk:** High-action UV dislocations on coarse lattices can trick local operators into assigning spurious fractional topological charges.
- **Mitigation:** The $L^1$-ball transport radius $R \ge 2$ gives GELT an extended multi-hop stencil that can distinguish true topological instantons (radius $\rho \sim a$) from unphysical single-plaquette dislocations.

---

### Plan 8: Gauge-Equivariant Neural Quantum States (NQS) for Hamiltonian Lattice Gauge Theory

#### 1. Physics Motivation & The Open Problem
Hamiltonian lattice gauge theory (Kogut–Susskind formulation) provides the fundamental theoretical foundation for quantum simulation, tensor network algorithms, and real-time non-equilibrium dynamics of gauge theories. In this continuous-time formulation, the state of the gauge field is a wavefunction $|\Psi[U]\rangle$. The physical Hilbert space $\mathcal{H}_{\text{phys}}$ is strictly restricted by **Gauss's law** at every lattice site:
$$\hat{G}^a(x) |\Psi\rangle = 0 \quad \forall x, a$$
where $\hat{G}^a(x) = \sum_\mu (\hat{E}_\mu^a(x) - \hat{E}_{-\mu}^a(x))$ is the non-Abelian divergence of the chromoelectric electric field. Constructing a scalable, expressive variational ansatz $|\Psi_\theta\rangle$ that satisfies Gauss's law exactly for continuous gauge groups ($SU(2), SU(3)$) has been an intractable challenge for standard Neural Quantum States (NQS) and tensor networks in $D \ge 2+1$.

#### 2. Core Hypothesis & GELT-Specific Mechanism
In the link basis $|U\rangle$, Gauss's law is mathematically identical to local gauge invariance of the wavefunction amplitude: $\Psi[\Omega U \Omega^\dagger] = \Psi[U]$. Because GELT's scalar outputs are **identically gauge invariant to machine precision**, GELT defines an exact, unconstrained Neural Quantum State on the physical Hilbert space $\mathcal{H}_{\text{phys}}$. Using Variational Monte Carlo (VMC) and Stochastic Reconfiguration (SR), GELT can solve for the ground-state wavefunction and real-time string dynamics without truncating the infinite-dimensional chromoelectric Hilbert space.

#### 3. Mathematical & Architectural Formulation
- **Kogut–Susskind Hamiltonian:** On a $d$-dimensional spatial lattice:
  $$\hat{H} = \frac{g^2}{2} \sum_{x, i} \sum_{a=1}^{N_c^2-1} \left( \hat{E}_i^a(x) \right)^2 - \frac{1}{2g^2} \sum_{\vec{x}, i < j} \operatorname{Re}\Tr\left[ \hat{P}_{ij}(\vec{x}) \right]$$
  where $\hat{E}_i^a(x)$ acts as the Lie derivative on link $U_i(x)$: $\hat{E}_i^a(x) U_j(y) = \delta_{xy} \delta_{ij} \tau^a U_i(x)$.
- **GELT Wavefunction Ansatz:** The complex ground-state amplitude is parameterized as:
  $$\Psi_\theta[U] = \exp\left( - S_R[U; \theta] + i \, S_I[U; \theta] \right)$$
  where $S_R[U; \theta]$ and $S_I[U; \theta]$ are scalar outputs generated by twin GELT transformer encoders.
  Because both $S_R$ and $S_I$ are composed entirely of gauge-invariant attention scores and traces, $\hat{G}^a(x) |\Psi_\theta\rangle = 0$ holds identically for all parameter values $\theta$.
- **Variational Monte Carlo & Local Energy:** Sample spatial configurations $U \sim |\Psi_\theta[U]|^2$ using Metropolis sweeps on the spatial manifold. The local energy is:
  $$E_{\text{loc}}[U] = \frac{\hat{H} \Psi_\theta[U]}{\Psi_\theta[U]} = \frac{g^2}{2} \sum_{x, i, a} \left( \frac{\mathcal{D}_{i, a}^2 \Psi_\theta}{\Psi_\theta} \right) - \frac{1}{2g^2} \sum_\square \operatorname{Re}\Tr P_\square$$
  The chromoelectric kinetic term simplifies via automatic differentiation:
  $$\frac{\mathcal{D}_{i, a} \Psi_\theta}{\Psi_\theta} = - \mathcal{D}_{i, a} S_R + i \mathcal{D}_{i, a} S_I$$
  where $\mathcal{D}_{i, a} \equiv \left. \frac{d}{ds} \right|_{s=0} S(U_i(x) \to e^{i s \tau_a} U_i(x))$.
- **Optimization via Stochastic Reconfiguration (Natural Gradient):** Parameters are updated via the quantum Fisher information metric $S_{ij}$:
  $$\theta \leftarrow \theta - \eta \, S^{-1} \nabla_\theta \langle \hat{H} \rangle, \quad S_{ij} = \langle \mathcal{O}_i^* \mathcal{O}_j \rangle - \langle \mathcal{O}_i^* \rangle \langle \mathcal{O}_j \rangle, \quad \mathcal{O}_i \equiv \frac{\partial \log \Psi_\theta}{\partial \theta_i}$$

#### 4. Implementation & Computational Roadmap
1. **Milestone 1 (1+1D $U(1)$ and $SU(2)$ VMC):** Implement Kogut–Susskind VMC on a 1D chain of links; compare ground-state energy against exact diagonalization.
2. **Milestone 2 (2+1D $SU(2)$ Hamiltonian Engine):** Implement the automatic-differentiation chromoelectric kinetic operator on a $8 \times 8$ spatial grid.
3. **Milestone 3 (Ground State & Confinement Phase):** Optimize GELT-NQS across coupling regimes $g^2 \in [0.5, 2.0]$, measuring the ground-state energy density and spatial string tension.
4. **Milestone 4 (Real-Time Quantum Quench):** Integrate the time-dependent variational principle (TDVP) to simulate real-time string breaking following the creation of a static quark-antiquark pair.

#### 5. Scientific Impact & Novelty
- First exact gauge-invariant attention Neural Quantum State for non-Abelian lattice gauge theories.
- Bypasses the local Hilbert space truncation required by classical tensor network methods (MPS, PEPS).

#### 6. Feasibility & Risk Analysis
- **Primary Risk:** Computation of the quantum geometric tensor $S_{ij}$ scales quadratically with parameter count $N_{\text{params}}$.
- **Mitigation:** GELT is parameter-efficient ($\sim 1.6 \times 10^4$ parameters). Use MinSR (conjugate-gradient iterative solver) to compute parameter updates without inverting the full $N_{\text{params}} \times N_{\text{params}}$ matrix.

---

### Plan 9: Curved Spacetimes & Simplicial Gauge Fields on Arbitrary Geometries

#### 1. Physics Motivation & The Open Problem
Standard lattice gauge theory is fundamentally wedded to flat hypercubic grids. However, major frontiers in modern theoretical physics require formulating gauge theories on **curved backgrounds and non-trivial topologies**:
- **Holographic duality (AdS/CFT):** Discretizing gauge theories on hyperbolic spaces ($AdS_4$) to study strongly coupled boundary conformal field theories.
- **Quantum Gravity & Regge Calculus:** Discretizing spacetime as a simplicial complex (triangulated 4-manifold) where link lengths and curvature fluctuate dynamically.
- **Spherical and Toroidal Compactifications:** Simulating gauge fields on $S^2 \times S^2$ or curved cosmological backgrounds.
Classical lattice algorithms fail on irregular triangulations because standard convolutional stencils and shift operators (`torch.roll`) rely entirely on Cartesian translation invariance.

#### 2. Core Hypothesis & GELT-Specific Mechanism
GELT is fundamentally a **graph attention network with holonomic parallel transport**. Its shortest-path DP transport algorithm does not require regular Cartesian axes—it only requires a directed graph of vertices $V$ and edges $E$ endowed with gauge group holonomies $U_e$. By replacing hypercubic offsets with graph-geodesic shortest paths and replacing RoPE with Laplace–Beltrami spectral positional encodings, GELT generalizes naturally to arbitrary curved simplicial manifolds.

#### 3. Mathematical & Architectural Formulation
- **Simplicial Gauge Complex:** Spacetime is discretized as a 4-dimensional simplicial complex $\mathcal{K} = (V, E, F, T)$.
  - Gauge links $U_{uv} \in G$ live on directed edges $e = (u, v) \in E$, with $U_{vu} = U_{uv}^\dagger$.
  - Curvature is concentrated on 2-dimensional triangular faces $f = (u, v, w) \in F$, with plaquette holonomy $P_f = U_{uv} U_{vw} U_{wu}$.
  - Gauge action (Regge-Wilson action):
    $$S[U] = \sum_{f \in F} \beta_f \left( 1 - \frac{1}{N_c} \operatorname{Re}\Tr P_f \right)$$
- **Geodesic Parallel Transport Engine on Graphs:** For any vertex pair $(u, v)$ within graph-geodesic distance $d(u, v) \le R$:
  $$T_{u \leftarrow v} = \frac{1}{N_{\text{paths}}(u, v)} \sum_{p \in \mathcal{P}_{\text{shortest}}(u, v)} \prod_{e \in p} U_e$$
  This is computed via Dijkstra/Bellman-Ford dynamic programming over the edge adjacency matrix.
- **Manifold-Spectral Positional Encoding:** On a curved manifold, Cartesian coordinates do not exist. We compute the graph Laplacian $\mathcal{L} = D - A$ and its spectral decomposition:
  $$\mathcal{L} \phi_k = \lambda_k \phi_k, \quad k = 1, \dots, d_{\text{spectral}}$$
  The relative positional encoding between vertices $u$ and $v$ is given by the spectral distance and heat-kernel embedding:
  $$\gamma(u, v) = \sum_{k=1}^{d_{\text{spectral}}} e^{-\lambda_k \tau} \left( \phi_k(u) - \phi_k(v) \right)^2$$
- **Gauge-Invariant Simplicial Attention:**
  $$\alpha_{u \to v} = \operatorname{Softmax}_v \left( \frac{1}{\sqrt{d_k}} \operatorname{Re}\Tr\left[ Q(u)^\dagger T_{u \leftarrow v} K(v) T_{u \leftarrow v}^\dagger \right] + W_{\text{bias}} \gamma(u, v) \right)$$

#### 4. Implementation & Computational Roadmap
1. **Milestone 1 (Graph Transport Library):** Build a generalized graph parallel transport module operating on PyTorch Geometric (`torch_geometric`) data structures.
2. **Milestone 2 (2D Spherical Lattice $S^2$):** Discretize a 2D sphere using an icosahedral geodesic grid; simulate $U(1)$ and $SU(2)$ gauge fields and verify that GELT recovers rotationally symmetric observables.
3. **Milestone 3 (Hyperbolic Lattice $AdS_3$ / $AdS_4$):** Implement regular tessellations of hyperbolic space (e.g. $\{p, q\}$ hyperbolic tilings); compute Wilson loop scaling to measure boundary-to-boundary correlators.
4. **Milestone 4 (Simplicial Glueball Mass Extraction):** Measure the scalar glueball mass on a triangulated 4-sphere, demonstrating independence of the triangulated mesh choice.

#### 5. Scientific Impact & Novelty
- First gauge-equivariant transformer capable of simulating non-Abelian lattice gauge theories on arbitrary curved geometries and simplicial meshes.
- Establishes a direct numerical bridge between machine learning and discrete quantum gravity / holographic AdS/CFT models.

#### 6. Feasibility & Risk Analysis
- **Primary Risk:** Dynamic programming on irregular graphs cannot use vectorized `torch.roll` shift buffers.
- **Mitigation:** Precompute sparse shortest-path indices and path edge lists offline; parallel transport tensor contractions are executed using sparse matrix-matrix multiplies (SpMM) on GPU.

---

### Plan 10: Machine-Learned Wilsonian Renormalization Group Flows and Exact Non-Perturbative $\beta$-Functions

#### 1. Physics Motivation & The Open Problem
Wilson's modern understanding of quantum field theory defines a theory by its flow under the **Renormalization Group (RG)**: integrating out short-distance degrees of freedom generates an effective action $S_{\text{eff}}[U]$ along a renormalized trajectory in the infinite-dimensional space of gauge-invariant operators:
$$\frac{d S}{d \ln \mu} = \beta(S)$$
In classical lattice field theory, real-space RG blocking (e.g. Swendsen, Hasenfratz–Hasenfratz) requires severe, ad-hoc truncations of the action to a few hand-picked planar loops. Identifying the non-perturbative $\beta$-function, discovering fixed points (such as the conformal window in many-flavor gauge theories for Beyond-the-Standard-Model physics), and extracting anomalous dimensions have been severely bottlenecked by these uncontrolled operator truncations.

#### 2. Core Hypothesis & GELT-Specific Mechanism
GELT’s stacked attention blocks provide an exact mathematical realization of real-space coarse-graining: each block roughly doubles the reachable loop length ($1 \times 1 \to 2 \times 2 \to 4 \times 4$), exactly mirroring an iterated block-spin step. By formulating GELT as a gauge-covariant coarse-graining transformation, the network can learn the optimal real-space RG flow that preserves long-distance physical observables while systematically integrating out UV modes.

#### 3. Mathematical & Architectural Formulation
- **Real-Space Gauge Blocking Transformation:** Let $U$ be a link configuration on lattice $\Lambda$ with spacing $a$. A blocking transformation maps $U$ to a coarse configuration $U'$ on lattice $\Lambda'$ with spacing $a' = 2a$:
  $$U'_\mu(x') = \mathcal{F}_{\text{GELT}}[U]_\mu(x')$$
  where $x' = 2x$. The coarse link must transform under coarse gauge transformations $\Omega'(x') = \Omega(2x)$ as:
  $$U'_\mu(x') \to \Omega'(x') \, U'_\mu(x') \, \Omega^{\prime \dagger}(x' + \hat{\mu})$$
- **GELT Block-Link Construction:** The coarse link is constructed from the matrix-bilinear value output of GELT connecting site $x$ to site $x + 2\hat{\mu}$:
  $$V_\mu(x) = \sum_{y \in B_R(x)} \alpha_{x \to y}[U] \, Q(x)^\dagger \, T_{x \leftarrow y}[U] \, V(y) \, T_{x \leftarrow y}^\dagger[U]$$
  Projecting $V_\mu(x)$ onto $SU(N_c)$ yields the blocked gauge link:
  $$U'_\mu(x') = \Pi_{SU(N_c)}\left( V_\mu(2x) \right)$$
- **Information-Preserving RG Loss:** The blocking transformation is optimized by requiring that long-distance physical observables (e.g. large Wilson loops $W(R, T)$ or Polyakov correlators) match identically between fine and coarse lattices:
  $$\mathcal{L}_{\text{RG}} = \sum_{R, T \ge 2} \left| \langle W(2R, 2T) \rangle_{\text{fine}} - \langle W(R, T) \rangle_{\text{blocked}} \right|^2$$
- **Extraction of the Non-Perturbative $\beta$-Function:**
  $$\Delta \beta = \beta(a) - \beta(2a) = \frac{\partial \beta}{\partial \ln a} \ln 2$$
  By iterating the blocking step across multiple ensembles, GELT maps out the non-perturbative step-scaling function $\sigma(u)$ and locates non-trivial infrared fixed points ($\beta(g^*) = 0$) signaling conformal transitions.

#### 4. Implementation & Computational Roadmap
1. **Milestone 1 (Gauge-Covariant Blocking Layer):** Implement the $2\times$ decimation layer with $SU(N_c)$ unitary projection and verify exact coarse gauge covariance.
2. **Milestone 2 (2D $O(3)$ / Non-Linear Sigma Model Benchmark):** Validate the learned RG flow on the 2D $O(3)$ model, verifying recovery of the known asymptotic freedom $\beta$-function.
3. **Milestone 3 (4D $SU(2)$ Non-Perturbative Step Scaling):** Apply GELT-RG to $16^4 \to 8^4$ and $32^4 \to 16^4$ ensembles, computing the step-scaling function $\sigma(u)$ and comparing against the 2-loop and 3-loop perturbative $\beta$-functions.
4. **Milestone 4 (Conformal Window Probe):** Apply the framework to $SU(3)$ with $N_f = 12$ fundamental flavors to resolve the active controversy regarding whether the theory is conformal or chirally broken.

#### 5. Scientific Impact & Novelty
- Solves the longstanding problem of parameterizing real-space Renormalization Group flows without truncating to arbitrary small loop bases.
- Provides a clean, automated numerical tool for discovering conformal fixed points in strongly coupled gauge theories beyond the Standard Model.

#### 6. Feasibility & Risk Analysis
- **Primary Risk:** Coarse-grained links can suffer from entropy loss or contract to zero amplitude before group projection.
- **Mitigation:** The explicit $SU(N_c)$ polar SVD projection $\Pi_{SU(N_c)}$ preserves full unitarity, while the information-matching loss ensures that long-distance quantum fluctuations are strictly conserved.

---

## Parte C — GELT — 10 Future Routes (spark plans)

Explored from code capabilities only (`gelt/`, `scripts/`, `tests/`, `README.md`).
Not based on `notes/`. Date: 2026-09-07.

### 1. Gauge-equivariant flow sampler for critical slowing / freezing
- **Core idea:** Turn `GEMHSA` into a gauge-equivariant coupling layer for a normalizing flow / diffusion sampler that proposes global updates.
- **Why this repo:** `gelt/lattice.py:action`, `random_links`, `link_gauge_transformation` give exact log-prob + invariance tests for free; `sampler.py:mcmc_ensemble, integrated_autocorrelation_time` is the yardstick to beat (τ_int, acceptance).
- **First experiment:** Flow on 2D Z2 / SU(2) L=8 mapping Haar → Boltzmann at fixed β, trained by reverse-KL with exact `action()`; compare τ_int vs `metropolis_sweep` / `heatbath_overrelaxation_sweep`.
- **Success:** Independent configs where HMC freezes; lower τ_int at equal cost.

### 2. Learned perfect / improved action on coarse lattices
- **Core idea:** Train `GELT(reduction="sum")` to regress a fine-lattice action / long Wilson loops from coarse links — data-driven Symanzik improvement.
- **Why this repo:** `data.py:build_plaquette_datasets(target=...)` supports arbitrary `target(configs,group)`; `rectangular_wilson_loop(R,T,mu,nu)` gives R×T labels; `staple_sum(xi)` handles anisotropy.
- **First experiment:** Coarse SU(2) links → predict 3×3 loop / fine action; test scaling violation vs Wilson action on Creutz ratios.
- **Success:** Cheap coarse ensembles with fine physics.

### 3. Surrogate HMC / ML Metropolis filter
- **Core idea:** Use GELT as fast ΔS predictor inside HMC: propose with cheap surrogate, accept with exact `action()`.
- **Why this repo:** `action(beta,xi)` is exact and batched; `train_cnn.py` / `train_gelt.py` loops are the template for a surrogate regressor; gauge invariance guarantees detailed balance isn't broken by architecture.
- **First experiment:** Predict `S(U')-S(U)` for near-identity SU(2) proposals; measure effective speedup = acceptance × cost ratio.
- **Success:** Practical accelerator, not just better operator.

### 4. Full J^PC + torelon tower, not just scalars
- **Core idea:** Generalize `glueball.py:glueball_operator(R,T)` (currently spatial-plane sum → 0++) to 2++, 0-+, 1+-, torelons and Polyakov-loop correlators for string tension.
- **Why this repo:** `rectangular_wilson_loop`, `zero_momentum`, `connected_correlator`, `effective_mass`, GEVP helpers are all channel-agnostic; only operator construction changes.
- **First experiment:** Add `polyakov_loop(U,time_axis)` + winding spatial loops; get torelon mass / string tension on anisotropic SU(2) where 0++ already plateaus.
- **Success:** From one mass to spectroscopy.

### 5. Finite-temperature deconfinement probe
- **Core idea:** Per-site `GELT(reduction="none")` outputting the Polyakov-loop field, trained to locate β_c from susceptibility peak.
- **Why this repo:** `random_links(Lt=...)`, `mcmc_ensemble(Lt=...)` already make N_t × L^3 thermal lattices; `Trace→MLP→none` gives `(B,*Λ)` field for free; `_last_alpha` gives spatial maps of center domains.
- **First experiment:** Scan N_t=4,6 at SU(2), learn |P| susceptibility vs β; compare to direct measurement.
- **Success:** New physics regime with ~20 lines of new observable code.

### 6. Learned equivariant cooler / UV-denoiser
- **Core idea:** Replace iterated `ape_smear / cool` with a shallow GELT mapping thin → smooth links, trained to preserve long loops while killing UV noise.
- **Why this repo:** `glueball.ape_smear(directions=...)`, `topology.cool/cooled_charge_density` are the classical teacher + evaluation; transport `T` is already the correct equivariant receptive field for smoothing.
- **First experiment:** Supervised thin U → APE×6 U (equivariant L2 on links + loop-loss on 1×1,2×2); eval: does denoised Q go integer faster with fewer I-A annihilations?
- **Success:** Differentiable smearing usable inside training loops.

### 7. U(1) Coulomb + SU(3) real-world push
- **Core idea:** Fill the two registry gaps: U(1) heat-bath + SU(N≥3) Cabibbo-Marinari in `sampler.py:_PROPOSAL_FN/_SWEEP_FN`.
- **Why this repo:** `GaugeGroup` ABC + explicit `A@B/dagger` + `SU.random/project` already port verbatim; only proposals/sweeps are SU(2)-locked; `validate_sampler_{z2,su2}.py` is the test template (U(1) has exact 2D plaquette too).
- **First experiment:** U(1) 4D monopole / photon transition; SU(3) β-scan vs known crossover; rerun invariance tests in complex.
- **Success:** Unlocks real QCD-adjacent physics.

### 8. Masked-link foundation model
- **Core idea:** BERT-style self-supervised pretraining: mask random links, predict missing plaquettes / staples, then fine-tune on tiny labeled ensembles across β, L.
- **Why this repo:** `haar_ensemble + mcmc_ensemble` is an infinite data engine; `data.py:structured=True/False` + `ChannelLift` decouple width from physics; no new sampler needed.
- **First experiment:** Pretrain on Haar SU(2) D=4, fine-tune action / Wilson loop at β=2.4 with 10× fewer configs; transfer vs from-scratch.
- **Success:** Breaks per-β retraining habit.

### 9. Fermion extension: Dirac preconditioner
- **Core idea:** Keep configs quenched, add Wilson/staggered Dirac matrix built from `U`, train GELT to predict propagator / precondition CG inversion.
- **Why this repo:** Links layout `(D,*Λ,nc,nc)` + `torch.roll` shifts are exactly what a Dirac operator needs; `GEMHSA` adjoint transport is the right prior for a gauge-covariant preconditioner.
- **First experiment:** On 2D SU(2), learn M⁻¹·η → ψ for random sources; metric = CG iterations saved at fixed residual.
- **Success:** Entry to full QCD relevance without dynamical fermions.

### 10. Distill + scale: symbolic loop basis + blocked attention
- **Core idea (two halves, one theme):** (a) distill trained `W_out` into explicit short-loop basis via sparse regression for human-readable operators; (b) fix memory wall with hierarchical attention — local R=2 GELT → block links → second GELT on blocked lattice.
- **Why this repo:** (a) value path `Q_v†·Ṽ` provably builds loops, `Trace` makes them scalars — distillation target exists; (b) `_nbr_idx` cubic + `T:(N,n_off,*Λ,nc,nc)` is the scaling bottleneck, blocking is the standard LGT answer.
- **First experiment:** (a) LASSO of learned Ō(t) onto loop basis {plaq, rect, chair}; (b) 2× blocking then R=2 again = effective R=4 at R=2 cost; test mass / overlap vs full R=4.
- **Success:** Interpretability + physical volumes in one move.
