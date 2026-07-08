# Slide per il prof — GELT come operatore variazionale per il glueball 0⁺⁺

Bozza del contenuto (max 10 slide). Target: esperto di ML/AI e fisica
sperimentale delle particelle, **non** esperto di LGT — ogni concetto
reticolare va introdotto in una riga, il linguaggio ML può essere tecnico.
Le figure citate sono i PNG nella root del repo.

---

## Slide 1 — Titolo

**Una rete neurale gauge-equivariante come operatore variazionale:
spettroscopia del glueball 0⁺⁺ su reticolo**

- Tesi magistrale — GELT (Gauge-Equivariant Lattice Transformer)
- Sottotitolo in una riga: *la rete impara, senza supervisione, l'operatore
  che misura la massa del glueball meglio del metodo classico standard.*

---

## Slide 2 — Il problema fisico (per un non-LGT)

**Cos'è un glueball e perché serve il reticolo**

- La QCD prevede stati legati di soli gluoni ("glueball"); il più leggero è
  lo scalare 0⁺⁺. Non sono calcolabili in teoria delle perturbazioni →
  l'unico metodo da principi primi è la **lattice gauge theory**: spaziotempo
  discretizzato, campi di gauge = matrici sui link, integrale funzionale
  campionato via Monte Carlo.
- La massa si estrae dal decadimento euclideo di un correlatore a due punti:
  `C(Δt) = ⟨Ō(t+Δt) Ō(t)⟩ − ⟨Ō⟩² ~ e^(−m·Δt)` — analogo di una vita media:
  fittare un esponenziale nel tempo (euclideo, non reale).
- **Il collo di bottiglia è l'operatore Ō**: qualunque funzione
  gauge-invariante dei link con i numeri quantici giusti "vede" il glueball,
  ma un operatore mal costruito è dominato dagli stati eccitati e il segnale
  muore nel rumore prima che il ground state emerga.
- Setup qui: SU(2) pura gauge (banco di prova standard, niente quark),
  reticolo anisotropo 12³×24, β=2.4, ξ=a_s/a_t=3 (passo temporale più fine
  → più punti utili sul decadimento), 2000 configurazioni.

*Figura: `glueball_validation.png` (pannelli in basso: C(Δ) e m_eff del
baseline classico — mostra il problema: l'operatore "thin" non plateau-a.)*

---

## Slide 3 — Il metodo classico che vogliamo battere

**Smearing + GEVP: lo stato dell'arte variazionale "hand-crafted"**

- Ricetta standard (Morningstar–Peardon): si costruiscono a mano N varianti
  dell'operatore con diversi livelli di **APE smearing** (media locale
  iterata dei link — un blur gauge-covariante che sopprime l'UV), poi si
  risolve un problema agli autovalori generalizzato (**GEVP**) sulla matrice
  dei correlatori per trovare la combinazione lineare ottimale.
- È esattamente un metodo variazionale: cerca nel *sottospazio lineare* dei
  N operatori quello con maggiore sovrapposizione col ground state.
- Limite intrinseco: la base è scelta a mano e la ricerca è lineare in
  quella base. **Domanda della tesi: una rete può cercare nello spazio non
  lineare di tutti gli operatori gauge-invarianti e fare meglio?**

---

## Slide 4 — Il modello ML

**GELT: un transformer gauge-equivariante sui link del reticolo**

- Architettura tipo attention sul grafo del reticolo, con equivarianza di
  gauge **esatta by construction** (non imparata, non approssimata):
  score di attenzione = tracce gauge-invarianti `Re Tr[Q†K̃]`, valori
  trasportati parallelamente tra siti (media su tutti i cammini minimi),
  path dei valori matrix-bilineare. Analogo reticolare dell'equivarianza
  delle GNN/CNN equivarianti, ma per un gruppo di gauge *locale* (una copia
  di SU(2) per sito — molto più restrittivo di una simmetria globale).
- Vincolo fisico chiave: la rete opera **per timeslice, solo sui link
  spaziali** (3D). Se vedesse la direzione temporale potrebbe barare sulla
  loss (slide 5). Input: plaquette a più livelli di smearing (0,2,4,6) —
  12 canali matriciali.
- Output: uno scalare gauge-invariante per timeslice → è un operatore
  Ō_GELT(t) a tutti gli effetti, direttamente confrontabile col classico.

*Figura: uno schema a blocchi dell'architettura (da disegnare: link →
ChannelLift → blocchi attention → Trace → MLP → Ō(t)).*

---

## Slide 5 — La loss: variazionale, unsupervised, con garanzia

**Training senza etichette: la loss convergente È la fisica**

- Loss di Rayleigh: `L = −C(1)/C(0)` calcolata sul correlatore
  dell'operatore prodotto dalla rete sul batch. Nessun target, nessuna
  label: **unsupervised** nel senso pieno — la "supervisione" è la
  meccanica quantistica.
- Il punto elegante (transfer matrix): inserendo un set completo di
  autostati, `C(Δ) = Σₙ |⟨n|Ô|0⟩|² e^(−Eₙ·Δ)` con tutti i pesi **positivi**
  ed energie `Eₙ ≥ m`. Quindi `C(1)/C(0) ≤ e^(−m·a_t)`: la loss ha un
  **floor fisico invalicabile** e minimizzarla = massimizzare la
  sovrapposizione col ground state. Convergenza al floor ⇒
  `−log(−loss) = m·a_t`. *La loss finale è la misura.*
- È il motivo del vincolo 3D per-timeslice: un campo recettivo temporale
  romperebbe la disuguaglianza e la rete potrebbe spingere la massa
  apparente verso 0. (Regola d'oro LGT: "mai smearare nel tempo",
  applicata alla rete stessa.)
- Per ML: il bound rende il benchmark **one-sided** — si può fare peggio
  del vero, mai meglio. Nessun rischio di "reward hacking" fisico.

---

## Slide 6 — Risultato principale

**Stessa massa, operatore migliore — con significatività**

*Figura principale: `glueball_gelt_GELT_beats_GEVP.png` (m_eff(Δ): GELT vs
GEVP classico) + `glueball_overlap_run5.png` (fit cosh e pannello overlap).*

- **Massa** (fit cosh su Δ∈[2,7], 400 config di test, jackknife a blocchi):
  - GELT: `m·a_t = 0.332 ± 0.027`
  - GEVP classico proiettato: `m·a_t = 0.340 ± 0.030`
  - Differenza correlata Δm compatibile con 0 → **stessa fisica** (deve
    esserlo: la massa è fissata dalla teoria, non dall'operatore).
- **Qualità dell'operatore** = frazione di sovrapposizione col ground state
  `A₀` (1 = operatore perfetto):
  - GELT: `A₀ = 0.903 ± 0.047` — GEVP: `A₀ = 0.837 ± 0.056`
  - Differenza correlata (stesse config, gli errori comuni si cancellano):
    `ΔA₀ = +0.066 ± 0.031`
- Al punto più sensibile (Δ=1): `m_eff(GELT) − m_eff(GEVP) = −0.028 ± 0.007`
  → **3.9σ**. Per il bound variazionale, più basso = strettamente più
  vicino al vero.
- La loss satura il floor: `val −0.6185 ↔ m·a_t ≈ 0.33`, in accordo col
  plateau GEVP indipendente — il check che tutto il meccanismo è onesto.

---

## Slide 7 — Replica indipendente

**Il risultato non è un colpo di fortuna del dataset**

*Figura: `glueball_overlap_ens1.png`.*

- Intera pipeline rieseguita da zero: nuovo ensemble Monte Carlo (seed
  indipendente) + training da zero.
  - GELT `A₀ = 1.013 ± 0.062` vs GEVP `0.925 ± 0.071`;
    `ΔA₀ = +0.089 ± 0.030` (2.9σ); a Δ=1: `−0.038 ± 0.008` (**4.5σ**).
- **Combinando i due ensemble: `ΔA₀ = +0.078 ± 0.022` → 3.6σ.**
- Lezione metodologica interessante per ML: il *valore* della loss non
  replica (ogni ensemble ha il suo floor: il secondo legge m ≈ 0.395 per
  fluttuazione statistica), ma la **qualità dell'operatore sì** — la rete
  satura il floor del *suo* ensemble in entrambi i casi. La metrica giusta
  è A₀, non la loss.

---

## Slide 8 — Quanto è buono? E i limiti

**Onestà sul perimetro del risultato**

Cosa possiamo dire:
- Prima evidenza (in questo setup) che un transformer gauge-equivariante,
  addestrato unsupervised, **batte il metodo variazionale classico** come
  qualità di operatore (3.6σ su due ensemble), a parità di massa estratta.
- Il vantaggio è genuino: la GEVP "allargata" (base classica + operatore
  GELT) collassa sull'operatore GELT puro — la rete non è ridondante con
  la base classica, la contiene e la estende.

Limiti (in ordine di importanza):
- **SU(2), non SU(3)**: teoria di prova standard, non la QCD. Pura gauge
  (quenched): niente quark, quindi niente mixing glueball–mesoni.
- **Nessun limite al continuo**: un solo β, un solo volume; il reticolo
  spaziale è grossolano (β_s = 0.8) e l'anisotropia è quella bare (ξ=3 non
  rinormalizzato) → `m·a_t ≈ 0.33` non è convertibile in MeV. Il claim è
  il *confronto di metodi a parità di reticolo*, non un numero fisico.
- Solo il canale 0⁺⁺ (slide 9).
- Baseline L-CNN (rete equivariante concorrente, Favoni et al.) a parametri
  appaiati sullo stesso task: ancora da completare.

---

## Slide 9 — Domande prevedibili (e risposte)

**FAQ da fisico**

- *"E le masse degli altri glueball (2⁺⁺, 0⁻⁺, …)?"* — Non ancora. Servono
  operatori nelle altre rappresentazioni irriducibili del gruppo cubico:
  l'output attuale della rete è uno scalare rotazionalmente simmetrico →
  proietta sul canale A₁⁺⁺ (il 0⁺⁺). Estensione naturale: più teste di
  output con simmetrie di proiezione diverse, o combinazioni non banali di
  Wilson loop in input. Gli *eccitati dello stesso canale* invece sono già
  a portata: più operatori GELT indipendenti + GEVP sulla base imparata.
- *"Come sapete che è meglio se non conoscete la massa vera?"* — Bound
  variazionale (slide 5): ogni operatore dà m_eff ≥ m, quindi a Δ fissato
  più basso = migliore, senza bisogno del valore vero. E le due pipeline
  concordano sul plateau.
- *"La rete potrebbe overfittare le fluttuazioni dell'ensemble?"* — Split
  train/test rigido: tutti i numeri quotati vengono da 400 configurazioni
  mai viste in training, con jackknife a blocchi (autocorrelazione MC).
- *"Perché un transformer e non una CNN?"* — Esiste già la L-CNN
  equivariante; l'attention aggiunge campo recettivo globale in pochi layer
  e, soprattutto, **interpretabilità**: gli score di attenzione sono tracce
  gauge-invarianti, cioè osservabili fisiche (slide 10).
- *"Quanto costa rispetto al metodo classico?"* — Il training (ore su una
  V100) si paga una volta; l'inferenza è comparabile allo smearing. Ma il
  costo dominante per entrambi è il campionamento Monte Carlo, identico.

---

## Slide 10 — Prossimi passi

**Dal risultato al programma di tesi**

1. **Baseline L-CNN a parametri appaiati** sullo stesso task per-timeslice
   (chiude il confronto "attention vs convoluzione equivariante").
2. Terzo ensemble (seed 2, già in coda) — tre misure indipendenti di A₀.
3. **Il cuore della tesi — l'attention map come misura fisica**: essendo
   gli score gauge-invarianti, la mappa di attenzione è un'osservabile.
   Programma: lunghezza di correlazione emergente ℓ_att(β) vs ξ(β) fisica,
   localizzazione dell'attenzione sui lumps topologici, specializzazione
   di teste/layer. (Interpretabilità *validata contro ground truth
   fisica* — raro in ML.)
4. Scala: SU(3), tuning dell'anisotropia rinormalizzata, più β/volumi per
   il limite al continuo; altri canali (slide 9).

**Take-home**: una rete equivariante addestrata *senza dati etichettati*,
con una loss il cui minimo è protetto da una disuguaglianza quantistica,
ha imparato un operatore migliore di quello costruito a mano in 40 anni di
pratica LGT — misurando la stessa fisica, come deve.
