# Brief presentazione — GELT (work in progress)

> **Per il designer delle slide (Claude Design).** Questo file è il *brief dei contenuti*
> per un talk di avanzamento (~15–20 slide) rivolto a un professore di fisica che **non** è
> esperto di teorie di gauge su reticolo. Slide pulite e ariose: un'idea per slide, bullet
> corti (non paragrafi), molto spazio bianco, equazioni rese bene. Il testo in prosa sotto
> ogni slide sono **note per chi parla / contesto**: vanno distillate, non riversate sulla
> slide.
>
> **Lingua:** scrivi le slide in **italiano**. I termini tecnici consolidati possono
> restare in inglese se è la forma standard (es. *Wilson loop*, *softmax*, *attention*,
> *transformer*, *plaquette*); le equazioni restano invariate.
>
> **Tono:** lavoro in corso, onesto. **Non** stiamo rivendicando risultati finali. Mostriamo
> (1) qual è l'architettura, (2) che fa una cosa che una rete normale dimostrabilmente non
> può fare, (3) un problema aperto onesto, (4) dove andiamo dopo.
>
> **Motivo visivo suggerito:** l'idea ricorrente è "matrici sui lati di una griglia,
> moltiplicate lungo piccoli loop". Un motivo a griglia/reticolo tenue fa da filo
> conduttore.

---

## Asset immagini (usa esattamente questi file)

| chiave | percorso | cosa mostra |
|---|---|---|
| `gelt_wilson` | `/Users/francescopassante/Desktop/gelt_results/wilson1x2_scatter/gelt_wilson1x2_su2_scatter_N500.png` | GELT predice il Wilson loop 1×2 SU(2): una diagonale sottilissima — praticamente perfetto. |
| `cnn_wilson` | `/Users/francescopassante/Desktop/gelt_results/wilson1x2_scatter/large_cnn_wilson1x2_su2_scatter_N500.png` | Una CNN **grande** sullo *stesso* target: una nuvola senza struttura, nessuna correlazione. Fallimento totale. |
| `cnn_action` | `/Users/francescopassante/Desktop/gelt_results/action_scatter/large_cnn_action_su3_scatter_N5000.png` | Una CNN grande che predice l'**azione**: diagonale stretta — la CNN la azzecca. (Controllo.) |

Usa `gelt_wilson` e `cnn_wilson` **affiancate** nella slide dei risultati chiave — il
contrasto è tutta la storia. Usa `cnn_action` nella slide che spiega *perché* l'azione è il
task di controllo "facile".

---

## Arco narrativo (la spina dorsale del talk)

1. I dati vivono sui **lati** di una griglia e sono **matriciali**, con un'enorme ridondanza
   incorporata (simmetria di gauge). Le risposte fisiche devono essere cieche a quella
   ridondanza.
2. Il mattone fisico più piccolo è la **plaquette** (un minuscolo loop 1×1). Osservabili più
   grandi (**Wilson loop**) si costruiscono **moltiplicando** plaquette/link *in ordine*
   lungo rettangoli più grandi.
3. Una CNN normale non sa moltiplicare matrici specifiche in ordine — quindi può sommare gli
   input (→ impara l'**azione**) ma **non sa costruire un Wilson loop**, per quanto grande
   sia.
4. **GELT** è una rete ad attention in cui ogni operazione rispetta la simmetria per
   costruzione, e il cui value path è un **prodotto matriciale** — esattamente l'operazione
   "costruisci un loop". Impara Wilson loop che una CNN non può toccare.
5. **Problema aperto onesto:** al momento gli *score* dell'attention contano pochissimo — un
   **bias** appreso e indipendente dall'input fa quasi tutto il lavoro. Il forte prior
   geometrico risolve il task quasi da solo.
6. **Prossimo passo:** smettere di inseguire Wilson loop a geometria fissa (sono un sanity
   check superato) e passare a una vera **osservabile fisica**, dove l'attention dipendente
   dall'input dovrebbe davvero guadagnarsi il posto.

---

# SLIDE

---

## Slide 1 — Titolo

**GELT — A Gauge-Equivariant Lattice Transformer**
Sottotitolo: *Imparare osservabili fisiche di teorie di gauge su reticolo con un'attention che rispetta la simmetria di gauge*
Footer: Tesi magistrale — **work in progress** · [nome] · [data]

Note: tenere pulito. Eventuale motivo tenue reticolo-con-matrici.

---

## Slide 2 — L'obiettivo in una frase

> Costruire una rete neurale per le **teorie di gauge su reticolo** che abbia la simmetria
> della fisica incorporata — e mostrare che può calcolare cose che una rete ordinaria
> dimostrabilmente non può.

Note: questa è la tesi in una riga. Tutto il resto è "qual è la fisica", "qual è
l'architettura", "cosa funziona", "cosa è ancora aperto".

---

## Slide 3 — Cos'è l'input? (1/2) — una griglia con matrici sui lati

Bullet della slide:
- Lo spaziotempo è discretizzato su una **griglia** (un reticolo).
- Le variabili non vivono sui *siti* — vivono sui **lati** (link) tra siti vicini.
- Ogni link porta una **matrice** `U_μ(x)` (un elemento di un gruppo: Z₂, U(1), SU(2),
  SU(3)…).

Note per chi parla (per il non-esperto):
- Immagina una griglia quadrata. Su ogni lato che connette due siti vicini sta una matrice
  `U_μ(x)`: sito `x`, direzione `μ`.
- **Cosa significa fisicamente?** È un *trasportatore parallelo*: dice come ruotare un
  vettore interno di "colore" mentre salti da un sito all'altro lungo quel lato. Muoversi
  contro il lato applica l'inverso, `U†`.
- Quindi il dato fondamentale è un campo di **matrici sui lati**, non numeri sui punti. Per
  SU(3) (la vera QCD) ognuna è una matrice complessa unitaria 3×3; nel nostro testbed SU(2),
  2×2.

Visual: una piccola griglia, frecce sui lati etichettate `U_μ(x)`, un lato evidenziato come
matrice.

---

## Slide 4 — Cos'è l'input? (2/2) — simmetria di gauge: una ridondanza incorporata

Bullet della slide:
- Su **ogni sito** puoi scegliere una "rotazione di frame" locale `Ω(x)` — e la fisica non
  cambia.
- I link trasformano come: `U_μ(x) → Ω(x) · U_μ(x) · Ω†(x+μ̂)`  *(ruota all'inizio,
  de-ruota alla fine)*.
- ⇒ **infinite** configurazioni di link descrivono la **stessa** fisica.
- **Qualsiasi osservabile fisica deve essere invariante sotto questo.**

Note per chi parla:
- Questa è la *simmetria di gauge*. È una ridondanza nel modo in cui descriviamo il sistema,
  non una proprietà del sistema stesso.
- Perché conta per il ML: la simmetria è **esattamente nota**. Se la rete non la rispetta,
  spreca capacità (e dati di training) a ri-imparare che due config legate da una rotazione
  di gauge sono "la stessa cosa", e può produrre output gauge-dipendenti privi di senso. Se
  **incorporiamo** la simmetria, la rete vede solo i gradi di libertà fisici.
- È la stessa idea dell'equivarianza per traslazione nelle CNN, ma per un gruppo di
  simmetria più ricco, locale e matriciale.

Visual: stessa config disegnata due volte, legate da `Ω` per-sito, con un badge "= stessa
fisica".

---

## Slide 5 — Il mattone: la plaquette (un loop 1×1)

Bullet della slide:
- L'oggetto gauge-significativo più piccolo: moltiplica i 4 link attorno a un quadrato
  elementare.
- `P_{μν}(x) = U_μ(x) · U_ν(x+μ̂) · U_μ†(x+ν̂) · U_ν†(x)`
- Sotto una rotazione di gauge trasforma **covariantemente al suo angolo**:
  `P(x) → Ω(x) · P(x) · Ω†(x)`  *(le Ω intermedie si cancellano)*.
- La sua **traccia** è completamente gauge-**invariante**: `Re Tr P(x)` — un numero reale,
  indipendente da `Ω`.

Note per chi parla:
- Percorri il loop: avanti lungo `μ`, avanti lungo `ν`, indietro lungo `μ`, indietro lungo
  `ν`. Ogni passo è una matrice di link (o il suo inverso). Il prodotto è la plaquette.
- La magia: quando fai una trasformazione di gauge, ogni `Ω` interna incontra il suo inverso
  in un sito condiviso e si cancella — sopravvive solo la `Ω(x)` all'*angolo base* del loop,
  su entrambi i lati. Quindi `P` ruota come un oggetto "attaccato al sito x". Campi del genere
  li chiamiamo **covarianti**.
- Prendi la traccia e anche quell'ultima `Ω` si cancella (la traccia è ciclica): ottieni un
  numero puro su cui *tutti* gli osservatori concordano. **Ecco perché le osservabili sono
  tracce di loop.**

Visual: il quadrato a quattro link con il prodotto ordinato scritto attorno.

---

## Slide 6 — Cosa diamo alla rete, e cosa le chiediamo di predire

Bullet della slide:
- **Input:** il campo delle **matrici** plaquette — una matrice `nc×nc` per quadrato, per
  sito (teniamo la *matrice intera*, non solo la sua traccia, così la rete può
  moltiplicarle).
- **Target (per ora):** un **Wilson loop** `W(R,T)` = traccia del prodotto ordinato dei link
  attorno a un rettangolo `R×T`.
- L'**azione di Wilson** `S = β Σ (1 − Re Tr P / n_c)` è il target più semplice: solo una
  **somma** pesata delle tracce di plaquette già presenti nell'input.

Note per chi parla:
- Diamo deliberatamente alla rete le **matrici** plaquette (campi covarianti
  `W(x) → Ω(x) W(x) Ω†(x)`), non le loro tracce. Le tracce sarebbero già scalari
  gauge-invarianti — facili ma morti. Tenere le matrici è ciò che permette alla rete di
  *moltiplicarle tra loro* per costruire loop più grandi.
- **Perché i Wilson loop contano fisicamente:** un loop di taglia `R×T` sonda la teoria a
  quella scala — loop più grandi codificano cose come la forza tra quark statici e il
  confinamento. Costruirne uno dalle plaquette 1×1 significa moltiplicare molte plaquette
  adiacenti *nell'ordine giusto*, trasportate a un angolo comune. Quel "moltiplica queste
  matrici specifiche in quest'ordine" è esattamente l'operazione attorno a cui è costruita
  la nostra architettura — ed esattamente ciò che una CNN ordinaria non può fare.
- **Dati:** le configurazioni sono generate con **Monte Carlo Metropolis** a un accoppiamento
  `β` — cioè la vera distribuzione di Boltzmann della teoria, non matrici casuali. Quindi i
  loop portano una struttura fisica genuina, `β`-dipendente.

---

## Slide 7 — Perché una CNN normale è lo strumento sbagliato

Bullet della slide:
- Una CNN tratta le entrate delle matrici come "canali" di immagine indipendenti e fa
  convoluzioni spaziali.
- **Non ha nessuna nozione** di "moltiplica *queste quattro/otto* matrici, in *quest'ordine*,
  a *questo* punto base".
- Inoltre **non sa nulla della simmetria di gauge** — dovrebbe imparare quella ridondanza
  dai dati.
- Previsione: può **sommare** input (→ l'azione) ma **non può costruire un loop**.

Note per chi parla:
- La moltiplicazione di matrici è non commutativa e dipendente dall'ordine; la covarianza di
  gauge richiede di trasportare tutto prima in un frame comune. Uno stack di convoluzioni +
  non-linearità puntuali è strutturalmente il primitivo sbagliato — può mescolare e sommare
  canali, non formare prodotti di gruppo ordinati legati a punti base.
- È esattamente il gap di bias induttivo di cui parla la tesi.

---

## Slide 8 — Task di controllo: l'azione *è* imparabile da una CNN

Bullet della slide:
- L'azione è una **somma lineare** di quantità (`Re Tr P`) **già presenti nell'input**.
- Una CNN grande la impara pulita. → immagine `cnn_action`
- ✅ Quindi quando più avanti una CNN *fallisce*, **non** è un problema di capacità — è un
  problema di *struttura*.

Note per chi parla:
- Mostra `cnn_action` (CNN grande, azione SU(3), N=5000): diagonale stretta, vero vs
  predetto.
- Punto: "sommare gli input" è ben alla portata di una CNN. Lo stabiliamo apposta, così il
  fallimento successivo sui Wilson loop non si può liquidare con "la CNN era troppo piccola".
- (Vediamo lo stesso per SU(2)/Z₂; il plot SU(3) grande è l'illustrazione più pulita.)

Immagine: `cnn_action` — grande, centrata. Didascalia: *"CNN grande, azione: facile — è solo
una somma degli input."*

---

## Slide 9 — Il task difficile: un Wilson loop. La CNN fallisce; GELT riesce

**Questa è la slide chiave. Due scatter plot affiancati.**

- Sinistra: `cnn_wilson` — CNN grande, Wilson loop 1×2 SU(2) → **nuvola senza struttura**.
- Destra: `gelt_wilson` — GELT, *stesso* target → **diagonale sottilissima**.

Riga didascalia della slide:
> Stesso target, stessi dati. La CNN — anche con **molti** parametri — non può ricostruire un
> prodotto ordinato di matrici. GELT, costruita per moltiplicare loop, lo predice quasi
> esattamente.

Note per chi parla:
- Entrambi gli assi sono "vero vs predetto" del valore del Wilson loop sul test set. Un
  predittore perfetto è la diagonale.
- La nuvola della CNN non è underfitting per mancanza di capacità — è un modello **grande**.
  Semplicemente le manca l'*operazione*. La linea quasi perfetta di GELT è il bias induttivo
  che paga.
- Questa è la prima evidenza reale che il design gauge-equivariante ci compra qualcosa che
  una rete generica non può ottenere.

---

## Slide 10 — L'architettura: GELT in un'immagine

Bullet della slide (ideale una pipeline verticale a diagramma):
1. **Matrici plaquette** `W(x)` (input covariante)
2. → blocchi **GEMHSA** (attention gauge-equivariante) ×N
3. → **Trace** (covariante → invariante)
4. → **MLP** sugli scalari invarianti
5. → **somma / media sui siti** → predizione scalare

Riga della slide:
> Ogni blocco mappa **campi matriciali covarianti → campi matriciali covarianti**; solo il
> readout finale prende una **traccia** per diventare gauge-invariante. L'equivarianza è
> esatta *per costruzione*.

Note per chi parla:
- Ci basiamo sul **framework L-CNN** (Favoni et al., 2012.12901), che dimostra che puoi
  raggiungere *qualsiasi* Wilson loop trasportando e moltiplicando ripetutamente campi
  covarianti.
- La deviazione di GELT: rimpiazzare lo stack convoluzione+bilineare di L-CNN con un blocco
  di **attention**, mantenendo l'unica operazione che dà l'espressività — un **prodotto
  matriciale** nel value path. Più dettagli nelle due slide seguenti.

---

## Slide 11 — Dentro un blocco: attention gauge-equivariante (GEMHSA)

Mostra come pipeline etichettata; ogni freccia traccia "covariante" vs "invariante".

1. **Augment & proiezione** — combinazioni lineari per-sito dei canali danno le matrici
   query/key/value `Q, K, V`. Lineare-nei-canali ⇒ ancora **covariante** (`→ Ω Q Ω†` ecc.).
2. **Trasporto parallelo** — per confrontare il sito `x` col vicino `x+Δx`, porta la matrice
   del vicino nel frame di `x`: `K̃ = T(x) · K(x+Δx) · T†(x)`.
   - `T` è il prodotto dei link lungo il cammino di connessione. Facciamo la **media su tutti
     i cammini reticolari più corti** entro una palla di raggio-Manhattan `R` (non un singolo
     cammino allineato agli assi) ⇒ un campo recettivo più ricco e non allineato agli assi in
     un solo blocco.
3. **Score** `s = Re Tr[ Q†(x) · K̃ ]` (+ un bias appreso per testa e per vicino).
   - Entrambi i fattori sono covarianti *in x*, quindi `Q†K̃ → Ω (Q†K̃) Ω†`; la **traccia
     cancella Ω** ⇒ lo score è gauge-**invariante**. *(Fisicamente: un correlatore a due
     loop.)*
4. **Softmax** sui vicini ⇒ pesi di attention `α` (scalari invarianti).
5. **Value path (la deviazione chiave):** output `= Q†(x) · Σ_vicini α · Ṽ`.
   - Un **prodotto matriciale** di due matrici covarianti ⇒ **covariante**. È il passo
     "moltiplica due loop in un loop più grande".
6. **Residuo + gate** — gate tramite una funzione di `Re Tr` (uno scalare invariante),
   sommato al residuo. Covarianza preservata.

Note per chi parla:
- La disciplina di tutto il talk: tracciare il tipo di simmetria attraverso ogni freccia. Le
  matrici restano **covarianti**; nel momento in cui prendiamo una traccia otteniamo uno
  scalare **invariante**, l'unica cosa su cui possiamo fare softmax o passare a un MLP.
- Verificato a **precisione macchina** nella test suite (rotazioni di gauge casuali e
  worst-case): l'output della rete è esattamente invariante.

---

## Slide 12 — Perché il value path *moltiplicativo* è il punto cruciale

Bullet della slide:
- Il value path di un transformer vanilla è una **somma pesata** dei value: `Σ α·Ṽ`.
- Una somma di loop **non** è un loop più grande — resta della stessa taglia.
- GELT moltiplica: `Q† · (Σ α·Ṽ)` — un **prodotto**, quindi due loop si fondono in un loop
  più grande.
- Impilare blocchi ⇒ loop di taglia crescente ⇒ (seguendo L-CNN) **qualsiasi** Wilson loop è
  raggiungibile.

Note per chi parla:
- Questo singolo cambiamento — prodotto invece di somma nel value path — è ciò che trasferisce
  l'argomento di universalità "loop-doubling" di L-CNN dentro il transformer. È la ragione
  per cui GELT può costruire i loop 1×2, 2×2, … che la CNN non può.

---

## Slide 13 — Risultati finora (SU(2), dati Metropolis)

Bullet della slide:
- Dati: **SU(2)**, configurazioni da **Metropolis MCMC** (vera distribuzione
  all'accoppiamento `β`).
- GELT predice i Wilson loop accuratamente, e abbiamo **scalato la taglia del loop**:
  **1×2 → 1×3 → 2×2 → 2×3 → 3×3.** ✅
- La **CNN baseline grande fallisce su tutti** (la nuvola 1×2 della Slide 9 è
  rappresentativa).
- L'**azione** è predicibile da entrambe — è il task di controllo lineare.

Note per chi parla:
- Lo scatter 1×2 (Slide 9) era il risultato già mostrato in precedenza. **Il progresso da
  allora: abbiamo esteso lo stesso successo a rettangoli più grandi — 1×3, 2×2, 2×3 e 3×3 —
  tutti in SU(2) su configurazioni campionate con Metropolis.** Ogni loop più grande richiede
  alla rete di moltiplicare più plaquette in ordine, e GELT tiene il passo; la CNN non si
  alza mai da terra.
- Framing: questa è ora una dimostrazione *robusta* del gap di bias induttivo, non un singolo
  target fortunato.

(Opzionale: se in seguito si vuole un pannello a multipli dei loop più grandi, si può
aggiungere — per ora il contrasto 1×2 porta il messaggio.)

---

## Slide 14 — Problema aperto onesto: l'attention non fa il lavoro

Bullet della slide:
- I diagnostici mostrano: **quasi tutto il lavoro è fatto da un bias appreso e indipendente
  dall'input** — *non* dallo score di attention dipendente dai dati `Re Tr[Q†K̃]`.
- La softmax sui vicini è di fatto **fissata dal bias**, non dalla configurazione.
- Interpretazione: per un **singolo Wilson loop a geometria fissa**, il pattern di attention
  ottimo *è* una selezione geometrica fissa ("moltiplica sempre lungo questo rettangolo"). Un
  bias statico codifica esattamente quello — quindi gli score non hanno nulla da aggiungere.
- ⇒ Abbiamo mostrato che il **prior equivariante** (trasporto + value moltiplicativo) è
  potente; **non** abbiamo ancora mostrato che il **meccanismo di attention** si guadagna il
  posto.

Note per chi parla:
- È il caveat WIP centrale, detto chiaramente. La rete si comporta più come una *convoluzione
  fissa appresa sui cammini di trasporto* che come attention basata sul contenuto.
- Non è tanto un bug quanto una proprietà del task: un loop fisso è geometricamente banale —
  la stessa ricetta funziona per ogni configurazione, quindi non serve routing dipendente
  dall'input.

---

## Slide 15 — Possibili soluzioni

Bullet della slide:
- **Cambiare il target** così che un singolo pattern fisso non possa risolverlo — una vera
  osservabile fisica (slide seguente). *La più promettente — attacca la radice.*
- **Training multi-target / multi-`β`:** un singolo prior geometrico statico non può essere
  ottimo per molti target insieme ⇒ forza routing dipendente dai dati.
- **Ribilanciare score vs bias:** inizializzazione Q/K più marcata (scala Q/K già
  disaccoppiata da V), regolarizzare/annichilire gradualmente il bias, temperatura sullo
  score.
- **Value path più ricco:** una proiezione `Q` separata e più generale; più cammini
  bilineari.

Note per chi parla:
- Framing onesto: l'architettura *funziona*; la domanda è se l'*attention* contribuisce oltre
  un prior geometrico. Il modo più pulito per scoprirlo è dare un task che *richieda* routing
  dipendente dall'input.

---

## Slide 16 — Perché smetteremo di inseguire i Wilson loop

Bullet della slide:
- I Wilson loop a geometria fissa erano un **sanity check** — ed è **passato** (la CNN non
  può, GELT sì, fino a 3×3 in SU(2)).
- Ma ognuno è un singolo target geometricamente banale → un **prior statico basta** → non
  mette sotto stress l'attention.
- L'obiettivo vero è una vera **osservabile fisica** — un funzionale non lineare su **molti**
  loop e scale:
  - carica topologica / suscettibilità `χ_t`,
  - potenziale quark-statico / tensione di stringa,
  - un parametro d'ordine vicino a una transizione di fase.
- È esattamente dove il **campo recettivo completo (palla L1)** e l'**attention dipendente
  dai dati** dovrebbero contare.

Note per chi parla:
- Il punto da far arrivare al professore: predire Wilson loop sempre più grandi ha rendimenti
  decrescenti — conferma la stessa capacità a taglia maggiore. La scienza interessante (e il
  test su se l'*attention* aiuti) inizia quando il target è una vera quantità fisica, non un
  loop scelto a mano.

---

## Slide 17 — Stato & prossimi passi

Bullet della slide:
- ✅ Architettura di attention gauge-equivariante, equivarianza **esatta** (precisione
  macchina).
- ✅ Batte una CNN grande sui Wilson loop che strutturalmente non può imparare — **SU(2),
  fino a 3×3**, su dati Metropolis.
- ✅ Il task di controllo (azione) conferma che i fallimenti della CNN sono **strutturali,
  non di capacità**.
- ⏳ Aperto: rendere l'**attention** (non solo il bias) dimostrabilmente rilevante.
- ⏭ Prossimo: una vera **osservabile fisica** come target (`χ_t`, tensione di stringa, …);
  scalare verso **SU(3)** e **3+1D**.

Note: chiudere sulla roadmap — è un talk di avanzamento, quindi la slide proiettata in avanti
è la chiusura giusta.

---

## Slide 18 — (opzionale) Backup: la simmetria, tracciata dall'inizio alla fine

Un singolo diagramma riassuntivo per il Q&A: input **covariante** → il trasporto mantiene
**covariante** → lo score è una **traccia** ⇒ **invariante** → softmax/MLP agiscono solo su
**invarianti** → readout **invariante**. "Covariante in ingresso, invariante in uscita, per
costruzione."
