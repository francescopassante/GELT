# Stato del candidato vortici Z₂ — sintesi per decidere il next step

Tutto su `main`, pushato, 204 test verdi.

## Fatto

- **§9 di `notes/where_attention_can_win.md`** = attempt 4, il candidato vortici Z₂. Rinumerate §9→§10→§11.
- **Pre-flight** (`gelt/vortex_targets.py`, `scripts/z2_vortex_preflight.py`, 20 test) girato sui 4 ensemble in cache: **5/5 porte, 4/4 β**.
- **Scala β fissata** (§9.6): primario **0.7520**, replica **0.7450** — primi e secondi su headroom, soffitto lineare basso e base rate bilanciato.
- **`PROBE_GROUP=z2`** dentro `probe_common.py`; tutti i bracci girano; percorso SU(2) bit-identico (verificato contro un worktree al commit precedente).

## Numeri chiave

Siti con vortice (lettura mascherata), portata Manhattan 8, geometria di produzione:

| β | p_neg | base | R² lin | R² local | headroom |
|---|---|---|---|---|---|
| 0.7450 | .054 | .587 | +.106 | +.747 | +.253 |
| **0.7520** | .044 | .428 | +.152 | +.671 | **+.329** |
| 0.7560 | .036 | .314 | +.156 | +.806 | +.194 |
| 0.7585 | .032 | .212 | +.280 | +.762 | +.239 |

## Tre misure che hanno cambiato il disegno

1. In Z₂ il trasporto mediato di GELT è una **maschera di vortice esatta**, `T² = (1 + P_encl)/2 ∈ {0,1}`; quello single-path e quello della L-CNN sono l'identità. → terzo meccanismo (**M3**) → disegno **2×2** `{softmax, frozen} × {average, single}`, non un A/B.
2. **Lettura mascherata obbligatoria**: il 91% dei siti non porta vortici; all-sites 0.65 contro masked 0.19.
3. **La L-CNN non ha un forward utilizzabile in Z₂ a 4 layer**: 0.36 → 3.6 → 7.8e4 → **1.4e21** all'inizializzazione (`inf` in produzione), mentre il campo di GELT resta a 1. È **M2 prima del training**. Fix: `conv_init_scale = 0.5` (default 1.0 invariato).

## Due premesse del file iniziale corrette

- **T1 era stato ritirato come circolare** (softmax = soft-argmax), quindi l'analogia su cui poggiava la proposta non regge. Aggiunto un **quinto criterio** (non-circolarità) e l'audit di V1 contro di esso. Il carico ora è sul gap routing − densità misurato (+0.48 … +0.65).
- Il campo dei vortici è **sparso** (p_neg ≈ 0.03–0.05), non denso: l'argomento "la L-CNN fallisce perché il campo è denso" non si applica. Il contro-argomento (confronto di oggetti estesi ≠ detection) è scritto come argomento, non come risultato.

## Aperto

- **Nessun braccio addestrato.** Serve uno **sweep LR × init per famiglia** prima della griglia — anche per rivedere il 0.5.
- **`R²(local)` è l'unica colonna non monotona in β** (.747 / .671 / .806 / .762). Se sia rumore sta nelle barre d'errore dei dump `results/z2_vortex/preflight_b*.pt`, non ancora lette.
- **Tensione strutturale**: il soffitto lineare **sale** verso β_c (.106 → .280), quindi i due β scelti sono i più lontani dalla criticità — non è una misura al punto critico.
