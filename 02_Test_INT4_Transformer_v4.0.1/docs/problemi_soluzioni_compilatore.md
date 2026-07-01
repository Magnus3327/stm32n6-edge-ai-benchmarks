# Problemi e soluzioni — STM32N6 / ST Edge AI Core v4.0.1
### Scope: SOLO lato compilatore ed export/estrazione. Nessuna modifica al modello o ai suoi layer.

> Regola di questo documento: una "soluzione" è accettata **solo** se agisce sul
> compilatore (flag/profili) o sull'estrazione del modello (come si genera ONNX/TFLite),
> **senza toccare l'architettura**. Se l'unico modo per superare un errore è modificare
> i layer del modello, allora NON è una soluzione: è un **LIMITE VERO** da esibire a prof/STM.

Metodologia: ogni modello testato con `stedgeai analyze --target stm32n6 --st-neural-art
<profilo>@user_neuralart.json`, dove i profili definiscono quali memorie sono disponibili
(RAM interna veloce vs RAM/flash esterne lente). Config board reale: NUCLEO-N657X0-Q,
ST32N657, ~3.75 MB RAM interna scrivibile + hyperRAM 32 MB / octoFlash 64 MB esterne.

---

## TABELLA RIASSUNTIVA (verdetti REALI dalla matrice, 4 profili significativi)

Legenda memoria: **int**=solo RAM interna (~3.75MB) · **eRAM**=+hyperRAM esterna 32MB ·
**eFla**=+flash esterna (pesi) · **all**=tutte le memorie, opt O3.

| Modello (config CV_Models) | int | eRAM | eFla | all | Problema | Compila? |
|---|---|---|---|---|---|---|
| YOLOv7 @640          | LAYER | LAYER | LAYER | LAYER | `ScatterND > 5 dim` | ❌ mai (P7) |
| ResNet34 @224        | NOFIT | OK | NOFIT | OK | RAM attivazioni | ✅ con eRAM |
| PIDNet-S @2048×1024  | NOFIT | OK | NOFIT | OK | RAM attivazioni | ✅ con eRAM |
| RTMDet-l @640        | ARTIFACT | ARTIFACT | ARTIFACT | ARTIFACT | `NonMaxSuppression` | ⏳ dopo fix NMS (P3) |
| YOLOv8n @640         | NOFIT | OK | NOFIT | OK | RAM attivazioni | ✅ con eRAM |
| YOLOv11n @640        | NOFIT | OK | NOFIT | OK | RAM attivazioni | ✅ con eRAM |
| YOLOv11s @640        | NOFIT | OK | NOFIT | OK | RAM attivazioni | ✅ con eRAM |
| YOLOv11m @640        | NOFIT | NOFIT | NOFIT | OK | RAM attivazioni (grande) | ✅ solo con tutte |
| YOLOv11l @640        | NOFIT | NOFIT | NOFIT | OK | RAM attivazioni (grande) | ✅ solo con tutte |
| YOLOv11x @640        | NOFIT | NOFIT | NOFIT | OK | RAM attivazioni (grande) | ✅ solo con tutte |
| UniFormer-S @224 (ONNX)        | NOFIT | NOFIT | NOFIT | OK | RAM attivazioni | ✅ solo con tutte |
| UniFormer-B @224 (ONNX)        | NOFIT | NOFIT | NOFIT | NOFIT | RAM attivazioni | ❌ non entra mai |
| MobileCLIP-S0 img @256 (ONNX)  | ARTIFACT | — | — | — | `ONNX malformed` | ❌ (export sporco) |
| MobileCLIP-B img @256 (ONNX)   | ARTIFACT | — | — | — | `list index` | ❌ (export sporco) |
| UniFormer-S @224 (TFLite nat.) | LAYER | — | — | — | `Slice on batch` | ❌ (P8, limite vero) |
| UniFormer-B @224 (TFLite nat.) | LAYER | — | — | — | `Slice on batch` | ❌ (P8, limite vero) |
| MobileCLIP-S0 img (TFLite nat.)| LAYER | — | — | — | `Slice on batch` | ❌ (P8, limite vero) |
| MobileCLIP-B img (TFLite nat.) | LAYER | — | — | — | `Slice on > 3 dim` | ❌ (P8, limite vero) |
| baseline transformer block     | COMPILER | — | — | — | crash `NoneType` | ❌ (P9, bug ST) |
| FastInst-D1 @576               | n/a | n/a | n/a | n/a | DCN non esportabile | ❌ (non esportabile) |

### Modelli che COMPILANO → candidati scheda (9)
- Con hyperRAM esterna (`extram`): **ResNet34, PIDNet-S, YOLOv8n, YOLOv11n, YOLOv11s**
- Solo con tutte le memorie (`allmems-O3`): **YOLOv11m, YOLOv11l, YOLOv11x, UniFormer-S (ONNX)**

Nota importante: **UniFormer-S in ONNX compila** (con tutte le memorie) mentre in TFLite
nativo no (`Slice on batch`). Per questo modello la via ONNX è preferibile alla "pulita".
Il LayerNorm decomposto dell'ONNX NON è fatale qui: era solo un problema di RAM.

---

## CATALOGO PROBLEMI → CAUSA → SOLUZIONE

### P1 — "RAM Size too low" / `total bytes left unallocated` (E103)
- **Chi:** tutte le CNN @risoluzione piena (YOLO@640, ResNet34@224, PIDNet@2048, UniFormer@224).
- **Causa:** le **attivazioni** superano i ~3.75 MB di RAM interna veloce del N6.
- **Soluzione COMPILER (non tocca il modello):**
  - profilo `n6-extram` / `n6-allmems-O3` → le attivazioni possono usare la **hyperRAM esterna** (32 MB) → **compila**. Costo: latenza + energia molto maggiori (hyperRAM ~24× più costosa in energia della interna).
  - flag utili: `--Ocache-opt`, `--Oauto-sched`, `--enable-virtual-mem-pools` (già nei profili), `--split-weights` per i pesi.
- **Limite VERO (esibibile):** se il requisito è "RAM interna veloce", il modello **non ci sta** a quella risoluzione. Le uniche leve reali (ridurre risoluzione o architettura) sono **fuori scope**. → limite fisico dell'hardware documentato con il numero esatto di MB richiesti.

### P2 — `allowzero different from default for Reshape`
- **Chi:** export FP32 di ResNet34/YOLOv7/MobileNet (e loro INT4 rigenerati dall'FP32).
- **Causa:** bug dell'exporter ONNX che marca `allowzero≠0` su un Reshape.
- **Soluzione EXPORT (non tocca il modello):** usare l'**INT8** (in quantizzazione l'attributo sparisce) oppure patchare l'ONNX impostando `allowzero=0`. Nessun cambiamento ai layer.
- Stato: già risolto negli INT8 (infatti gli INT8 arrivano al compilatore).

### P3 — `NOT IMPLEMENTED: Unsupported layer types: NonMaxSuppression`
- **Chi:** RTMDet-l (e qualsiasi detector esportato col post-processing).
- **Causa:** l'export include l'**NMS** (post-processing) nel grafo; l'NPU non fa NMS.
- **Soluzione EXPORT (non tocca il modello):** esportare **senza post-processing** — il grafo del modello (backbone+neck+head raw) resta identico, si rimuove solo il decode/NMS finale (che è post-elaborazione, non un layer del modello). Per ultralytics: `nms=False`. Per rtmdet/mmdet: export dei soli output raw della testa. → il verdetto reale (RAM o OK) emerge.

### P4 — `operands could not be broadcast (N,3,3,3) (3,1,1,1)` + "layer non quantizzato"
- **Chi:** YOLO TFLite generati via onnx2tf.
- **Causa:** onnx2tf inserisce una **normalizzazione input** (3,1,1,1) e non applica bene la quantizzazione. Artefatto del **convertitore**, non del modello.
- **Soluzione EXPORT (non tocca il modello):** usare l'**export TFLite nativo** dal framework (ultralytics `export(format="tflite", int8=True)` o PyTorch→TFLite diretto con litert-torch), evitando onnx2tf. Stesso modello, grafo pulito.

### P5 — LayerNorm decomposto (`Reduce/Sub/Mul/Sqrt/Reciprocal`, tutti "non quantizzati")
- **Chi:** transformer esportati in ONNX (MobileCLIP text, UniFormer).
- **Causa:** l'export ONNX **decompone** LayerNorm in operatori elementari → l'NPU non riconosce il pattern; il supporto `LayerNormalization` aggiunto in ST 4.0.x non si attiva.
- **Soluzione EXPORT (non tocca il modello):** export **TFLite nativo** (litert-torch) mantiene LayerNorm come operatore intero; oppure export ONNX con **opset ≥ 17** che emette l'op `LayerNormalization` nativo. Stessa matematica, grafo riconoscibile.

### P6 — `INTERNAL ERROR: ONNX malformed (shape inference fails)` / `TOOL ERROR: list index out of range` / `Unknown layer format` / `E010 Non-numerical elements`
- **Chi:** transformer in ONNX INT8 (MobileCLIP s0/b, ecc.).
- **Causa:** grafo ONNX **sporco** (shape dinamiche, costanti mal formate) prodotto dall'export/quantizzazione ONNX.
- **Soluzione EXPORT (non tocca il modello):** export **TFLite nativo** (PyTorch→TFLite diretto). Già verificato: elimina questi errori e porta il compilatore a un errore **netto e singolo** (vedi P8).

### P7 — `NOT IMPLEMENTED: ScatterND with data dimensions > 5 (with Batch)`
- **Chi:** YOLOv7 @640.
- **Causa:** la **testa di detection** di YOLOv7 genera uno `ScatterND` ad alta dimensionalità (indexing/assign nel decode).
- **Soluzione EXPORT (da provare, non tocca i layer):** esportare **senza il decode-head** (solo output raw della rete), come per l'NMS. Se lo ScatterND è nel post-processing → risolvibile in export. **Se invece è intrinseco al forward del modello → LIMITE VERO** (richiederebbe modifica del modello, fuori scope).

### P8 — `NOT IMPLEMENTED: Slice on batch dimension` / `Slice on more than 3 dimensions`
- **Chi:** MobileCLIP / UniFormer in **TFLite nativo** (grafo già pulito).
- **Causa:** l'attention fa `qkv[0/1/2]` = **slice sulla dimensione 0** di un tensore 5D; l'NPU non mappa lo slice su batch/>3D.
- **Soluzione:** l'unico modo per eliminarlo è **riscrivere l'attention** (es. 3 proiezioni Q/K/V separate) — cioè **modificare il modello** → **FUORI SCOPO**.
- **Prova diagnostica:** è stato verificato che con il rewrite QKV-separato (matematicamente identico, `max|diff|=0`) l'errore **sparisce**, il che **dimostra** che è un gap del compilatore su quel pattern di op, non un problema di export. → **LIMITE VERO del compilatore 4.0.1 sull'attention**, esibibile a STM.

### P9 — `INTERNAL ERROR: 'NoneType' object is not subscriptable`
- **Chi:** baseline_transformer_block (blocco attention completo).
- **Causa:** **crash interno** del compilatore su una combinazione di operatori (i singoli op — softmax, layernorm, mha, dense — passano isolati).
- **Soluzione COMPILER (non tocca il modello):** provare `--expand-softmax` (nuovo in 4.0.x, espande il softmax in op mappabili); ricompilare da TFLite nativo. Se persiste → **bug del compilatore da segnalare a STM** (esibibile: modello minimale che fa crashare il tool).

---

## FLAG / PROFILI COMPILATORE UTILI (riepilogo, tutti non invasivi)

| Flag / profilo | Effetto | Quando |
|---|---|---|
| `--st-neural-art n6-noextmem@...` | solo RAM interna | misurare il limite fisico vero |
| `--st-neural-art n6-extram@...` | + RAM esterna | far "entrare" i modelli RAM-bound (lento) |
| `--st-neural-art n6-extflash@...` | + flash esterna (pesi) | pesi grandi in flash |
| `--st-neural-art n6-allmems-O3@...` | tutte le memorie, opt max | footprint ottimizzato |
| `--Ocache-opt`, `--Oauto-sched` | ottimizzazione scheduling/cache | ridurre RAM/latenza |
| `--split-weights` | pesi come C-array separati | gestione flash |
| `--expand-softmax` | espande softmax in HW | sbloccare attention (P9) |
| `--Omax-ca-pipe N` | pipeline convoluzioni | throughput |

---

## CONCLUSIONE (per prof / ingegnere STM)

Errori **risolvibili senza toccare il modello** (tooling/export/compiler):
- P2 allowzero, P3 NMS, P4 onnx2tf, P5 LayerNorm decomposto, P6 ONNX sporco → **export nativo/pulito**
- P1 RAM → **profilo con RAM esterna** (a costo di latenza/energia)

Errori che sono **LIMITI VERI** (non superabili senza modificare il modello, quindi fuori scope):
- P1 in RAM interna (limite fisico ~3.75 MB)
- P7 ScatterND (se intrinseco al forward)
- P8 Slice on batch (gap compilatore sull'attention — dimostrato con prova diagnostica)
- P9 crash interno del compilatore (bug ST)
- FastInst-D1 (DCN non esportabile)
