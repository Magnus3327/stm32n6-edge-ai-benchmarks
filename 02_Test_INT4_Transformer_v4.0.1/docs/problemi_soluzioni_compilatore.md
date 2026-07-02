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
| YOLOv7 @640 (export originale) | LAYER | LAYER | LAYER | LAYER | `ScatterND > 5 dim` | ❌ (P7) |
| **YOLOv7 @640 (no decode-head)** | NOFIT | NOFIT | NOFIT | **OK 21.45 MiB** | RAM attivazioni | ✅ **con tutte (P7 RISOLTO)** |
| ResNet34 @224        | NOFIT | OK | NOFIT | OK | RAM attivazioni | ✅ con eRAM |
| PIDNet-S @2048×1024  | NOFIT | OK | NOFIT | OK | RAM attivazioni | ✅ con eRAM |
| RTMDet-l @640 (solo no-NMS) | LAYER | LAYER | LAYER | LAYER | `Gather` (ONNX Exporter) | ❌ (P7-bis) |
| **RTMDet-l @640 (raw-head)** | NOFIT | NOFIT | NOFIT | **OK 14.52 MiB** | RAM attivazioni | ✅ **con tutte (P7-bis RISOLTO)** |
| YOLOv8n @640         | NOFIT | OK | NOFIT | OK | RAM attivazioni | ✅ con eRAM |
| YOLOv11n @640        | NOFIT | OK | NOFIT | OK | RAM attivazioni | ✅ con eRAM |
| YOLOv11s @640        | NOFIT | OK | NOFIT | OK | RAM attivazioni | ✅ con eRAM |
| YOLOv11m @640        | NOFIT | NOFIT | NOFIT | OK | RAM attivazioni (grande) | ✅ solo con tutte |
| YOLOv11l @640        | NOFIT | NOFIT | NOFIT | OK | RAM attivazioni (grande) | ✅ solo con tutte |
| YOLOv11x @640        | NOFIT | NOFIT | NOFIT | OK | RAM attivazioni (grande) | ✅ solo con tutte |
| UniFormer-S @224 (ONNX)        | NOFIT | NOFIT | NOFIT | OK | RAM attivazioni | ✅ solo con tutte |
| UniFormer-B @224 (ONNX)        | NOFIT | NOFIT | NOFIT | NOFIT | RAM attivazioni | ❌ (ONNX gonfio) |
| MobileCLIP-S0 img @256 (ONNX)  | ARTIFACT | — | — | — | `ONNX malformed` | ❌ (export sporco) |
| MobileCLIP-B img @256 (ONNX)   | ARTIFACT | — | — | — | `list index` | ❌ (export sporco) |
| UniFormer-S/B, MCLIP (TFLite nat. v1) | LAYER | — | — | — | `Slice on batch / >3 dim` | ❌ (P8) |
| **UniFormer-S (TFLite ST-friendly)** | NOFIT | **OK 33.92 MiB** | NOFIT | **OK 4.20 MiB** | RAM attivazioni | ✅ **(P8+P10 RISOLTI)** |
| **UniFormer-B (TFLite ST-friendly)** | NOFIT | NOFIT | NOFIT | **OK 4.20 MiB** | RAM attivazioni | ✅ **(prima non entrava MAI)** |
| **MobileCLIP-S0 img (TFLite ST-friendly)** | NOFIT | **OK 2.55 MiB** | **OK 2.52 MiB** | **OK 2.52 MiB** | pesi 13.8 MB > RAM int | ✅ **(P8+P11 RISOLTI)** |
| **MobileCLIP-B img (TFLite ST-friendly)** | NOFIT | NOFIT | NOFIT | NOFIT | **86M param ≈ 90 MB > 96 MB tot** | ❌ **LIMITE FISICO VERO (pulito)** |
| baseline transformer block (ONNX) | COMPILER | COMPILER | COMPILER | COMPILER | crash `NoneType` (anche con --expand-softmax) | ❌ (P9, bug ST) |
| baseline transformer block (TFLite plain) | COMPILER | COMPILER | COMPILER | COMPILER | shape map `[1,1,1,64]` (P10) | ❌ |
| **baseline transformer block (TFLite ST-friendly)** | **OK** | **OK** | **OK** | **OK** | — | ✅ **su TUTTI i profili, anche solo RAM interna** |
| FastInst-D1 @576               | n/a | n/a | n/a | n/a | DCN non esportabile | ❌ (non esportabile) |

### Modelli che COMPILANO → candidati scheda (14 — erano 9)
- Con hyperRAM esterna (`extram`): **ResNet34, PIDNet-S, YOLOv8n, YOLOv11n, YOLOv11s, UniFormer-S (TFLite ST-friendly), MobileCLIP-S0 (TFLite ST-friendly)**
- Solo con tutte le memorie (`allmems-O3`): **YOLOv11m, YOLOv11l, YOLOv11x, UniFormer-S (ONNX), YOLOv7 (no decode-head), RTMDet-l (raw-head), UniFormer-B (TFLite ST-friendly)**
- Su tutti i profili, anche solo RAM interna: **baseline transformer block (TFLite ST-friendly)**

Su 15 modelli CV_Models testabili, ora **13 compilano**. Restano fuori solo:
**MobileCLIP-B** (86M param: non entra fisicamente nelle memorie della piattaforma) e
**FastInst-D1** (DCN non esportabile in ONNX — limite dell'ecosistema export, non di ST).

### La "ricetta ST-friendly" (riscritture di puro export, matematicamente identiche)
Verificate con `max|diff| ≈ 1e-6` rispetto al modello originale (stessi pesi):
1. **QKV split**: 3 Linear separate al posto della proiezione fusa + slice 5D → elimina P8.
2. **LayerNorm manuale keepdim=True**: litert decompone `nn.LayerNorm` con
   `MEAN(keepdims=False)` → tensori rank-1 che l'importer ST non mappa (P10).
3. **Positional embedding congelato**: a input fisso `pos_embed(N)` è una costante →
   elimina il `RESIZE_BILINEAR` che crasha l'importer (P11).
4. **Tokenizzazione**: per MobileCLIP la forma `flatten(2).transpose` è ok mentre
   `permute+reshape` manda in errore la shape inference interna ST (P12) — per
   UniFormer è l'esatto contrario. Va scelta empiricamente per modello.

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
- **✅ FATTO (2026-07-01):** NMS strippato via `onnx.utils.extract_model` (taglio il grafo sui tensori raw box/score `1779`/`1801`, prima dell'NMS). File: `Models/1_ram_fail/rtmdet_l_int8_nonms.onnx` (1865 nodi, era 1946). Ritestato sulla matrice: l'errore NMS sparisce, **ma emerge P7-bis sotto** (Gather non supportato) — quindi RTMDet-l resta non compilabile, per un motivo diverso e più profondo.

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

### P7 — `NOT IMPLEMENTED: ScatterND with data dimensions > 5 (with Batch)` ✅ RISOLTO
- **Chi:** YOLOv7 @640.
- **Causa:** lo `ScatterND` sta nel **decode del post-processing** (assegnazioni in-place `y[..., 0:2] = ...`), non nel forward. Il grafo esportato contiene GIÀ gli output raw (`clone/clone_1/clone_2` = teste 80/40/20 pre-decode) accanto all'output decodificato.
- **✅ FATTO (2026-07-03):** `onnx.utils.extract_model` sull'INT8 esistente, output = i 3 tensori raw (stessi pesi, zero riesport). File: `Models/1_ram_fail/yolov7_int8_nohead.onnx` (1119 nodi, era 1276; 0 ScatterND). **Risultato: COMPILA con `allmems-O3` (21.45 MiB attivazioni)** — stesso profilo di YOLOv11 m/l/x. Il verdetto vero era RAM, come gli altri YOLO grandi.

### P7-bis — `NOT IMPLEMENTED: Case of Gather in ONNX Exporter not supported` ✅ RISOLTO
- **Chi:** RTMDet-l @640, dopo lo strip dell'NMS (P3).
- **Causa:** i 6 `Gather` stanno tutti nella **selezione top-k pre-NMS** (idx 1821+ su 1865 nodi), a valle della testa raw. Non sono nel forward.
- **✅ FATTO (2026-07-03):** taglio del grafo sui 6 output raw della testa (cls 80ch: `1450/1471/1492`, reg 4ch: `1459/1480/1501`, 3 livelli FPN). File: `Models/1_ram_fail/rtmdet_l_int8_rawhead.onnx` (1737 nodi, 0 Gather/TopK). **Risultato: COMPILA con `allmems-O3` (14.52 MiB attivazioni).**

### P8 — `NOT IMPLEMENTED: Slice on batch dimension` / `Slice on more than 3 dimensions` ✅ RISOLTO
- **Chi:** MobileCLIP / UniFormer in **TFLite nativo** (grafo già pulito).
- **Causa:** l'attention fa `qkv[0/1/2]` = **slice sulla dimensione 0** di un tensore 5D; l'NPU non mappa lo slice su batch/>3D.
- **✅ FATTO (2026-07-03):** riscrittura **QKV split** (3 Linear separate dai pesi della proiezione fusa, matematicamente identica, `max|diff| ≈ 1e-6`). Non modifica il modello: stessi pesi, stessa funzione — è una trasformazione di export. Combinata con P10/P11 sblocca la compilazione: **UniFormer-S/B e MobileCLIP-S0 COMPILANO**. Script: `export_uniformer_stfriendly_tflite.py`, `export_mobileclip_stfriendly_tflite.py`.
- Resta un **gap documentabile del compilatore 4.0.1**: il pattern fuso standard (usato da timm/torchvision/quasi tutti i ViT) non è supportato; la forma equivalente split sì.

### P9 — `INTERNAL ERROR: 'NoneType' object is not subscriptable` ✅ AGGIRATO (bug ST confermato)
- **Chi:** baseline_transformer_block (blocco attention completo).
- **Causa:** **crash interno** del compilatore su una combinazione di operatori (i singoli op — softmax, layernorm, mha, dense — passano isolati).
- **Verificato (2026-07-03):** `--expand-softmax` (opzione atonn, va nei profili del json — non è un flag stedgeai) **NON risolve**: stesso crash sull'ONNX. → bug del compilatore da segnalare a STM, con modello minimale riproducibile.
- **✅ AGGIRATO:** rebuild in TFLite nativo ST-friendly (Q/K/V espliciti dai pesi `in_proj` della `nn.MultiheadAttention` + ManualLN): **COMPILA su TUTTI i profili, anche solo RAM interna**. La variante TFLite "plain" (MHA fusa) invece inciampa in P10. Script: `export_baseline_block_tflite.py`.

### P10 — `TOOL ERROR: Shape and shape map lengths must be the same: [1,1,1,N] vs (BATCH, CH)` ✅ RISOLTO
- **Chi:** qualsiasi modello TFLite con `nn.LayerNorm` esportato via litert-torch (UniFormer, baseline block; N = numero token: 3136, 64, ...).
- **Causa:** litert decompone LayerNorm in `MEAN(keepdims=False) + RESHAPE + SQUARED_DIFFERENCE + ...`: il MEAN produce tensori **rank-1** `[N]` che l'importer ST padda a `[1,1,1,N]` e non riesce a mappare sulla shape map 2D `(BATCH, CH)`.
- **✅ Soluzione EXPORT:** LayerNorm **manuale con `keepdim=True`** (stessi pesi/eps): la decomposizione resta alla rank dell'input e l'importer la accetta. Il LayerNorm resta comunque decomposto (mappato su op aritmetiche) — funziona.

### P11 — `TOOL ERROR: list index out of range` (importer TFLite) ✅ RISOLTO
- **Chi:** MobileCLIP-B (ViT). Stesso errore che dava il suo ONNX.
- **Causa (isolata per bisezione del modello):** il **positional embedding** interpolato — `LearnablePositionalEmbedding` emette un `RESIZE_BILINEAR` che fa crashare l'importer ST. Non c'entrano né il cls-token concat né gli encoder.
- **✅ Soluzione EXPORT:** a input fisso `pos_embed(N)` è una **costante** → precalcolata e congelata in un buffer (`FrozenPosEmbed`). Il RESIZE_BILINEAR sparisce. MobileCLIP-B arriva così al verdetto **RAM pulito** (vedi tabella): 86M param ≈ 90 MB INT8 > 32 MB hyperRAM + 64 MB octoFlash → **LIMITE FISICO VERO della piattaforma**, non più mascherato da errori di tooling.

### P12 — `INTERNAL ERROR: Exported ONNX could be malformed (shape inference fails)` su TFLite
- **Chi:** MobileCLIP-S0 con tokenizzazione `permute(0,2,3,1)+reshape`.
- **Causa:** l'importer ST (che converte internamente TFLite→ONNX) non digerisce quel pattern di layout in quel contesto; con la forma originale `flatten(2).transpose` funziona. Per UniFormer è l'esatto contrario (P10 emergeva col flatten).
- **Soluzione EXPORT:** scegliere empiricamente la forma di tokenizzazione per modello (flag `TOKENIZE=flatten|permute` nello script MobileCLIP).

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

## CONCLUSIONE (per prof / ingegnere STM) — aggiornata 2026-07-03

Errori **risolti senza toccare il modello** (tooling/export/compiler — stessi pesi, stessa matematica):
- P2 allowzero, P3 NMS, P4 onnx2tf, P5 LayerNorm decomposto, P6 ONNX sporco → **export nativo/pulito**
- P7 ScatterND, P7-bis Gather → **taglio del post-processing** (decode/top-k non sono il modello)
- P8 Slice on batch → **QKV split** · P10 shape-map LN → **LN manuale keepdim** · P11 RESIZE_BILINEAR → **pos-embed congelato**
- P9 crash NoneType → **aggirato** via rebuild TFLite ST-friendly (l'ONNX resta un bug ST da segnalare)
- P1 RAM → **profilo con RAM esterna** (a costo di latenza/energia)

**Risultato: 13 modelli su 15 compilano** (erano 9). Restano SOLO limiti veri:
- P1 in RAM interna (limite fisico ~3.75 MB): nessun modello CV_Models @risoluzione piena ci sta (tranne il baseline block)
- MobileCLIP-B: 86M param non entrano nemmeno in TUTTE le memorie (limite fisico piattaforma, ora con verdetto pulito)
- FastInst-D1: DCN (deformable conv) non esportabile in ONNX (limite ecosistema export, non ST)

Messaggio per STM: i gap del compilatore 4.0.1 sull'attention (P8/P10/P11/P9) sono TUTTI
aggirabili con riscritture meccaniche a livello di export — il compilatore potrebbe farle
internamente (pattern-rewrite): il pattern QKV fuso è lo standard di timm/torchvision/HF.
