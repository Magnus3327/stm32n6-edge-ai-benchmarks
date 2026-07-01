# Campagna di test — ST AI Studio / ST Edge AI Core v4.0.1 (build 20581)

Obiettivo: ri-testare i modelli su **AI Core 4.0.1** (prima era 4.0.0-20500) seguendo
i suggerimenti dei prof dopo la presentazione:
- **INT4** per i modelli con RAM fail
- **TFLite** per i modelli che non compilavano (transformer)

> ⚠️ Tutto in **INT8** come baseline; **niente FP32** (non è il target ed era solo un
> artefatto di build/ROM overflow). I modelli del gruppo RAM-fail si testano **anche in INT4**.

---

## ⚠️ LEGGERE PRIMA DI TESTARE — due avvertenze tecniche importanti

Ho verificato i file `.onnx` reali. Due dei suggerimenti vanno presi con cautela, e
conviene saperlo *prima* di bruciare una sessione su Windows:

### 1. INT4 (per la RAM) — quasi certamente NON risolve, ma va documentato
- L'export INT4 (onnxruntime) è **weight-only**: quantizza i *pesi*, non le *attivazioni*.
  Verificato: `pidnet_s_int4` ha gli **stessi identici** nodi Quantize/Dequantize dell'INT8
  (469/232). Le attivazioni restano int8/float.
- Il collo di bottiglia in npuRAM sono **le attivazioni**, non i pesi (che vanno in Flash).
  Quindi INT4, per costruzione, **non può ridurre la RAM di lavoro**.
- Inoltre l'INT4 introduce operatori nel dominio `com.microsoft` (MatMulNBits/QDQ a blocchi)
  che ST Edge AI Core 4.0 **non ingerisce** → `pidnet_s_int4` dà `E103 ... 32 MB unallocated`.
- ➡️ Va testato **una volta** per *documentare* nella tesi che il suggerimento non è
  applicabile su questa NPU (Neural-ART è int8-native). Le uniche leve reali per la RAM
  sono: **ridurre la risoluzione di input** o cambiare architettura.

### 2. TFLite via onnx2tf — NON è un vero bypass
- Lo script `export_tflite.py` fa `ONNX → onnx2tf → TFLite`: converte l'ONNX **già
  problematico**, quindi si porta dietro gli stessi difetti (LayerNorm decomposto in
  ReduceMean/Sub/Mul/Rsqrt, shape dinamiche). Per questo i TFLite transformer falliscono
  anche loro (`TOOL ERROR: list index out of range`, `0 vs. 116`).
- Il TFLite ha senso **solo se nativo** (esportato dal framework originale, non riconvertito):
  - YOLO: `ultralytics ... export(format="tflite", int8=True)` → grafo pulito.
  - MobileNet/classification: TFLite ufficiali esistono (Kaggle/TF Hub).
  - Per resnet/pidnet/rtmdet/mobileclip/uniformer **non** esiste un TFLite nativo ufficiale.
- Le due sottocartelle separano i due casi:
  - `tflite_native/`   → export pulito dal framework (da generare)
  - `tflite_onnx2tf/`  → conversione dell'ONNX (quelli vecchi, già falliti)

### Cosa è davvero un problema di EXPORT (fixabile), non del modello
- `allowzero` su Reshape (mobilenet/resnet34/yolov7 **fp32**) → sparisce in INT8.
- `NonMaxSuppression` nel grafo (rtmdet) → va esportato senza post-processing.
- ➡️ Per questo il gruppo "1_ram_fail" va testato in **INT8** (non fp32): lì l'allowzero
  non c'è e i modelli *arrivano* al compilatore (e solo allora si vede se è davvero RAM).

---

## Cosa è cambiato tra AI Core 4.0.0 (20500) e 4.0.1 (20581)

ST pubblica le note dettagliate a livello di **major (4.0.0)**; la **4.0.1 è una patch
di manutenzione** (build 20500→20581) senza changelog pubblico granulare. Voci rilevanti
documentate nella linea 4.0.x (utili per questa tesi):

**Nuovo supporto operatori (rilevante per i transformer):**
- `LayerNormalization (ONNX)` e `LayerNormalization (Keras)` — nuovo layer supportato.
- `Size (ONNX)`, `ADD_N (TFLite)`, miglior supporto tensori int64, constant propagation.
- `ReduceMean` migliorato.

**Neural-ART / NPU (rilevante per YOLO e softmax/attention):**
- Migliore riconoscimento **SWISH/SiLU** (auto con alpha==1.0) — usato da YOLO. Opzione
  `--SWISH-recognition` per alpha≠1.0.
- Riconoscimento **GELU** (tanh) espanso.
- **`--expand-softmax`**: espande il softmax in layer mappabili in hardware (utile per
  attention; può ridurre leggermente la precisione → validare).
- `--ec-optimize`: ottimizzazioni blob dell'epoch controller (sperimentale).
- `clone_dma` ora attivo di default (`--Ono-clone-dma` per disattivare).

**Fix di manutenzione (probabile contenuto 4.0.1):** allineamento memory pool, calcolo
limiti Stream Engine, configurazione indirizzi DMA, eligibilità broadcasting.

➡️ **Da provare nella nuova sessione:** ri-lanciare `baseline_transformer_block` e i
transformer con **`--expand-softmax`** attivo — la 4.0.x ha aggiunto LayerNorm+softmax
expansion, quindi è il singolo cambiamento che potrebbe sbloccare qualcosa rispetto alla
presentazione. Annotare se cambia l'esito.

Fonti: https://stedgeai-dc.st.com/assets/embedded-docs/release_note.html ·
https://www.st.com/en/development-tools/stedgeai-core.html

---

## Struttura

```
Test_v4.0.1/
  reports/            ← compila qui i .txt (uno per fase) + RIEPILOGO
  to_windows/         ← COPIA QUESTA su Windows e dai in pasto ad AI Studio
    fase0_golden_path/        (3)  validati INT8 — devono passare (controllo regressione 4.0.1)
    fase1_ram_fail/int8/      (10) CNN che fallivano per RAM — INT8
    fase1_ram_fail/int4/      (10) stessi, INT4 (test ipotesi prof)
    fase1_ram_fail/tflite_native/ (6) YOLO TFLite INT8 nativo @640 PULITO (ultralytics)
                                     — yolov7 escluso (non ultralytics)
    fase2_transformer/
        onnx_int8/            (6)  transformer ONNX INT8
        tflite_native/        ( )  TFLite PULITO — da generare su Colab/Linux:
                                   Models/scripts/export_transformer_tflite_COLAB.py
        tflite_onnx2tf/       (4)  TFLite vecchi (conversione ONNX, già falliti)
    fase3_baseline/
        conv/                 (8)  micro-benchmark conv
        transformer/          (8)  micro-benchmark operatori transformer
```

## Protocollo per ogni modello (AI Studio)
1. Generate / Analyze su board NUCLEO-N657X0-Q (o DK), target stm32n6.
2. Registra: **esito** (OK / RAM fail / Compiler fail), **RAM attivazioni (MB)**,
   **epoch** (HW/SW/tot), **latenza** se compila, e il **messaggio di errore esatto**.
3. Per i transformer prova anche con flag **`--expand-softmax`**.
4. Copia/incolla l'errore completo nel .txt della fase corrispondente.
