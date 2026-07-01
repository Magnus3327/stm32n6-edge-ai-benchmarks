# Sorgenti dei modelli — CV_Models.csv

Per ogni modello: da dove provengono i pesi, come sono stati scaricati, e come è stato generato l'ONNX/TFLite usato nei test.

---

## YOLOv7 — Object Detection (COCO 2017) — 36.9M params

**Pesi:** file `yolov7.pt`, scaricato da GitHub Releases ufficiale.
- URL: `https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt`
- Richiede anche il codice del repo per caricare il checkpoint (pickle con riferimenti ai moduli):
  `git clone --depth 1 https://github.com/WongKinYiu/yolov7.git`

**Export ONNX:**
`torch.onnx.export(model, torch.randn(1,3,640,640), opset_version=13, input_names=["images"])`
Script: `Models/scripts/export_models.py` → `export_yolov7()`

**Note:** Non è un modello ultralytics → nessun export TFLite nativo disponibile. Escluso dalla fase TFLite.

---

## ResNet34 — Image Classification (ImageNet1K) — 21.8M params

**Pesi:** scaricati automaticamente da `torchvision`.
- Fonte: `torchvision.models.ResNet34_Weights.IMAGENET1K_V1`
- Download automatico via PyTorch Hub alla prima esecuzione.

**Export ONNX:**
`torch.onnx.export(model, torch.randn(1,3,224,224), opset_version=13)`
Script: `Models/scripts/export_models.py` → `export_resnet34()`

**Note:** L'export FP32 aveva `allowzero≠0` su un nodo Reshape (bug dell'exporter). Scompare in INT8.

---

## PIDNet-S — Semantic Segmentation (Cityscapes) — 7.6M params

**Pesi:** ONNX precompilato scaricato da HuggingFace Hub.
- Repo HuggingFace: `oenpu/PIDNet_S_enlight_friendly_onnx`
- File: `PIDNet_S_enlight_friendly.onnx`
- Comando: `hf_hub_download(repo_id="oenpu/PIDNet_S_enlight_friendly_onnx", filename="PIDNet_S_enlight_friendly.onnx")`

**Alternativa (pesi PyTorch originali):**
- Repo ufficiale: `https://github.com/XuJiacong/PIDNet`
- Pesi: `PIDNet_S_Cityscapes_test.pt` (Google Drive, link nel README del repo)

Script: `Models/scripts/export_models.py` → `export_pidnet_s()`

---

## RTMDet-l — Object Detection (COCO 2017) — 52.3M params

**Pesi:** checkpoint scaricato da OpenMMLab (mmdetection).
- URL: `https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth`
- Framework: MMDetection (`mim install mmengine mmcv mmdet`)

**Export ONNX:** tramite `mmdet` + `init_detector()`, export diretto con `torch.onnx.export`.
Script: `Models/scripts/export_models.py` → `export_rtmdet_l()`

**Note:** L'export include NMS nel grafo per default → ST Edge AI non supporta NMS → necessario esportare senza post-processing.

---

## YOLOv8-n — Object Detection (COCO 2017) — 3.2M params

**Pesi:** scaricati automaticamente da `ultralytics`.
- Comando: `YOLO("yolov8n.pt")` → download automatico da GitHub Releases di ultralytics.

**Export ONNX:** `YOLO.export(format="onnx", imgsz=640, opset=13, nms=False, simplify=True)`
**Export TFLite INT8:** pipeline ONNX → onnx2tf (script `scratchpad/yolo_stage2b.py`)
Script ONNX: `Models/scripts/export_models.py` → `export_yolov8n()`

---

## FastInst-D1 — Instance Segmentation (COCO 2017) — 30M params

**Stato:** Non esportato. Backbone ResNet50-DCN usa Deformable Convolutions (DCN) non supportate in ONNX standard.
- Repo: `https://github.com/junjiehe96/FastInst`
- Framework: Detectron2 / Mask2Former

**Note per la tesi:** Questo modello non è nei test ST AI Studio perché l'export ONNX fallisce per le DCN. Documentato come limitazione dell'export, non del modello. FastInst-D1 non compare in nessuna cartella `to_windows/`.

---

## UniFormer (small, base) — Image Classification (ImageNet1K)

**Pesi:** scaricati dal repo HuggingFace `Sense-X/uniformer_image` nelle sessioni precedenti.
- Cache locale: `Models/.cache/uniformer/uniformer_small_in1k.pth`, `uniformer_base_in1k.pth`
- Codice modello: `Models/.cache/uniformer/uniformer.py` (dal repo ufficiale `Sense-X/UniFormer`)

**Parametri effettivi (dal codice):**
- `uniformer_small`: embed_dim=[64,128,320,512], depth=[3,4,8,3] → **21.5M params** (CSV indicava 2.4M — le varianti CSV sono configurazioni custom più piccole)
- `uniformer_base`: embed_dim=[64,128,320,512], depth=[5,8,20,7] → **49.8M params** (CSV indicava 5.6M)

**Export ONNX:** `torch.onnx.export(model, torch.randn(1,3,224,224), opset_version=13)`
Script: `Models/scripts/export_models.py` → `_export_uniformer()`

**Export TFLite INT8 (nativo):** `litert_torch.convert()` con PT2E quantization statica.
Eseguito su macOS arm64 con conda Python 3.12 + litert-torch 0.9.1.
Script: `Models/scripts/export_transformer_tflite_MACOS.py`
Output: `tflite_native/uniformer_small_int8.tflite` (22 MB), `uniformer_base_int8.tflite` (51 MB)

---

## YOLOv11-n/s/m/l/x — Object Detection (COCO 2017) — 2.6M–56.9M params

**Pesi:** scaricati automaticamente da `ultralytics`.
- Comandi: `YOLO("yolo11n.pt")`, `YOLO("yolo11s.pt")`, ecc. → download automatico.
- Cache locale: `Models/.cache/yolo11n.pt`, ecc.

**Export ONNX:** `YOLO.export(format="onnx", imgsz=640, opset=13, nms=False, simplify=True)`
**Export TFLite INT8:** pipeline 2-stage:
1. `scratchpad/yolo_stage1_onnx.py`: ultralytics → ONNX pulito @640
2. `scratchpad/yolo_stage2b.py`: onnx2tf v2.5 con `output_integer_quantized_tflite=True` → TFLite INT8

Script ONNX: `Models/scripts/export_models.py` → `_export_yolov11()`

**Output TFLite:**
- `yolov11n_int8.tflite` (3.20 MB), `yolov11s_int8.tflite` (10.16 MB)
- `yolov11m_int8.tflite` (20.98 MB), `yolov11l_int8.tflite` (26.44 MB)
- `yolov11x_int8.tflite` (58.32 MB)

---

## MobileCLIP-S0 e MobileCLIP-B — Image-Text Encoding (DataCompDR) — 11.4M+42.4M / 86.3M+63.4M params

**Pesi:** scaricati da HuggingFace Hub nelle sessioni precedenti.
- Cache locale: `/Users/matteo/.cache/mobileclip/mobileclip_s0.pt`, `mobileclip_b.pt`
- Repo originale Apple: `https://github.com/apple/ml-mobileclip`
- Package Python: `pip install git+https://github.com/apple/ml-mobileclip.git`

**Export ONNX:** esporta image encoder e text encoder separatamente.
- Image encoder: input `(1,3,256,256)`
- Text encoder: input `(1,77)` token ids
Script: `Models/scripts/export_models.py` → `_export_mobileclip()`

**Export TFLite INT8 (nativo):** `litert_torch.convert()` con PT2E quantization statica.
Eseguito su macOS arm64 con conda Python 3.12 + litert-torch 0.9.1.
Script: `Models/scripts/export_transformer_tflite_MACOS.py`
Output: `tflite_native/mobileclip_s0_image_int8.tflite` (13.85 MB), `mobileclip_b_image_int8.tflite` (90.48 MB)

**Note:** Solo l'image encoder è incluso nei test TFLite (il text encoder ha dimensione token variabile, difficile da quantizzare staticamente senza dati reali).

---

## Riepilogo file generati per i test

| Fase | Modello | File | Formato | Come generato |
|------|---------|------|---------|---------------|
| fase1/int4 | tutti RAM-fail | *_int4.onnx | ONNX | onnxruntime weight-only INT4 |
| fase1/tflite | YOLO v8/11 | *_int8.tflite | TFLite | ultralytics→ONNX→onnx2tf |
| fase2/onnx | mobileclip, uniformer | *_int8.onnx | ONNX | torch.onnx.export + quantization |
| fase2/tflite | mobileclip, uniformer | *_int8.tflite | TFLite | litert-torch PT2E (nativo macOS) |
| fase3 | transformer block | *_int8.onnx | ONNX | modello sintetico, script dedicato |
