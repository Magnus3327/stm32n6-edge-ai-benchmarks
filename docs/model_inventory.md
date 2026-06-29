# 📦 Inventario Modelli CV — Baseline Performance

> Aggiornato: 2026-06-16  
> Output directory: `Models/exported/` — tutti i file sono **self-contained** (nessun `.data` esterno)

---

## ✅ Modelli Esportati (13 file ONNX)

| # | Modello | Task | Params | Dataset | Input Shape | Dim. File | Percorso |
|---|---------|------|--------|---------|-------------|-----------|---------|
| 1 | **ResNet34** | Image Classification | 21.8M | ImageNet1K | `1×3×224×224` | 83.3 MB | `Models/exported/resnet34.onnx` |
| 2 | **YOLOv8-n** | Object Detection | 3.2M | COCO 2017 | `1×3×640×640` | 12.3 MB | `Models/exported/yolov8n.onnx` |
| 3 | **YOLOv11-n** | Object Detection | 2.6M | COCO 2017 | `1×3×640×640` | 10.2 MB | `Models/exported/yolov11n.onnx` |
| 4 | **YOLOv11-s** | Object Detection | 9.4M | COCO 2017 | `1×3×640×640` | 36.3 MB | `Models/exported/yolov11s.onnx` |
| 5 | **YOLOv11-m** | Object Detection | 20.1M | COCO 2017 | `1×3×640×640` | 76.9 MB | `Models/exported/yolov11m.onnx` |
| 6 | **YOLOv11-l** | Object Detection | 25.3M | COCO 2017 | `1×3×640×640` | 97.0 MB | `Models/exported/yolov11l.onnx` |
| 7 | **YOLOv11-x** | Object Detection | 56.9M | COCO 2017 | `1×3×640×640` | 217.5 MB | `Models/exported/yolov11x.onnx` |
| 8 | **YOLOv7** | Object Detection | 36.9M | COCO 2017 | `1×3×640×640` | 144.1 MB | `Models/exported/yolov7.onnx` |
| 9 | **PIDNet-S** | Semantic Segmentation | 7.6M | Cityscapes | `1×3×1024×1024` | 30.9 MB | `Models/exported/pidnet_s.onnx` |
| 10 | **MobileCLIP-S0** (image enc.) | Image-Text Encoding | 11.4M | DataCompDR | `1×3×256×256` | 44.1 MB | `Models/exported/mobileclip_s0_image.onnx` |
| 11 | **MobileCLIP-S0** (text enc.) | Image-Text Encoding | 42.4M | DataCompDR | `1×77` (token ids) | 162.1 MB | `Models/exported/mobileclip_s0_text.onnx` |
| 12 | **MobileCLIP-B** (image enc.) | Image-Text Encoding | 86.3M | DataCompDR | `1×3×256×256` | 330.1 MB | `Models/exported/mobileclip_b_image.onnx` |
| 13 | **MobileCLIP-B** (text enc.) | Image-Text Encoding | 63.4M | DataCompDR | `1×77` (token ids) | 242.9 MB | `Models/exported/mobileclip_b_text.onnx` |

**Totale occupato: ~1.49 GB**

> [!NOTE]
> MobileCLIP è esportato in due file separati per modello (image encoder + text encoder), in quanto hanno input/output diversi e vengono usati indipendentemente a runtime.
>
> PIDNet-S è stato ottenuto come ONNX precompilato da HuggingFace (`oenpu/PIDNet_S_enlight_friendly_onnx`) poiché il link Google Drive originale del repository è offline.

---

## ❌ Modelli Non Esportati (Tier 3 — Setup Complesso)

| Modello | Task | Params | Motivo |
|---------|------|--------|--------|
| **RTMDet-l** | Object Detection | 52.3M | Richiede `mmdet`, `mmengine`, `mmcv` (ecosistema OpenMMLab). L'export ONNX diretto è inaffidabile — raccomandato `mmdeploy`. |
| **FastInst-D1** | Instance Segmentation | 30M | Usa **Deformable Convolutions (DCN)** non supportate in ONNX standard. Richiede TensorRT o Detectron2 tracing. |
| **UniFormer-T** | Image Classification | 1.8M | Variante custom non disponibile pubblicamente. Le versioni standard in `timm` hanno parametri diversi (22M+). |
| **UniFormer-S** | Image Classification | 2.4M | Come sopra. |
| **UniFormer-M** | Image Classification | 5.6M | Come sopra. |
| **UniFormer-L** | Image Classification | 10M | Come sopra. |

### Come esportare i modelli Tier 3

**RTMDet-l:**
```bash
pip install -U openmim
mim install mmengine mmcv mmdet
python download_and_export_models.py --models rtmdet_l
# Oppure con mmdeploy per export affidabile:
pip install mmdeploy mmdeploy-runtime-onnxruntime
```

**FastInst-D1:**
```bash
# Non esportabile in ONNX standard (DCN). Usare TensorRT:
pip install torch2trt
# oppure Detectron2 tracing
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

**UniFormer (t/s/m/l):**
Le varianti del CSV (1.8M–10M) sono configurazioni custom leggere non pubblicate.  
I modelli standard UniFormer-S (22M) e UniFormer-B (50M) sono disponibili in `timm`.

---

## 🧹 Pulizia Eseguita

| Azione | File Coinvolti |
|--------|----------------|
| Rimossi duplicati dalla root | `yolo11{n,s,m,l,x}.onnx`, `yolo11{n,s,m,l,x}.pt`, `yolov8n.onnx`, `yolov8n.pt` (12 file, ~560 MB liberati) |
| Rimossi `.DS_Store` | 4 file rimossi |
| Consolidati file external data | `resnet34`, `yolov7`, `mobileclip_s0_image`, `mobileclip_s0_text`, `mobileclip_b_image`, `mobileclip_b_text` — ora tutti self-contained |

---

## 📁 Struttura Cartelle Finale

```
Modelli/
├── CV_Models.csv                    # Specifica modelli
├── download_and_export_models.py    # Script export (aggiornato)
├── generate_nb.py                   # Generatore notebook
├── requirements_export.txt          # Dipendenze
├── README.md
├── Models/
│   ├── exported/                    # ← tutti i modelli ONNX (13 file)
│   │   ├── resnet34.onnx            (83.3 MB)
│   │   ├── yolov7.onnx              (144.1 MB)
│   │   ├── yolov8n.onnx             (12.3 MB)
│   │   ├── yolov11n.onnx            (10.2 MB)
│   │   ├── yolov11s.onnx            (36.3 MB)
│   │   ├── yolov11m.onnx            (76.9 MB)
│   │   ├── yolov11l.onnx            (97.0 MB)
│   │   ├── yolov11x.onnx            (217.5 MB)
│   │   ├── pidnet_s.onnx            (30.9 MB)
│   │   ├── mobileclip_s0_image.onnx (44.1 MB)
│   │   ├── mobileclip_s0_text.onnx  (162.1 MB)
│   │   ├── mobileclip_b_image.onnx  (330.1 MB)
│   │   └── mobileclip_b_text.onnx   (242.9 MB)
│   ├── .cache/                      # Pesi e repo clonati (cache locale)
│   │   ├── yolov7.pt
│   │   ├── yolov7/                  # repo git
│   │   └── PIDNet/                  # repo git
│   └── baseline_conv2d_*.tflite     # Modelli baseline TFLite
├── Output/
├── Results/
└── Report/
```
