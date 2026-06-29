# Modelli per ST AI Studio — STM32N657 Neural-ART™ NPU

NPU specs: 600 GOPS INT8 · ~1.5 MB npuRAM attivazioni · 4 bank da ~448 KB

---

## Struttura cartelle

```
Models/
├── scripts/                         ← script Python (export, quantize, patch)
│
├── 0_validated/                     ← funzionano sull'NPU
│   ├── fp32/
│   └── int8/
│
├── 1_ram_fail/                      ← overflow RAM attivazioni (input grande)
│   ├── fp32/
│   ├── int8/
│   └── int4/
│
├── 2_transformer_fail/              ← operatori Transformer non supportati
│   ├── fp32/
│   └── int8/
│
└── 3_baseline/                      ← microbenchmark singoli operatori
    ├── conv/                        ← .tflite (già testati)
    └── transformer/                 ← .onnx (da testare)
```

Note tecniche: vedi `../docs/` (UNIFORMER_NOTE.txt, FASTINST_NOTE.txt, npu_hardware_limits.md)

---

## 0_validated — Funzionano sull'NPU

| File | Input | Formato |
|------|-------|---------|
| `mobilenet_v2_224.onnx` | 224×224×3 | FP32 |
| `mobilenet_v2_224_int8.onnx` | 224×224×3 | INT8 |
| `yolov8n_micro_224.onnx` | 224×224×3 | FP32 (input ridotto da 640) |
| `yolov8n_micro_224_int8.onnx` | 224×224×3 | INT8 |
| `yolov11n_micro_224.onnx` | 224×224×3 | FP32 (input ridotto da 640) |
| `yolov11n_micro_224_int8.onnx` | 224×224×3 | INT8 |

---

## 1_ram_fail — Overflow RAM (attivazioni > 1.5 MB)

Tutti disponibili in FP32 + INT8 + INT4. Testare l'INT4 per vedere se risolve il RAM overflow.

| Modello | Params | Input | FP32 | INT8 | INT4 |
|---------|--------|-------|:----:|:----:|:----:|
| YOLOv7 | 36.9M | 640×640×3 | ✅ | ✅ | ✅ |
| YOLOv8-n | 3.2M | 640×640×3 | ✅ | ✅ | ✅ |
| YOLOv11-n | 2.6M | 640×640×3 | ✅ | ✅ | ✅ |
| YOLOv11-s | 9.4M | 640×640×3 | ✅ | ✅ | ✅ |
| YOLOv11-m | 20.1M | 640×640×3 | ✅ | ✅ | ✅ |
| YOLOv11-l | 25.3M | 640×640×3 | ✅ | ✅ | ✅ |
| YOLOv11-x | 56.9M | 640×640×3 | ✅ | ✅ | ✅ |
| PIDNet-S | 7.6M | 2048×1024×3 | ✅ | ✅ | ✅ |
| RTMDet-l | 52.3M | 640×640×3 | ✅ | ✅ | ✅ |
| ResNet34 * | 21.8M | 224×224×3 | ✅ | ✅ | ✅ |

\* ResNet34: il file si chiama `resnet34_fixed` (versione con patch Reshape). Fallisce per im2col 7×7, non per RAM.

**INT4**: `weight_type=QInt4, activation_type=QInt8` — pesi 50% più leggeri vs INT8.

---

## 2_transformer_fail — Operatori Transformer non supportati

| Modello | Params | Input | FP32 | INT8 | Note |
|---------|--------|-------|:----:|:----:|------|
| MobileCLIP-S0 image | 11.4M | 256×256×3 | ✅ | ✅ | |
| MobileCLIP-S0 text | 42.4M | seq 77 | ✅ | ✅ | |
| MobileCLIP-B image | 86.3M | 256×256×3 | ✅ | ✅ | |
| MobileCLIP-B text | 63.4M | seq 77 | ✅ | ✅ | |
| UniFormer-Small | 21.5M | 224×224×3 | ✅ | ✅ | variante pubblica più simile al CSV |
| UniFormer-Base | 49.8M | 224×224×3 | ✅ | ✅ | |
| UniFormer-XXS ⚙️ | 10.2M | 224×224×3 | — | — | `python scripts/export_uniformer.py --variants light` |
| UniFormer-XS ⚙️ | 16.5M | 224×224×3 | — | — | `python scripts/export_uniformer.py --variants light` |
| FastInst-D1 R50 ⚙️ | 30M | 576×576×3 | — | — | Richiede setup Detectron2, vedi `scripts/export_fastinst.py` |

⚙️ = da generare con script (vedi Setup più sotto)

---

## 3_baseline — Microbenchmark operatori singoli

### conv/ — già testati (TFLite, da AI Studio / CubeMX)
`baseline_conv2d_*.tflite`, `baseline_depthwise_3x3_int8.tflite`

### transformer/ — da testare in ST AI Studio (ONNX INT8)

| File | Operatore | Input shape |
|------|-----------|-------------|
| `baseline_dense_int8.onnx` | Linear (FC) 128→128 + ReLU | (1, 128) |
| `baseline_layernorm_fused_int8.onnx` | LayerNorm opset18 fused | (1, 64, 128) |
| `baseline_layernorm_opset13_int8.onnx` | LayerNorm decomposed | (1, 64, 128) |
| `baseline_layernorm_opset17_int8.onnx` | LayerNorm opset17 | (1, 64, 128) |
| `baseline_softmax_int8.onnx` | Softmax attention weights | (1, 4, 64, 64) |
| `baseline_mha_int8.onnx` | MultiHeadAttention (4 heads) | (1, 64, 128) |
| `baseline_transformer_block_int8.onnx` | Encoder block completo | (1, 64, 128) |
| `baseline_conv_mha_hybrid_int8.onnx` | Conv2D patchifier + MHA | (1, 3, 64, 64) |

**Scopo**: caricare uno alla volta in AI Studio per isolare QUALE operatore causa COMPILER FAIL.

---

## Setup — Scripts

```bash
cd "Baseline Performance/Modelli"
source .venv/bin/activate
```

| Script | Cosa fa | Comando |
|--------|---------|---------|
| `export_models.py` | Scarica YOLOv7/8/11, PIDNet, MobileCLIP | `python Models/scripts/export_models.py` |
| `quantize_int8.py` | INT8 quantization (fp32 → int8) | `python Models/scripts/quantize_int8.py` |
| `quantize_int4.py` | INT4 weight-only (fp32 → int4) | `python Models/scripts/quantize_int4.py` |
| `patch_onnx.py` | IR downgrade + fix Reshape | `python Models/scripts/patch_onnx.py --all` |
| `generate_transformer_baselines.py` | Genera baseline ONNX transformer | `python Models/scripts/generate_transformer_baselines.py` |
| `export_uniformer.py` | Export UniFormer Standard + Light | `python Models/scripts/export_uniformer.py` |
| `export_fastinst.py` | Export FastInst-D1 R50 | Richiede Detectron2 (vedi sotto) |

### Setup FastInst (manuale)

```bash
# 1. Installa Detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 2. Clona FastInst repo
git clone https://github.com/junjiehe96/FastInst.git /tmp/FastInst
cd /tmp/FastInst && pip install -e .

# 3. Scarica checkpoint R50 no-DCN (116 MB)
mkdir -p Models/.cache/fastinst
wget -O Models/.cache/fastinst/fastinst_R50_576.pth \
  https://github.com/junjiehe96/FastInst/releases/download/v0.1.0/fastinst_R50_ppm-fpn_x1_576_34.9.pth

# 4. Esporta
python Models/scripts/export_fastinst.py
```

### Setup UniFormer-Light (manuale — sandbox blocca .pth pickle)

```bash
python Models/scripts/export_uniformer.py --variants light
```

---

## Piano di test su ST AI Studio (Windows)

### Fase 1 — Operatore singolo che fallisce (baseline transformer)
1. `baseline_dense_int8.onnx` → deve passare
2. `baseline_layernorm_fused_int8.onnx` → fallisce? (LayerNorm opset18)
3. `baseline_softmax_int8.onnx` → fallisce? (Softmax)
4. `baseline_mha_int8.onnx` → fallisce? (Attention)
5. `baseline_transformer_block_int8.onnx` → fallisce? (blocco completo)

### Fase 2 — Modelli reali Transformer
6. `MobileCLIP-S0 image INT8` → quale errore esatto?
7. `UniFormer-Small INT8` → stesso errore?

### Fase 3 — RAM fail: INT4 risolve?
8. `yolov8n_int8.onnx` (640×640) → RAM overflow
9. `yolov8n_int4.onnx` (640×640) → INT4 risolve?
10. `rtmdet_l_int8.onnx` → RAM overflow
11. `rtmdet_l_int4.onnx` → INT4 risolve?
