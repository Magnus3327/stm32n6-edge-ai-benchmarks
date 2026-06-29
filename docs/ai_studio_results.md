# Risultati Test ST AI Studio — STM32N657 Neural-ART™ NPU

**Data test:** ___________  
**Versione ST AI Studio:** ___________  
**Target:** STM32N657 · 600 GOPS INT8 · ~1.5 MB npuRAM attivazioni

---

## Fase 1 — Baseline operatori Transformer

> Obiettivo: trovare il primo operatore che causa COMPILER FAIL

| # | File | Esito | Errore / Note |
|---|------|:-----:|---------------|
| 1 | `3_baseline/transformer/baseline_dense_int8.onnx` | | |
| 2 | `3_baseline/transformer/baseline_layernorm_opset13_int8.onnx` | | |
| 3 | `3_baseline/transformer/baseline_layernorm_fused_int8.onnx` | | |
| 4 | `3_baseline/transformer/baseline_softmax_int8.onnx` | | |
| 5 | `3_baseline/transformer/baseline_mha_int8.onnx` | | |
| 6 | `3_baseline/transformer/baseline_transformer_block_int8.onnx` | | |
| 7 | `3_baseline/transformer/baseline_conv_mha_hybrid_int8.onnx` | | |

**Primo operatore che fallisce:** ___________

**Messaggio di errore esatto:**
```
(incolla qui)
```

---

## Fase 2 — Modelli validati (devono funzionare)

| # | File | Esito | RAM att. (MB) | Note |
|---|------|:-----:|:-------------:|------|
| 8 | `0_validated/fp32/mobilenet_v2_224.onnx` | | | |
| 9 | `0_validated/int8/mobilenet_v2_224_int8.onnx` | | | |
| 10 | `0_validated/fp32/yolov8n_micro_224.onnx` | | | |
| 11 | `0_validated/int8/yolov8n_micro_224_int8.onnx` | | | |
| 12 | `0_validated/fp32/yolov11n_micro_224.onnx` | | | |
| 13 | `0_validated/int8/yolov11n_micro_224_int8.onnx` | | | |

---

## Fase 3 — RAM fail: INT4 risolve?

> Obiettivo: verificare se INT4 riduce abbastanza le attivazioni per entrare nei ~1.5 MB npuRAM

### YOLOv8-n (input 640×640)

| Formato | File | Esito | RAM att. (MB) | Errore |
|---------|------|:-----:|:-------------:|--------|
| FP32 | `1_ram_fail/fp32/yolov8n_640.onnx` | | | |
| INT8 | `1_ram_fail/int8/yolov8n_int8.onnx` | | | |
| INT4 | `1_ram_fail/int4/yolov8n_int4.onnx` | | | |

### RTMDet-l (input 640×640)

| Formato | File | Esito | RAM att. (MB) | Errore |
|---------|------|:-----:|:-------------:|--------|
| FP32 | `1_ram_fail/fp32/rtmdet_l.onnx` | | | |
| INT8 | `1_ram_fail/int8/rtmdet_l_int8.onnx` | | | |
| INT4 | `1_ram_fail/int4/rtmdet_l_int4.onnx` | | | |

### YOLOv7 (input 640×640)

| Formato | File | Esito | RAM att. (MB) | Errore |
|---------|------|:-----:|:-------------:|--------|
| INT8 | `1_ram_fail/int8/yolov7_int8.onnx` | | | |
| INT4 | `1_ram_fail/int4/yolov7_int4.onnx` | | | |

### PIDNet-S (input 2048×1024)

| Formato | File | Esito | RAM att. (MB) | Errore |
|---------|------|:-----:|:-------------:|--------|
| INT8 | `1_ram_fail/int8/pidnet_s_int8.onnx` | | | |
| INT4 | `1_ram_fail/int4/pidnet_s_int4.onnx` | | | |

### ResNet34 (im2col fail)

| Formato | File | Esito | Errore |
|---------|------|:-----:|--------|
| FP32 | `1_ram_fail/fp32/resnet34_fixed.onnx` | | |
| INT8 | `1_ram_fail/int8/resnet34_fixed_int8.onnx` | | |

**Note RAM fail:**
```
(note generali su quali modelli entrano/non entrano nella RAM)
```

---

## Fase 4 — Transformer fail: ONNX vs TFLite

> Obiettivo: verificare se TFLite bypassa il COMPILER FAIL o dà lo stesso errore

### MobileCLIP-S0 (image encoder, 256×256)

| Formato | File | Esito | Errore |
|---------|------|:-----:|--------|
| ONNX INT8 | `2_transformer_fail/int8/mobileclip_s0_image_int8.onnx` | | |
| TFLite INT8 | `2_transformer_fail/tflite/mobileclip_s0_image_int8.tflite` | | |

### MobileCLIP-B (image encoder, 256×256)

| Formato | File | Esito | Errore |
|---------|------|:-----:|--------|
| ONNX INT8 | `2_transformer_fail/int8/mobileclip_b_image_int8.onnx` | | |
| TFLite INT8 | `2_transformer_fail/tflite/mobileclip_b_image_int8.tflite` | | |

### UniFormer-Small (224×224)

| Formato | File | Esito | Errore |
|---------|------|:-----:|--------|
| ONNX INT8 | `2_transformer_fail/int8/uniformer_small_int8.onnx` | | |
| TFLite INT8 | `2_transformer_fail/tflite/uniformer_small_int8.tflite` | | |

### UniFormer-Base (224×224)

| Formato | File | Esito | Errore |
|---------|------|:-----:|--------|
| ONNX INT8 | `2_transformer_fail/int8/uniformer_base_int8.onnx` | | |
| TFLite INT8 | `2_transformer_fail/tflite/uniformer_base_int8.tflite` | | |

**Errore Transformer (copia qui il messaggio esatto da AI Studio):**
```
(incolla qui — uguale per tutti i modelli transformer?)
```

**TFLite bypassa il problema?**  ☐ Sì  ☐ No  ☐ Errore diverso

---

## Riepilogo finale

| Categoria | N. modelli | ✅ OK | ❌ RAM fail | ❌ Compiler fail |
|-----------|:----------:|:-----:|:-----------:|:----------------:|
| 0_validated | 6 | | | |
| 1_ram_fail (INT8) | 5 | | | |
| 1_ram_fail (INT4) | 5 | | | |
| 2_transformer_fail (ONNX) | 4 | | | |
| 2_transformer_fail (TFLite) | 4 | | | |
| 3_baseline | 7 | | | |

**Conclusioni:**

```
(scrivi qui le osservazioni principali)
```
