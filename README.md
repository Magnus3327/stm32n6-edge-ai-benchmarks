# NPU Performance Benchmark & Analysis — STM32N6

This repository is part of a curricular internship project at the **University of Trento**, carried out within the **DISI** (Department of Information Engineering and Computer Science) under the supervision of professors **Vella** and **Yildirim**.

---

## Project Overview

The project focuses on the optimization and performance evaluation of deep learning operators (single-layer microbenchmarks) deployed on the **Neural-ART™ NPU** of the **STM32N657** microcontroller (Cortex-M55 + 600 GOPS INT8 NPU).

The study compares:
- **Actual hardware measurements (Baseline)** vs. **static compiler predictions (Estimated)**
- Two compiler generations from STMicroelectronics: **AI Core v2.2** (via STM32CubeMX / X-CUBE-AI) and **AI Core v4.0** (via STM32 AI Studio)
- Two execution platforms: the local **STM32N6 Nucleo** board and the **STM32N6 Discovery Kit (DK)** board accessed via **ST Developer Cloud**

A second phase of the project extends the analysis to **real-world Computer Vision models** (YOLO, ResNet, MobileCLIP, PIDNet), evaluating their deployability on the NPU and identifying the architectural constraints that prevent deployment.

---

## Repository Structure

```
Modelli/
├── CV_Models.csv                     # Specification of CV models evaluated
├── model_inventory.md                # Inventory of exported ONNX models (13 files)
├── quantized_inventory.md            # Inventory of INT8 quantized ONNX models
├── npu_hardware_limits.md            # NPU hardware limits reference (SRAM layout, etc.)
├── generate_nb.py                    # Script to programmatically generate the Jupyter Notebook
├── requirements_export.txt           # Python dependencies for model export
├── README.md
├── Models/
│   ├── exported/                     # ONNX FP32 models (13 files, ~1.49 GB total)
│   │   ├── resnet34.onnx             (83.3 MB)
│   │   ├── yolov7.onnx               (144.1 MB)
│   │   ├── yolov8n.onnx              (12.3 MB)
│   │   ├── yolov11{n,s,m,l,x}.onnx  (10–218 MB)
│   │   ├── pidnet_s.onnx             (30.9 MB)
│   │   ├── mobileclip_s0_{image,text}.onnx
│   │   └── mobileclip_b_{image,text}.onnx
│   ├── quantized_int8/               # INT8 QDQ quantized models (70–75% smaller)
│   ├── baseline/                     # Baseline TFLite single-layer microbenchmark models
│   ├── .cache/                       # Local model weights and cloned repos (cache)
│   ├── download_and_export_models.py # Script to download and export ONNX models
│   ├── generate_final_models.py      # Generates the final model set for benchmarking
│   ├── quantize_int8.py              # Static INT8 quantization (QDQ format)
│   ├── downgrade_ir.py               # Patches ONNX IR version 10 → 9 for CubeMX v2.2
│   ├── fix_onnx_reshape.py           # Removes 'allowzero' from Reshape nodes
│   ├── fix_slide_nums.py             # Utility: fixes slide numbering in the HTML report
│   ├── reorder_slides.py             # Utility: reorders slides in the HTML report
│   └── export_resnet_fixed.py        # Exports ResNet34 with fixed batch size & no Shape op
├── Output/                           # Raw compiler logs, compilation reports, DevCloud outputs
│   └── Output_CV_Models/             # Error logs from CV model deployment attempts
├── Results/
│   ├── baseline/                     # Measured latencies, RAM/Flash footprint on hardware
│   └── estimated/                    # Static RAM and Flash estimates from the compilers
└── Report/
    ├── NPU_Performance_Analysis.ipynb  # Jupyter Notebook with comparison tables and plots
    ├── NPU_Performance_Analysis.html   # Exported HTML report
    ├── NPU_Performance_Analysis.pdf    # Exported PDF report (Chrome headless)
    ├── presentation.html               # 19-slide internship presentation (HTML/CSS)
    ├── NPU_Presentation.pdf            # Exported PDF presentation
    └── plot/                           # Exported chart images (latency, RAM, MACC, distributions)
```

---

## Microbenchmark Setup

Eight single-layer operators were benchmarked (operator-level microbenchmarks), all with a `64×64×3` INT8 input:

| # | Layer | MACC (M) |
|---|-------|-----------|
| 1 | Depthwise Conv2D 3×3 | 0.34 |
| 2 | Conv2D 16f 3×3 | 1.80 |
| 3 | Conv2D 16f 3×3 + MaxPool | 1.80 |
| 4 | Conv2D 32f 1×1 (pointwise) | 0.46 |
| 5 | Conv2D 32f 3×3 | 3.60 |
| 6 | Conv2D 32f 5×5 | 9.96 |
| 7 | Conv2D 64f 3×3 | 7.21 |
| 8 | Conv2D 32f 7×7 | 19.66 |

> **Note:** The 7×7 Conv2D fails to compile on AI Studio v4.0 and DevCloud v4.0 (OOM during compilation). Only CubeMX v2.2 handles it successfully.

---

## Key Findings

1. **Compiler version matters more than the board.** CubeMX v2.2 outperforms AI Studio v4.0 on the Nucleo board for most layers. The newer v4.0 "Epoch Controller" produces suboptimal tiling for some configurations.

2. **The physical wall is ~1.5 MB of SRAM.** Activation tensors exceeding this limit trigger multi-epoch tiling (performance loss) or a hard compilation failure.

3. **Standard Conv2D 3×3 is the NPU sweet spot.** Single-epoch execution, predictable throughput, accurate static estimates, and linear MACC scaling. The Neural-ART architecture was built for this pattern.

4. **Memory-bound layers are problematic.** Pointwise (1×1) and Depthwise (3×3) convolutions have a RAM/MACC ratio 100–1000× higher than standard convolutions. Compiler static RAM estimates are off by 100–900% for these layers.

5. **Flash footprint is compiler-independent.** The same INT8 model produces identical binary weights across all platforms. Flash is a property of the model, not the compiler.

6. **No state-of-the-art CV model deploys out-of-the-box.** All 10 CV models in `CV_Models.csv` fail due to RAM overflow, 7×7 kernel im2col explosion, or unsupported Transformer operators.

---

## CV Model Deployment Audit

All models were exported to ONNX, patched, quantized to INT8, and submitted to the ST Edge AI compilers. Results:

| Category | Models | Failure Reason |
|----------|--------|----------------|
| **RAM FAIL** | YOLOv7, YOLOv8-n, YOLOv11-n/s/m/l/x, PIDNet-S, RTMDet-l | Max activation > npuRAM budget at native resolution |
| **OOM FAIL** | ResNet34 | 7×7 kernel im2col buffer ≈ 1.84 MB exceeds SRAM |
| **Compiler FAIL** | MobileCLIP-S0/B (image & text) | Dynamic shapes, unsupported Transformer/LayerNorm operators |

### ONNX Patches Applied

Before compilation, the following patches were applied to the ONNX graphs:

| Script | Purpose |
|--------|---------|
| `Models/downgrade_ir.py` | Downgrades IR version 10 → 9 (required by CubeMX v2.2) |
| `Models/fix_onnx_reshape.py` | Removes `allowzero` attribute from all Reshape nodes |
| `Models/export_resnet_fixed.py` | Fixes batch size to 1 and removes dynamic Shape operators in ResNet34 |

---

## Golden Path — Successful Real-World Deployment

**MobileNetV2 @ 224×224** was successfully compiled and measured on the **STM32N6570-DK** board using **CubeMX v2.2**:

| Metric | Value |
|--------|-------|
| Measured latency | **20.694 ms** (avg over 10 runs, range 20.689–20.708 ms) |
| Throughput | **48.32 inferences/second** |
| Hardware epochs | 58 total (55 NPU + 3 CPU) |
| NPU offload | **94.8%** (55/58 layers) |
| RAM (activations) | **1.19 MiB** (validate mode) |
| Flash (weights) | **3.39 MiB** on OctoFlash |
| npuRAM4/5 utilization | **98.44%** (441/448 KB each) |
| Quantization quality | cosine similarity = 0.9976, NSE = 0.9950 |

**YOLOv8n / YOLOv11n @ 224×224** also compile successfully at reduced resolution (max activation drops from ~1.6 MB to ~200 KB).

### NPU Deployment Rules of Thumb

| ✅ Do | ❌ Avoid |
|-------|---------|
| Pure CNN with 3×3 or 5×5 kernels | Kernels ≥ 7×7 (im2col RAM explosion) |
| Input resolution s.t. max activation < 448 KB/npuRAM bank | Transformer, Self-Attention, LayerNorm (not mapped to NPU) |
| Weights on OctoFlash (up to 112 MB) | Dynamic shapes in ONNX graphs |
| Depthwise Separable Convolutions | ONNX IR version > 9 with CubeMX v2.2 |

---

## Getting Started

### Prerequisites

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install nbformat nbconvert pandas seaborn matplotlib ipykernel jinja2 ipython
```

Or install from the requirements file:

```bash
pip install -r requirements_export.txt
```

### Regenerate the Analysis Notebook

```bash
# 1. Generate the notebook structure
python3 generate_nb.py

# 2. Execute all cells in-place
jupyter nbconvert --to notebook --execute --inplace Report/NPU_Performance_Analysis.ipynb

# 3. Export to HTML
jupyter nbconvert --to html Report/NPU_Performance_Analysis.ipynb

# 4. Export to PDF (requires Google Chrome)
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --no-sandbox \
  --print-to-pdf="Report/NPU_Performance_Analysis.pdf" \
  "Report/NPU_Performance_Analysis.html"
```

### Export the Presentation to PDF

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --disable-gpu --no-sandbox \
  --print-to-pdf="Report/NPU_Presentation.pdf" \
  --print-to-pdf-no-header --no-pdf-header-footer \
  "file:///$(pwd)/Report/presentation.html"
```

### Apply ONNX Patches to New Models

```bash
source .venv/bin/activate
python Models/fix_onnx_reshape.py    # Remove 'allowzero' from Reshape nodes
python Models/downgrade_ir.py        # Downgrade IR version 10 → 9
```

---

## Quantized Models (INT8)

All exported FP32 ONNX models were statically quantized to INT8 (QDQ format) using `onnxruntime.quantization`. File size reduction is approximately **70–75%** across all models.

> **Calibration note:** Quantization was performed with random (dummy) data. This is sufficient for hardware profiling (latency/memory), but for accuracy evaluation, calibration with real dataset images (ImageNet, COCO) is recommended.

| Model | FP32 Size | INT8 Size | Reduction |
|-------|-----------|-----------|-----------|
| ResNet34 | 83.3 MB | 21.0 MB | ~74.8% |
| YOLOv7 | 144.1 MB | 36.8 MB | ~74.4% |
| YOLOv8-n | 12.3 MB | 3.4 MB | ~72.3% |
| YOLOv11-n | 10.2 MB | 3.0 MB | ~70.6% |
| YOLOv11-s | 36.3 MB | 9.5 MB | ~73.8% |
| YOLOv11-m | 76.9 MB | 19.8 MB | ~74.2% |
| YOLOv11-l | 97.0 MB | 25.1 MB | ~74.1% |
| YOLOv11-x | 217.5 MB | 55.3 MB | ~74.6% |
| PIDNet-S | 30.9 MB | 7.6 MB | ~75.4% |
| MobileCLIP-S0 (image) | 44.1 MB | 11.4 MB | ~74.1% |
| MobileCLIP-S0 (text) | 162.1 MB | 42.5 MB | ~73.8% |
| MobileCLIP-B (image) | 330.1 MB | 86.6 MB | ~73.8% |
| MobileCLIP-B (text) | 242.9 MB | 63.5 MB | ~73.8% |

---

**Author:** Matteo Miglio — Student ID / Matricola: 243947  
**University of Trento** — DISI  
**Supervisors:** Prof. Vella, Prof. Yildirim
