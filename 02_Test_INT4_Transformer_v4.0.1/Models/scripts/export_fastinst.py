#!/usr/bin/env python3
"""
Export FastInst-D1 (R50, senza DCN) → ONNX + INT8.

MODELLO: FastInst-D1 con backbone ResNet-50 standard (NO Deformable Convolutions)
  Params  : 30M
  Input   : 576×576×3
  Task    : Instance Segmentation (COCO)
  Categoria: COMPILER FAIL (Transformer decoder queries)
  Fonte   : https://github.com/junjiehe96/FastInst

SETUP RICHIESTO (eseguire UNA VOLTA):
  # 1. Installa Detectron2 (richiede Python 3.8–3.11, torch >= 2.0)
  pip install 'git+https://github.com/facebookresearch/detectron2.git'

  # 2. Clona il repo FastInst
  git clone https://github.com/junjiehe96/FastInst.git /tmp/FastInst
  cd /tmp/FastInst && pip install -e .

  # 3. Scarica il checkpoint R50 (no DCN) — 116 MB
  wget -O Models/.cache/fastinst/fastinst_R50_576.pth \\
    https://github.com/junjiehe96/FastInst/releases/download/v0.1.0/fastinst_R50_ppm-fpn_x1_576_34.9.pth

  # 4. Esegui questo script
  python Models/scripts/export_fastinst.py

NOTA: Se il download da wget fallisce, usa il browser:
  https://github.com/junjiehe96/FastInst/releases/tag/v0.1.0
  → fastinst_R50_ppm-fpn_x1_576_34.9.pth
  → Salva in: Models/.cache/fastinst/fastinst_R50_576.pth
"""

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("export_fastinst")

CHECKPOINT = Path(__file__).parent.parent / ".cache" / "fastinst" / "fastinst_R50_576.pth"
FASTINST_REPO = Path("/tmp/FastInst")
CONFIG = "configs/coco/instance-segmentation/fastinst_R50_ppm-fpn_x1_576.yaml"

OUT_FP32 = Path(__file__).parent.parent / "2_transformer_fail" / "fp32"
OUT_INT8 = Path(__file__).parent.parent / "2_transformer_fail" / "int8"

INPUT_H, INPUT_W = 576, 576


def check_requirements():
    missing = []
    try:
        import detectron2
    except ImportError:
        missing.append("detectron2  →  pip install 'git+https://github.com/facebookresearch/detectron2.git'")

    if not FASTINST_REPO.exists():
        missing.append(f"FastInst repo  →  git clone https://github.com/junjiehe96/FastInst.git {FASTINST_REPO}")

    if not CHECKPOINT.exists():
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        missing.append(
            f"Checkpoint  →  scarica da https://github.com/junjiehe96/FastInst/releases/download/v0.1.0/"
            f"fastinst_R50_ppm-fpn_x1_576_34.9.pth\n"
            f"   Salva come: {CHECKPOINT}"
        )

    if missing:
        log.error("Requirements mancanti:")
        for m in missing:
            log.error(f"  ✗ {m}")
        sys.exit(1)


def build_model():
    sys.path.insert(0, str(FASTINST_REPO))
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    import fastinst  # noqa: F401 — registers models

    cfg = get_cfg()
    cfg.merge_from_file(str(FASTINST_REPO / CONFIG))
    cfg.MODEL.WEIGHTS = str(CHECKPOINT)
    cfg.MODEL.DEVICE = "cpu"
    cfg.INPUT.MIN_SIZE_TEST = INPUT_H
    cfg.INPUT.MAX_SIZE_TEST = INPUT_W
    cfg.freeze()

    predictor = DefaultPredictor(cfg)
    return predictor.model


def export_to_onnx(model) -> str:
    import torch
    import numpy as np

    OUT_FP32.mkdir(parents=True, exist_ok=True)
    out_path = str(OUT_FP32 / "fastinst_d1_r50.onnx")

    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W)

    log.info(f"Esportando ONNX → {out_path}")
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            out_path,
            opset_version=18,
            input_names=["image"],
            output_names=["output"],
            dynamo=False,
        )
    sz = os.path.getsize(out_path) / 1024 / 1024
    log.info(f"✅ FP32: {sz:.1f} MB → {out_path}")
    return out_path


def quantize_int8(fp32_path: str) -> None:
    import numpy as np
    import onnxruntime
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    class DR(CalibrationDataReader):
        def __init__(self, n=10):
            self.n = n; self.cur = 0
            s = onnxruntime.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
            self.iname = s.get_inputs()[0].name
        def get_next(self):
            if self.cur < self.n:
                self.cur += 1
                return {self.iname: np.random.uniform(0, 1, (1, 3, INPUT_H, INPUT_W)).astype(np.float32)}

    OUT_INT8.mkdir(parents=True, exist_ok=True)
    int8_path = str(OUT_INT8 / "fastinst_d1_r50_int8.onnx")
    log.info(f"Quantizzando INT8 → {int8_path}")
    quantize_static(
        fp32_path, int8_path, DR(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    sz = os.path.getsize(int8_path) / 1024 / 1024
    log.info(f"✅ INT8: {sz:.1f} MB → {int8_path}")


def main():
    check_requirements()

    fp32_out = OUT_FP32 / "fastinst_d1_r50.onnx"
    int8_out = OUT_INT8 / "fastinst_d1_r50_int8.onnx"

    if fp32_out.exists() and int8_out.exists():
        log.info("⏭️  FastInst già presente, skip.")
        return

    log.info("Costruendo modello FastInst-D1 R50...")
    model = build_model()

    fp32_path = export_to_onnx(model)
    quantize_int8(fp32_path)

    log.info("\n✅ Fatto!")
    log.info(f"  FP32 → {fp32_out}")
    log.info(f"  INT8  → {int8_out}")
    log.info(
        "\n  NOTA: Se l'export fallisce per operatori Transformer nel decoder,\n"
        "  FastInst cade nella categoria COMPILER FAIL (stessa di MobileCLIP).\n"
        "  Il fallimento stesso è il dato di interesse per la tesi."
    )


if __name__ == "__main__":
    main()
