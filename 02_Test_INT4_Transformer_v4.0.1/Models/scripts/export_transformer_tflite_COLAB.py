#!/usr/bin/env python3
"""
Export PULITO transformer -> INT8 TFLite via ai-edge-torch (PyTorch -> TFLite diretto).

PERCHE' QUESTO SCRIPT NON GIRA SU macOS:
  ai-edge-torch e' di fatto Linux-only e supporta Python 3.9-3.12. La macchina di
  sviluppo ha solo Python 3.13/3.14 su macOS -> impossibile. Va eseguito su:
    - Google Colab (consigliato, gratis, Linux + Python 3.11), oppure
    - una qualsiasi VM/box Linux con Python 3.11/3.12.

PERCHE' "pulito" e non come prima:
  Il vecchio export_tflite.py faceva ONNX -> onnx2tf -> TFLite, cioe' convertiva
  l'ONNX gia' problematico (LayerNorm decomposto, shape dinamiche). Qui andiamo
  PyTorch -> TFLite diretto, mantenendo LayerNorm/attention come operatori interi.

USO (in una cella Colab):
    !pip install -q ai-edge-torch ai-edge-litert timm
    !pip install -q git+https://github.com/apple/ml-mobileclip.git
    !python export_transformer_tflite_COLAB.py

Output: ./tflite_native/<modello>_int8.tflite  -> poi copiali in
        Models/Test_v4.0.1/to_windows/fase2_transformer/tflite_native/
"""
import os
from pathlib import Path
import numpy as np
import torch

OUT = Path("tflite_native"); OUT.mkdir(exist_ok=True)

import ai_edge_torch
from ai_edge_torch.quantize.pt2e_quantizer import PT2EQuantizer, get_symmetric_quantization_config
from ai_edge_torch.quantize.quant_config import QuantConfig
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e


def to_int8_tflite(module: torch.nn.Module, sample: tuple, name: str):
    """PyTorch nn.Module -> INT8 TFLite (PT2E static quant, calib su dati random)."""
    module = module.eval()
    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False)
    )
    exported = torch.export.export_for_training(module, sample).module()
    prepared = prepare_pt2e(exported, quantizer)
    # --- calibrazione (sostituisci con immagini reali per accuratezza) ---
    for _ in range(16):
        prepared(*[torch.rand_like(s) for s in sample])
    converted = convert_pt2e(prepared)
    edge = ai_edge_torch.convert(
        converted, sample, quant_config=QuantConfig(pt2e_quantizer=quantizer)
    )
    path = OUT / f"{name}_int8.tflite"
    edge.export(str(path))
    print(f"OK  {name:22s} -> {path}  ({path.stat().st_size/1e6:.2f} MB)")


# ------------------------------------------------------------------ MobileCLIP
def export_mobileclip(variant: str):
    """variant in {'s0','b'} -> esporta l'image encoder (256x256)."""
    import mobileclip
    model, _, _ = mobileclip.create_model_and_transforms(f"mobileclip_{variant}")
    enc = getattr(model, "image_encoder", None) or getattr(model, "visual")
    sample = (torch.randn(1, 3, 256, 256),)
    to_int8_tflite(enc, sample, f"mobileclip_{variant}_image")


# ------------------------------------------------------------------- UniFormer
def export_uniformer(variant: str):
    """variant in {'small','base'} -> classifier 224x224. Richiede i pesi/codice
    UniFormer (vedi Models/scripts/export_uniformer.py per il caricamento esatto:
    hf_repo='Sense-X/uniformer_image'). Qui placeholder via timm se disponibile."""
    import timm
    name = f"uniformer_{variant}"
    try:
        model = timm.create_model(name, pretrained=True)
    except Exception as e:
        print(f"SKIP {name}: caricalo come in export_uniformer.py ({e})")
        return
    sample = (torch.randn(1, 3, 224, 224),)
    to_int8_tflite(model, sample, name)


if __name__ == "__main__":
    for v in ("s0", "b"):
        try:
            export_mobileclip(v)
        except Exception as e:
            print(f"FAIL mobileclip_{v}: {e}")
    for v in ("small", "base"):
        try:
            export_uniformer(v)
        except Exception as e:
            print(f"FAIL uniformer_{v}: {e}")
    print("\nCopia ./tflite_native/*.tflite in "
          "Models/Test_v4.0.1/to_windows/fase2_transformer/tflite_native/")
