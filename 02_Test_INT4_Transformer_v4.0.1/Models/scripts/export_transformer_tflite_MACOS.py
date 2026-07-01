"""
Export transformer -> INT8 TFLite via litert-torch (PyTorch -> TFLite diretto).

Gira su macOS arm64 con conda Python 3.12 + litert-torch 0.9.1:
    conda activate py312_aiedge
    pip install -q litert-torch timm
    pip install git+https://github.com/apple/ml-mobileclip.git
    python Models/scripts/export_transformer_tflite_MACOS.py

Output: ./tflite_native/<modello>_int8.tflite  -> poi copiali in
        Models/Test_v4.0.1/to_windows/fase2_transformer/tflite_native/
"""
import os
from pathlib import Path
import numpy as np
import torch

OUT = Path("tflite_native"); OUT.mkdir(exist_ok=True)

import litert_torch
from litert_torch.quantize.pt2e_quantizer import PT2EQuantizer, get_symmetric_quantization_config
from litert_torch.quantize.quant_config import QuantConfig
from torchao.quantization.pt2e import quantize_pt2e


def to_int8_tflite(module: torch.nn.Module, sample: tuple, name: str):
    """PyTorch nn.Module -> INT8 TFLite (PT2E static quant, calib su dati random)."""
    module = module.eval()
    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False)
    )
    exported = torch.export.export(module, sample).module()
    prepared = quantize_pt2e.prepare_pt2e(exported, quantizer)
    for _ in range(16):
        prepared(*[torch.rand_like(s) for s in sample])
    converted = quantize_pt2e.convert_pt2e(prepared, fold_quantize=False)
    edge = litert_torch.convert(
        converted, sample, quant_config=QuantConfig(pt2e_quantizer=quantizer)
    )
    path = OUT / f"{name}_int8.tflite"
    edge.export(str(path))
    print(f"OK  {name:22s} -> {path}  ({path.stat().st_size/1e6:.2f} MB)")


def export_mobileclip(variant: str):
    """variant in {'s0','b'} -> esporta l'image encoder (256x256)."""
    import mobileclip
    model, _, _ = mobileclip.create_model_and_transforms(f"mobileclip_{variant}")
    enc = getattr(model, "image_encoder", None) or getattr(model, "visual")
    sample = (torch.randn(1, 3, 256, 256),)
    to_int8_tflite(enc, sample, f"mobileclip_{variant}_image")


def export_uniformer(variant: str):
    """variant in {'small','base'} -> carica da Models/.cache/uniformer/ (locale)."""
    import sys
    # uniformer .pth/.py vivono in 01_Presentazione (pesi condivisi, non duplicati)
    cache_dir = Path(__file__).parent.parent.parent.parent / "01_Presentazione" / "Models" / ".cache" / "uniformer"
    sys.path.insert(0, str(cache_dir))
    try:
        from uniformer import uniformer_small, uniformer_base
        factory = {"small": uniformer_small, "base": uniformer_base}[variant]
        model = factory(pretrained=False)
        pth = cache_dir / f"uniformer_{variant}_in1k.pth"
        state = torch.load(str(pth), map_location="cpu", weights_only=False)
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
    finally:
        sys.path.remove(str(cache_dir))
    sample = (torch.randn(1, 3, 224, 224),)
    to_int8_tflite(model, sample, f"uniformer_{variant}")


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
