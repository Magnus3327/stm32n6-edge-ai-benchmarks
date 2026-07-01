#!/usr/bin/env python3
"""
Export UniFormer variants con pesi pretrained → ONNX + INT8.

VARIANTI ESPORTATE:
  Standard (Sense-X/uniformer_image):
    uniformer_small   : depth=[3,4,8,3],  embed_dim=[64,128,320,512]  → 21.5M params
    uniformer_base    : depth=[5,8,20,7], embed_dim=[64,128,320,512]  → 49.8M params

  Light (Andy1621/uniformer_light — varianti più piccole ufficiali):
    uniformer_xxs     : depth=[2,5,8,2],  embed_dim=[56,112,224,448]  → 10.2M params
    uniformer_xs      : depth=[3,5,9,3],  embed_dim=[64,128,256,512]  → 16.5M params

NOTE sulle varianti del CSV:
  Il CSV chiede (t): 1.8M, (s): 2.4M, (m): 5.6M, (l): 10M.
  Queste configurazioni NON esistono come checkpoint pubblici.
  Le varianti Light (XXS/XS = 10.2M/16.5M) sono le più piccole mai rilasciate
  dagli autori originali, stessa architettura (DWConv + Self-Attention + LayerNorm).
  UniFormer-XXS (10.2M) è il match migliore per la "l: 10M" del CSV.

NOTE TECNICHE (UniFormer-Light):
  Il modello originale usa token pruning dinamico (easy_gather su global_attn)
  che non è esportabile in ONNX. Questo script lo DISABILITA impostando
  prune_ratio=[[],[],[],[]] → il modello diventa statico e ONNX-esportabile.
  I pesi rimangono identici (il pruning è solo logica di inferenza, non pesi).

USO:
    source .venv/bin/activate
    python Models/scripts/export_uniformer.py

    # Solo Light
    python Models/scripts/export_uniformer.py --variants light

    # Solo Standard
    python Models/scripts/export_uniformer.py --variants standard
"""

import argparse
import logging
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("export_uniformer")

CACHE = Path(__file__).parent.parent / ".cache" / "uniformer"
OUT_FP32 = Path(__file__).parent.parent / "2_transformer_fail" / "fp32"
OUT_INT8 = Path(__file__).parent.parent / "2_transformer_fail" / "int8"

STANDARD_CONFIGS = {
    "uniformer_small": dict(
        depth=[3, 4, 8, 3], embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4, qkv_bias=True,
        hf_repo="Sense-X/uniformer_image", weight_file="uniformer_small_in1k.pth",
        model_py="uniformer.py",
    ),
    "uniformer_base": dict(
        depth=[5, 8, 20, 7], embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4, qkv_bias=True,
        hf_repo="Sense-X/uniformer_image", weight_file="uniformer_base_in1k.pth",
        model_py="uniformer.py",
    ),
}

LIGHT_CONFIGS = {
    "uniformer_xxs": dict(
        depth=[2, 5, 8, 2],
        embed_dim=[56, 112, 224, 448],
        head_dim=28,
        mlp_ratio=[3, 3, 3, 3],
        qkv_bias=True,
        conv_stem=True,
        # Pruning DISABILITATO per ONNX export (originale: [[], [], [1,0.5,...], [0.5,0.5]])
        prune_ratio=[[], [], [], []],
        trade_off=[[], [], [], []],
        hf_repo="Andy1621/uniformer_light",
        weight_file="uniformer_xxs_160_in1k.pth",
        model_py="uniformer_light.py",
        params_m=10.2,
    ),
    "uniformer_xs": dict(
        depth=[3, 5, 9, 3],
        embed_dim=[64, 128, 256, 512],
        head_dim=32,
        mlp_ratio=[3, 3, 3, 3],
        qkv_bias=True,
        conv_stem=True,
        prune_ratio=[[], [], [], []],
        trade_off=[[], [], [], []],
        hf_repo="Andy1621/uniformer_light",
        weight_file="uniformer_xs_160_in1k.pth",
        model_py="uniformer_light.py",
        params_m=16.5,
    ),
}

LIGHT_MODEL_PY_URL = (
    "https://raw.githubusercontent.com/Sense-X/UniFormer/main/"
    "image_classification/models/uniformer_light.py"
)


class DummyDataReader(CalibrationDataReader):
    def __init__(self, model_path: str, n: int = 10):
        self.n = n
        self.cur = 0
        s = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.iname = s.get_inputs()[0].name

    def get_next(self):
        if self.cur < self.n:
            self.cur += 1
            return {self.iname: np.random.uniform(-1, 1, (1, 3, 224, 224)).astype(np.float32)}
        return None


def download_file(url: str, dst: Path) -> bool:
    try:
        log.info(f"  Scaricando {dst.name} da {url[:60]}...")
        urllib.request.urlretrieve(url, str(dst))
        return True
    except Exception as e:
        log.error(f"  ❌ Download fallito: {e}")
        return False


def download_from_hf(repo: str, filename: str, dst: Path) -> bool:
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        path = hf_hub_download(repo, filename)
        shutil.copy2(path, dst)
        return True
    except Exception as e:
        log.error(f"  ❌ HF download fallito ({repo}/{filename}): {e}")
        return False


def export_standard(name: str, cfg: dict) -> bool:
    """Export varianti standard (Sense-X) con uniformer.py."""
    CACHE.mkdir(parents=True, exist_ok=True)

    # Download model code
    code_path = CACHE / cfg["model_py"]
    if not code_path.exists():
        if not download_from_hf(cfg["hf_repo"], cfg["model_py"], code_path):
            return False

    # Download weights
    weight_path = CACHE / cfg["weight_file"]
    if not weight_path.exists():
        if not download_from_hf(cfg["hf_repo"], cfg["weight_file"], weight_path):
            return False

    sys.path.insert(0, str(CACHE))
    try:
        from uniformer import UniFormer
        model_kwargs = {k: v for k, v in cfg.items()
                        if k not in ("hf_repo", "weight_file", "model_py")}
        model = UniFormer(**model_kwargs)
        state = torch.load(str(weight_path), map_location="cpu", weights_only=False)
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
    finally:
        sys.path.remove(str(CACHE))

    return _export_and_quantize(model, name)


def export_light(name: str, cfg: dict) -> bool:
    """Export varianti Light (Andy1621) con uniformer_light.py (pruning disabilitato)."""
    import uniformer_light as ul_mod

    CACHE.mkdir(parents=True, exist_ok=True)

    # Download uniformer_light.py da GitHub
    light_py = CACHE / "uniformer_light.py"
    if not light_py.exists():
        if not download_file(LIGHT_MODEL_PY_URL, light_py):
            return False

    # Download weights da HF
    weight_path = CACHE / cfg["weight_file"]
    if not weight_path.exists():
        log.info(f"  Peso {cfg['weight_file']} non trovato, scaricando da HF...")
        if not download_from_hf(cfg["hf_repo"], cfg["weight_file"], weight_path):
            # Fallback: Google Drive link per xxs_160
            gd_ids = {
                "uniformer_xxs_160_in1k.pth": "1CsXCzfVFPyM2mY8BA_1fO3q_VBngnF44",
                "uniformer_xs_160_in1k.pth":  "1CwQ91fjuAu3Rhn3zV4vjAmQ8v36AfYPT",
            }
            gd_id = gd_ids.get(cfg["weight_file"])
            if gd_id:
                log.warning(f"  Prova manuale: scarica da Google Drive")
                log.warning(f"  https://drive.google.com/file/d/{gd_id}/view")
                log.warning(f"  Salva come: {weight_path}")
            return False

    sys.path.insert(0, str(CACHE))
    try:
        # Import fresco per evitare conflitti con uniformer standard
        if "uniformer_light" in sys.modules:
            del sys.modules["uniformer_light"]
        import uniformer_light as ul_mod

        model_kwargs = {k: v for k, v in cfg.items()
                        if k not in ("hf_repo", "weight_file", "model_py", "params_m")}
        model = ul_mod.UniFormer_Light(**model_kwargs)

        state = torch.load(str(weight_path), map_location="cpu", weights_only=False)
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        log.info(f"  Params: {params:.1f}M (atteso ~{cfg.get('params_m', '?')}M)")
    finally:
        sys.path.remove(str(CACHE))

    # Reset global state prima di export
    try:
        import uniformer_light
        uniformer_light.global_attn = 0
        uniformer_light.token_indices = None
    except Exception:
        pass

    return _export_and_quantize(model, name)


def _export_and_quantize(model: torch.nn.Module, name: str) -> bool:
    model.eval()
    fp32_path = str(OUT_FP32 / f"{name}.onnx")
    int8_path = str(OUT_INT8 / f"{name}_int8.onnx")

    log.info(f"  Esportando ONNX FP32 → {fp32_path}")
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                torch.randn(1, 3, 224, 224),
                fp32_path,
                opset_version=18,
                input_names=["input"],
                output_names=["output"],
                dynamo=False,
            )
        sz = os.path.getsize(fp32_path) / 1024 / 1024
        log.info(f"  ✅ FP32: {sz:.1f} MB")
    except Exception as e:
        log.error(f"  ❌ Export FP32 fallito: {e}")
        return False

    log.info(f"  Quantizzando INT8 → {int8_path}")
    try:
        quantize_static(
            fp32_path, int8_path,
            DummyDataReader(fp32_path),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
        )
        sz = os.path.getsize(int8_path) / 1024 / 1024
        log.info(f"  ✅ INT8: {sz:.1f} MB")
    except Exception as e:
        log.error(f"  ❌ Quantizzazione INT8 fallita: {e}")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", choices=["all", "standard", "light"], default="all")
    args = parser.parse_args()

    OUT_FP32.mkdir(parents=True, exist_ok=True)
    OUT_INT8.mkdir(parents=True, exist_ok=True)

    results = {}

    if args.variants in ("all", "standard"):
        for name, cfg in STANDARD_CONFIGS.items():
            log.info(f"\n{'='*55}\n  {name}\n{'='*55}")
            if (OUT_FP32 / f"{name}.onnx").exists() and (OUT_INT8 / f"{name}_int8.onnx").exists():
                log.info("  ⏭️  Skip (già presente)")
                results[name] = "SKIP"
                continue
            ok = export_standard(name, cfg)
            results[name] = "OK" if ok else "FAIL"

    if args.variants in ("all", "light"):
        for name, cfg in LIGHT_CONFIGS.items():
            log.info(f"\n{'='*55}\n  {name} (~{cfg.get('params_m','?')}M params)\n{'='*55}")
            if (OUT_FP32 / f"{name}.onnx").exists() and (OUT_INT8 / f"{name}_int8.onnx").exists():
                log.info("  ⏭️  Skip (già presente)")
                results[name] = "SKIP"
                continue
            ok = export_light(name, cfg)
            results[name] = "OK" if ok else "FAIL"

    print("\n" + "="*55 + " RIEPILOGO " + "="*55)
    for name, status in results.items():
        icon = "✅" if status == "OK" else ("⏭️" if status == "SKIP" else "❌")
        print(f"  {icon}  {name:<30} [{status}]")
    print(f"\n  FP32 → {OUT_FP32}")
    print(f"  INT8  → {OUT_INT8}")


if __name__ == "__main__":
    main()
