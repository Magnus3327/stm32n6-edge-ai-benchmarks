#!/usr/bin/env python3
"""
Converti modelli transformer ONNX → TFLite (FP32 + INT8).

Usa onnx2tf che converte ONNX → TensorFlow SavedModel → TFLite.
I professori suggeriscono TFLite perché ST AI Studio potrebbe
accettare operatori Transformer meglio via TFLite che via ONNX.

SETUP (una volta sola — usa .venv_tf con Python 3.13):
    /opt/homebrew/bin/python3.13 -m venv .venv_tf
    source .venv_tf/bin/activate
    pip install onnx2tf tensorflow

USO:
    source .venv_tf/bin/activate
    python Models/scripts/export_tflite.py

    # Solo un modello
    python Models/scripts/export_tflite.py --model uniformer_small

Output: Models/2_transformer_fail/tflite/
  {nome}_fp32.tflite       ← FP32 (stesso comportamento dell'ONNX)
  {nome}_int8.tflite       ← INT8 quantizzato (per test NPU)
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("export_tflite")

SRC_DIR = Path(__file__).parent.parent / "2_transformer_fail" / "fp32"
OUT_DIR = Path(__file__).parent.parent / "2_transformer_fail" / "tflite"

# Modelli da convertire (da SRC_DIR/*.onnx)
# Escludi i modelli > 300 MB per evitare OOM durante la conversione
MODELS = [
    "mobileclip_s0_image",
    "mobileclip_s0_text",
    "uniformer_small",
    "uniformer_base",
    "mobileclip_b_image",
    "mobileclip_b_text",
]


def check_onnx2tf():
    try:
        import onnx2tf
        import tensorflow
        log.info(f"onnx2tf {onnx2tf.__version__}, TF {tensorflow.__version__}")
        return True
    except ImportError as e:
        log.error(f"onnx2tf / tensorflow non trovati: {e}")
        log.error("Esegui: source .venv_tf/bin/activate && pip install onnx2tf tensorflow")
        return False


def convert_to_tflite(onnx_path: Path, out_name: str) -> dict:
    """Converte un file ONNX in FP32 e INT8 TFLite."""
    tmp_dir = OUT_DIR / f"_tmp_{out_name}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    fp32_out = OUT_DIR / f"{out_name}_fp32.tflite"
    int8_out = OUT_DIR / f"{out_name}_int8.tflite"
    results = {}

    # --- FP32 TFLite ---
    if fp32_out.exists():
        log.info(f"  ⏭️  FP32 già presente: {fp32_out.name}")
        results["fp32"] = "SKIP"
    else:
        log.info(f"  Convertendo FP32 → {fp32_out.name}")
        try:
            import onnx2tf
            onnx2tf.convert(
                input_onnx_file_path=str(onnx_path),
                output_folder_path=str(tmp_dir),
                output_tfv1_pb=False,
                output_saved_model=True,
                copy_onnx_input_output_names_to_tflite=True,
                non_verbose=True,
            )
            # onnx2tf crea saved_model, poi convertiamo in TFLite
            import tensorflow as tf
            saved_model_path = str(tmp_dir)
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
            tflite_model = converter.convert()
            fp32_out.write_bytes(tflite_model)
            sz = fp32_out.stat().st_size / 1024 / 1024
            log.info(f"  ✅ FP32: {sz:.1f} MB → {fp32_out.name}")
            results["fp32"] = "OK"
        except Exception as e:
            log.error(f"  ❌ FP32 fallito: {e}")
            results["fp32"] = f"FAIL: {e}"

    # --- INT8 TFLite ---
    if int8_out.exists():
        log.info(f"  ⏭️  INT8 già presente: {int8_out.name}")
        results["int8"] = "SKIP"
    elif results.get("fp32") in ("OK", "SKIP") and (tmp_dir.exists() or fp32_out.exists()):
        log.info(f"  Quantizzando INT8 → {int8_out.name}")
        try:
            import tensorflow as tf
            import numpy as np

            saved_model_path = str(tmp_dir)
            if not tmp_dir.exists():
                log.warning("  SavedModel tmp non presente, skip INT8 (esegui senza --model per rigenerare)")
                results["int8"] = "SKIP"
            else:
                converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                ]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8

                # Calibration data — immagini dummy
                interp = tf.lite.Interpreter(
                    model_path=str(fp32_out) if fp32_out.exists() else None,
                    model_content=None,
                )
                # Usa il saved model per capire la shape dell'input
                spec = tf.saved_model.load(saved_model_path)
                input_shapes = {}
                try:
                    for key, val in spec.signatures["serving_default"].structured_input_signature[1].items():
                        input_shapes[key] = [d.value or 1 for d in val.shape.dims]
                except Exception:
                    input_shapes = None

                def representative_dataset():
                    for _ in range(10):
                        if input_shapes:
                            yield {k: np.random.uniform(-1, 1, v).astype(np.float32)
                                   for k, v in input_shapes.items()}
                        else:
                            yield [np.random.uniform(-1, 1, (1, 224, 224, 3)).astype(np.float32)]

                converter.representative_dataset = representative_dataset
                tflite_model = converter.convert()
                int8_out.write_bytes(tflite_model)
                sz = int8_out.stat().st_size / 1024 / 1024
                log.info(f"  ✅ INT8: {sz:.1f} MB → {int8_out.name}")
                results["int8"] = "OK"
        except Exception as e:
            log.error(f"  ❌ INT8 fallito: {e}")
            results["int8"] = f"FAIL: {e}"
    else:
        results["int8"] = "SKIP (FP32 fallito)"

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Converti solo questo modello (es: uniformer_small)")
    args = parser.parse_args()

    if not check_onnx2tf():
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    to_convert = [args.model] if args.model else MODELS
    all_results = {}

    for name in to_convert:
        onnx_path = SRC_DIR / f"{name}.onnx"
        if not onnx_path.exists():
            log.warning(f"  ⚠️  File non trovato: {onnx_path}")
            all_results[name] = {"fp32": "MISSING", "int8": "MISSING"}
            continue

        sz_mb = onnx_path.stat().st_size / 1024 / 1024
        log.info(f"\n{'='*55}\n  {name} ({sz_mb:.0f} MB)\n{'='*55}")
        all_results[name] = convert_to_tflite(onnx_path, name)

    print("\n" + "="*55 + " RIEPILOGO TFLITE " + "="*55)
    print(f"  {'Modello':<35} {'FP32':<20} {'INT8'}")
    print(f"  {'─'*35} {'─'*20} {'─'*20}")
    for name, res in all_results.items():
        fp32 = res.get("fp32", "—")
        int8 = res.get("int8", "—")
        fp32_icon = "✅" if fp32 in ("OK","SKIP") else "❌"
        int8_icon = "✅" if int8 in ("OK","SKIP") else "❌"
        print(f"  {name:<35} {fp32_icon} {fp32:<18} {int8_icon} {int8}")
    print(f"\n  Output: {OUT_DIR}")
    print(f"\n  NOTA: Se onnx2tf fallisce su LayerNorm/MHA,")
    print(f"  il fallimento stesso conferma il COMPILER FAIL su ST AI Studio.")


if __name__ == "__main__":
    main()
