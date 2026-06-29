#!/usr/bin/env python3
"""
Quantizzazione INT4 (Weight-only) dei modelli RAM-fail.

Strategia: pesi → INT4, attivazioni → INT8.
  - I pesi occupano 4× meno della versione FP32 (50% rispetto a INT8).
  - Le attivazioni restano INT8 → il muro della RAM dipende dalle attivazioni,
    non dai pesi. Tuttavia alcuni compilatori NPU (incluso ST Edge AI) trattano
    i layer INT4-weight in modo diverso e potrebbero usare tiling più aggressivo.
  - Se ST AI Studio accetta INT4 nativamente, le attivazioni intermedie potrebbero
    venire compresse a 4 bit dal compilatore, riducendo la RAM di 2×.
  - Nota: INT4 pieno (pesi + attivazioni) non è standard ONNX/onnxruntime e
    richiederebbe un compilatore con supporto esplicito.

Uso:
    # Quantizza tutti i modelli in 1_ram_fail/int8 → 1_ram_fail/int4
    python quantize_int4.py

    # Quantizza un modello specifico
    python quantize_int4.py --input path/to/model.onnx --output path/to/model_int4.onnx

    # Quantizza una cartella specifica
    python quantize_int4.py --dir Models/1_ram_fail/fp32
"""

import argparse
import glob
import logging
import os
from pathlib import Path

import numpy as np
import onnxruntime
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("quantize_int4")


class DummyDataReader(CalibrationDataReader):
    """Dati casuali per calibrazione statica (sufficiente per profiling latenza/RAM)."""

    def __init__(self, model_path: str, num_samples: int = 10):
        self.num_samples = num_samples
        self.current_sample = 0
        session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.inputs = {}
        for inp in session.get_inputs():
            shape = [s if isinstance(s, int) else 1 for s in inp.shape]
            self.inputs[inp.name] = {"shape": shape, "type": inp.type}

    def get_next(self):
        if self.current_sample < self.num_samples:
            self.current_sample += 1
            feed = {}
            for name, info in self.inputs.items():
                if "int64" in info["type"] or "int32" in info["type"]:
                    dtype = np.int64 if "int64" in info["type"] else np.int32
                    feed[name] = np.random.randint(0, 100, size=info["shape"]).astype(dtype)
                else:
                    feed[name] = np.random.uniform(-1.0, 1.0, size=info["shape"]).astype(np.float32)
            return feed
        return None


def quantize_int4(input_path: str, output_path: str) -> bool:
    log.info(f"INT4 quantizing: {os.path.basename(input_path)}")
    try:
        dr = DummyDataReader(input_path, num_samples=10)
        quantize_static(
            model_input=input_path,
            model_output=output_path,
            calibration_data_reader=dr,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,   # attivazioni INT8 (standard NPU)
            weight_type=QuantType.QInt4,        # pesi INT4 (50% rispetto a INT8)
        )
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        int8_path = output_path.replace("int4", "int8")
        if os.path.exists(int8_path):
            int8_size = os.path.getsize(int8_path) / (1024 * 1024)
            reduction = (1 - size_mb / int8_size) * 100
            log.info(f"  ✅ {size_mb:.1f} MB (INT8: {int8_size:.1f} MB, -{reduction:.0f}%)")
        else:
            log.info(f"  ✅ {size_mb:.1f} MB")
        return True
    except Exception as e:
        log.error(f"  ❌ Fallito: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Quantizzazione INT4 (weight-only) per ST NPU.")
    parser.add_argument("--input", type=str, help="File ONNX singolo (FP32 o INT8)")
    parser.add_argument("--output", type=str, help="File output INT4 (se --input è specificato)")
    parser.add_argument("--dir", type=str, help="Cartella con file .onnx da quantizzare")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.parent
    out_dir = script_dir / "1_ram_fail" / "int4"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Modalità file singolo ---
    if args.input:
        out = args.output or str(out_dir / Path(args.input).name.replace(".onnx", "_int4.onnx"))
        quantize_int4(args.input, out)
        return

    # --- Modalità cartella ---
    if args.dir:
        in_dir = Path(args.dir)
    else:
        # Default: quantizza i modelli FP32 (più stabili) da 1_ram_fail/fp32
        in_dir = script_dir / "1_ram_fail" / "fp32"

    files = sorted(in_dir.glob("*.onnx"))
    if not files:
        log.warning(f"Nessun .onnx in {in_dir}")
        return

    # Modelli con external data (.data) non possono essere caricati senza
    # load_external_data=True — per ora si saltano con un avviso chiaro.
    skip_external = {"resnet34_fixed.onnx"}

    log.info(f"Trovati {len(files)} modelli in {in_dir}")
    results = {}
    for f in files:
        if f.name in skip_external:
            log.warning(f"  ⏭️  Skip {f.name} (ha external data, consolidare prima con patch_onnx.py)")
            results[f.name] = "SKIP"
            continue
        out_name = f.stem + "_int4.onnx"
        out_path = out_dir / out_name
        if out_path.exists():
            log.info(f"  ⏭️  Skip {f.name} (già presente in int4/)")
            results[f.name] = "SKIP"
            continue
        ok = quantize_int4(str(f), str(out_path))
        results[f.name] = "OK" if ok else "FAIL"

    print("\n══════════════════════ RIEPILOGO INT4 ══════════════════════")
    for name, status in results.items():
        icon = "✅" if status == "OK" else ("⏭️" if status == "SKIP" else "❌")
        print(f"  {icon}  {name}")
    ok_count = sum(1 for s in results.values() if s == "OK")
    print(f"\n  Completati: {ok_count}/{len(results)}")
    print(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()
