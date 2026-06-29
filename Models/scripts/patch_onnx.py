#!/usr/bin/env python3
"""
Patch ONNX models for compatibility with ST Edge AI Core (CubeMX v2.2 and AI Studio v4.0).

Due patch disponibili:
  1. downgrade_ir  — Porta l'IR version 10 → 9 (obbligatorio per CubeMX v2.2)
  2. fix_reshape   — Rimuove l'attributo 'allowzero' dai nodi Reshape (crash su entrambi i compiler)

Uso:
    python patch_onnx.py --all                          # applica entrambe le patch a tutti i .onnx
    python patch_onnx.py --fix reshape                  # solo fix_reshape
    python patch_onnx.py --fix ir                       # solo downgrade IR
    python patch_onnx.py --input my_model.onnx          # file singolo
    python patch_onnx.py --dir Models/1_ram_fail/fp32   # cartella specifica
"""

import argparse
import glob
import os
from pathlib import Path

import onnx


def downgrade_ir(model: onnx.ModelProto, target_ir: int = 9) -> onnx.ModelProto:
    if model.ir_version > target_ir:
        print(f"  IR {model.ir_version} → {target_ir}")
        model.ir_version = target_ir
    return model


def fix_reshape_allowzero(model: onnx.ModelProto) -> onnx.ModelProto:
    fixed = 0
    for node in model.graph.node:
        if node.op_type == "Reshape":
            for attr in list(node.attribute):
                if attr.name == "allowzero":
                    node.attribute.remove(attr)
                    fixed += 1
    if fixed:
        print(f"  Rimosso 'allowzero' da {fixed} nodo/i Reshape")
    return model


def patch_file(path: str, do_ir: bool, do_reshape: bool, target_ir: int = 9) -> None:
    print(f"\n[PATCH] {os.path.basename(path)}")
    try:
        model = onnx.load(path, load_external_data=False)
    except Exception as e:
        print(f"  ⚠️  Caricamento fallito (potrebbe avere external data): {e}")
        return

    if do_ir:
        model = downgrade_ir(model, target_ir)
    if do_reshape:
        model = fix_reshape_allowzero(model)

    onnx.save(model, path)
    print(f"  ✅ Salvato")


def main():
    parser = argparse.ArgumentParser(description="Patch ONNX per ST Edge AI Core.")
    parser.add_argument("--all", action="store_true", help="Applica tutte le patch disponibili")
    parser.add_argument("--fix", nargs="+", choices=["ir", "reshape"],
                        help="Patch specifiche: ir, reshape")
    parser.add_argument("--input", type=str, help="File ONNX singolo")
    parser.add_argument("--dir", type=str, help="Cartella contenente file .onnx")
    parser.add_argument("--target-ir", type=int, default=9,
                        help="Versione IR target (default: 9 per CubeMX v2.2)")
    args = parser.parse_args()

    do_ir = args.all or (args.fix and "ir" in args.fix)
    do_reshape = args.all or (args.fix and "reshape" in args.fix)

    if not do_ir and not do_reshape:
        do_ir = True
        do_reshape = True
        print("[INFO] Nessuna patch specificata — applicando tutte le patch per default")

    # Raccolta file da patchare
    files = []
    if args.input:
        files = [args.input]
    elif args.dir:
        files = sorted(glob.glob(os.path.join(args.dir, "*.onnx")))
    else:
        # Default: patcha tutte le cartelle del progetto
        script_dir = Path(__file__).parent.parent
        for pattern in [
            "1_ram_fail/fp32/*.onnx",
            "1_ram_fail/int8/*.onnx",
            "2_transformer_fail/fp32/*.onnx",
            "2_transformer_fail/int8/*.onnx",
            "0_validated/fp32/*.onnx",
            "0_validated/int8/*.onnx",
        ]:
            files.extend(sorted(glob.glob(str(script_dir / pattern))))

    if not files:
        print("Nessun file .onnx trovato.")
        return

    print(f"\nPatch da applicare: {'IR downgrade' if do_ir else ''} {'fix_reshape' if do_reshape else ''}")
    print(f"File trovati: {len(files)}")

    for f in files:
        patch_file(f, do_ir, do_reshape, args.target_ir)

    print(f"\n✅ Patch completate su {len(files)} file.")


if __name__ == "__main__":
    main()
