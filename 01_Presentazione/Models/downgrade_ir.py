import onnx
import glob
import os

def downgrade_ir():
    files = glob.glob("Models/quantized_int8/*.onnx")
    for file in files:
        model = onnx.load(file)
        # Se la versione IR è maggiore di 9, forzala a 9
        if model.ir_version > 9:
            print(f"Downgrade {os.path.basename(file)}: IR {model.ir_version} -> 9")
            model.ir_version = 9
            onnx.save(model, file)
        else:
            print(f"OK {os.path.basename(file)}: IR {model.ir_version}")

if __name__ == "__main__":
    downgrade_ir()
