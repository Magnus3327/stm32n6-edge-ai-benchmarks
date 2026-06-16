import os
from ultralytics import YOLO
import onnxruntime
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader
import numpy as np

class DummyDataReader(CalibrationDataReader):
    def __init__(self, model_path, num_samples=5):
        self.num_samples = num_samples
        self.current_sample = 0
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.inputs = {}
        for inp in session.get_inputs():
            shape = [s if isinstance(s, int) else 1 for s in inp.shape]
            self.inputs[inp.name] = {'shape': shape, 'type': inp.type}
            
    def get_next(self):
        if self.current_sample < self.num_samples:
            self.current_sample += 1
            feed_dict = {}
            for name, info in self.inputs.items():
                feed_dict[name] = np.random.uniform(-1.0, 1.0, size=info['shape']).astype(np.float32)
            return feed_dict
        return None

def main():
    print("--- 1. Esportazione YOLOv11-n (Risoluzione 224x224) ---")
    model = YOLO("yolo11n.pt")
    # Esporta in formato ONNX con dimensione immagine ridotta
    onnx_path = model.export(format="onnx", imgsz=224, opset=13)
    onnx_path = str(onnx_path)
    print(f"Modello esportato: {onnx_path}")
    
    # Preparazione per la quantizzazione
    out_dir = os.path.join("quantized_int8")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "yolov11n_micro_224_int8.onnx")
    
    print("\n--- 2. Quantizzazione Statica INT8 (QDQ) ---")
    dr = DummyDataReader(onnx_path)
    try:
        quantize_static(
            model_input=onnx_path,
            model_output=out_path,
            calibration_data_reader=dr,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8
        )
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"✅ Successo! Il modello Micro INT8 è pronto.")
        print(f"File salvato in: {out_path} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"❌ Errore durante la quantizzazione: {e}")

if __name__ == "__main__":
    main()
