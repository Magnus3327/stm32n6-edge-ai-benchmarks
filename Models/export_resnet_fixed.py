import torch
import torchvision.models as models
import os
import onnx
import sys

print("--- 1. Esportazione ResNet34 (Batch Size Fissa) ---")
model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
model.eval()
out = "Models/exported/resnet34_fixed.onnx"

# Esportiamo SENZA dynamic_axes per rimuovere l'operatore Shape dinamico
with torch.no_grad():
    torch.onnx.export(
        model,
        torch.randn(1, 3, 224, 224),
        out,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
    )
print(f"Modello esportato: {out}")

print("\n--- 2. Slimming del modello ONNX ---")
try:
    import onnxslim
except ImportError:
    os.system(f"{sys.executable} -m pip install onnxslim")

os.system(f"onnxslim {out} {out}")

print("\n--- 3. Quantizzazione Statica INT8 (QDQ) ---")
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

out_quant = "Models/quantized_int8/resnet34_fixed_int8.onnx"
dr = DummyDataReader(out)
try:
    quantize_static(
        model_input=out,
        model_output=out_quant,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8
    )
    size_mb = os.path.getsize(out_quant) / (1024 * 1024)
    print(f"✅ Successo! Il modello INT8 è pronto e semplificato.")
    print(f"File salvato in: {out_quant} ({size_mb:.2f} MB)")
except Exception as e:
    print(f"❌ Errore durante la quantizzazione: {e}")
