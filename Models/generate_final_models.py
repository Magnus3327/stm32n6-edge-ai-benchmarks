import os
import torch
import torchvision.models as models
from ultralytics import YOLO
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader
import numpy as np
import shutil

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

def quantize_model(input_onnx, output_onnx):
    print(f"Quantizing {input_onnx} to {output_onnx}...")
    dr = DummyDataReader(input_onnx)
    quantize_static(
        model_input=input_onnx,
        model_output=output_onnx,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8
    )

def main():
    os.makedirs("Models/exported", exist_ok=True)
    os.makedirs("Models/quantized_int8", exist_ok=True)

    # 1. YOLOv8n Micro (224x224)
    print("\n--- 1. Esportazione YOLOv8n Micro ---")
    model_yolo = YOLO("yolov8n.pt")
    yolo_out = model_yolo.export(format="onnx", imgsz=224, opset=13, simplify=True)
    yolo_final_exported = "Models/exported/yolov8n_micro_224.onnx"
    if os.path.exists(yolo_out):
        shutil.move(yolo_out, yolo_final_exported)

    # 2. MobileNetV2 (224x224, CNN Pura)
    print("\n--- 2. Esportazione MobileNetV2 (Batch fisso) ---")
    model_mbv2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model_mbv2.eval()
    mbv2_out = "Models/exported/mobilenet_v2_224.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model_mbv2,
            torch.randn(1, 3, 224, 224),
            mbv2_out,
            opset_version=13,
            input_names=["input"],
            output_names=["output"],
        )
    os.system(f"onnxslim {mbv2_out} {mbv2_out}")

    # 3. Quantizzazione
    print("\n--- 3. Quantizzazione INT8 ---")
    quant_yolo = "Models/quantized_int8/yolov8n_micro_224_int8.onnx"
    quant_mbv2 = "Models/quantized_int8/mobilenet_v2_224_int8.onnx"
    
    quantize_model(yolo_final_exported, quant_yolo)
    quantize_model(mbv2_out, quant_mbv2)

    # 4. Patch anti-allowzero
    print("\n--- 4. Patch ONNX 'allowzero' ---")
    os.system("python Models/fix_onnx_reshape.py")

    print(f"\n✅ Tutto fatto! I modelli sono pronti in Models/quantized_int8/:")
    print(f" - {os.path.basename(quant_yolo)}")
    print(f" - {os.path.basename(quant_mbv2)}")

if __name__ == "__main__":
    main()
