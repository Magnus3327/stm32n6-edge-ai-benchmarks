import os
import glob
import logging
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader, QuantFormat
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("quantize")

class DummyDataReader(CalibrationDataReader):
    """
    Fornisce dati casuali per la calibrazione statica. 
    Dato che l'obiettivo è valutare le performance su NPU e non la vera accuratezza 
    su un dataset di validazione, i dati casuali sono sufficienti per permettere al
    quantizzatore di calcolare dei range e inserire i nodi QuantizeLinear/DequantizeLinear.
    """
    def __init__(self, model_path, num_samples=5):
        self.num_samples = num_samples
        self.current_sample = 0
        
        # Usiamo InferenceSession solo per estrarre facilmente shape e tipo degli input
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.inputs = {}
        for inp in session.get_inputs():
            shape = inp.shape
            # Sostituiamo le dimensioni dinamiche (None o stringhe) con 1
            shape = [s if isinstance(s, int) else 1 for s in shape]
            self.inputs[inp.name] = {'shape': shape, 'type': inp.type}
            
    def get_next(self):
        if self.current_sample < self.num_samples:
            self.current_sample += 1
            feed_dict = {}
            for name, info in self.inputs.items():
                shape = info['shape']
                typ = info['type']
                # Tipi interi (ad es. per i token di testo)
                if 'int64' in typ or 'int32' in typ:
                    dtype = np.int64 if 'int64' in typ else np.int32
                    feed_dict[name] = np.random.randint(0, 100, size=shape).astype(dtype)
                else:
                    # Immagini / Float standard
                    feed_dict[name] = np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)
            return feed_dict
        return None

def quantize_model(input_path, output_path):
    log.info(f"Quantizing {os.path.basename(input_path)} ...")
    dr = DummyDataReader(input_path, num_samples=10)
    
    try:
        quantize_static(
            model_input=input_path,
            model_output=output_path,
            calibration_data_reader=dr,
            quant_format=QuantFormat.QDQ, 
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8
        )
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        log.info(f"✅ Successo: Salvato in {os.path.basename(output_path)} ({size_mb:.1f} MB)")
    except Exception as e:
        log.error(f"❌ Fallito: Impossibile quantizzare {os.path.basename(input_path)}")
        log.error(f"Motivo: {e}")

if __name__ == "__main__":
    # Percorsi configurati sulla base della cartella Models/exported
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(SCRIPT_DIR, "exported")
    output_dir = os.path.join(SCRIPT_DIR, "quantized_int8")
    
    os.makedirs(output_dir, exist_ok=True)
    
    onnx_files = sorted(glob.glob(os.path.join(input_dir, "*.onnx")))
    if not onnx_files:
        log.warning(f"Nessun file ONNX trovato in {input_dir}")
        exit(0)
        
    log.info(f"Trovati {len(onnx_files)} modelli. Inizio quantizzazione statica INT8 (QDQ)...")
    
    for f in onnx_files:
        name = os.path.basename(f)
        out_name = name.replace(".onnx", "_int8.onnx")
        out_path = os.path.join(output_dir, out_name)
        
        if os.path.exists(out_path):
            log.info(f"Salta {name}, modello già quantizzato.")
            continue
            
        quantize_model(f, out_path)
