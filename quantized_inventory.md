# 📦 Inventario Modelli Quantizzati (INT8)

> **Data Creazione:** 2026-06-16
> **Formato:** ONNX (Quantize-Dequantize QDQ)
> **Precisione:** `INT8` (Attivazioni e Pesi)
> **Directory Output:** `Models/quantized_int8/` (o `quantized_int8/` in base all'ultimo aggiornamento)

Tutti i modelli esportati sono stati processati con uno script di quantizzazione statica (`quantize_int8.py`) sfruttando la libreria `onnxruntime.quantization`. La quantizzazione permette di convertire i parametri matematici in Float32 (FP32) in interi a 8 bit (INT8).

### 📉 Risultati della Quantizzazione

Come mostrato nella tabella seguente, la dimensione dei file si è **ridotta di circa il 70-75%** per tutti i modelli.

| Modello | Dimensione FP32 (Originale) | Dimensione INT8 | Fattore di riduzione |
|---------|-----------------------------|-----------------|----------------------|
| **YOLOv11-x** | 217.5 MB | **55.3 MB** | ~ 74.6% |
| **YOLOv11-l** | 97.0 MB | **25.1 MB** | ~ 74.1% |
| **YOLOv11-m** | 76.9 MB | **19.8 MB** | ~ 74.2% |
| **YOLOv11-s** | 36.3 MB | **9.5 MB** | ~ 73.8% |
| **YOLOv11-n** | 10.2 MB | **3.0 MB** | ~ 70.6% |
| **YOLOv8-n** | 12.3 MB | **3.4 MB** | ~ 72.3% |
| **YOLOv7** | 144.1 MB | **36.8 MB** | ~ 74.4% |
| **ResNet34** | 83.3 MB | **21.0 MB** | ~ 74.8% |
| **PIDNet-S** | 30.9 MB | **7.6 MB** | ~ 75.4% |
| **MobileCLIP-S0** (Image) | 44.1 MB | **11.4 MB** | ~ 74.1% |
| **MobileCLIP-S0** (Text) | 162.1 MB | **42.5 MB** | ~ 73.8% |
| **MobileCLIP-B** (Image) | 330.1 MB | **86.6 MB** | ~ 73.8% |
| **MobileCLIP-B** (Text) | 242.9 MB | **63.5 MB** | ~ 73.8% |

---

## 🛠️ Utilizzo con STM32Cube.AI / ST Edge AI Core

I modelli sono ora nel formato raccomandato (`QuantFormat.QDQ`) dai compilatori per le moderne NPU.

> [!TIP]
> **Come iniziare:** Quando carichi questi modelli su AI Studio, inizia sempre da quelli più piccoli (ad es. `yolov11n_int8.onnx` o `yolov8n_int8.onnx`). Entrambi pesano **circa 3 MB** in INT8, il che li rende eccellenti candidati per i test di `Memory footprint` iniziali senza scatenare errori legati ai limiti RAM dei microcontrollori standard.

> [!WARNING]
> **Dati di Calibrazione:** La quantizzazione è stata eseguita fornendo tensori generati casualmente (dummy data). Per l'obiettivo di testare le **prestazioni hardware, la latenza e l'uso della memoria (profiling)** questo approccio è ottimale in quanto i tempi di inferenza su NPU restano identici. Tuttavia, qualora l'obiettivo finale divenisse l'accuratezza (mAP), sarà necessario fornire allo script di quantizzazione immagini reali prese dai dataset di test (come COCO o ImageNet).
