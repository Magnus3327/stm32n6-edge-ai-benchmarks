import tensorflow as tf
import numpy as np
import os

# Parametri globali
INPUT_SHAPE = (64, 64, 3)
BATCH_SIZE = 1
NUM_CLASSES = 10
MODELS_DIR = "."

def representative_data_gen():
    """
    Genera dati casuali per calibrare i pesi e le attivazioni durante la quantizzazione INT8.
    """
    for _ in range(100):
        # Dati randomici tra -1 e 1 come esempio di input normalizzato
        data = np.random.rand(1, *INPUT_SHAPE).astype(np.float32) * 2.0 - 1.0
        yield [data]

def convert_to_tflite_int8(model, filename):
    """
    Converte un modello Keras in TFLite con Full Integer Quantization (INT8).
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Impostazioni per l'ottimizzazione e quantizzazione INT8
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    
    # Limita le operazioni supportate a INT8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Imposta tipo di input e output a INT8
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    filepath = os.path.join(MODELS_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(tflite_model)
    print(f"Salvato modello quantizzato: {filepath}")

def build_conv2d_model():
    """Modello con un singolo strato Conv2D (3x3)."""
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    # Output layer fittizio per avere una fine
    outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inputs, outputs, name="conv2d_3x3")

def build_depthwise_model():
    """Modello con un singolo strato DepthwiseConv2D (3x3)."""
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(inputs)
    outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inputs, outputs, name="depthwise_3x3")

def build_conv2d_1x1_model():
    """Modello con un singolo strato Conv2D Pointwise (1x1)."""
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = tf.keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inputs, outputs, name="conv2d_1x1")

def build_conv2d_pool_model():
    """Modello con Conv2D seguito da MaxPooling2D."""
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inputs, outputs, name="conv2d_pool")

def build_conv2d_custom_model(filters, kernel_size):
    """Modello con un Conv2D personalizzabile in filtri e dimensione kernel."""
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(inputs)
    outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inputs, outputs, name=f"conv2d_{filters}f_{kernel_size[0]}x{kernel_size[1]}")

def main():
    print("Generazione dei modelli baseline Keras...")
    
    models = {
        "baseline_conv2d_3x3_int8.tflite": build_conv2d_model(),
        "baseline_depthwise_3x3_int8.tflite": build_depthwise_model(),
        "baseline_conv2d_1x1_int8.tflite": build_conv2d_1x1_model(),
        "baseline_conv2d_pool_int8.tflite": build_conv2d_pool_model(),
        # Nuovi modelli per testare il carico e i filtri sulla NPU
        "baseline_conv2d_16f_3x3_int8.tflite": build_conv2d_custom_model(16, (3, 3)),
        "baseline_conv2d_64f_3x3_int8.tflite": build_conv2d_custom_model(64, (3, 3)),
        "baseline_conv2d_32f_5x5_int8.tflite": build_conv2d_custom_model(32, (5, 5)),
        "baseline_conv2d_32f_7x7_int8.tflite": build_conv2d_custom_model(32, (7, 7)),
    }
    
    print("\nInizio conversione e quantizzazione (Full Integer INT8)...")
    for filename, model in models.items():
        print(f"Elaborazione di {model.name}...")
        convert_to_tflite_int8(model, filename)
        
    print("\nTutti i modelli sono stati generati e salvati con successo in formato .tflite.")

if __name__ == "__main__":
    main()
