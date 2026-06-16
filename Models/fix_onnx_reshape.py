import onnx
import glob
import os

def fix_allowzero(model_path):
    print(f"Analizzando {os.path.basename(model_path)}...")
    model = onnx.load(model_path)
    modified = False
    
    for node in model.graph.node:
        if node.op_type == "Reshape":
            attr_to_remove = None
            for attr in node.attribute:
                if attr.name == "allowzero":
                    attr_to_remove = attr
                    break
            
            if attr_to_remove is not None:
                # ST Edge AI Core non supporta allowzero=1
                # Lo rimuoviamo per forzare il fallback al default (0)
                node.attribute.remove(attr_to_remove)
                modified = True
                print(f"  -> Rimossa proprietà 'allowzero' dal nodo: {node.name}")
                
    if modified:
        # Salva sovrascrivendo il file
        onnx.save(model, model_path)
        print("  ✅ Modello patchato e salvato.\n")
    else:
        print("  Nessuna modifica necessaria.\n")

def main():
    # Cerca tutti i file ONNX quantizzati
    onnx_files = glob.glob("Models/quantized_int8/*.onnx")
    if not onnx_files:
        print("Nessun file ONNX trovato.")
        return
        
    for file in onnx_files:
        fix_allowzero(file)

if __name__ == "__main__":
    main()
