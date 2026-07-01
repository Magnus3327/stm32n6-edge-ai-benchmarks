#!/usr/bin/env python3
"""
Genera microbenchmark baseline per operatori Transformer/FC sul ST NPU.

Analogia con i baseline conv (3_baseline/conv/*.tflite):
  - 8 layer conv singoli → benchmark convoluzioni
  - 7 operatori transformer singoli → benchmark transformer
  - Stesso input: dimensioni piccole, INT8 quantizzato

Output: Models/3_baseline/transformer/
  baseline_dense_int8.onnx              Dense (MatMul) 128→128
  baseline_layernorm_opset13_int8.onnx  LayerNorm DECOMPOSED (Reduce/Sub/Mul/Sqrt) ← replica MobileCLIP
  baseline_layernorm_opset17_int8.onnx  LayerNorm FUSED (nodo singolo) ← test ipotesi prof
  baseline_softmax_int8.onnx            Softmax su attention weights (4 heads, 64 token)
  baseline_mha_int8.onnx                MultiHeadAttention completo (64 token, 128 dim, 4 head)
  baseline_transformer_block_int8.onnx  Encoder Block completo (MHA + FFN + LayerNorm)
  baseline_conv_mha_hybrid_int8.onnx    Conv2D + MHA (come MobileCLIP image encoder)

Dipendenze: torch, onnxruntime (già installati nel .venv)
TensorFlow NON richiesto — tutti i modelli sono in formato ONNX.

Per TFLite:
  pip install tensorflow==2.17
  python generate_conv_baselines.py   # genera i .tflite conv (già in 3_baseline/conv/)

Uso:
    python generate_transformer_baselines.py
    python generate_transformer_baselines.py --opset 17  # solo LayerNorm fused
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import torch.nn as nn
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("baselines_transformer")

# Dimensioni usate in tutti i baseline transformer
SEQ_LEN = 64       # numero di token
EMBED_DIM = 128    # dimensione embedding
NUM_HEADS = 4      # teste attention
FFN_DIM = 256      # hidden dim FFN (2× EMBED_DIM)
BATCH = 1


# ═══════════════════════════════════════════════════════════════
#  Modelli Keras-style come nn.Module
# ═══════════════════════════════════════════════════════════════

class DenseModel(nn.Module):
    """Singolo layer Dense (Linear) — il 'connected' richiesto dai prof."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class LayerNormModel(nn.Module):
    """LayerNorm isolato — l'operatore che fallisce su MobileCLIP."""
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(EMBED_DIM)

    def forward(self, x):
        return self.ln(x)


class SoftmaxModel(nn.Module):
    """Softmax su tensore di attention weights — (B, H, T, T)."""
    def forward(self, x):
        return torch.softmax(x, dim=-1)


class MHAModel(nn.Module):
    """MultiHeadAttention con Q/K/V espliciti — esportabile in ONNX."""
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.k = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.v = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.out = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.scale = (EMBED_DIM // NUM_HEADS) ** -0.5

    def forward(self, x):
        B, T, D = x.shape
        d_h = D // NUM_HEADS
        q = self.q(x).reshape(B, T, NUM_HEADS, d_h).transpose(1, 2)
        k = self.k(x).reshape(B, T, NUM_HEADS, d_h).transpose(1, 2)
        v = self.v(x).reshape(B, T, NUM_HEADS, d_h).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        return self.out((attn @ v).transpose(1, 2).reshape(B, T, D))


class TransformerBlockModel(nn.Module):
    """Transformer Encoder Block: LayerNorm + MHA + LayerNorm + FFN."""
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.mha = nn.MultiheadAttention(EMBED_DIM, NUM_HEADS, batch_first=True)
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.ffn = nn.Sequential(
            nn.Linear(EMBED_DIM, FFN_DIM),
            nn.GELU(),
            nn.Linear(FFN_DIM, EMBED_DIM),
        )

    def forward(self, x):
        # MHA con residual
        x = x + self.mha(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        # FFN con residual
        x = x + self.ffn(self.ln2(x))
        return x


class ConvMHAHybridModel(nn.Module):
    """
    CNN + MHA Hybrid — replica struttura MobileCLIP image encoder:
    patch di immagine 8×8 → flatten → MHA (Q/K/V espliciti) → output.
    Input: (B, C, H, W) immagine 64×64×3
    """
    PATCH = 8

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, EMBED_DIM, kernel_size=self.PATCH, stride=self.PATCH)
        self.q = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.k = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.v = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.out_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.ln = nn.LayerNorm(EMBED_DIM)
        self.scale = (EMBED_DIM // NUM_HEADS) ** -0.5
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        patches = self.patch_embed(x)                             # (B, 128, 8, 8)
        B, C, H, W = patches.shape
        tokens = patches.flatten(2).transpose(1, 2)              # (B, 64, 128)
        T, D, d_h = tokens.shape[1], EMBED_DIM, EMBED_DIM // NUM_HEADS
        q = self.q(tokens).reshape(B, T, NUM_HEADS, d_h).transpose(1, 2)
        k = self.k(tokens).reshape(B, T, NUM_HEADS, d_h).transpose(1, 2)
        v = self.v(tokens).reshape(B, T, NUM_HEADS, d_h).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        tokens = self.out_proj((attn @ v).transpose(1, 2).reshape(B, T, D))
        tokens = self.ln(tokens)                                  # (B, 64, 128)
        return self.pool(tokens.transpose(1, 2)).squeeze(-1)      # (B, 128)


# ═══════════════════════════════════════════════════════════════
#  Export + Quantizzazione
# ═══════════════════════════════════════════════════════════════

class DummyDataReader(CalibrationDataReader):
    def __init__(self, fp32_path: str, dummy_input: np.ndarray, num_samples: int = 20):
        self.num_samples = num_samples
        self.current_sample = 0
        self.dummy_input = dummy_input
        session = onnxruntime.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
        self.input_name = session.get_inputs()[0].name

    def get_next(self):
        if self.current_sample < self.num_samples:
            self.current_sample += 1
            noise = (self.dummy_input + np.random.randn(*self.dummy_input.shape) * 0.1).astype(np.float32)
            return {self.input_name: noise}
        return None


def export_and_quantize(
    model: nn.Module,
    dummy_input: torch.Tensor,
    name: str,
    out_dir: Path,
    opset: int = 18,
) -> bool:
    log.info(f"Generando {name} (opset {opset}) ...")
    model.eval()

    fp32_path = out_dir / f"_tmp_{name}_fp32.onnx"
    out_path = out_dir / f"{name}_int8.onnx"

    # 1. Export ONNX FP32 — legacy exporter (dynamo=False) per controllo opset preciso
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(fp32_path),
                opset_version=opset,
                input_names=["input"],
                output_names=["output"],
                dynamo=False,
            )
    except Exception as e:
        log.error(f"  ❌ Export ONNX fallito: {e}")
        return False

    # 2. Quantizzazione INT8
    try:
        dummy_np = dummy_input.numpy()
        dr = DummyDataReader(str(fp32_path), dummy_np, num_samples=20)
        quantize_static(
            model_input=str(fp32_path),
            model_output=str(out_path),
            calibration_data_reader=dr,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
        )
        size_kb = out_path.stat().st_size / 1024
        log.info(f"  ✅ {out_path.name} ({size_kb:.1f} KB)")
    except Exception as e:
        log.error(f"  ❌ Quantizzazione fallita, salvo FP32 come fallback: {e}")
        fp32_path.rename(out_dir / f"{name}_fp32.onnx")
        return False
    finally:
        fp32_path.unlink(missing_ok=True)

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opset", type=int, default=17,
                        help="Opset ONNX principale (default 17 per LayerNorm fused)")
    args = parser.parse_args()

    out_dir = Path(__file__).parent.parent / "3_baseline" / "transformer"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dummy inputs ──────────────────────────────────────────
    dummy_seq = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)          # (1, 64, 128)
    dummy_attn_w = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, SEQ_LEN)  # (1, 4, 64, 64)
    dummy_img = torch.randn(BATCH, 3, 64, 64)                   # (1, 3, 64, 64)
    dummy_vec = torch.randn(BATCH, EMBED_DIM)                   # (1, 128)

    # ── Catalogo modelli ──────────────────────────────────────
    # Ogni entry: (nome_file, modello, dummy_input, opset)
    # Nota: torch 2.12 esporta nativo a opset 18.
    # LayerNorm opset 13 (decomposed) e opset 18 (fused) → test ipotesi dei prof.
    models_to_gen = [
        # 1. Dense/FC — il "connected" richiesto dai prof
        ("baseline_dense", DenseModel(), dummy_vec, 18),

        # 2. LayerNorm FUSED (opset 18) — un singolo nodo LayerNormalization
        #    Ipotesi dei prof: questo potrebbe essere accettato da AI Studio
        ("baseline_layernorm_fused", LayerNormModel(), dummy_seq, 18),

        # 3. Softmax su attention weights (B, H, T, T)
        ("baseline_softmax", SoftmaxModel(), dummy_attn_w, 18),

        # 4. MultiHeadAttention completo
        ("baseline_mha", MHAModel(), dummy_seq, 18),

        # 5. Transformer Encoder Block completo (LayerNorm + MHA + FFN)
        ("baseline_transformer_block", TransformerBlockModel(), dummy_seq, 18),

        # 6. Conv2D + MHA Hybrid — simula il patchifier dei ViT (come MobileCLIP)
        ("baseline_conv_mha_hybrid", ConvMHAHybridModel(), dummy_img, 18),
    ]

    results = {}
    for name, model, dummy, opset in models_to_gen:
        ok = export_and_quantize(model, dummy, name, out_dir, opset)
        results[name] = "OK" if ok else "FAIL"

    print("\n═══════════════════ RIEPILOGO BASELINE TRANSFORMER ═══════════════════")
    print(f"  Output: {out_dir}\n")
    print(f"  {'Modello':<45} {'Stato'}")
    print(f"  {'─'*45} {'─'*10}")
    for name, status in results.items():
        icon = "✅" if status == "OK" else "❌"
        print(f"  {icon}  {name:<43}")

    print(f"\n  Nota:")
    print(f"  opset 18 → LayerNorm FUSED (nodo singolo LayerNormalization)")
    print(f"  Ipotesi prof: ST AI Studio v4.0 supporta opset 18 e potrebbe accettare")
    print(f"  LayerNorm come nodo singolo — da confrontare con i FAIL di MobileCLIP.")
    print(f"\n  Carica questi .onnx in ST AI Studio per vedere quale op mappa su NPU!")


if __name__ == "__main__":
    main()
