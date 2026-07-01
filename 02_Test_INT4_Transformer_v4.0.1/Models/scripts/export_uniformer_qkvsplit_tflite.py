"""
B.2 FIX — UniFormer attention senza slice sul batch.

Problema ST: "NOT IMPLEMENTED: Slice on batch dimension" perche' l'attention originale fa:
    qkv = self.qkv(x).reshape(B,N,3,heads,d).permute(2,0,3,1,4)
    q,k,v = qkv[0], qkv[1], qkv[2]      # <-- slice su dim-0 di un tensore 5D
Fix: QKV fuso -> 3 Linear separate (q_proj/k_proj/v_proj), reshape a 4D con
permute(0,2,1,3) (il batch resta dim-0, nessuno slice). Matematicamente identico.

Uso: .venv312/bin/python Models/scripts/export_uniformer_qkvsplit_tflite.py
Output: tflite_native/uniformer_small_qkvsplit_int8.tflite
"""
import sys, types
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent.parent  # 02_Test_INT4_Transformer_v4.0.1/
REPO = ROOT.parent
CACHE = REPO / "01_Presentazione" / "Models" / ".cache" / "uniformer"  # pesi condivisi
OUT = ROOT / "tflite_native"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(CACHE))
from uniformer import uniformer_small


def attn_forward_split(self, x):
    """Attention senza slice: q/k/v da Linear separate, tensori 4D."""
    B, N, C = x.shape
    h = self.num_heads
    d = C // h
    q = self.q_proj(x).reshape(B, N, h, d).permute(0, 2, 1, 3)
    k = self.k_proj(x).reshape(B, N, h, d).permute(0, 2, 1, 3)
    v = self.v_proj(x).reshape(B, N, h, d).permute(0, 2, 1, 3)
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    return x


def split_qkv(model):
    """Sostituisce ogni Attention.qkv fuso con q_proj/k_proj/v_proj separate."""
    n = 0
    for m in model.modules():
        if type(m).__name__ == "Attention" and hasattr(m, "qkv"):
            C = m.qkv.in_features
            has_b = m.qkv.bias is not None
            W = m.qkv.weight.data  # [3C, C] ordinato q|k|v
            q = nn.Linear(C, C, bias=has_b)
            k = nn.Linear(C, C, bias=has_b)
            v = nn.Linear(C, C, bias=has_b)
            q.weight.data = W[0:C].clone()
            k.weight.data = W[C:2 * C].clone()
            v.weight.data = W[2 * C:3 * C].clone()
            if has_b:
                b = m.qkv.bias.data
                q.bias.data = b[0:C].clone()
                k.bias.data = b[C:2 * C].clone()
                v.bias.data = b[2 * C:3 * C].clone()
            m.q_proj, m.k_proj, m.v_proj = q, k, v
            del m.qkv
            m.forward = types.MethodType(attn_forward_split, m)
            n += 1
    return n


def build(pth):
    model = uniformer_small(pretrained=False)
    state = torch.load(str(pth), map_location="cpu", weights_only=False)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    return model.eval()


if __name__ == "__main__":
    pth = CACHE / "uniformer_small_in1k.pth"

    # 1) modello originale (riferimento) e modello patchato
    ref = build(pth)
    patched = build(pth)
    nblk = split_qkv(patched)
    print(f"Attention block ristrutturati (QKV split): {nblk}")

    # 2) validazione numerica: stesso output entro tolleranza
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        yr = ref(x)
        yp = patched(x)
    max_diff = (yr - yp).abs().max().item()
    print(f"max |diff| ref vs patched: {max_diff:.3e}  (atteso ~0)")
    assert max_diff < 1e-3, "La surgery ha cambiato il risultato!"

    # 3) export INT8 TFLite via litert-torch
    import litert_torch
    from litert_torch.quantize.pt2e_quantizer import PT2EQuantizer, get_symmetric_quantization_config
    from litert_torch.quantize.quant_config import QuantConfig
    from torchao.quantization.pt2e import quantize_pt2e

    sample = (torch.randn(1, 3, 224, 224),)
    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False))
    exported = torch.export.export(patched, sample).module()
    prepared = quantize_pt2e.prepare_pt2e(exported, quantizer)
    for _ in range(16):
        prepared(*[torch.rand_like(s) for s in sample])
    converted = quantize_pt2e.convert_pt2e(prepared, fold_quantize=False)
    edge = litert_torch.convert(converted, sample, quant_config=QuantConfig(pt2e_quantizer=quantizer))
    path = OUT / "uniformer_small_qkvsplit_int8.tflite"
    edge.export(str(path))
    print(f"OK -> {path}  ({path.stat().st_size/1e6:.2f} MB)")
