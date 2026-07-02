"""
B.2 FIX v2 — UniFormer "ST-friendly": QKV split + layout NHWC per LN/token.

Quattro riscritture di puro layout/export (matematicamente identiche, stessi pesi):
  1. Attention: QKV fuso -> 3 Linear separate (elimina "Slice on batch dimension")
  2. PatchEmbed: flatten(2).transpose -> permute(0,2,3,1); LN sull'ultima dim
  3. SABlock: tokenizzazione via permute+reshape invece di flatten+transpose
  4. LayerNorm manuale con keepdim=True: litert decompone nn.LayerNorm con
     MEAN(keepdims=False) -> tensori rank-1 [3136] che l'importer ST padda a
     [1,1,1,3136] e non mappa su (BATCH,CH). Con keepdim la decomposizione
     resta 3D/4D e l'importer la accetta.

Uso: .venv312/bin/python Models/scripts/export_uniformer_stfriendly_tflite.py [small|base]
Output: tflite_native/uniformer_<v>_stfriendly_int8.tflite
"""
import sys, types
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent.parent  # 02_Test_INT4_Transformer_v4.0.1/
CACHE = ROOT / "Models" / ".cache" / "uniformer"
OUT = ROOT / "tflite_native"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(CACHE))
from uniformer import uniformer_small, uniformer_base


class ManualLayerNorm(nn.Module):
    """LayerNorm sull'ultima dim con keepdim=True (stessi pesi/eps di nn.LayerNorm).

    litert decompone nn.LayerNorm in MEAN(keepdims=False)+RESHAPE: i tensori
    rank-1 risultanti non passano l'importer ST. Qui la decomposizione resta
    alla stessa rank dell'input.
    """
    def __init__(self, ln: nn.LayerNorm):
        super().__init__()
        self.weight = nn.Parameter(ln.weight.data.clone())
        self.bias = nn.Parameter(ln.bias.data.clone())
        self.eps = ln.eps

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        xc = x - mu
        var = (xc * xc).mean(-1, keepdim=True)
        return xc * torch.rsqrt(var + self.eps) * self.weight + self.bias


def replace_layernorms(model):
    n = 0
    for mod in model.modules():
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.LayerNorm):
                setattr(mod, name, ManualLayerNorm(child))
                n += 1
    return n


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


def patchembed_forward_nhwc(self, x):
    """LN in layout NHWC: niente flatten a N=H*W token (stesso risultato)."""
    x = self.proj(x)
    x = x.permute(0, 2, 3, 1)      # B,H,W,C
    x = self.norm(x)               # LN su C (identico a LN sui token)
    x = x.permute(0, 3, 1, 2).contiguous()
    return x


def sablock_forward_nhwc(self, x):
    """Tokenizzazione via permute+reshape (stesso ordine dei token del flatten)."""
    x = x + self.pos_embed(x)
    B, C, H, W = x.shape
    x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
    x = x + self.drop_path(self.attn(self.norm1(x)))
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
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


def patch_layout(model):
    """PatchEmbed e SABlock in layout NHWC."""
    np_, ns = 0, 0
    for m in model.modules():
        if type(m).__name__ == "PatchEmbed":
            m.forward = types.MethodType(patchembed_forward_nhwc, m)
            np_ += 1
        elif type(m).__name__ == "SABlock":
            m.forward = types.MethodType(sablock_forward_nhwc, m)
            ns += 1
    return np_, ns


def build(variant):
    fn = uniformer_small if variant == "small" else uniformer_base
    pth = CACHE / f"uniformer_{variant}_in1k.pth"
    model = fn(pretrained=False)
    state = torch.load(str(pth), map_location="cpu", weights_only=False)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    return model.eval()


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "small"

    ref = build(variant)
    patched = build(variant)
    nblk = split_qkv(patched)
    np_, ns = patch_layout(patched)
    nln = replace_layernorms(patched)
    print(f"[{variant}] QKV split: {nblk} | PatchEmbed NHWC: {np_} | SABlock NHWC: {ns} | ManualLN: {nln}")

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        yr = ref(x)
        yp = patched(x)
    max_diff = (yr - yp).abs().max().item()
    print(f"max |diff| ref vs patched: {max_diff:.3e}  (atteso ~0)")
    assert max_diff < 1e-3, "La surgery ha cambiato il risultato!"

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
    path = OUT / f"uniformer_{variant}_stfriendly_int8.tflite"
    edge.export(str(path))
    print(f"OK -> {path}  ({path.stat().st_size/1e6:.2f} MB)")
