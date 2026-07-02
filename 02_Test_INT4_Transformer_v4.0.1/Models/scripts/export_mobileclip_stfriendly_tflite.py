"""
B.2 FIX — MobileCLIP image encoder "ST-friendly" (stessa ricetta di UniFormer).

Riscritture di puro export (matematicamente identiche, stessi pesi):
  1. mci.MHSA (S0): qkv fuso + unbind(0) su tensore 5D -> "Slice on batch dim".
     Fix: 3 Linear separate + tokenizzazione via permute+reshape.
  2. transformer.MultiHeadAttention (B): qkv[:, :, i] su 5D -> "Slice on > 3 dim".
     Fix: 3 Linear separate, tensori sempre 4D.
  3. nn.LayerNorm -> ManualLayerNorm keepdim=True (l'importer ST non mappa i
     tensori rank-1 della decomposizione MEAN di litert).

Uso: .venv312/bin/python Models/scripts/export_mobileclip_stfriendly_tflite.py [s0|b]
Output: tflite_native/mobileclip_<v>_image_stfriendly_int8.tflite
"""
import sys, types
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent.parent  # 02_Test_INT4_Transformer_v4.0.1/
OUT = ROOT / "tflite_native"; OUT.mkdir(exist_ok=True)

import mobileclip
from mobileclip.models.mci import MHSA
from mobileclip.modules.common.transformer import MultiHeadAttention


class ManualLayerNorm(nn.Module):
    """LayerNorm sull'ultima dim con keepdim=True (stessi pesi/eps)."""
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


def _make_split(linear_qkv: nn.Linear, C: int):
    """Da un Linear fuso [3C, C] a tre Linear [C, C] (ordine q|k|v)."""
    has_b = linear_qkv.bias is not None
    W = linear_qkv.weight.data
    outs = []
    for i in range(3):
        l = nn.Linear(C, C, bias=has_b)
        l.weight.data = W[i * C:(i + 1) * C].clone()
        if has_b:
            l.bias.data = linear_qkv.bias.data[i * C:(i + 1) * C].clone()
        outs.append(l)
    return outs


import os
TOKENIZE = os.environ.get("TOKENIZE", "permute")  # permute | flatten


def mhsa_forward_split(self, x):
    """mci.MHSA senza slice 5D: input (B,C,H,W), token via permute+reshape."""
    B, C, H, W = x.shape
    N = H * W
    h, d = self.num_heads, self.head_dim
    if TOKENIZE == "flatten":
        t = torch.flatten(x, start_dim=2).transpose(-2, -1)
    else:
        t = x.permute(0, 2, 3, 1).reshape(B, N, C)
    q = self.q_proj(t).reshape(B, N, h, d).permute(0, 2, 1, 3)
    k = self.k_proj(t).reshape(B, N, h, d).permute(0, 2, 1, 3)
    v = self.v_proj(t).reshape(B, N, h, d).permute(0, 2, 1, 3)
    attn = (q * self.scale) @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    out = (attn @ v).transpose(1, 2).reshape(B, N, C)
    out = self.proj(out)
    out = self.proj_drop(out)
    if TOKENIZE == "flatten":
        return out.transpose(-2, -1).reshape(B, C, H, W)
    return out.reshape(B, H, W, C).permute(0, 3, 1, 2)


def mha_forward_split(self, x_q, x_kv=None, key_padding_mask=None,
                      attn_mask=None, *args, **kwargs):
    """transformer.MultiHeadAttention (solo self-attention, senza maschere)."""
    B, S, C = x_q.shape
    h, d = self.num_heads, self.head_dim
    q = self.q_proj(x_q).reshape(B, S, h, d).transpose(1, 2)
    k = self.k_proj(x_q).reshape(B, S, h, d).transpose(1, 2)
    v = self.v_proj(x_q).reshape(B, S, h, d).transpose(1, 2)
    attn = (q * self.scaling) @ k.transpose(-1, -2)
    attn = self.softmax(attn)
    attn = self.attn_dropout(attn)
    out = (attn @ v).transpose(1, 2).reshape(B, S, C)
    return self.out_proj(out)


def split_attention(enc):
    nm, nh = 0, 0
    for m in enc.modules():
        if isinstance(m, MHSA):
            C = m.qkv.in_features
            m.q_proj, m.k_proj, m.v_proj = _make_split(m.qkv, C)
            del m.qkv
            m.forward = types.MethodType(mhsa_forward_split, m)
            nm += 1
        elif isinstance(m, MultiHeadAttention):
            C = m.embed_dim
            m.q_proj, m.k_proj, m.v_proj = _make_split(m.qkv_proj, C)
            del m.qkv_proj
            m.forward = types.MethodType(mha_forward_split, m)
            nh += 1
    return nm, nh


class FrozenPosEmbed(nn.Module):
    """Positional embedding precalcolato: a input fisso, pos_embed(N) e' una
    costante. Evita il RESIZE_BILINEAR che fa crashare l'importer ST
    ('list index out of range')."""
    def __init__(self, pos):
        super().__init__()
        self.register_buffer("pos", pos)

    def forward(self, seq_len, *args, **kwargs):
        return self.pos


def freeze_pos_embed(enc, img_size=256):
    vit = getattr(enc, "model", None)
    if vit is None or not hasattr(vit, "pos_embed"):
        return False
    with torch.no_grad():
        p = vit.patch_emb(torch.zeros(1, 3, img_size, img_size))
        n_patches = p.shape[-2] * p.shape[-1]
        pos = vit.pos_embed(n_patches).detach().clone()
    vit.pos_embed = FrozenPosEmbed(pos)
    return True


def find_pretrained(variant):
    for p in (Path.home() / ".cache" / "mobileclip" / f"mobileclip_{variant}.pt",
              ROOT / "Models" / ".cache" / "mobileclip" / f"mobileclip_{variant}.pt"):
        if p.exists():
            return str(p)
    return None


def build(variant):
    pt = find_pretrained(variant)
    model, _, _ = mobileclip.create_model_and_transforms(
        f"mobileclip_{variant}", pretrained=pt, reparameterize=True)
    print(f"pretrained: {pt or 'NON TROVATO (pesi random, verdetto compile comunque valido)'}")
    enc = getattr(model, "image_encoder", None) or model.visual
    return enc.eval()


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "s0"

    import copy
    ref = build(variant)
    patched = copy.deepcopy(ref)
    nm, nh = split_attention(patched)
    nln = replace_layernorms(patched)
    npos = freeze_pos_embed(patched)
    print(f"[{variant}] MHSA split: {nm} | MHA split: {nh} | ManualLN: {nln} | FrozenPos: {npos}")

    x = torch.randn(1, 3, 256, 256)
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

    sample = (torch.randn(1, 3, 256, 256),)
    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False))
    exported = torch.export.export(patched, sample).module()
    prepared = quantize_pt2e.prepare_pt2e(exported, quantizer)
    for _ in range(16):
        prepared(*[torch.rand_like(s) for s in sample])
    converted = quantize_pt2e.convert_pt2e(prepared, fold_quantize=False)
    edge = litert_torch.convert(converted, sample, quant_config=QuantConfig(pt2e_quantizer=quantizer))
    path = OUT / f"mobileclip_{variant}_image_stfriendly_int8.tflite"
    edge.export(str(path))
    print(f"OK -> {path}  ({path.stat().st_size/1e6:.2f} MB)")
