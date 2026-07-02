"""
B.3 FIX — baseline_transformer_block in TFLite nativo (litert-torch).

L'ONNX INT8 fa crashare il compilatore ST ('NoneType' not subscriptable, P9),
anche con --expand-softmax. Qui due varianti TFLite native:
  - plain:      blocco identico all'ONNX (nn.MultiheadAttention fusa)
  - stfriendly: stessi pesi, ma Q/K/V espliciti (dai pesi in_proj della MHA)
                + LayerNorm manuale keepdim=True (ricetta UniFormer/MobileCLIP)

Stessa architettura del generatore ONNX: LN+MHA+LN+FFN, dim 128, 4 teste,
input (1, 64, 128).

Uso: .venv312/bin/python Models/scripts/export_baseline_block_tflite.py
"""
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "tflite_native"; OUT.mkdir(exist_ok=True)

SEQ_LEN, EMBED_DIM, NUM_HEADS, FFN_DIM, BATCH = 64, 128, 4, 256, 1


class TransformerBlockModel(nn.Module):
    """Identico a generate_transformer_baselines.py."""
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
        x = x + self.mha(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ffn(self.ln2(x))
        return x


class ManualLayerNorm(nn.Module):
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


class ExplicitMHA(nn.Module):
    """Self-attention con Q/K/V espliciti, pesi presi da nn.MultiheadAttention."""
    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        D = mha.embed_dim
        self.h = mha.num_heads
        self.d = D // self.h
        self.scale = self.d ** -0.5
        W, b = mha.in_proj_weight.data, mha.in_proj_bias.data
        self.q = nn.Linear(D, D); self.k = nn.Linear(D, D); self.v = nn.Linear(D, D)
        self.q.weight.data, self.q.bias.data = W[:D].clone(), b[:D].clone()
        self.k.weight.data, self.k.bias.data = W[D:2*D].clone(), b[D:2*D].clone()
        self.v.weight.data, self.v.bias.data = W[2*D:].clone(), b[2*D:].clone()
        self.out = nn.Linear(D, D)
        self.out.weight.data = mha.out_proj.weight.data.clone()
        self.out.bias.data = mha.out_proj.bias.data.clone()

    def forward(self, x):
        B, T, D = x.shape
        q = self.q(x).reshape(B, T, self.h, self.d).transpose(1, 2)
        k = self.k(x).reshape(B, T, self.h, self.d).transpose(1, 2)
        v = self.v(x).reshape(B, T, self.h, self.d).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        return self.out((attn @ v).transpose(1, 2).reshape(B, T, D))


class BlockSTFriendly(nn.Module):
    """Stessi pesi del blocco plain, forma ST-friendly."""
    def __init__(self, src: TransformerBlockModel):
        super().__init__()
        self.ln1 = ManualLayerNorm(src.ln1)
        self.mha = ExplicitMHA(src.mha)
        self.ln2 = ManualLayerNorm(src.ln2)
        self.ffn = src.ffn

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


def to_int8_tflite(module, sample, name):
    import litert_torch
    from litert_torch.quantize.pt2e_quantizer import PT2EQuantizer, get_symmetric_quantization_config
    from litert_torch.quantize.quant_config import QuantConfig
    from torchao.quantization.pt2e import quantize_pt2e
    module = module.eval()
    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False))
    exported = torch.export.export(module, sample).module()
    prepared = quantize_pt2e.prepare_pt2e(exported, quantizer)
    for _ in range(16):
        prepared(*[torch.rand_like(s) for s in sample])
    converted = quantize_pt2e.convert_pt2e(prepared, fold_quantize=False)
    edge = litert_torch.convert(converted, sample, quant_config=QuantConfig(pt2e_quantizer=quantizer))
    path = OUT / f"{name}_int8.tflite"
    edge.export(str(path))
    print(f"OK {name} -> {path} ({path.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    torch.manual_seed(0)
    plain = TransformerBlockModel().eval()
    friendly = BlockSTFriendly(plain).eval()

    x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
    with torch.no_grad():
        d = (plain(x) - friendly(x)).abs().max().item()
    print(f"max |diff| plain vs stfriendly: {d:.3e}")
    assert d < 1e-4

    sample = (torch.randn(BATCH, SEQ_LEN, EMBED_DIM),)
    for name, m in (("baseline_tf_block_plain", plain),
                    ("baseline_tf_block_stfriendly", friendly)):
        try:
            to_int8_tflite(m, sample, name)
        except Exception as e:
            print(f"FAIL {name}: {e}")
