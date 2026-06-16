#!/usr/bin/env python3
"""
Download and export CV models from CV_Models.csv to ONNX format.

Automatizza il download dei pesi pre-addestrati e l'export in formato ONNX
per tutti i modelli elencati nel file CV_Models.csv.

I modelli sono organizzati in 3 livelli di complessità:

  Tier 1 (completamente automatico):
    resnet34, yolov8n, yolov11n/s/m/l/x

  Tier 2 (semi-automatico, potrebbe richiedere download manuale dei pesi):
    yolov7, pidnet_s, mobileclip_s0, mobileclip_b

  Tier 3 (complesso, potrebbe richiedere setup aggiuntivo):
    rtmdet_l, fastinst_d1, uniformer_t/s/m/l

Uso:
    # Esporta tutti i modelli
    python download_and_export_models.py --all

    # Esporta modelli specifici
    python download_and_export_models.py --models resnet34 yolov8n yolov11n

    # Esporta solo modelli Tier 1 (facili)
    python download_and_export_models.py --all --tier 1

    # Mostra modelli disponibili
    python download_and_export_models.py --list

Dipendenze:
    pip install -r requirements_export.txt

Output:
    I modelli vengono esportati in ./Models/exported/<nome_modello>.onnx
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR / "exported"
CACHE_DIR = SCRIPT_DIR / ".cache"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("export")


# ---------------------------------------------------------------------------
# Funzioni di utilità
# ---------------------------------------------------------------------------
def ensure_dir(p: Path):
    """Crea una directory se non esiste."""
    p.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, desc: str = ""):
    """Scarica un file da URL se non esiste già in cache."""
    if dest.exists():
        log.info(f"  Cache hit: {dest.name}")
        return
    log.info(f"  Download {desc or dest.name} ...")
    ensure_dir(dest.parent)
    urllib.request.urlretrieve(url, str(dest))
    size_mb = dest.stat().st_size / (1024 * 1024)
    log.info(f"  Scaricato: {dest.name} ({size_mb:.1f} MB)")


def clone_repo(url: str, dest: Path):
    """Shallow-clone di un repository git se non esiste già."""
    if dest.exists():
        log.info(f"  Repo in cache: {dest.name}")
        return
    log.info(f"  Cloning {url} ...")
    ensure_dir(dest.parent)
    subprocess.check_call(
        ["git", "clone", "--depth", "1", url, str(dest)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log.info(f"  Clonato: {dest.name}")


def verify_onnx(path: str) -> bool:
    """Verifica che un modello ONNX sia valido."""
    import onnx

    model = onnx.load(path, load_external_data=True)
    onnx.checker.check_model(model)
    size_mb = os.path.getsize(path) / (1024 * 1024)

    # Mostra info input/output
    graph = model.graph
    inputs = [f"{i.name}: {[d.dim_value for d in i.type.tensor_type.shape.dim]}" for i in graph.input]
    outputs = [o.name for o in graph.output]
    log.info(f"  ✅ Verificato: {Path(path).name} ({size_mb:.1f} MB)")
    log.info(f"     Input:  {inputs}")
    log.info(f"     Output: {outputs}")
    return True


def consolidate_onnx(path: str) -> str:
    """
    Se l'export ha generato file external data (.onnx.data), li consolida
    in un singolo file ONNX self-contained ed elimina i file .data orfani.
    """
    import onnx

    data_file = path + ".data"
    if not os.path.exists(data_file):
        return path  # già self-contained

    log.info(f"  Consolidamento external data: {Path(path).name} ...")
    model = onnx.load(path, load_external_data=True)
    tmp = path + ".tmp"
    onnx.save(model, tmp, save_as_external_data=False)
    onnx.checker.check_model(tmp)
    os.replace(tmp, path)
    os.remove(data_file)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    log.info(f"  -> {Path(path).name} ({size_mb:.1f} MB) self-contained ✅")
    return path


# ===================================================================
#  TIER 1 — Completamente automatici
# ===================================================================

def export_resnet34() -> str:
    """
    ResNet34 · ImageNet1K · 21.8M params
    Input: 1×3×224×224
    Fonte: torchvision (download automatico dei pesi)
    """
    import torch
    import torchvision.models as models

    log.info("Export ResNet34 ...")

    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    model.eval()

    out = str(OUTPUT_DIR / "resnet34.onnx")
    with torch.no_grad():
        torch.onnx.export(
            model,
            torch.randn(1, 3, 224, 224),
            out,
            opset_version=13,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
    consolidate_onnx(out)
    verify_onnx(out)
    return out


def export_yolov8n() -> str:
    """
    YOLOv8-n · COCO 2017 · 3.2M params
    Input: 1×3×640×640
    Fonte: ultralytics (download automatico)
    """
    from ultralytics import YOLO

    log.info("Export YOLOv8-n ...")

    model = YOLO("yolov8n.pt")
    result = model.export(format="onnx", imgsz=640, opset=13)
    result = str(result)

    dest = str(OUTPUT_DIR / "yolov8n.onnx")
    if os.path.abspath(result) != os.path.abspath(dest):
        shutil.copy2(result, dest)
    verify_onnx(dest)
    return dest


def _export_yolov11(variant: str) -> str:
    """
    YOLOv11 variant · COCO 2017
    Input: 1×3×640×640
    Fonte: ultralytics (download automatico)
    """
    from ultralytics import YOLO

    log.info(f"Export YOLOv11-{variant} ...")

    model = YOLO(f"yolo11{variant}.pt")
    result = model.export(format="onnx", imgsz=640, opset=13)
    result = str(result)

    dest = str(OUTPUT_DIR / f"yolov11{variant}.onnx")
    if os.path.abspath(result) != os.path.abspath(dest):
        shutil.copy2(result, dest)
    verify_onnx(dest)
    return dest


def export_yolov11n() -> str:
    """YOLOv11-n · 2.6M params"""
    return _export_yolov11("n")


def export_yolov11s() -> str:
    """YOLOv11-s · 9.4M params"""
    return _export_yolov11("s")


def export_yolov11m() -> str:
    """YOLOv11-m · 20.1M params"""
    return _export_yolov11("m")


def export_yolov11l() -> str:
    """YOLOv11-l · 25.3M params"""
    return _export_yolov11("l")


def export_yolov11x() -> str:
    """YOLOv11-x · 56.9M params"""
    return _export_yolov11("x")


# ===================================================================
#  TIER 2 — Semi-automatici
# ===================================================================

def export_yolov7() -> str:
    """
    YOLOv7 · COCO 2017 · 36.9M params
    Input: 1×3×640×640

    Richiede clone del repo ufficiale perché i pesi (pickle)
    fanno riferimento ai moduli del codebase.
    """
    import torch

    log.info("Export YOLOv7 ...")

    repo = CACHE_DIR / "yolov7"
    clone_repo("https://github.com/WongKinYiu/yolov7.git", repo)

    weights = CACHE_DIR / "yolov7.pt"
    download_file(
        "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt",
        weights,
        "YOLOv7 weights",
    )

    # Servono i moduli del repo per caricare il checkpoint
    sys.path.insert(0, str(repo))
    try:
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
        model = ckpt["model"].float().eval()

        out = str(OUTPUT_DIR / "yolov7.onnx")
        with torch.no_grad():
            torch.onnx.export(
                model,
                torch.randn(1, 3, 640, 640),
                out,
                opset_version=13,
                input_names=["images"],
                output_names=["output"],
            )
        verify_onnx(out)
        return out
    finally:
        sys.path.remove(str(repo))


def export_pidnet_s() -> str:
    """
    PIDNet-S · Cityscapes · 7.6M params
    Input: 1×3×1024×1024

    Strategia di download pesi (in ordine di priorità):
      1. HuggingFace Hub (oenpu/PIDNet_S_enlight_friendly_onnx) — ONNX precompilato
      2. Google Drive (link originale PIDNet README) — potrebbe non funzionare

    Il modello ONNX di HuggingFace è già esportato e verificato.
    Se vuoi rigenerare l'ONNX dai pesi originali, scarica manualmente:
      https://github.com/XuJiacong/PIDNet
    e posiziona i pesi in:
      Models/.cache/PIDNet/pretrained_models/cityscapes/PIDNet_S_Cityscapes_test.pt
    """
    import shutil

    log.info("Export PIDNet-S ...")

    out = str(OUTPUT_DIR / "pidnet_s.onnx")

    # --- Strategia 1: ONNX precompilato da HuggingFace ---
    try:
        from huggingface_hub import hf_hub_download

        log.info("  Download ONNX precompilato da HuggingFace ...")
        hf_path = hf_hub_download(
            repo_id="oenpu/PIDNet_S_enlight_friendly_onnx",
            filename="PIDNet_S_enlight_friendly.onnx",
        )
        shutil.copy2(hf_path, out)
        verify_onnx(out)
        return out
    except Exception as e:
        log.warning(f"  HuggingFace fallito: {e}. Tentativo fallback Google Drive...")

    # --- Strategia 2: Export da pesi originali (Google Drive) ---
    import torch

    repo = CACHE_DIR / "PIDNet"
    clone_repo("https://github.com/XuJiacong/PIDNet.git", repo)

    weights_dir = repo / "pretrained_models" / "cityscapes"
    ensure_dir(weights_dir)
    weights = weights_dir / "PIDNet_S_Cityscapes_test.pt"

    if not weights.exists():
        log.info("  Tentativo download pesi da Google Drive ...")
        try:
            import gdown
            gdown.download(
                id="1hIBp_8maRr60-B3PF0NVtaA6TYBvO4y-",
                output=str(weights),
                quiet=False,
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Download automatico dei pesi PIDNet-S fallito: {e}\n"
                f"Scaricare manualmente dal README di PIDNet e salvare in:\n"
                f"  {weights}"
            )

    if not weights.exists():
        raise FileNotFoundError(f"Pesi non trovati: {weights}")

    sys.path.insert(0, str(repo))
    try:
        from models.pidnet import get_pred_model

        model = get_pred_model("pidnet-s", 19)  # 19 classi Cityscapes
        state = torch.load(str(weights), map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval()

        with torch.no_grad():
            torch.onnx.export(
                model,
                torch.randn(1, 3, 1024, 2048),
                out,
                opset_version=13,
                input_names=["input"],
                output_names=["output"],
            )
        consolidate_onnx(out)
        verify_onnx(out)
        return out
    finally:
        sys.path.remove(str(repo))


def _export_mobileclip(variant: str) -> str:
    """
    MobileCLIP · DataCompDR
    Input image: 1×3×256×256

    Esporta l'image encoder e il text encoder separatamente.
    Richiede il package 'mobileclip' di Apple:
      pip install git+https://github.com/apple/ml-mobileclip.git
    """
    import torch

    variant_label = variant.upper()
    log.info(f"Export MobileCLIP-{variant_label} ...")

    # --- Caricamento modello ---
    try:
        import mobileclip

        model_name = f"mobileclip_{variant}"
        model, _, preprocess = mobileclip.create_model_and_transforms(
            model_name, pretrained=f"/Users/matteo/.cache/mobileclip/{model_name}.pt"
        )
        model.eval()
    except ImportError:
        raise ImportError(
            "Package 'mobileclip' non trovato. Installare con:\n"
            "  pip install git+https://github.com/apple/ml-mobileclip.git"
        )
    except Exception:
        # Fallback: prova open_clip se mobileclip non funziona
        try:
            import open_clip

            variant_map = {"s0": "MobileCLIP-S0", "b": "MobileCLIP-B-LT"}
            model_name = variant_map.get(variant, f"MobileCLIP-{variant_label}")
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained="datacompdr"
            )
            model.eval()
        except Exception as e2:
            raise RuntimeError(
                f"Impossibile caricare MobileCLIP-{variant_label}.\n"
                f"Provato mobileclip e open_clip, entrambi falliti.\n"
                f"Ultimo errore: {e2}\n"
                f"Installare: pip install git+https://github.com/apple/ml-mobileclip.git"
            )

    # --- Export image encoder ---
    out_img = str(OUTPUT_DIR / f"mobileclip_{variant}_image.onnx")

    # L'attributo per l'image encoder varia tra versioni del package
    if hasattr(model, "image_encoder"):
        img_enc = model.image_encoder
    elif hasattr(model, "visual"):
        img_enc = model.visual
    else:
        raise AttributeError(
            "Impossibile trovare l'image encoder nel modello MobileCLIP. "
            "Attributi disponibili: " + str([a for a in dir(model) if not a.startswith("_")])
        )

    with torch.no_grad():
        dummy_image = torch.randn(1, 3, 256, 256)
        torch.onnx.export(
            img_enc,
            dummy_image,
            out_img,
            opset_version=13,
            input_names=["image"],
            output_names=["image_features"],
        )
    consolidate_onnx(out_img)
    verify_onnx(out_img)

    # --- Export text encoder (best effort) ---
    out_txt = str(OUTPUT_DIR / f"mobileclip_{variant}_text.onnx")
    try:
        if hasattr(model, "text_encoder"):
            txt_enc = model.text_encoder
        elif hasattr(model, "text"):
            txt_enc = model.text
        else:
            log.warning("  ⚠️  Text encoder non trovato, skip export testo")
            return out_img

        dummy_text = torch.randint(0, 49408, (1, 77))
        with torch.no_grad():
            torch.onnx.export(
                txt_enc,
                dummy_text,
                out_txt,
                opset_version=13,
                input_names=["text"],
                output_names=["text_features"],
            )
        consolidate_onnx(out_txt)
        verify_onnx(out_txt)
        log.info(f"  Image encoder: {out_img}")
        log.info(f"  Text encoder:  {out_txt}")
    except Exception as e:
        log.warning(f"  ⚠️  Export text encoder fallito: {e}")
        log.warning("  (Image encoder esportato con successo)")

    return out_img


def export_mobileclip_s0() -> str:
    """MobileCLIP-S0 · 11.4M+42.4M params (image+text)"""
    return _export_mobileclip("s0")


def export_mobileclip_b() -> str:
    """MobileCLIP-B · 86.3M+63.4M params (image+text)"""
    return _export_mobileclip("b")


# ===================================================================
#  TIER 3 — Complessi / Richiedono setup aggiuntivo
# ===================================================================

def export_rtmdet_l() -> str:
    """
    RTMDet-l · COCO 2017 · 52.3M params
    Input: 1×3×640×640

    Parte dell'ecosistema MMDetection. Richiede mmdet, mmengine, mmcv.
    Installazione:
      pip install -U openmim
      mim install mmengine mmcv mmdet

    ⚠️  L'export ONNX diretto da mmdet è complicato. Per un export affidabile
    si consiglia mmdeploy.
    """
    import torch

    log.info("Export RTMDet-l ...")

    try:
        from mmdet.apis import init_detector
        import mmdet
    except ImportError:
        raise ImportError(
            "mmdet non trovato. Installare con:\n"
            "  pip install -U openmim\n"
            "  mim install mmengine mmcv mmdet"
        )

    # --- Download checkpoint ---
    ckpt_url = (
        "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
        "rtmdet_l_8xb32-300e_coco/"
        "rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"
    )
    ckpt_path = CACHE_DIR / "rtmdet_l.pth"
    download_file(ckpt_url, ckpt_path, "RTMDet-l checkpoint")

    # --- Trova config file ---
    mmdet_root = Path(mmdet.__file__).parent
    config_candidates = [
        mmdet_root / ".mim" / "configs" / "rtmdet" / "rtmdet_l_8xb32-300e_coco.py",
        mmdet_root / "configs" / "rtmdet" / "rtmdet_l_8xb32-300e_coco.py",
        mmdet_root.parent / "configs" / "rtmdet" / "rtmdet_l_8xb32-300e_coco.py",
    ]

    config_path = None
    for candidate in config_candidates:
        if candidate.exists():
            config_path = candidate
            break

    if config_path is None:
        # Prova a scaricarlo dal repo
        config_url = (
            "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/"
            "configs/rtmdet/rtmdet_l_8xb32-300e_coco.py"
        )
        config_path = CACHE_DIR / "rtmdet_l_config.py"
        if not config_path.exists():
            download_file(config_url, config_path, "RTMDet-l config")

    log.info(f"  Config: {config_path}")

    # --- Tentativo export ---
    # I modelli mmdet hanno un'interfaccia forward complessa.
    # L'export diretto spesso fallisce. Proviamo comunque.
    try:
        model = init_detector(str(config_path), str(ckpt_path), device="cpu")
        model.eval()

        # MMDet wrappa il modello, dobbiamo accedere al backbone
        out = str(OUTPUT_DIR / "rtmdet_l.onnx")

        # Prova export con il wrapper di mmdet per ONNX se disponibile
        try:
            from mmdet.utils import register_all_modules
            register_all_modules()
        except ImportError:
            pass

        # Export diretto — potrebbe fallire per custom ops
        dummy = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy,
                out,
                opset_version=13,
                input_names=["input"],
                output_names=["output"],
            )
        verify_onnx(out)
        return out

    except Exception as e:
        log.error(f"  Export diretto fallito: {e}")
        raise RuntimeError(
            f"Export ONNX diretto di RTMDet-l fallito.\n"
            f"Errore: {e}\n"
            f"\n"
            f"Per un export affidabile, usare mmdeploy:\n"
            f"  pip install mmdeploy mmdeploy-runtime-onnxruntime\n"
            f"  python -m mmdeploy.tools.deploy \\\n"
            f"    detection_onnxruntime_static.py \\\n"
            f"    {config_path} \\\n"
            f"    {ckpt_path} \\\n"
            f"    demo.jpg \\\n"
            f"    --work-dir {OUTPUT_DIR}/rtmdet_l_mmdeploy"
        )


def export_fastinst_d1() -> str:
    """
    FastInst-D1 · COCO 2017 · 30M params
    Input: 1×3×576×576

    ⚠️  LIMITAZIONE: FastInst usa Deformable Convolutions (DCN) che NON sono
    supportate nell'export ONNX standard. L'export probabilmente fallirà.

    Backbone: ResNet50-DCN
    Framework: Detectron2 / Mask2Former

    Alternative:
    1. TensorRT supporta DCN nativamente
    2. Sostituire i layer DCN con Conv2D standard (perde accuratezza)
    3. Usare il tracing di Detectron2 (supporto limitato)
    """
    log.info("Export FastInst-D1 ...")

    try:
        import detectron2  # noqa: F401
    except ImportError:
        raise ImportError(
            "detectron2 non trovato. Installare con:\n"
            "  pip install 'git+https://github.com/facebookresearch/detectron2.git'\n"
            "\n"
            "⚠️  NOTA: FastInst-D1 usa DCN (Deformable Convolutions) che\n"
            "non sono supportate in ONNX standard. L'export potrebbe\n"
            "comunque fallire. Considerare le alternative elencate sopra."
        )

    # Clone FastInst repo
    repo = CACHE_DIR / "FastInst"
    clone_repo("https://github.com/junjiehe96/FastInst.git", repo)

    raise NotImplementedError(
        "FastInst-D1 usa Deformable Convolutions (DCN) che non sono\n"
        "supportate nell'export ONNX standard.\n"
        "\n"
        "Opzioni disponibili:\n"
        "\n"
        "1. Export TensorRT (supporta DCN nativamente):\n"
        "   Usare torch2trt o il toolkit TensorRT di NVIDIA\n"
        "\n"
        "2. Export con Detectron2 tracing:\n"
        "   from detectron2.export import TracingAdapter\n"
        f"   Vedere: {repo}/README.md\n"
        "\n"
        "3. Sostituire DCN con Conv2D standard:\n"
        "   Modificare il backbone ResNet50-DCN per usare Conv2D normali\n"
        "   (non richiede re-training per test di latenza)\n"
        "\n"
        "Repo: https://github.com/junjiehe96/FastInst"
    )


def _export_uniformer(variant: str) -> str:
    """
    UniFormer · ImageNet1K · Input 1×3×224×224

    Il CSV elenca varianti custom leggere (t/s/m/l) con 1.8M-10M params.
    I modelli UniFormer standard sono più grandi (S: 22M, B: 50M).

    Strategia:
    1. Prova timm (se il modello è disponibile)
    2. Fallback: clone del repo ufficiale

    Nota: per test di latenza su NPU, i pesi random sono sufficienti
    (la latenza non dipende dai valori dei pesi).
    """
    import torch

    log.info(f"Export UniFormer-{variant} ...")

    # --- Tentativo 1: timm ---
    try:
        import timm

        available = timm.list_models("uniformer*")
        log.info(f"  Modelli UniFormer in timm: {available if available else 'nessuno'}")

        # Mappa le varianti del CSV ai nomi timm
        timm_map = {
            "t": ["uniformer_tiny", "uniformer_small"],
            "s": ["uniformer_small", "uniformer_small_plus"],
            "m": ["uniformer_base", "uniformer_base_ls"],
            "l": ["uniformer_large", "uniformer_base"],
        }

        model = None
        candidates = timm_map.get(variant, ["uniformer_small"])
        for candidate in candidates:
            if candidate in available:
                log.info(f"  Usando timm model: {candidate}")
                model = timm.create_model(candidate, pretrained=True)
                break

        if model is None:
            # Prova il primo candidato senza pretrained
            for candidate in candidates:
                if candidate in available:
                    log.info(f"  Usando timm model (senza pretrained): {candidate}")
                    model = timm.create_model(candidate, pretrained=False)
                    break

        if model is not None:
            model.eval()
            out = str(OUTPUT_DIR / f"uniformer_{variant}.onnx")
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    torch.randn(1, 3, 224, 224),
                    out,
                    opset_version=13,
                    input_names=["input"],
                    output_names=["output"],
                )
            verify_onnx(out)

            # Avviso: il modello timm potrebbe non corrispondere esattamente
            actual_params = sum(p.numel() for p in model.parameters()) / 1e6
            csv_params = {"t": 1.8, "s": 2.4, "m": 5.6, "l": 10.0}
            log.warning(
                f"  ⚠️  Modello timm ha {actual_params:.1f}M params "
                f"(CSV indica {csv_params.get(variant, '?')}M)."
            )
            log.warning(
                f"  Se i params non corrispondono, le varianti del CSV sono"
                f" configurazioni custom più leggere."
            )
            return out

        log.info("  Nessun modello UniFormer trovato in timm")
        raise ImportError("UniFormer non disponibile in timm")

    except ImportError as e:
        log.info(f"  Approccio timm non riuscito: {e}")

    # --- Tentativo 2: repo ufficiale ---
    repo = CACHE_DIR / "UniFormer"
    clone_repo("https://github.com/Sense-X/UniFormer.git", repo)

    # Il repo ha sottodirectory per task diversi
    model_dir = repo / "image_classification"
    if not model_dir.exists():
        # Struttura del repo potrebbe variare
        model_dir = repo

    sys.path.insert(0, str(model_dir))
    try:
        # Prova ad importare il modulo del modello
        try:
            from models.uniformer import uniformer_small, uniformer_base
        except ImportError:
            try:
                from uniformer import uniformer_small, uniformer_base
            except ImportError:
                raise ImportError(
                    f"Impossibile importare modelli UniFormer dal repo.\n"
                    f"Verificare la struttura del repo in: {repo}"
                )

        factory_map = {
            "t": uniformer_small,   # Più vicino a tiny
            "s": uniformer_small,
            "m": uniformer_base,
            "l": uniformer_base,
        }

        factory = factory_map.get(variant, uniformer_small)
        model = factory()
        model.eval()

        out = str(OUTPUT_DIR / f"uniformer_{variant}.onnx")
        with torch.no_grad():
            torch.onnx.export(
                model,
                torch.randn(1, 3, 224, 224),
                out,
                opset_version=13,
                input_names=["input"],
                output_names=["output"],
            )
        verify_onnx(out)

        actual_params = sum(p.numel() for p in model.parameters()) / 1e6
        log.warning(
            f"  ⚠️  Modello UniFormer standard usato ({actual_params:.1f}M params).\n"
            f"  Le varianti del CSV (1.8-10M params) sono custom e più piccole."
        )
        return out

    except Exception as e:
        raise RuntimeError(
            f"Export UniFormer-{variant} fallito: {e}\n"
            f"\n"
            f"Le varianti nel CSV (t: 1.8M, s: 2.4M, m: 5.6M, l: 10M) sono\n"
            f"configurazioni custom più leggere dei modelli standard.\n"
            f"Parametri di configurazione dal CSV:\n"
            f"  C = [(t):64, (s,m):128, (l):192]\n"
            f"  C_max = [(t,s):192, (m):320, (l):384]\n"
            f"\n"
            f"Per creare queste varianti custom servono le definizioni\n"
            f"di modello esatte (non disponibili pubblicamente)."
        )
    finally:
        if str(model_dir) in sys.path:
            sys.path.remove(str(model_dir))


def export_uniformer_t() -> str:
    """UniFormer-T · 1.8M params"""
    return _export_uniformer("t")


def export_uniformer_s() -> str:
    """UniFormer-S · 2.4M params"""
    return _export_uniformer("s")


def export_uniformer_m() -> str:
    """UniFormer-M · 5.6M params"""
    return _export_uniformer("m")


def export_uniformer_l() -> str:
    """UniFormer-L · 10M params"""
    return _export_uniformer("l")


# ===================================================================
#  Registry dei modelli
# ===================================================================
MODELS = {
    # Tier 1 — Completamente automatici
    "resnet34": {
        "fn": export_resnet34,
        "tier": 1,
        "desc": "ResNet34 (21.8M) · Image Classification",
        "deps": "torch, torchvision",
    },
    "yolov8n": {
        "fn": export_yolov8n,
        "tier": 1,
        "desc": "YOLOv8-n (3.2M) · Object Detection",
        "deps": "ultralytics",
    },
    "yolov11n": {
        "fn": export_yolov11n,
        "tier": 1,
        "desc": "YOLOv11-n (2.6M) · Object Detection",
        "deps": "ultralytics",
    },
    "yolov11s": {
        "fn": export_yolov11s,
        "tier": 1,
        "desc": "YOLOv11-s (9.4M) · Object Detection",
        "deps": "ultralytics",
    },
    "yolov11m": {
        "fn": export_yolov11m,
        "tier": 1,
        "desc": "YOLOv11-m (20.1M) · Object Detection",
        "deps": "ultralytics",
    },
    "yolov11l": {
        "fn": export_yolov11l,
        "tier": 1,
        "desc": "YOLOv11-l (25.3M) · Object Detection",
        "deps": "ultralytics",
    },
    "yolov11x": {
        "fn": export_yolov11x,
        "tier": 1,
        "desc": "YOLOv11-x (56.9M) · Object Detection",
        "deps": "ultralytics",
    },
    # Tier 2 — Semi-automatici
    "yolov7": {
        "fn": export_yolov7,
        "tier": 2,
        "desc": "YOLOv7 (36.9M) · Object Detection",
        "deps": "torch (+ git clone repo)",
    },
    "pidnet_s": {
        "fn": export_pidnet_s,
        "tier": 2,
        "desc": "PIDNet-S (7.6M) · Semantic Segmentation",
        "deps": "torch, gdown (+ git clone repo)",
    },
    "mobileclip_s0": {
        "fn": export_mobileclip_s0,
        "tier": 2,
        "desc": "MobileCLIP-S0 (11.4M+42.4M) · Image-Text Encoding",
        "deps": "mobileclip / open_clip",
    },
    "mobileclip_b": {
        "fn": export_mobileclip_b,
        "tier": 2,
        "desc": "MobileCLIP-B (86.3M+63.4M) · Image-Text Encoding",
        "deps": "mobileclip / open_clip",
    },
    # Tier 3 — Complessi
    "rtmdet_l": {
        "fn": export_rtmdet_l,
        "tier": 3,
        "desc": "RTMDet-l (52.3M) · Object Detection",
        "deps": "mmdet, mmengine, mmcv (via mim)",
    },
    "fastinst_d1": {
        "fn": export_fastinst_d1,
        "tier": 3,
        "desc": "FastInst-D1 (30M) · Instance Segmentation ⚠️ DCN",
        "deps": "detectron2 (DCN non supportato in ONNX)",
    },
    "uniformer_t": {
        "fn": export_uniformer_t,
        "tier": 3,
        "desc": "UniFormer-T (1.8M) · Image Classification",
        "deps": "timm / UniFormer repo",
    },
    "uniformer_s": {
        "fn": export_uniformer_s,
        "tier": 3,
        "desc": "UniFormer-S (2.4M) · Image Classification",
        "deps": "timm / UniFormer repo",
    },
    "uniformer_m": {
        "fn": export_uniformer_m,
        "tier": 3,
        "desc": "UniFormer-M (5.6M) · Image Classification",
        "deps": "timm / UniFormer repo",
    },
    "uniformer_l": {
        "fn": export_uniformer_l,
        "tier": 3,
        "desc": "UniFormer-L (10M) · Image Classification",
        "deps": "timm / UniFormer repo",
    },
}


# ===================================================================
#  Main
# ===================================================================
def main():
    global OUTPUT_DIR  # dichiarata prima di qualsiasi uso

    parser = argparse.ArgumentParser(
        description="Download e export dei modelli CV in formato ONNX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all", action="store_true", help="Esporta tutti i modelli"
    )
    group.add_argument(
        "--models",
        nargs="+",
        metavar="NOME",
        help="Esporta modelli specifici (usa --list per vedere i nomi)",
    )
    group.add_argument(
        "--list", action="store_true", help="Mostra i modelli disponibili"
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3],
        help="Esporta solo modelli fino al tier specificato (1=facili, 3=tutti)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Directory di output (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    # --- Lista modelli ---
    if args.list:
        tier_labels = {
            1: "✅ Completamente Automatico",
            2: "⚠️  Semi-Automatico",
            3: "🔧 Complesso / Setup Manuale",
        }
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║              MODELLI DISPONIBILI PER EXPORT                ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        for tier in [1, 2, 3]:
            print(f"\n  Tier {tier} — {tier_labels[tier]}:")
            print(f"  {'─' * 56}")
            for name, info in MODELS.items():
                if info["tier"] == tier:
                    print(f"    {name:18s} │ {info['desc']}")
                    print(f"    {'':18s} │ Deps: {info['deps']}")
            print()
        return

    # --- Configura output dir ---
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir).resolve()

    # --- Determina modelli da esportare ---
    if args.all:
        to_export = list(MODELS.keys())
    else:
        to_export = args.models
        for name in to_export:
            if name not in MODELS:
                parser.error(
                    f"Modello sconosciuto: '{name}'. Usa --list per i nomi disponibili."
                )

    # Filtra per tier se specificato
    max_tier = args.tier or 3
    to_export = [n for n in to_export if MODELS[n]["tier"] <= max_tier]

    if not to_export:
        print("Nessun modello selezionato per l'export.")
        return

    # --- Crea directory ---
    ensure_dir(OUTPUT_DIR)
    ensure_dir(CACHE_DIR)

    # --- Export ---
    print(f"\n{'═' * 60}")
    print(f"  Export di {len(to_export)} modelli in ONNX")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'═' * 60}\n")

    results = {}
    for i, name in enumerate(to_export, 1):
        info = MODELS[name]
        print(f"\n┌─ [{i}/{len(to_export)}] Tier {info['tier']} ─ {name}")
        print(f"│  {info['desc']}")
        print(f"│  Deps: {info['deps']}")
        print(f"└{'─' * 58}")

        t0 = time.time()
        try:
            path = info["fn"]()
            elapsed = time.time() - t0
            results[name] = ("✅", path, f"{elapsed:.1f}s")
            log.info(f"  Completato in {elapsed:.1f}s")
        except NotImplementedError as e:
            elapsed = time.time() - t0
            first_line = str(e).split("\n")[0]
            results[name] = ("⛔", first_line, f"{elapsed:.1f}s")
            log.warning(f"  Non implementato: {first_line}")
            log.info(f"  Dettagli:\n{e}")
        except Exception as e:
            elapsed = time.time() - t0
            first_line = str(e).split("\n")[0]
            results[name] = ("❌", first_line, f"{elapsed:.1f}s")
            log.error(f"  Fallito: {first_line}")
            log.debug(traceback.format_exc())

    # --- Riepilogo ---
    print(f"\n{'═' * 60}")
    print(f"  RIEPILOGO EXPORT")
    print(f"{'═' * 60}")

    n_ok = sum(1 for s, _, _ in results.values() if s == "✅")
    n_skip = sum(1 for s, _, _ in results.values() if s == "⛔")
    n_fail = sum(1 for s, _, _ in results.values() if s == "❌")

    print(f"\n  ✅ Successo:        {n_ok}")
    print(f"  ⛔ Non supportato:  {n_skip}")
    print(f"  ❌ Fallito:         {n_fail}")
    print()

    for name, (status, info, elapsed) in results.items():
        tier = MODELS[name]["tier"]
        print(f"  {status} [{tier}] {name:18s} │ {elapsed:>6s} │ {info}")

    print(f"\n  Output directory: {OUTPUT_DIR}")

    # Lista file esportati
    exported = list(OUTPUT_DIR.glob("*.onnx"))
    if exported:
        print(f"\n  File ONNX esportati ({len(exported)}):")
        for f in sorted(exported):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    📦 {f.name} ({size_mb:.1f} MB)")

    print()


if __name__ == "__main__":
    main()
