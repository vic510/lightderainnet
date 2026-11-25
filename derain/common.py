"""Shared helpers for model loading, inference and environment logging."""
import argparse
import datetime
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import build_arch


def add_model_args(p: argparse.ArgumentParser):
    g = p.add_argument_group("Model (choose one: --ckpt, --arch, --ext-*)")
    g.add_argument("--ckpt", help="checkpoint (.pt) produced by this project's train.py")
    g.add_argument("--arch", default="light", help="built-in architecture used when --ckpt is not given (random weights; only suitable for speed tests)")
    g.add_argument("--width", type=int, default=32)
    g.add_argument("--blocks", type=int, default=6)
    g.add_argument("--ext-repo", help="path to an external model repo, e.g. a cloned DRSformer")
    g.add_argument("--ext-module", help="module path, e.g. basicsr.models.archs.DRSformer_arch")
    g.add_argument("--ext-class", help="class name, e.g. DRSformer")
    g.add_argument("--ext-kwargs", default="{}", help="constructor kwargs (JSON)")
    g.add_argument("--ext-weights", help="external model weights file")
    g.add_argument("--pad-multiple", type=int, default=8, help="pad height and width to a multiple of this before inference")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def _extract_state_dict(obj):
    for key in ("state_dict", "params_ema", "params", "model"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            return obj[key]
    return obj


def load_checkpoint_model(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["model_cfg"]
    model = build_arch(cfg["arch"], **{k: v for k, v in cfg.items() if k != "arch"})
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), {"source": "ckpt", "path": str(path), **cfg}


def load_model(args):
    """Build the model from CLI arguments and return (model, info). info is written to the results file for traceability."""
    device = args.device
    if args.ckpt:
        return load_checkpoint_model(args.ckpt, device)
    if args.ext_repo:
        sys.path.insert(0, str(Path(args.ext_repo).resolve()))
        cls = getattr(importlib.import_module(args.ext_module), args.ext_class)
        model = cls(**json.loads(args.ext_kwargs))
        if args.ext_weights:
            sd = _extract_state_dict(torch.load(args.ext_weights, map_location="cpu", weights_only=False))
            model.load_state_dict(sd)
        info = {"source": "external", "repo": args.ext_repo, "class": args.ext_class,
                "kwargs": args.ext_kwargs, "weights": args.ext_weights}
        return model.to(device).eval(), info
    model = build_arch(args.arch, width=args.width, blocks=args.blocks)
    info = {"source": "random-init", "arch": args.arch, "width": args.width, "blocks": args.blocks}
    return model.to(device).eval(), info


def forward_padded(model: nn.Module, x: torch.Tensor, multiple: int = 8) -> torch.Tensor:
    h, w = x.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return model(x)[..., :h, :w]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def env_info(device="cuda"):
    info = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": platform.processor(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        info["gpu"] = prop.name
        info["gpu_mem_mib"] = prop.total_memory // 2 ** 20
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10)
            info["driver"] = out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return info


def append_jsonl(path, record: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
