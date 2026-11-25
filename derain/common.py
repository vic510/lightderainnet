"""模型載入、推論與環境紀錄的共用函式。"""
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
    g = p.add_argument_group("模型（三擇一：--ckpt、--arch、--ext-*）")
    g.add_argument("--ckpt", help="本專案 train.py 產生的 checkpoint (.pt)")
    g.add_argument("--arch", default="light", help="未給 --ckpt 時建立的自建架構（隨機權重，只適合測速）")
    g.add_argument("--width", type=int, default=32)
    g.add_argument("--blocks", type=int, default=6)
    g.add_argument("--ext-repo", help="外部模型程式庫路徑，例如 git clone 下來的 DRSformer")
    g.add_argument("--ext-module", help="模組路徑，例如 basicsr.models.archs.DRSformer_arch")
    g.add_argument("--ext-class", help="類別名稱，例如 DRSformer")
    g.add_argument("--ext-kwargs", default="{}", help="建構參數（JSON）")
    g.add_argument("--ext-weights", help="外部模型權重檔")
    g.add_argument("--pad-multiple", type=int, default=8, help="推論前將長寬補到此倍數")
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
    """依命令列參數建立模型，回傳 (model, info)。info 會寫進結果檔以便追溯。"""
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
