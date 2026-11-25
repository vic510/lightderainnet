"""Computes PSNR / SSIM on a paired test set.

Example:
  python -m derain.evaluate --ckpt runs/light32/best.pt --data D:/datasets/Rain100L/test --precision fp32
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch

from .common import add_model_args, append_jsonl, env_info, forward_padded, load_model
from .data import find_pairs, read_image, to_tensor, to_uint8, write_image
from .metrics import psnr, quantize, ssim

DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


@torch.inference_mode()
def run_eval(model, pairs, device, precision="fp32", pad_multiple=8, y_channel=True, crop=0, save_dir=None):
    dtype = DTYPES[precision]
    rows = []
    for rp, cp in pairs:
        rain = to_tensor(read_image(rp)).unsqueeze(0).to(device, dtype)
        clean = to_tensor(read_image(cp)).unsqueeze(0).to(device)
        out = quantize(forward_padded(model, rain, pad_multiple).float())
        rows.append({
            "image": rp.name,
            "psnr": psnr(out, clean, y_channel, crop).item(),
            "ssim": ssim(out, clean, y_channel, crop).item(),
            "input_psnr": psnr(rain.float(), clean, y_channel, crop).item(),
        })
        if save_dir:
            write_image(Path(save_dir) / f"{rp.stem}.png", to_uint8(out[0]))
    return rows


def summarize(rows):
    n = len(rows)
    return {k: sum(r[k] for r in rows) / n for k in ("psnr", "ssim", "input_psnr")} | {"n_images": n}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    p.add_argument("--data", required=True, help="test set root (containing rain/ and norain/ etc.)")
    p.add_argument("--rain-dir")
    p.add_argument("--clean-dir")
    p.add_argument("--precision", default="fp32", choices=list(DTYPES))
    p.add_argument("--rgb", action="store_true", help="compute on RGB (default: Y channel)")
    p.add_argument("--crop", type=int, default=0, help="border pixels to crop before computing metrics")
    p.add_argument("--save-images", help="folder to write derained images to")
    p.add_argument("--out", default="results", help="results output folder")
    p.add_argument("--tag", default="", help="experiment tag, e.g. B-fp32")
    args = p.parse_args()

    model, model_info = load_model(args)
    if args.precision == "fp16":
        model = model.half()
    pairs = find_pairs(args.data, args.rain_dir, args.clean_dir)
    t0 = time.perf_counter()
    rows = run_eval(model, pairs, args.device, args.precision, args.pad_multiple,
                    not args.rgb, args.crop, args.save_images)
    summary = summarize(rows)

    out = Path(args.out)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = out / f"eval_{args.tag or 'run'}_{stamp}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    record = {
        "kind": "eval", "tag": args.tag, "model": model_info, "data": str(args.data),
        "precision": args.precision, "metric_space": "RGB" if args.rgb else "Y", "crop": args.crop,
        "quantized_8bit": True, **summary, "per_image_csv": str(csv_path),
        "wall_time_s": round(time.perf_counter() - t0, 2), "env": env_info(args.device),
    }
    append_jsonl(out / "evals.jsonl", record)
    print(json.dumps({k: record[k] for k in ("tag", "precision", "n_images", "psnr", "ssim", "input_psnr")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
