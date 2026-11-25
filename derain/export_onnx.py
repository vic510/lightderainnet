"""Exports a fixed-size ONNX model so TensorRT (trtexec) can measure the optimized engine's latency.

Example:
  python -m derain.export_onnx --ckpt runs/light32/best.pt --size 720x1280 --onnx exports/light32_720p.onnx
  trtexec --onnx=exports/light32_720p.onnx --fp16 --warmUp=500 --iterations=200 --saveEngine=exports/light32_720p_fp16.engine
Height and width must already be multiples of --pad-multiple (720x1280 works; use 1088 instead of 1080, or resize first).
"""
import argparse
from pathlib import Path

import torch

from .common import add_model_args, load_model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    p.add_argument("--size", default="720x1280")
    p.add_argument("--onnx", required=True)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    h, w = (int(v) for v in args.size.lower().split("x"))
    if h % args.pad_multiple or w % args.pad_multiple:
        raise SystemExit(f"{h}x{w} is not a multiple of {args.pad_multiple}; adjust the size")
    model, _ = load_model(args)
    x = torch.rand(1, 3, h, w, device=args.device)
    Path(args.onnx).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, x, args.onnx, input_names=["rainy"], output_names=["derained"],
                      opset_version=args.opset, dynamo=False)
    print(f"Wrote {args.onnx}")


if __name__ == "__main__":
    main()
