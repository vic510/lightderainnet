"""Cascade baselines: deraining (w32, Rain13K weights) and dehazing (AOD/UCL) in both orders, Outdoor-Rain Test1, same metrics.
Usage (from the repo root): python tools/cascade_eval.py derain IN_DIR OUT_DIR   # derain with runs/light32 and save the images
                  python tools/cascade_eval.py score DIR [DIR...]       # compute PSNR/SSIM against GT
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from derain.common import load_checkpoint_model
from derain.data import read_image, to_tensor, write_image, to_uint8
from derain.common import forward_padded
from derain.metrics import psnr, quantize, ssim

GT = Path("../datasets/allweather/outdoor_rain/test/rainstreak/gt")
INP = Path("../datasets/allweather/outdoor_rain/test/rainstreak/input")


def derain(src, dst):
    m = load_checkpoint_model("runs/light32/best.pt")[0].cuda().eval().half()
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(src).glob("*.png")):
        x = to_tensor(read_image(p)).unsqueeze(0).cuda().half()
        with torch.no_grad():
            o = quantize(forward_padded(m, x, 8).float())
        write_image(dst / p.name, to_uint8(o[0]))


def score(d):
    r = []
    for p in sorted(GT.glob("*.png")):
        g = to_tensor(read_image(p)).unsqueeze(0).cuda()
        o = to_tensor(read_image(Path(d) / p.name)).unsqueeze(0).cuda()
        r.append((psnr(o, g, True, 0).item(), ssim(o, g, True, 0).item()))
    return np.mean(r, 0), len(r)


if __name__ == "__main__":
    if sys.argv[1] == "derain":
        derain(sys.argv[2], Path(sys.argv[3]))
    else:
        for d in sys.argv[2:]:
            (a, b), n = score(d)
            print(f"{d}\tPSNR {a:.2f}\tSSIM {b:.3f}\tn={n}")
