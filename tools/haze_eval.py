"""Dehazing test-set evaluation on RESIDE-6K test (848 pairs matched by file name; synthetic haze from indoor ITS + outdoor OTS): AOD-Net, UCL-Dehaze and our w32 300k.
Same metrics as the other tables (Y-channel PSNR/SSIM, 8-bit quantized). UCL needs fake_B generated first with UCL's test.py (see --ucl-dir).
Usage (from the repo root): python tools/haze_eval.py [--ucl-dir DIR]
"""
import argparse, json, pathlib, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "."); sys.path.insert(0, "dehaze/run")
from derain.common import forward_padded, load_checkpoint_model
from derain.data import read_image, to_tensor
from derain.metrics import psnr, quantize, ssim
import aod_pytorch as A

D = pathlib.Path("../datasets/haze/reside6k/test")
p = argparse.ArgumentParser(); p.add_argument("--ucl-dir"); p.add_argument("--out", default="results/haze_eval.jsonl")
args = p.parse_args()
names = sorted(set(f.name for f in (D / "hazy").iterdir()) & set(f.name for f in (D / "GT").iterdir()))
aod = A.AODNet()
for n, (w, b) in A.load_caffemodel(pathlib.Path("dehaze/AOD-Net-master/AOD_Net.caffemodel")).items():
    getattr(aod, n).weight.data = torch.from_numpy(w); getattr(aod, n).bias.data = torch.from_numpy(b.reshape(-1))
aod.cuda().eval()
ours = load_checkpoint_model("runs/light32_mixed300k/last.pt")[0].cuda().eval()
res = {"input": [], "aod": [], "ours_w32_mixed300k": []}
if args.ucl_dir: res["ucl"] = []
def score(img, gt):
    o = quantize(img.cuda()); return psnr(o, gt, True, 0).item(), ssim(o, gt, True, 0).item()
for n in names:
    gt = to_tensor(read_image(D / "GT" / n)).unsqueeze(0).cuda()
    im = Image.open(D / "hazy" / n).convert("RGB")
    x = to_tensor(np.asarray(im)).unsqueeze(0).cuda()
    res["input"].append(score(x, gt))
    with torch.no_grad():
        res["aod"].append(score(aod(x).clamp(0, 1), gt))
        res["ours_w32_mixed300k"].append(score(forward_padded(ours, x, 8).clamp(0, 1), gt))
    if args.ucl_dir:
        u = Image.open(pathlib.Path(args.ucl_dir) / (pathlib.Path(n).stem + ".png")).convert("RGB").resize(im.size, Image.BICUBIC)
        res["ucl"].append(score(to_tensor(np.asarray(u)).unsqueeze(0), gt))
out = {"set": "RESIDE-6K test (name-paired)", "n": len(names)}
for k, v in res.items(): out[k] = dict(zip(("psnr", "ssim"), np.mean(v, 0).tolist()))
print(json.dumps(out, ensure_ascii=False))
open(args.out, "a").write(json.dumps(out, ensure_ascii=False) + "\n")
