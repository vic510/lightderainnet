"""External-method comparison: quality and 720p latency on the same machine with the same metrics (Y-channel PSNR/SSIM, 8-bit quantized).
- Restormer (CVPR 2022, official Rain13K deraining weights): quality + speed
- TransWeather (CVPR 2022): no official weights released, so only speed is measured (random weights) and quality is cited from the paper
External code lives in third_party/ (not in git); weights are in third_party/w/.

Usage (from the repo root): python tools/compare_sota.py [--speed-only]
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from derain.common import forward_padded, load_checkpoint_model  # noqa: E402
from derain.evaluate import run_eval  # noqa: E402
from derain.data import find_pairs  # noqa: E402

TP = Path(__file__).resolve().parents[1] / "third_party"
OR = "../datasets/allweather/outdoor_rain/test/rainstreak"
R13 = "../datasets/Rain13K/test"
SETS = {"OutdoorRain": OR, "Rain100L": f"{R13}/Rain100L", "Rain100H": f"{R13}/Rain100H"}


def restormer():
    sys.path.insert(0, str(TP / "Restormer/basicsr/models/archs"))
    from restormer_arch import Restormer
    m = Restormer(LayerNorm_type="WithBias")
    m.load_state_dict(torch.load(TP / "w/restormer_deraining.pth", map_location="cpu")["params"])
    return m


def transweather():
    sys.modules.setdefault("mmcv", types.ModuleType("mmcv"))
    cnn = types.ModuleType("mmcv.cnn")
    cnn.ConvModule = None  # imported by the source but unused
    sys.modules["mmcv.cnn"] = cnn
    sys.path.insert(0, str(TP / "TransWeather"))
    from transweather_model import Transweather
    return Transweather()


def wgws(setting="Setting1_K1"):
    """WGWS-Net (CVPR 2023); Setting1 is trained with Outdoor-Rain (rain + haze), flag [0,1,0] = rain (official test script HRain)."""
    sys.path.insert(0, str(TP / "w_wgws/repo"))
    from networks.Network_Stage2_K1_Flag import UNet
    net = UNet(base_channel=18, num_res=6)
    net.load_state_dict(torch.load(TP / f"w_wgws/ckpt/{setting}.pth", map_location="cpu", weights_only=True))

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def half(self):  # dynamic-conv weights are fp32 and are not converted to half precision; latency is FP32
            return self

        def forward(self, x):
            return self.net(x.float(), flag=[0, 1, 0])

    return Wrap()


def weatherremover():
    """WeatherRemover (IEEE TAI 2026) official outdoor_rain weights (the original ckpt.pth is a tar.gz holding four sets; extract outdoor_rain into w_wr/)."""
    sys.path.insert(0, str(TP / "w_wr/repo"))
    from model.cmformer import CMFormer
    m = CMFormer()
    sd = torch.load(TP / "w_wr/outdoor_rain_ckpt", map_location="cpu", weights_only=True)
    m.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()})
    return m


@torch.no_grad()
def latency(model, h=720, w=1280, iters=50, warmup=10, mult=8):
    x = torch.rand(1, 3, h, w, device="cuda", dtype=torch.float16)
    for _ in range(warmup):
        forward_padded(model, x, mult)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        forward_padded(model, x, mult)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--speed-only", action="store_true")
    p.add_argument("--only", nargs="*", help="only run these model names")
    p.add_argument("--out", default="results/compare_sota.jsonl")
    args = p.parse_args()
    torch.backends.cudnn.benchmark = True
    models = {
        "ours_w32_mixed300k": (lambda: load_checkpoint_model("runs/light32_mixed300k/last.pt")[0], True, 8),
        "restormer": (restormer, True, 8),
        "transweather": (transweather, False, 32),
        "wgws_setting1": (wgws, True, 16),
        "weatherremover_outdoor": (weatherremover, True, 8),
    }
    for name, (build, has_weights, mult) in models.items():
        if args.only and name not in args.only:
            continue
        m = build().cuda().eval().half()
        r = {"model": name, "params_M": sum(q.numel() for q in m.parameters()) / 1e6,
             "latency_720p_fp16_ms": latency(m, mult=mult)}
        if has_weights and not args.speed_only:
            for s, d in SETS.items():
                rows = run_eval(m, find_pairs(d), "cuda", precision="fp16", pad_multiple=mult)
                r[s] = {"psnr": sum(x["psnr"] for x in rows) / len(rows), "ssim": sum(x["ssim"] for x in rows) / len(rows)}
        print(json.dumps(r, ensure_ascii=False), flush=True)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
