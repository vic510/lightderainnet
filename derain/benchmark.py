"""Model inference latency / throughput / GPU memory benchmark (model only; excludes file I/O and encode/decode).

Method: fixed input size, repeated runs after warm-up, each timed with CUDA events and synchronized;
reports mean / median / p95. Use video.py to measure the full video pipeline.

Example:
  python -m derain.benchmark --ckpt runs/light32/best.pt --size 720x1280 1080x1920 --precision fp32 fp16 --tag B
"""
import argparse
import copy
import json
import statistics
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from .common import add_model_args, append_jsonl, count_params, env_info, forward_padded, load_model

DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


def parse_size(s):
    h, w = s.lower().split("x")
    return int(h), int(w)


def percentile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(q / 100 * (len(xs) - 1)))]


@torch.inference_mode()
def measure(model, x, pad, warmup, iters):
    cuda = x.is_cuda
    for _ in range(warmup):
        forward_padded(model, x, pad)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        if cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            forward_padded(model, x, pad)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            t = time.perf_counter()
            forward_padded(model, x, pad)
            times.append((time.perf_counter() - t) * 1000)
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else None
    return times, peak


@torch.inference_mode()
def count_gflops(model, x, pad):
    with FlopCounterMode(display=False) as fc:
        forward_padded(model, x, pad)
    return fc.get_total_flops() / 1e9


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    p.add_argument("--size", nargs="+", default=["720x1280"], help="input size HxW; may be given multiple times")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--precision", nargs="+", default=["fp32", "fp16"], choices=list(DTYPES))
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--cudnn-benchmark", action="store_true", help="let cuDNN auto-select the fastest convolution algorithm")
    p.add_argument("--compile", action="store_true", help="torch.compile (Windows needs triton installed separately and may not be supported)")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--out", default="results/benchmarks.jsonl")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    base, model_info = load_model(args)
    env = env_info(args.device)
    params = count_params(base)
    for size in args.size:
        h, w = parse_size(size)
        try:
            gflops = round(count_gflops(base, torch.rand(1, 3, h, w, device=args.device), args.pad_multiple), 2)
        except torch.cuda.OutOfMemoryError:
            gflops = None
            torch.cuda.empty_cache()
        for prec in args.precision:
            if prec == "fp16" and args.device == "cpu":
                print("[skip] fp16 is not benchmarked on CPU")
                continue
            model = copy.deepcopy(base).to(dtype=DTYPES[prec])
            x = torch.rand(args.batch, 3, h, w, device=args.device, dtype=DTYPES[prec])
            if args.channels_last:
                model, x = model.to(memory_format=torch.channels_last), x.to(memory_format=torch.channels_last)
            if args.compile:
                model = torch.compile(model)
            record = {"kind": "benchmark", "tag": args.tag, "model": model_info, "params": params,
                      "gflops_per_image": gflops, "size": [h, w], "batch": args.batch,
                      "precision": prec, "channels_last": args.channels_last,
                      "cudnn_benchmark": args.cudnn_benchmark, "compile": args.compile,
                      "warmup": args.warmup, "iters": args.iters, "timing": "model forward only, CUDA events"}
            try:
                times, peak = measure(model, x, args.pad_multiple, args.warmup, args.iters)
            except torch.cuda.OutOfMemoryError:
                record["status"] = "OOM"
                torch.cuda.empty_cache()
            except Exception as e:  # e.g. torch.compile unsupported on this platform
                record["status"] = f"error: {type(e).__name__}: {e}"
            else:
                med = statistics.median(times)
                record.update({
                    "status": "ok",
                    "latency_ms": {"mean": round(statistics.mean(times), 3), "median": round(med, 3),
                                   "p95": round(percentile(times, 95), 3), "min": round(min(times), 3),
                                   "std": round(statistics.stdev(times), 3) if len(times) > 1 else 0.0},
                    "throughput_fps": round(1000.0 / med * args.batch, 2),
                    "peak_mem_mib": round(peak, 1) if peak is not None else None,
                })
            record["env"] = env
            append_jsonl(args.out, record)
            summary = {k: record.get(k) for k in ("tag", "size", "precision", "status", "latency_ms",
                                                  "throughput_fps", "peak_mem_mib", "gflops_per_image")}
            print(json.dumps(summary, ensure_ascii=False))
            del model, x
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
