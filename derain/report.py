"""Turns results/*.jsonl into Markdown tables that can be pasted into a report (all numbers come from the recorded results).

Example:
  python -m derain.report --results results > results/summary.md
"""
import argparse
import json
from pathlib import Path


def load(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def model_name(m):
    if m.get("source") == "external":
        return m.get("class", "external")
    name = f"{m.get('arch', '?')}-w{m.get('width', '?')}b{m.get('blocks', '?')}"
    return name + (" (random weights)" if m.get("source") == "random-init" else "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args()
    root = Path(args.results)

    benches = load(root / "benchmarks.jsonl")
    if benches:
        env = benches[-1]["env"]
        print(f"## Model inference latency (batch=1, model forward only)\n\nEnvironment: {env.get('gpu', 'CPU')}, driver {env.get('driver', '?')}, "
              f"PyTorch {env['torch']}, CUDA {env['cuda_runtime']}\n")
        print("| Tag | Model | Params | Resolution | Precision | GFLOPs | median (ms) | p95 (ms) | FPS | Peak memory (MiB) | Status |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in benches:
            lat = r.get("latency_ms") or {}
            print(f"| {r['tag']} | {model_name(r['model'])} | {r['params']:,} | {r['size'][0]}x{r['size'][1]} | "
                  f"{r['precision']}{' CL' if r.get('channels_last') else ''} | {r.get('gflops_per_image')} | "
                  f"{lat.get('median', '')} | {lat.get('p95', '')} | {r.get('throughput_fps', '')} | "
                  f"{r.get('peak_mem_mib', '')} | {r['status']} |")
        print()

    evals = load(root / "evals.jsonl")
    if evals:
        print("## Quality (test-set average)\n")
        print("| Tag | Model | Dataset | Images | Precision | Metric space | Input PSNR | PSNR (dB) | SSIM |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in evals:
            print(f"| {r['tag']} | {model_name(r['model'])} | {Path(r['data']).as_posix()} | {r['n_images']} | "
                  f"{r['precision']} | {r['metric_space']} | {r['input_psnr']:.2f} | {r['psnr']:.2f} | {r['ssim']:.4f} |")
        print()

    videos = load(root / "videos.jsonl")
    if videos:
        print("## Full video pipeline\n")
        print("| Tag | Mode | Precision | Frames | Throughput FPS | End-to-end latency median (ms) | p95 (ms) | Reaches source FPS |")
        print("|---|---|---|---|---|---|---|---|")
        for r in videos:
            lat = r.get("end_to_end_latency_ms") or {}
            mode = r["mode"] + ("(paced at source FPS)" if r.get("paced") else "")
            print(f"| {r['tag']} | {mode} | {r['precision']} | {r['frames']} | {r['throughput_fps']} | "
                  f"{lat.get('median', '')} | {lat.get('p95', '')} | {'yes' if r['realtime'] else 'no'} |")


if __name__ == "__main__":
    main()
