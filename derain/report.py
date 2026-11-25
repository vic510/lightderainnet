"""把 results/*.jsonl 整理成 Markdown 表格，可直接貼進期中報告（數字皆來自實際紀錄）。

範例：
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
    return name + (" (隨機權重)" if m.get("source") == "random-init" else "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args()
    root = Path(args.results)

    benches = load(root / "benchmarks.jsonl")
    if benches:
        env = benches[-1]["env"]
        print(f"## 模型推論延遲（batch=1，僅模型 forward）\n\n環境：{env.get('gpu', 'CPU')}，driver {env.get('driver', '?')}，"
              f"PyTorch {env['torch']}，CUDA {env['cuda_runtime']}\n")
        print("| 代號 | 模型 | 參數量 | 解析度 | 精度 | GFLOPs | median (ms) | p95 (ms) | FPS | 峰值記憶體 (MiB) | 狀態 |")
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
        print("## 畫質（測試集平均）\n")
        print("| 代號 | 模型 | 資料集 | 張數 | 精度 | 指標空間 | 輸入 PSNR | PSNR (dB) | SSIM |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in evals:
            print(f"| {r['tag']} | {model_name(r['model'])} | {Path(r['data']).as_posix()} | {r['n_images']} | "
                  f"{r['precision']} | {r['metric_space']} | {r['input_psnr']:.2f} | {r['psnr']:.2f} | {r['ssim']:.4f} |")
        print()

    videos = load(root / "videos.jsonl")
    if videos:
        print("## 完整影片流程\n")
        print("| 代號 | 模式 | 精度 | 幀數 | 吞吐量 FPS | 端到端延遲 median (ms) | p95 (ms) | 達來源幀率 |")
        print("|---|---|---|---|---|---|---|---|")
        for r in videos:
            lat = r.get("end_to_end_latency_ms") or {}
            mode = r["mode"] + ("（依幀率送幀）" if r.get("paced") else "")
            print(f"| {r['tag']} | {mode} | {r['precision']} | {r['frames']} | {r['throughput_fps']} | "
                  f"{lat.get('median', '')} | {lat.get('p95', '')} | {'是' if r['realtime'] else '否'} |")


if __name__ == "__main__":
    main()
