# Derain: GPU-Accelerated Image Deraining Experiments

PyTorch experiment code for image deraining: training a lightweight model, knowledge distillation, PSNR/SSIM evaluation, model latency benchmarking, full video-pipeline measurement, and ONNX/TensorRT export. Every result is written to `results/*.jsonl` together with hardware and software versions, so it stays traceable.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130   # adjust to your CUDA version
pip install -r requirements.txt
```

Training takes a long time; use `tmux` or `nohup`:

```bash
nohup python -m derain.train ... --out runs/light32 > runs/light32.log 2>&1 &
```

Run all commands from the repo root.

> **Path note:** OpenCV on Windows cannot read/write videos at non-ASCII paths. Keep videos and datasets under plain ASCII paths such as `D:\datasets\`. Image I/O already handles non-ASCII paths.

## Modules

| Module | Purpose |
| --- | --- |
| `derain/models.py` | Built-in lightweight model `LightDerainNet` (adjustable width/depth for structural-compression experiments) |
| `derain/data.py` | Paired dataset loading (Rain100L/H, Rain200L/H, Rain1400, SPA-Data, ...; pairs by the number in the file name) |
| `derain/metrics.py` | PSNR and SSIM (default: 8-bit quantized, computed on the Y channel) |
| `derain/train.py` | Training; `--teacher` enables knowledge distillation, `--amp` mixed precision |
| `derain/evaluate.py` | Test-set quality evaluation; writes a per-image CSV and averages |
| `derain/benchmark.py` | Model forward only: mean/median/p95 latency, FPS, peak GPU memory, GFLOPs |
| `derain/video.py` | Full video pipeline: `sequential` (per-stage timings) and `pipelined` (CPU encode/decode overlapped with GPU inference) |
| `derain/export_onnx.py` | ONNX export for TensorRT `trtexec` |
| `derain/report.py` | Turns results into Markdown tables |
| `derain/synth.py` | Synthetic rain images/videos, **for smoke tests only; do not report these results** |

## Dataset

Rain13K goes in `../datasets/Rain13K` and is downloaded by `tools/fetch_rain13k.py`.
The source is the unofficial Hugging Face mirror `dronefreak/Rain13K` (license marked unknown); use it for academic purposes only and cite the original papers.
Anonymous downloads run at roughly 4 files per second, about 36,000 files in total, so expect 2-3 hours.

`tools/train_after_download.sh` waits for the download to finish, checks the pair counts, then starts training `runs/light32` automatically.

## Typical workflow

```powershell
# 0. Smoke test (no dataset needed)
python -m derain.synth dataset --out data/smoke --n 40 --size 128
python -m derain.train --train-root data/smoke/train --val-root data/smoke/test --iters 200 --val-every 100 --batch 4 --patch 96 --workers 0 --out runs/smoke

# 1. Train the baseline (Rain200L as an example)
python -m derain.train --train-root D:/datasets/Rain200L/train --val-root D:/datasets/Rain200L/test --amp --out runs/light32

# 2. Quality evaluation (FP32 and FP16, to compare the quality impact of precision)
python -m derain.evaluate --ckpt runs/light32/best.pt --data D:/datasets/Rain200L/test --precision fp32 --tag B-fp32
python -m derain.evaluate --ckpt runs/light32/best.pt --data D:/datasets/Rain200L/test --precision fp16 --tag D-fp16

# 3. Model latency (several resolutions x precisions)
python -m derain.benchmark --ckpt runs/light32/best.pt --size 480x640 720x1280 --precision fp32 fp16 --tag B
python -m derain.benchmark --ckpt runs/light32/best.pt --size 720x1280 --precision fp16 --channels-last --cudnn-benchmark --tag C

# 4. Structural compression: width-16 student + knowledge distillation
python -m derain.train ... --width 16 --teacher runs/light32/best.pt --kd-weight 0.5 --out runs/light16_kd

# 5. Full video pipeline
python -m derain.video --ckpt runs/light32/best.pt --input D:/videos/rain.mp4 --mode sequential --tag V-seq
python -m derain.video --ckpt runs/light32/best.pt --input D:/videos/rain.mp4 --mode pipelined --precision fp16 --output D:/videos/out.mp4 --side-by-side --tag V-pipe

# 6. Summarize into tables
python -m derain.report --results results > results/summary.md
```

## External models (DRSformer, SSDRNet, ...)

External models are not reimplemented here; they are loaded straight from their official repos so they match the published implementation. Example with DRSformer:

```powershell
git clone https://github.com/cschenxiang/DRSformer D:/repos/DRSformer
# download the pretrained weights following that repo's README first
python -m derain.benchmark --ext-repo D:/repos/DRSformer --ext-module basicsr.models.archs.DRSformer_arch --ext-class DRSformer --ext-weights D:/repos/DRSformer/pretrained_models/xxx.pth --size 480x640 --precision fp32 fp16 --tag DRSformer
```

`--ext-kwargs` passes constructor arguments as JSON. Common keys such as `params` and `state_dict` in the weights file are unwrapped automatically.
The external repo may need extra dependencies (see its README). DRSformer can run out of memory at high resolution on 4 GB of VRAM; the run is recorded as `OOM` instead of crashing.

## Measurement rules (state these when reporting)

- **Model latency:** batch=1, fixed size, 20 warm-up runs then 100 timed runs, each timed with CUDA events and synchronized; report median and p95. Excludes file I/O and encode/decode.
- **Full pipeline:** in `sequential`, latency is the time from reading a frame to writing it. In `pipelined`, it is the time from when the frame is read to when it is written, including queueing. Throughput = frames / total time.
  - When testing `pipelined` on a video file, reading is much faster than a real camera, so frames pile up in the queue and latency is overestimated. Use `--pace` to feed frames at the source frame rate to get camera-scenario latency.
  - Measure throughput (maximum processing speed) without `--pace` and latency (camera scenario) with `--pace`; report them separately.
- **Quality:** output is quantized to 8-bit and measured on the Y channel, no border crop by default. Numbers using RGB or a crop are not directly comparable with other settings.
- FP16 changes numeric values, so re-evaluate quality; do not judge by speed alone.
