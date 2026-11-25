# Evaluation protocol

All numbers in this repository follow the rules below. State them whenever you report a result.

- **Model latency:** batch=1, fixed size, 20 warm-up runs then 100 timed runs, each timed with CUDA events and synchronized; report median and p95. Excludes file I/O and encode/decode.
- **Full pipeline:** in `sequential`, latency is the time from reading a frame to writing it. In `pipelined`, it is the time from when the frame is read to when it is written, including queueing. Throughput = frames / total time.
  - When testing `pipelined` on a video file, reading is much faster than a real camera, so frames pile up in the queue and latency is overestimated. Use `--pace` to feed frames at the source frame rate to get camera-scenario latency.
  - Measure throughput (maximum processing speed) without `--pace` and latency (camera scenario) with `--pace`; report them separately.
- **Quality:** output is quantized to 8-bit and measured on the Y channel, no border crop by default. Numbers using RGB or a crop are not directly comparable with other settings.
- FP16 changes numeric values, so re-evaluate quality; do not judge by speed alone.

## Metrics

- **PSNR / SSIM** (`derain/metrics.py`): output quantized to 8-bit, computed on the Y channel of YCbCr (ITU-R BT.601, same as MATLAB `rgb2ycbcr`), no border crop, SSIM as in Wang et al. 2004 (Gaussian window 11, sigma 1.5, K1=0.01, K2=0.03).
- **Downstream detection** (`tools/downstream_det.py`): Outdoor-Rain Test1 has no box labels, so YOLO11s detections on the clean ground truth (conf >= 0.25, 2,115 boxes) are used as pseudo-labels. mAP50 / mAP50-95 (COCO-style 101-point interpolation) and recall (conf >= 0.25, IoU 0.5) are then computed for the rainy input and for each model's output. This measures *closeness to detection on clean images*, not absolute accuracy against human labels.
- **Latency** (`derain/benchmark.py`, `tools/compare_sota.py`): 720p, batch 1, FP16 unless noted, one machine (NVIDIA GB10) for all models. WGWS-Net contains FP32-only dynamic convolutions, so its latency is FP32.

## Threats to validity

- Official weights of the compared models were trained on different data. Differences therefore mix architecture and data distribution. Only the TransWeather row (and our own models) share the same training set.
- The same-budget (30k-iteration) retraining of all baselines is **not** reported: large models had not converged (e.g. Restormer's validation PSNR was still rising), so the comparison would not be fair. A converged same-data comparison is future work.
- mAP50 uses pseudo-labels from a detector on clean images and is a relative metric.
- Single training seed; no confidence intervals. Differences below about 0.1 dB should not be over-interpreted.
- Training data contains no pure atmospheric-scattering haze; see the limitation on RESIDE-6K in the README.
