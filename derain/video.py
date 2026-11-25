"""Frame-by-frame video deraining; measures full-pipeline FPS and end-to-end latency.

Two modes:
  sequential  read -> preprocess -> inference -> postprocess -> write, one frame at a time; per-stage timings are recorded.
  pipelined   reading and writing each run in their own thread, overlapping with GPU inference (CPU/GPU parallelism); measures the throughput gain.
Pre/post-processing (uint8->float, BGR->RGB, normalization) runs on the GPU; the CPU only does encode/decode.

Example:
  python -m derain.video --ckpt runs/light32/best.pt --input rainy.mp4 --output derained.mp4 --mode pipelined --precision fp16
"""
import argparse
import json
import queue
import statistics
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .benchmark import percentile
from .common import add_model_args, append_jsonl, env_info, forward_padded, load_model

DTYPES = {"fp32": torch.float32, "fp16": torch.float16}
_STOP = object()


def open_video(src):
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else str(src))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {src} (OpenCV on Windows may not support non-ASCII paths; use an ASCII path)")
    return cap


class Processor:
    """GPU-side preprocessing + inference + postprocessing."""

    def __init__(self, model, device, dtype, pad, resize=None):
        self.model, self.device, self.dtype, self.pad, self.resize = model, device, dtype, pad, resize
        self.cuda = str(device).startswith("cuda")

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    @torch.inference_mode()
    def __call__(self, frame_bgr: np.ndarray, timings=None):
        t0 = time.perf_counter()
        if self.resize:
            frame_bgr = cv2.resize(frame_bgr, (self.resize[1], self.resize[0]), interpolation=cv2.INTER_AREA)
        src = torch.from_numpy(frame_bgr)
        if self.cuda:
            src = src.pin_memory()
        x = src.to(self.device, non_blocking=True)
        x = x.flip(-1).permute(2, 0, 1).unsqueeze(0).to(self.dtype) / 255.0   # BGR HWC -> RGB BCHW
        self._sync()
        t1 = time.perf_counter()
        y = forward_padded(self.model, x, self.pad)
        self._sync()
        t2 = time.perf_counter()
        out = (y[0].clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).flip(-1)  # -> BGR HWC
        out = out.contiguous().cpu().numpy()
        t3 = time.perf_counter()
        if timings is not None:
            timings["pre_ms"].append((t1 - t0) * 1000)
            timings["infer_ms"].append((t2 - t1) * 1000)
            timings["post_ms"].append((t3 - t2) * 1000)
        return frame_bgr, out


def compose(inp, out, side_by_side):
    return np.concatenate([inp, out], axis=1) if side_by_side else out


def run_sequential(cap, proc, writer_factory, max_frames, side_by_side):
    timings = {k: [] for k in ("read_ms", "pre_ms", "infer_ms", "post_ms", "write_ms")}
    latencies, writer, n = [], None, 0
    t_start = time.perf_counter()
    while n < max_frames:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        timings["read_ms"].append((time.perf_counter() - t0) * 1000)
        inp, out = proc(frame, timings)
        t1 = time.perf_counter()
        frame_out = compose(inp, out, side_by_side)
        if writer is None:
            writer = writer_factory(frame_out.shape)
        if writer is not None:
            writer.write(frame_out)
        timings["write_ms"].append((time.perf_counter() - t1) * 1000)
        latencies.append((time.perf_counter() - t0) * 1000)
        n += 1
    return n, time.perf_counter() - t_start, latencies, timings, writer


def run_pipelined(cap, proc, writer_factory, max_frames, side_by_side, qsize, pace_fps=None):
    q_in, q_out = queue.Queue(maxsize=qsize), queue.Queue(maxsize=qsize)
    latencies, state = [], {"writer": None, "error": None}

    def reader():
        n, t0 = 0, time.perf_counter()
        while n < max_frames:
            if pace_fps:  # simulate a camera: frame n is only "captured" at n/fps seconds
                time.sleep(max(0.0, t0 + n / pace_fps - time.perf_counter()))
            ok, frame = cap.read()
            if not ok:
                break
            q_in.put((time.perf_counter(), frame))
            n += 1
        q_in.put(_STOP)

    def writer_loop():
        try:
            while (item := q_out.get()) is not _STOP:
                t_read, frame_out = item
                if state["writer"] is None:
                    state["writer"] = writer_factory(frame_out.shape)
                if state["writer"] is not None:
                    state["writer"].write(frame_out)
                latencies.append((time.perf_counter() - t_read) * 1000)
        except Exception as e:  # record the error for the main thread and keep draining the queue so the main thread does not block
            state["error"] = e
            while q_out.get() is not _STOP:
                pass

    threads = [threading.Thread(target=reader, daemon=True), threading.Thread(target=writer_loop, daemon=True)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    n = 0
    while (item := q_in.get()) is not _STOP:
        t_read, frame = item
        inp, out = proc(frame)
        q_out.put((t_read, compose(inp, out, side_by_side)))
        n += 1
    q_out.put(_STOP)
    for t in threads:
        t.join()
    if state["error"]:
        raise state["error"]
    return n, time.perf_counter() - t_start, latencies, None, state["writer"]


def stats(xs):
    return {"mean": round(statistics.mean(xs), 3), "median": round(statistics.median(xs), 3),
            "p95": round(percentile(xs, 95), 3)} if xs else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    p.add_argument("--input", required=True, help="video file or camera index (e.g. 0)")
    p.add_argument("--output", help="output video (.mp4); if omitted, only measure without saving")
    p.add_argument("--mode", default="pipelined", choices=["sequential", "pipelined"])
    p.add_argument("--precision", default="fp32", choices=list(DTYPES))
    p.add_argument("--resize", help="resize to HxW before inference, e.g. 720x1280")
    p.add_argument("--max-frames", type=int, default=10 ** 9)
    p.add_argument("--warmup", type=int, default=10, help="number of warm-up runs on the first frame before measuring")
    p.add_argument("--queue", type=int, default=4, help="queue length in pipelined mode")
    p.add_argument("--pace", action="store_true",
                   help="in pipelined mode, feed frames at the source frame rate (simulating a camera) so latency does not include queueing caused by reading the file too fast")
    p.add_argument("--side-by-side", action="store_true", help="write a side-by-side output (original | derained)")
    p.add_argument("--out", default="results/videos.jsonl")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    model, model_info = load_model(args)
    model = model.to(dtype=DTYPES[args.precision])
    resize = tuple(int(v) for v in args.resize.lower().split("x")) if args.resize else None
    proc = Processor(model, args.device, DTYPES[args.precision], args.pad_multiple, resize)

    cap = open_video(args.input)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, first = cap.read()
    if not ok:
        raise IOError("The video has no frames")
    for _ in range(args.warmup):
        proc(first)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def writer_factory(shape):
        if not args.output:
            return None
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        w = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (shape[1], shape[0]))
        if not w.isOpened():
            raise IOError(f"Cannot create output video {args.output}")
        return w

    if args.mode == "sequential":
        n, wall, lat, timings, writer = run_sequential(cap, proc, writer_factory, args.max_frames, args.side_by_side)
    else:
        n, wall, lat, timings, writer = run_pipelined(cap, proc, writer_factory, args.max_frames,
                                                      args.side_by_side, args.queue, src_fps if args.pace else None)
    cap.release()
    if writer is not None:
        writer.release()

    record = {
        "kind": "video", "tag": args.tag, "model": model_info, "input": str(args.input), "mode": args.mode,
        "precision": args.precision, "resize": resize, "paced": args.pace, "frames": n, "source_fps": src_fps,
        "wall_time_s": round(wall, 3), "throughput_fps": round(n / wall, 2) if wall else None,
        "end_to_end_latency_ms": stats(lat),
        "stage_ms": {k: stats(v) for k, v in timings.items()} if timings else None,
        "realtime": (n / wall) >= 0.98 * src_fps if wall else None,  # tolerate timing error when using --pace
        "timing": "sequential: read→write per frame; pipelined: read timestamp→written, includes queueing",
        "env": env_info(args.device),
    }
    append_jsonl(args.out, record)
    print(json.dumps({k: record[k] for k in ("tag", "mode", "precision", "frames", "throughput_fps",
                                             "end_to_end_latency_ms", "stage_ms", "realtime")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
