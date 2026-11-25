"""Generates synthetic rain images / rainy videos, used only as a smoke test that the code runs.

Note: the synthetic data is very simplistic; results must not be reported as model performance. Use public datasets such as Rain100L/Rain200H/SPA-Data for real experiments.

Example:
  python -m derain.synth dataset --out data/smoke --n 40 --size 128
  python -m derain.synth video --out data/smoke_video.mp4 --frames 90 --size 360x640
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from .data import read_image, write_image


def procedural_scene(h, w, rng):
    """Smooth gradient + random color blocks to mimic a structured scene."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([(xx / w), (yy / h), 1 - xx / w], axis=-1) * rng.uniform(0.3, 0.8, 3)
    for _ in range(rng.integers(5, 12)):
        color = rng.uniform(0, 1, 3).tolist()
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), int(rng.integers(5, max(6, min(h, w) // 4))), color, -1)
        else:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(10, w // 2)), y + int(rng.integers(10, h // 2))),
                          color, -1)
    return np.clip(cv2.GaussianBlur(img, (0, 0), 1.0), 0, 1)


def rain_layer(h, w, rng, density=0.004, length=None, angle=None):
    """Random points -> directional motion blur -> rain streaks. Returns HxW, range [0,1]."""
    length = length or int(rng.integers(9, 25))
    angle = angle if angle is not None else float(rng.uniform(-20, 20))
    drops = (rng.random((h, w)) < density).astype(np.float32) * rng.uniform(0.6, 1.0, (h, w)).astype(np.float32)
    k = np.zeros((length, length), np.float32)
    k[:, length // 2] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    k /= k.sum() + 1e-8
    streaks = cv2.filter2D(drops, -1, k) * length * 0.5
    return np.clip(streaks, 0, 1)


def add_rain(clean, rng, **kw):
    r = rain_layer(*clean.shape[:2], rng, **kw)[..., None]
    return np.clip(clean * (1 - 0.6 * r) + 0.9 * r, 0, 1)


def make_dataset(args):
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    sources = sorted(Path(args.clean_dir).glob("*")) if args.clean_dir else []
    for split, n in (("train", args.n), ("test", max(4, args.n // 5))):
        for i in range(1, n + 1):
            if sources:
                clean = read_image(sources[(i - 1) % len(sources)]).astype(np.float32) / 255.0
            else:
                clean = procedural_scene(args.size, args.size, rng)
            rainy = add_rain(clean, rng)
            write_image(out / split / "norain" / f"norain-{i}.png", (clean * 255).round().astype(np.uint8))
            write_image(out / split / "rain" / f"rain-{i}.png", (rainy * 255).round().astype(np.uint8))
    print(f"Synthetic data written to {out} (train/test, each with rain/ and norain/)")


def make_video(args):
    rng = np.random.default_rng(args.seed)
    h, w = (int(v) for v in args.size.lower().split("x"))
    scene = procedural_scene(h, w * 2, rng)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    if not vw.isOpened():
        raise IOError(f"Cannot create video {args.out} (use an ASCII path)")
    angle = float(rng.uniform(-15, 15))
    for f in range(args.frames):
        x0 = int(f / max(1, args.frames - 1) * w)        # slow camera pan
        frame = add_rain(scene[:, x0:x0 + w], rng, angle=angle)
        vw.write(cv2.cvtColor((frame * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"Synthetic rainy video written to {args.out} ({args.frames} frames, {h}x{w})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--out", required=True)
    d.add_argument("--n", type=int, default=40)
    d.add_argument("--size", type=int, default=128)
    d.add_argument("--clean-dir", help="optional: use your own clean images instead of procedural scenes")
    d.add_argument("--seed", type=int, default=0)
    v = sub.add_parser("video")
    v.add_argument("--out", required=True)
    v.add_argument("--frames", type=int, default=90)
    v.add_argument("--size", default="360x640")
    v.add_argument("--fps", type=float, default=30.0)
    v.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    make_dataset(args) if args.cmd == "dataset" else make_video(args)


if __name__ == "__main__":
    main()
