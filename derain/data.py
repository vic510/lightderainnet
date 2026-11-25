"""Paired deraining datasets (Rain13K, Rain100L/H, Rain200L/H, Rain1400, SPA-Data, etc.).

Layout: root/<rain dir>/... and root/<clean dir>/... (subfolders are searched recursively).
Pairs by file name when both sides share it; otherwise by the first number in the name,
so rain-001.png <-> norain-001.png, norain-1x2.png <-> norain-1.png, 1_3.jpg <-> 1.jpg all match.
"""
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
RAIN_DIRS = ("rain", "input", "rainy", "data", "degraded")
CLEAN_DIRS = ("norain", "target", "gt", "clean", "groundtruth")


def read_image(path) -> np.ndarray:
    """Read as RGB uint8. imdecode is used so non-ASCII Windows paths work."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_image(path, rgb: np.ndarray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"Cannot write image: {path}")
    buf.tofile(str(path))


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """CHW [0,1] tensor -> HWC uint8."""
    return (t.clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()


def _find_dir(root: Path, names):
    for n in names:
        if (root / n).is_dir():
            return root / n
    raise FileNotFoundError(f"None of the folders {names} found under {root}; specify them with --rain-dir/--clean-dir")


def _list_images(d: Path):
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS)


def _image_id(p: Path):
    m = re.search(r"\d+", p.stem)
    return int(m.group()) if m else p.stem


def _index(paths, key):
    out = {}
    for p in paths:
        k = key(p)
        if k in out:
            return None  # duplicates exist, so this key cannot be used for pairing
        out[k] = p
    return out


def find_pairs(root, rain_dir=None, clean_dir=None):
    """Prefer pairing by full file name (e.g. Rain13K uses identical names); fall back to the first number in the name."""
    root = Path(root)
    rd = root / rain_dir if rain_dir else _find_dir(root, RAIN_DIRS)
    cd = root / clean_dir if clean_dir else _find_dir(root, CLEAN_DIRS)
    rain_imgs, clean_imgs = _list_images(rd), _list_images(cd)
    clean = _index(clean_imgs, lambda p: p.stem)
    key = lambda p: p.stem  # noqa: E731
    if clean is None or not all(p.stem in clean for p in rain_imgs):
        clean, key = _index(clean_imgs, _image_id), _image_id
        if clean is None:
            raise ValueError(f"Clean image names or numbers in {cd} are duplicated; cannot pair")
    pairs, missing = [], []
    for p in rain_imgs:
        k = key(p)
        if k in clean:
            pairs.append((p, clean[k]))
        else:
            missing.append(p)
    if not pairs:
        raise FileNotFoundError(f"No pairable images between {rd} and {cd}")
    if missing:
        print(f"[warning] {len(missing)} rain images have no matching clean image and were skipped (e.g. {missing[0].name})")
    return pairs


class PairedDataset(Dataset):
    def __init__(self, pairs, patch=None, augment=False):
        self.pairs = pairs
        self.patch = patch
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        rp, cp = self.pairs[i]
        rain, clean = read_image(rp), read_image(cp)
        if rain.shape != clean.shape:
            raise ValueError(f"Size mismatch: {rp.name} {rain.shape} vs {cp.name} {clean.shape}")
        if self.patch:
            rain, clean = self._random_crop(rain, clean, self.patch)
        if self.augment:
            if random.random() < 0.5:
                rain, clean = rain[:, ::-1], clean[:, ::-1]
            if random.random() < 0.5:
                rain, clean = rain[::-1], clean[::-1]
        return to_tensor(rain), to_tensor(clean), rp.name

    @staticmethod
    def _random_crop(a, b, size):
        h, w = a.shape[:2]
        if h < size or w < size:
            ph, pw = max(0, size - h), max(0, size - w)
            pad = ((0, ph), (0, pw), (0, 0))
            a, b = np.pad(a, pad, mode="reflect"), np.pad(b, pad, mode="reflect")
            h, w = a.shape[:2]
        y, x = random.randint(0, h - size), random.randint(0, w - size)
        return a[y:y + size, x:x + size], b[y:y + size, x:x + size]
