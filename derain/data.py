"""成對去雨資料集（Rain13K、Rain100L/H、Rain200L/H、Rain1400、SPA-Data 等）。

資料夾結構：root/<rain 目錄>/... 與 root/<clean 目錄>/...（子資料夾會遞迴搜尋）。
兩邊同名時直接以檔名配對；否則以檔名中的第一個數字配對，
因此 rain-001.png ↔ norain-001.png、norain-1x2.png ↔ norain-1.png、1_3.jpg ↔ 1.jpg 都能對上。
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
    """讀成 RGB uint8。用 imdecode 才能讀含中文的 Windows 路徑。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"無法讀取影像：{path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_image(path, rgb: np.ndarray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"無法寫入影像：{path}")
    buf.tofile(str(path))


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """CHW [0,1] tensor -> HWC uint8。"""
    return (t.clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()


def _find_dir(root: Path, names):
    for n in names:
        if (root / n).is_dir():
            return root / n
    raise FileNotFoundError(f"{root} 下找不到 {names} 任一資料夾，請用 --rain-dir/--clean-dir 指定")


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
            return None  # 有重複，這種鍵不能用來配對
        out[k] = p
    return out


def find_pairs(root, rain_dir=None, clean_dir=None):
    """優先用完整檔名配對（如 Rain13K 兩邊同名）；不行才用檔名中的第一個數字。"""
    root = Path(root)
    rd = root / rain_dir if rain_dir else _find_dir(root, RAIN_DIRS)
    cd = root / clean_dir if clean_dir else _find_dir(root, CLEAN_DIRS)
    rain_imgs, clean_imgs = _list_images(rd), _list_images(cd)
    clean = _index(clean_imgs, lambda p: p.stem)
    key = lambda p: p.stem  # noqa: E731
    if clean is None or not all(p.stem in clean for p in rain_imgs):
        clean, key = _index(clean_imgs, _image_id), _image_id
        if clean is None:
            raise ValueError(f"{cd} 的乾淨影像檔名或編號有重複，無法配對")
    pairs, missing = [], []
    for p in rain_imgs:
        k = key(p)
        if k in clean:
            pairs.append((p, clean[k]))
        else:
            missing.append(p)
    if not pairs:
        raise FileNotFoundError(f"{rd} 與 {cd} 之間沒有可配對的影像")
    if missing:
        print(f"[警告] {len(missing)} 張雨圖找不到對應乾淨圖，已略過（例如 {missing[0].name}）")
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
            raise ValueError(f"尺寸不一致：{rp.name} {rain.shape} vs {cp.name} {clean.shape}")
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
