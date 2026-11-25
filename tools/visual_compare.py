"""Visual comparison: input | original w32 (Rain13K) | Restormer | mixed w32 300k | GT. Usage (from the repo root): python tools/visual_compare.py"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from derain.common import forward_padded, load_checkpoint_model  # noqa: E402
from compare_sota import restormer  # noqa: E402

OR = Path("../datasets/allweather/outdoor_rain/test/rainstreak")
NAMES = sorted(p.name for p in (OR / "input").iterdir())[::150][:5]
OUT = Path("results/visual_compare"); OUT.mkdir(parents=True, exist_ok=True)

models = {
    "Original-w32 (Rain13K)": (load_checkpoint_model("runs/light32/best.pt")[0], 8),
    "Restormer (official)": (restormer(), 8),
    "Ours mixed w32 300k": (load_checkpoint_model("runs/light32_mixed300k/last.pt")[0], 8),
}
for k in models:
    models[k] = (models[k][0].cuda().eval().half(), models[k][1])

def run(m, mult, img):
    x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[None].float().div(255).cuda().half()
    with torch.no_grad():
        y = forward_padded(m, x, mult).clamp(0, 1)
    return Image.fromarray((y[0].permute(1, 2, 0).float().cpu().numpy() * 255 + .5).astype(np.uint8))

for n in NAMES:
    inp = Image.open(OR / "input" / n).convert("RGB")
    gt = Image.open(OR / "gt" / n).convert("RGB")
    cols = [("Input", inp)] + [(k, run(m, mu, inp)) for k, (m, mu) in models.items()] + [("GT", gt)]
    w, h = inp.size; s = 0.5
    tw, th = int(w * s), int(h * s)
    sheet = Image.new("RGB", (tw * len(cols), th + 24), "white")
    d = ImageDraw.Draw(sheet)
    for i, (t, im) in enumerate(cols):
        sheet.paste(im.resize((tw, th)), (i * tw, 24)); d.text((i * tw + 4, 6), t, fill="black")
    sheet.save(OUT / f"{Path(n).stem}.jpg", quality=90)
    print(n)
