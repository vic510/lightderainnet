"""訓練自建輕量模型（可選知識蒸餾，用於結構壓縮實驗）。

範例：
  python -m derain.train --train-root D:/datasets/Rain200L/train --val-root D:/datasets/Rain200L/test --out runs/light32
  # 壓縮：寬度 16 的學生模型，以寬度 32 的模型為老師
  python -m derain.train ... --width 16 --teacher runs/light32/best.pt --kd-weight 0.5 --out runs/light16_kd
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .common import count_params, env_info, load_checkpoint_model
from .data import PairedDataset, find_pairs
from .evaluate import run_eval, summarize
from .metrics import ssim_loss
from .models import build_arch


def infinite(loader):
    while True:
        yield from loader


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-root", required=True)
    p.add_argument("--val-root", required=True)
    p.add_argument("--arch", default="light")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--iters", type=int, default=50000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ssim-weight", type=float, default=0.1, help="損失 = L1 + w·(1-SSIM)")
    p.add_argument("--teacher", help="知識蒸餾老師模型 checkpoint")
    p.add_argument("--kd-weight", type=float, default=0.5)
    p.add_argument("--amp", action="store_true", help="混合精度訓練（較快、省記憶體）")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--val-max", type=int, default=0, help="驗證最多用幾張（0=全部）")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", help="從 last.pt 接續")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device
    use_amp = args.amp and device.startswith("cuda")

    model_cfg = {"arch": args.arch, "width": args.width, "blocks": args.blocks}
    model = build_arch(**model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    teacher = load_checkpoint_model(args.teacher, device)[0] if args.teacher else None

    start_iter, best = 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_iter, best = ck["iter"], ck["best_psnr"]

    train_pairs = find_pairs(args.train_root)
    val_pairs = find_pairs(args.val_root)
    if args.val_max:
        val_pairs = val_pairs[: args.val_max]
    loader = DataLoader(PairedDataset(train_pairs, args.patch, augment=True), batch_size=args.batch,
                        shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.workers > 0)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "model_cfg": model_cfg, "params": count_params(model),
                   "n_train": len(train_pairs), "n_val": len(val_pairs), "env": env_info(device)},
                  f, ensure_ascii=False, indent=2)
    print(f"參數量 {count_params(model):,}；訓練 {len(train_pairs)} 對、驗證 {len(val_pairs)} 對")

    def save(name, it):
        torch.save({"model_cfg": model_cfg, "state_dict": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "iter": it, "best_psnr": best}, out / name)

    log_f = open(out / "log.csv", "a", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    if start_iter == 0:
        log.writerow(["iter", "loss", "lr", "val_psnr", "val_ssim", "elapsed_s"])
    t0, running = time.time(), 0.0
    batches = infinite(loader)
    model.train()
    for it in range(start_iter + 1, args.iters + 1):
        rain, clean, _ = next(batches)
        rain, clean = rain.to(device, non_blocking=True), clean.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            pred = model(rain)
            loss = F.l1_loss(pred, clean)
            if teacher is not None:
                with torch.no_grad():
                    target_t = teacher(rain)
                loss = loss + args.kd_weight * F.l1_loss(pred, target_t)
        if args.ssim_weight > 0:  # SSIM 以 float32 計算較穩定
            loss = loss + args.ssim_weight * ssim_loss(pred.float(), clean)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        running += loss.item()
        lr = sched.get_last_lr()[0]

        if it % args.log_every == 0:
            avg = running / args.log_every
            running = 0.0
            log.writerow([it, round(avg, 5), lr, "", "", round(time.time() - t0, 1)])
            log_f.flush()
            print(f"[{it}/{args.iters}] loss {avg:.4f}  lr {lr:.2e}")
        if it % args.val_every == 0 or it == args.iters:
            model.eval()
            s = summarize(run_eval(model, val_pairs, device))
            model.train()
            if s["psnr"] > best:
                best = s["psnr"]
                save("best.pt", it)
            save("last.pt", it)
            log.writerow([it, "", lr, round(s["psnr"], 3), round(s["ssim"], 4), round(time.time() - t0, 1)])
            log_f.flush()
            print(f"[{it}] val PSNR {s['psnr']:.3f} dB  SSIM {s['ssim']:.4f}  (best {best:.3f})")
    log_f.close()


if __name__ == "__main__":
    main()
