"""Downstream detection evaluation: Outdoor-Rain Test1 has no box annotations,
so YOLO detections on the clean GT (conf>=0.25) serve as pseudo-labels; we then measure mAP50, mAP50-95 and recall of the rainy input and of each model's output.
The numbers show how close detection on the restored image is to detection on the clean image; they are not absolute mAP against human annotations.

Usage (from the repo root): python tools/downstream_det.py --gt DIR --variant input=DIR --variant w32=DIR ...
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou
from ultralytics import YOLO

IOU_TH = np.arange(0.5, 0.96, 0.05)


def detect(model, files, conf, batch=16):
    out = {}
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        for f, r in zip(chunk, model.predict([str(f) for f in chunk], conf=conf, verbose=False, half=True)):
            b = r.boxes
            out[f.name] = (b.xyxy.cpu(), b.cls.cpu().long(), b.conf.cpu())
    return out


def ap_at(recall, precision):
    # COCO-style 101-point interpolation
    mpre = np.maximum.accumulate(np.concatenate([precision, [0]])[::-1])[::-1]
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, rs, side="left")
    return float(np.mean([mpre[i] if i < len(recall) else 0 for i in idx]))


def evaluate(gt, pred):
    classes = sorted({int(c) for _, cl, _ in gt.values() for c in cl})
    aps = np.zeros((len(classes), len(IOU_TH)))
    tp_total = n_gt_total = 0
    for ci, c in enumerate(classes):
        scores, tps, n_gt = [], [], 0
        for name, (gb, gc, _) in gt.items():
            g = gb[gc == c]
            n_gt += len(g)
            pb, pc, ps = pred.get(name, (torch.zeros(0, 4), torch.zeros(0), torch.zeros(0)))
            m = pc == c
            p, s = pb[m], ps[m]
            order = s.argsort(descending=True)
            p, s = p[order], s[order]
            tp = np.zeros((len(p), len(IOU_TH)), bool)
            if len(p) and len(g):
                iou = box_iou(p, g).numpy()
                for ti, th in enumerate(IOU_TH):
                    used = np.zeros(len(g), bool)
                    for k in range(len(p)):
                        cand = np.where((iou[k] >= th) & ~used)[0]
                        if len(cand):
                            j = cand[iou[k, cand].argmax()]
                            used[j] = tp[k, ti] = True
            scores.append(s.numpy())
            tps.append(tp)
        s = np.concatenate(scores)
        tp = np.concatenate(tps) if tps else np.zeros((0, len(IOU_TH)))
        order = np.argsort(-s)
        tp = tp[order]
        for ti in range(len(IOU_TH)):
            ctp = np.cumsum(tp[:, ti])
            recall = ctp / max(n_gt, 1)
            precision = ctp / np.arange(1, len(ctp) + 1)
            aps[ci, ti] = ap_at(recall, precision)
        # recall at conf>=0.25 and IoU 0.5
        keep = s[order] >= 0.25
        tp_total += int(tp[keep, 0].sum())
        n_gt_total += n_gt
    return {"mAP50": aps[:, 0].mean(), "mAP50_95": aps.mean(), "recall@0.25": tp_total / max(n_gt_total, 1),
            "n_classes": len(classes), "n_gt_boxes": n_gt_total}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True)
    p.add_argument("--variant", action="append", required=True, help="name=folder")
    p.add_argument("--weights", default="yolo11s.pt")
    p.add_argument("--out", default="results/downstream_det.jsonl")
    args = p.parse_args()
    model = YOLO(args.weights)
    files = sorted(Path(args.gt).glob("*.png"))
    gt = detect(model, files, 0.25)
    for v in args.variant:
        name, d = v.split("=", 1)
        vf = [Path(d) / f.name for f in files]
        r = {"variant": name, "weights": args.weights, **evaluate(gt, detect(model, vf, 0.001))}
        print(json.dumps(r, ensure_ascii=False))
        with open(args.out, "a") as fh:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
