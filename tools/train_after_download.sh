#!/usr/bin/env bash
# 等 Rain13K 下載程序結束 → 檢查配對 → 背景訓練 LightDerainNet w32。
# 用法（在 code/ 下）：nohup bash tools/train_after_download.sh <下載程序 PID> > runs/light32_pipeline.log 2>&1 &
set -euo pipefail
PID=${1:?需要下載程序的 PID}
DATA=../datasets/Rain13K
PY=.venv/bin/python

while kill -0 "$PID" 2>/dev/null; do sleep 60; done
if [ "$(grep -c '^DONE' ../datasets/download.log)" -lt 2 ]; then
    echo "下載未完成（log 裡沒有兩個 DONE），不開始訓練"; exit 1
fi

$PY - <<'EOF'
from derain.data import find_pairs
for split in ["train", "test/Rain100L", "test/Rain100H", "test/Test100", "test/Test1200", "test/Test2800"]:
    print(split, len(find_pairs(f"../datasets/Rain13K/{split}")))
EOF

$PY -m derain.train --train-root $DATA/train --val-root $DATA/test/Rain100L \
    --width 32 --blocks 6 --patch 128 --batch 16 --iters 100000 --lr 2e-4 --amp \
    --workers 8 --val-every 5000 --log-every 500 --out runs/light32
