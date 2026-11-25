# 去雨實驗程式（GPU 加速專題）

影像去雨（derain）的 PyTorch 實驗程式：自建輕量模型訓練、知識蒸餾、PSNR/SSIM 評估、模型延遲量測、完整影片流程量測，以及 ONNX/TensorRT 匯出。
訓練、PSNR/SSIM 評估、模型延遲量測、完整影片流程量測，以及 ONNX/TensorRT 匯出。
所有結果都會連同硬體與軟體版本，自動寫進 `results/*.jsonl`，之後可以追溯。

## 安裝

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130   # 依你的 CUDA 版本調整
pip install -r requirements.txt
```

訓練要跑很久，建議用 `tmux` 或 `nohup`：

```bash
nohup python -m derain.train ... --out runs/light32 > runs/light32.log 2>&1 &
```

以下指令都在 `code/` 目錄下執行。

> **路徑注意**：OpenCV 在 Windows 上讀寫影片時，不支援含中文的路徑。影片和資料集建議放在純英文路徑，例如 `D:\datasets\`。影像讀寫已經處理過中文路徑。

## 檔案

| 模組 | 用途 |
| --- | --- |
| `derain/models.py` | 自建輕量模型 `LightDerainNet`（寬度、深度可調，用來做結構壓縮實驗） |
| `derain/data.py` | 成對資料集載入（Rain100L/H、Rain200L/H、Rain1400、SPA-Data 等，用檔名數字配對） |
| `derain/metrics.py` | PSNR、SSIM（預設 8-bit 量化後在 Y 通道計算） |
| `derain/train.py` | 訓練；`--teacher` 可做知識蒸餾，`--amp` 是混合精度 |
| `derain/evaluate.py` | 測試集畫質評估，輸出每張圖的 CSV 和平均值 |
| `derain/benchmark.py` | 只量模型 forward：mean/median/p95 延遲、FPS、峰值 GPU 記憶體、GFLOPs |
| `derain/video.py` | 完整影片流程：`sequential`（含各階段耗時）和 `pipelined`（CPU 編解碼與 GPU 推論平行） |
| `derain/export_onnx.py` | 匯出 ONNX，給 TensorRT `trtexec` 量測 |
| `derain/report.py` | 把結果整理成 Markdown 表格 |
| `derain/synth.py` | 產生合成雨圖和影片，**只用來確認程式能跑，結果不能寫進報告** |

## 資料集

Rain13K 放在 `../datasets/Rain13K`，由 `tools/fetch_rain13k.py` 下載。
來源是 Hugging Face 的非官方鏡像 `dronefreak/Rain13K`，授權標示為 unknown，只做學術使用，並引用原始論文。
匿名下載大約每秒 4 個檔案，全部約 36,000 個檔案，需要 2～3 小時。

`tools/train_after_download.sh` 會等下載結束、檢查配對張數，再自動開始訓練 `runs/light32`。
下載進度看 `../datasets/download.log`，訓練進度看 `runs/light32_pipeline.log`。

## 典型流程

```powershell
# 0. 煙霧測試（不需要資料集）
python -m derain.synth dataset --out data/smoke --n 40 --size 128
python -m derain.train --train-root data/smoke/train --val-root data/smoke/test --iters 200 --val-every 100 --batch 4 --patch 96 --workers 0 --out runs/smoke

# 1. 訓練自建 baseline（以 Rain200L 為例）
python -m derain.train --train-root D:/datasets/Rain200L/train --val-root D:/datasets/Rain200L/test --amp --out runs/light32

# 2. 畫質評估（FP32 與 FP16 各一次，比較精度造成的品質差異）
python -m derain.evaluate --ckpt runs/light32/best.pt --data D:/datasets/Rain200L/test --precision fp32 --tag B-fp32
python -m derain.evaluate --ckpt runs/light32/best.pt --data D:/datasets/Rain200L/test --precision fp16 --tag D-fp16

# 3. 模型延遲（多個解析度 × 精度）
python -m derain.benchmark --ckpt runs/light32/best.pt --size 480x640 720x1280 --precision fp32 fp16 --tag B
python -m derain.benchmark --ckpt runs/light32/best.pt --size 720x1280 --precision fp16 --channels-last --cudnn-benchmark --tag C

# 4. 結構壓縮：寬度 16 的學生模型 + 知識蒸餾
python -m derain.train ... --width 16 --teacher runs/light32/best.pt --kd-weight 0.5 --out runs/light16_kd

# 5. 完整影片流程
python -m derain.video --ckpt runs/light32/best.pt --input D:/videos/rain.mp4 --mode sequential --tag V-seq
python -m derain.video --ckpt runs/light32/best.pt --input D:/videos/rain.mp4 --mode pipelined --precision fp16 --output D:/videos/out.mp4 --side-by-side --tag V-pipe

# 6. 整理成表格
python -m derain.report --results results > results/summary.md
```

## 外部模型（DRSformer、SSDRNet 等）

外部模型不在這裡重寫，而是從官方程式庫直接載入，確保它跟論文是同一份實作。以 DRSformer 為例：

```powershell
git clone https://github.com/cschenxiang/DRSformer D:/repos/DRSformer
# 先依該 repo README 下載預訓練權重
python -m derain.benchmark --ext-repo D:/repos/DRSformer --ext-module basicsr.models.archs.DRSformer_arch --ext-class DRSformer --ext-weights D:/repos/DRSformer/pretrained_models/xxx.pth --size 480x640 --precision fp32 fp16 --tag DRSformer
```

`--ext-kwargs` 可以用 JSON 傳建構參數。權重檔裡的 `params`、`state_dict` 等常見鍵會自動解開。
外部 repo 可能有額外的相依套件，需要依它的 README 安裝。DRSformer 在 4 GB VRAM 上跑高解析度時可能 OOM，程式會記錄成 `OOM`，不會中斷。

## 量測規則（寫報告時請一併寫上）

- **模型延遲**：batch=1、固定尺寸、warm-up 20 次後量 100 次。每次用 CUDA Event 計時並同步，回報 median 與 p95。不含讀檔、編解碼。
- **完整流程**：`sequential` 的延遲是一幀從讀取到寫出的時間。`pipelined` 是從讀到該幀、到寫出的時間，包含排隊。吞吐量 = 幀數 ÷ 總時間。
  - 用影片檔測 `pipelined` 時，讀檔比真實攝影機快很多，幀會塞在佇列裡，延遲因此被高估。這種情況用 `--pace` 依來源幀率送幀，量到的才是攝影機情境的延遲。
  - 吞吐量（最多能處理多快）不加 `--pace`；延遲（攝影機情境）加 `--pace`。兩者分開報告。
- **畫質**：輸出先量化成 8-bit，在 Y 通道計算，預設不裁邊。換成 RGB 或裁邊的數字，不能跟其他設定直接比較。
- FP16 會改變數值，所以要重新評估畫質，不能只看速度。
