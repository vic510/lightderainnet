"""下載 Rain13K（Hugging Face 鏡像 dronefreak/Rain13K，非官方，授權標示 unknown，僅供學術使用並引用原始論文）。

資料夾保持鏡像原樣：train/{input,target}/part_*/ 與 test/<測試集>/{input,target}/。
derain.data.find_pairs 會自動辨認 input/target 並遞迴搜尋 part_*，不需要再整理。
（該鏡像的 parquet 自動轉檔是空的，所以只能逐檔下載。）

用法：
  python tools/fetch_rain13k.py --out ../datasets/Rain13K --part test    # 先抓測試集（約 4,300 對）
  python tools/fetch_rain13k.py --out ../datasets/Rain13K --part train   # 訓練集（13,711 對）
"""
import argparse

from huggingface_hub import snapshot_download

PATTERNS = {"test": ["test/**", "README.md"], "train": ["train/**"], "all": ["*"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--part", default="all", choices=list(PATTERNS))
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()
    path = snapshot_download("dronefreak/Rain13K", repo_type="dataset", local_dir=args.out,
                             allow_patterns=PATTERNS[args.part], max_workers=args.workers)
    print("DONE", path)


if __name__ == "__main__":
    main()
