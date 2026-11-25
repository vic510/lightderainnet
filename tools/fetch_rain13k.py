"""Downloads Rain13K (Hugging Face mirror dronefreak/Rain13K; unofficial, license marked unknown; academic use only, cite the original papers).

The folder layout mirrors the source: train/{input,target}/part_*/ and test/<test set>/{input,target}/.
derain.data.find_pairs auto-detects input/target and searches part_* recursively, so no reorganizing is needed.
(The mirror's auto-converted parquet is empty, so files must be downloaded one by one.)

Usage:
  python tools/fetch_rain13k.py --out ../datasets/Rain13K --part test    # test set first (about 4,300 pairs)
  python tools/fetch_rain13k.py --out ../datasets/Rain13K --part train   # training set (13,711 pairs)
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
