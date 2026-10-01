"""Prepare train-only statistics for the FR3/Wuji reader before torchrun."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openwam.dataloader.fr3_wuji import compute_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out", default="outputs/fr3_wuji/normalization_stats.npy")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    payload = compute_stats(args.dataset_dir, args.out, validation_fraction=args.validation_fraction, seed=args.seed)
    print(f"Saved {Path(args.out).resolve()} ({payload['train_episodes']} training episodes)")


if __name__ == "__main__":
    main()
