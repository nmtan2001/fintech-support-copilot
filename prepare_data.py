"""Download the official BANKING77 splits without modifying their membership."""

import argparse
from pathlib import Path

from fintech_support.data import download_data, read_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    download_data(args.data_dir)
    for split in ("train", "test"):
        frame = read_split(args.data_dir, split)
        print(f"{split}: {len(frame)} samples; {frame['category'].nunique()} intents")


if __name__ == "__main__":
    main()
