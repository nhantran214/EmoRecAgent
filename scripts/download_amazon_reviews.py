#!/usr/bin/env python3
"""
Download Amazon Reviews 2023 from Hugging Face (McAuley-Lab/Amazon-Reviews-2023).

Dataset overview:
  - ~571M reviews, 54.5M users, 48.2M items across 33 categories
  - Total size on Hugging Face: ~750 GB
  - Docs: https://amazon-reviews-2023.github.io/

Two download modes:
  1. files   - Download raw JSONL/CSV files via huggingface_hub (recommended)
  2. dataset - Load via `datasets` API and optionally export to JSONL

Examples:
  # List categories and estimated sizes
  python scripts/download_amazon_reviews.py --list-categories

  # Download reviews + metadata for one small category (~100 MB compressed)
  python scripts/download_amazon_reviews.py --category All_Beauty --data-type review meta

  # Download pre-split benchmark data (train/valid/test)
  python scripts/download_amazon_reviews.py --category All_Beauty --data-type 0core_timestamp

  # Download shared mapping files
  python scripts/download_amazon_reviews.py --shared-files
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"

CATEGORIES = [
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Arts_Crafts_and_Sewing",
    "Automotive",
    "Baby_Products",
    "Beauty_and_Personal_Care",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Digital_Music",
    "Electronics",
    "Gift_Cards",
    "Grocery_and_Gourmet_Food",
    "Handmade_Products",
    "Health_and_Household",
    "Health_and_Personal_Care",
    "Home_and_Kitchen",
    "Industrial_and_Scientific",
    "Kindle_Store",
    "Magazine_Subscriptions",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Software",
    "Sports_and_Outdoors",
    "Subscription_Boxes",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Video_Games",
    "Unknown",
]

# Approximate number of ratings per category (from dataset card).
CATEGORY_RATINGS = {
    "All_Beauty": 701_500,
    "Amazon_Fashion": 2_500_000,
    "Appliances": 2_100_000,
    "Arts_Crafts_and_Sewing": 9_000_000,
    "Automotive": 20_000_000,
    "Baby_Products": 6_000_000,
    "Beauty_and_Personal_Care": 23_900_000,
    "Books": 29_500_000,
    "CDs_and_Vinyl": 4_800_000,
    "Cell_Phones_and_Accessories": 20_800_000,
    "Clothing_Shoes_and_Jewelry": 66_000_000,
    "Digital_Music": 130_400,
    "Electronics": 43_900_000,
    "Gift_Cards": 152_400,
    "Grocery_and_Gourmet_Food": 14_300_000,
    "Handmade_Products": 664_200,
    "Health_and_Household": 25_600_000,
    "Health_and_Personal_Care": 494_100,
    "Home_and_Kitchen": 67_400_000,
    "Industrial_and_Scientific": 5_200_000,
    "Kindle_Store": 25_600_000,
    "Magazine_Subscriptions": 71_500,
    "Movies_and_TV": 17_300_000,
    "Musical_Instruments": 3_000_000,
    "Office_Products": 12_800_000,
    "Patio_Lawn_and_Garden": 16_500_000,
    "Pet_Supplies": 16_800_000,
    "Software": 4_900_000,
    "Sports_and_Outdoors": 19_600_000,
    "Subscription_Boxes": 16_200,
    "Tools_and_Home_Improvement": 27_000_000,
    "Toys_and_Games": 16_300_000,
    "Video_Games": 4_600_000,
    "Unknown": 63_800_000,
}

# Config names used by `datasets.load_dataset(...)` (dataset mode).
DATASET_CONFIG_PREFIXES = {
    "review": "raw_review",
    "meta": "raw_meta",
    "0core_rating_only": "0core_rating_only",
    "0core_timestamp": "0core_timestamp",
    "0core_timestamp_w_his": "0core_timestamp_w_his",
    "5core_rating_only": "5core_rating_only",
    "5core_timestamp": "5core_timestamp",
    "5core_timestamp_w_his": "5core_timestamp_w_his",
}

SHARED_FILES = [
    "all_categories.txt",
    "asin2category.json",
]


def build_config_name(data_type: str, category: str) -> str:
    prefix = DATASET_CONFIG_PREFIXES[data_type]
    return f"{prefix}_{category}"


def build_file_patterns(data_types: list[str], categories: list[str]) -> list[str]:
    """Map logical data types to actual paths in the HF dataset repo."""
    patterns: list[str] = []
    for category in categories:
        for data_type in data_types:
            if data_type == "review":
                patterns.append(f"raw/review_categories/{category}.jsonl")
            elif data_type == "meta":
                patterns.append(f"raw/meta_categories/meta_{category}.jsonl")
            elif data_type == "0core_rating_only":
                patterns.append(f"benchmark/0core/rating_only/{category}.csv")
            elif data_type == "0core_timestamp":
                patterns.append(f"benchmark/0core/timestamp/{category}.*.csv")
            elif data_type == "0core_timestamp_w_his":
                patterns.append(f"benchmark/0core/timestamp_w_his/{category}.*.csv")
            elif data_type == "5core_rating_only":
                patterns.append(f"benchmark/5core/rating_only/{category}.csv")
            elif data_type == "5core_timestamp":
                patterns.append(f"benchmark/5core/timestamp/{category}.*.csv")
            elif data_type == "5core_timestamp_w_his":
                patterns.append(f"benchmark/5core/timestamp_w_his/{category}.*.csv")
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
    return patterns


def list_categories() -> None:
    print(f"{'Category':<35} {'#Ratings':>12}")
    print("-" * 50)
    for category in CATEGORIES:
        ratings = CATEGORY_RATINGS.get(category, "?")
        print(f"{category:<35} {ratings:>12,}")
    print()
    print("Tip: start with All_Beauty or Digital_Music for quick testing.")
    print("Full dataset is ~750 GB — download one category at a time.")


def download_raw_files(
    categories: list[str],
    data_types: list[str],
    output_dir: Path,
    token: str | None,
) -> None:
    from huggingface_hub import snapshot_download

    allow_patterns = build_file_patterns(data_types, categories)

    print(f"Downloading to: {output_dir.resolve()}")
    print(f"Categories: {', '.join(categories)}")
    print(f"Data types: {', '.join(data_types)}")
    print(f"Patterns: {allow_patterns}")

    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(output_dir),
        allow_patterns=allow_patterns,
        token=token,
    )
    print(f"Done. Files saved under: {path}")


def download_shared_files(output_dir: Path, token: str | None) -> None:
    from huggingface_hub import hf_hub_download

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in SHARED_FILES:
        print(f"Downloading {filename} ...")
        cached = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=str(output_dir),
            token=token,
        )
        print(f"  -> {cached}")


def load_via_datasets(
    categories: list[str],
    data_types: list[str],
    output_dir: Path,
    export_jsonl: bool,
    max_samples: int | None,
) -> None:
    import datasets

    datasets.logging.set_verbosity_error()

    for category in categories:
        for data_type in data_types:
            config = build_config_name(data_type, category)
            print(f"\nLoading config: {config}")

            dataset = datasets.load_dataset(
                REPO_ID,
                config,
                trust_remote_code=True,
            )

            if export_jsonl:
                out_path = output_dir / f"{config}.jsonl"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                _export_to_jsonl(dataset, out_path, max_samples)
                print(f"Exported to {out_path}")
            else:
                print(dataset)
                split = next(iter(dataset.keys()))
                print(f"Sample from split '{split}':")
                print(dataset[split][0])


def _export_to_jsonl(dataset, out_path: Path, max_samples: int | None) -> None:
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for split_name in dataset.keys():
            split = dataset[split_name]
            for row in split:
                record = dict(row)
                record["_split"] = split_name
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                if max_samples is not None and count >= max_samples:
                    print(f"  Stopped at max_samples={max_samples}")
                    return
    print(f"  Wrote {count:,} rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Amazon Reviews 2023 from Hugging Face.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="Print available categories and exit.",
    )
    parser.add_argument(
        "--category",
        nargs="+",
        choices=CATEGORIES,
        metavar="CATEGORY",
        help="One or more categories to download (e.g. All_Beauty Books).",
    )
    parser.add_argument(
        "--data-type",
        nargs="+",
        choices=list(DATASET_CONFIG_PREFIXES.keys()),
        default=["review"],
        help="Data type(s) to download. Default: review.",
    )
    parser.add_argument(
        "--mode",
        choices=["files", "dataset"],
        default="files",
        help="files: raw HF files (default). dataset: load via datasets API.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/amazon-reviews-2023"),
        help="Local output directory.",
    )
    parser.add_argument(
        "--shared-files",
        action="store_true",
        help="Download all_categories.txt and asin2category.json.",
    )
    parser.add_argument(
        "--export-jsonl",
        action="store_true",
        help="(dataset mode) Export loaded data to JSONL files.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="(dataset mode + --export-jsonl) Limit exported rows per config.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (optional, for gated/private assets).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_categories:
        list_categories()
        return 0

    if args.shared_files:
        download_shared_files(args.output_dir, args.token)
        if not args.category:
            return 0

    if not args.category:
        print(
            "Error: specify --category CATEGORY or use --list-categories.",
            file=sys.stderr,
        )
        return 1

    if args.mode == "files":
        download_raw_files(
            categories=args.category,
            data_types=args.data_type,
            output_dir=args.output_dir,
            token=args.token,
        )
    else:
        load_via_datasets(
            categories=args.category,
            data_types=args.data_type,
            output_dir=args.output_dir,
            export_jsonl=args.export_jsonl,
            max_samples=args.max_samples,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
