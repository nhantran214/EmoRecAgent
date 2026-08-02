#!/usr/bin/env python3
"""
Convert unpacked Yelp JSON → Amazon-Reviews-2023-shaped JSONL for EmoRecAgent.

Does **not** change the recommender method: only rewrites fields so existing
``build_dataset`` / ABSA / Stage-1–2 pipelines can run unchanged.

Inputs (from ``download_yelp_open_dataset.py``):
  data/yelp-open-dataset/raw/yelp_json/{review,business}.json

Outputs (paths match ``configs/categories/Yelp.yaml``):
  data/yelp-open-dataset/raw/review_categories/Yelp.jsonl
  data/yelp-open-dataset/raw/meta_categories/meta_Yelp.jsonl

Examples:
  python3 scripts/prepare_yelp_dataset.py
  python3 scripts/prepare_yelp_dataset.py --cities Philadelphia --categories Restaurants
  python3 scripts/prepare_yelp_dataset.py --max-reviews 100000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from emorecagent.data.yelp import (
    BUSINESS_FILENAMES,
    REVIEW_FILENAMES,
    convert_businesses,
    convert_reviews,
    find_yelp_json,
)

DEFAULT_RAW = Path("data/yelp-open-dataset/raw")
DEFAULT_YELP_DIR = DEFAULT_RAW / "yelp_dataset"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--yelp-json-dir",
        type=Path,
        default=DEFAULT_YELP_DIR,
        help="Directory containing Yelp review/business JSON",
    )
    parser.add_argument(
        "--review-out",
        type=Path,
        default=DEFAULT_RAW / "review_categories" / "Yelp.jsonl",
    )
    parser.add_argument(
        "--meta-out",
        type=Path,
        default=DEFAULT_RAW / "meta_categories" / "meta_Yelp.jsonl",
    )
    parser.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help="Optional city filter (match business.city, case-insensitive)",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Optional category substrings (e.g. Restaurants Bars)",
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=None,
        help="Cap converted reviews (smoke / debug)",
    )
    args = parser.parse_args()

    yelp_dir: Path = args.yelp_json_dir
    review_src = find_yelp_json(yelp_dir, REVIEW_FILENAMES)
    business_src = find_yelp_json(yelp_dir, BUSINESS_FILENAMES)
    if review_src is None:
        raise SystemExit(
            f"No review JSON under {yelp_dir} "
            f"(looked for {REVIEW_FILENAMES}).\n"
            "Run: python3 scripts/download_yelp_open_dataset.py --archive ..."
        )
    if business_src is None:
        raise SystemExit(
            f"No business JSON under {yelp_dir} "
            f"(looked for {BUSINESS_FILENAMES})."
        )

    print(f"business: {business_src}")
    n_meta, keep_ids = convert_businesses(
        business_src,
        args.meta_out,
        cities=args.cities,
        categories_substr=args.categories,
    )
    print(f"Wrote {n_meta} meta rows → {args.meta_out}")

    filter_ids = keep_ids if (args.cities or args.categories) else None
    if filter_ids is not None and not filter_ids:
        raise SystemExit("City/category filter matched zero businesses.")

    print(f"reviews:  {review_src}")
    n_rev = convert_reviews(
        review_src,
        args.review_out,
        keep_business_ids=filter_ids,
        max_reviews=args.max_reviews,
    )
    print(f"Wrote {n_rev} review rows → {args.review_out}")
    if n_rev == 0:
        print("ERROR: zero reviews converted.", file=sys.stderr)
        raise SystemExit(1)
    print(
        "\nNext (same method as Amazon categories):\n"
        "  export CONFIG=configs/categories/Yelp.yaml\n"
        "  python3 scripts/build_dataset.py --config \"$CONFIG\" --log-dir logs/Yelp\n"
        "  python3 scripts/run_absa.py --config \"$CONFIG\" --log-dir logs/Yelp\n"
    )


if __name__ == "__main__":
    main()
