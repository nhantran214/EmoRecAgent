#!/usr/bin/env python3
"""
Download / unpack the Yelp Open Dataset (educational).

License / ToS: you must accept Yelp's terms on the official page before use:
  https://business.yelp.com/data/resources/open-dataset/

Yelp does not publish a stable anonymous CDN URL. Typical workflow:

  1. Download the JSON TAR from the page above (browser).
  2. Point this script at the archive:

       python3 scripts/download_yelp_open_dataset.py \\
         --archive ~/Downloads/yelp_dataset.tar

  Optional: if you have a direct URL (institutional mirror), pass ``--url``.

Extracts into:
  data/yelp-open-dataset/raw/yelp_dataset/   (yelp_academic_dataset_*.json)

Then build the processed split directly (native Yelp rows are normalized on read):

  export CONFIG=configs/categories/Yelp.yaml
  python3 scripts/build_dataset.py --config \"$CONFIG\" --log-dir logs/Yelp
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

DEFAULT_OUT = Path("data/yelp-open-dataset")
EXTRACT_SUBDIR = Path("raw/yelp_dataset")
DATASET_PAGE = "https://business.yelp.com/data/resources/open-dataset/"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} → {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 — user-supplied URL
    print(f"Saved {dest} ({dest.stat().st_size / 1e9:.2f} GB)")


def _extract_tar(archive: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} → {dest_dir}")
    with tarfile.open(archive, mode="r:*") as tf:
        # Python 3.12+ supports filter=; keep portable.
        try:
            tf.extractall(dest_dir, filter="data")  # type: ignore[call-arg]
        except TypeError:
            tf.extractall(dest_dir)
    print("Extract done.")


def _summarize(extract_dir: Path) -> None:
    json_files = sorted(extract_dir.rglob("*.json"))
    if not json_files:
        print(
            f"WARNING: no *.json under {extract_dir}. "
            "Check that the archive is the Yelp JSON package.",
            file=sys.stderr,
        )
        return
    print("JSON files:")
    for p in json_files:
        rel = p.relative_to(extract_dir) if p.is_relative_to(extract_dir) else p
        print(f"  {rel}  ({p.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Local Yelp JSON TAR (after accepting ToS on the dataset page)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Optional direct download URL (institutional mirror / your copy)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Dataset root (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Only download (or verify archive); do not extract",
    )
    args = parser.parse_args()

    out_root: Path = args.output_dir
    extract_dir = out_root / EXTRACT_SUBDIR
    archives_dir = out_root / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)

    archive = args.archive
    if args.url:
        dest = archives_dir / Path(args.url).name
        if not dest.name.endswith((".tar", ".tgz", ".tar.gz", ".zip")):
            dest = archives_dir / "yelp_dataset.tar"
        _download(args.url, dest)
        archive = dest

    if archive is None:
        print(
            "Yelp Open Dataset requires accepting the license in a browser.\n"
            f"  1. Open {DATASET_PAGE}\n"
            "  2. Download the JSON package (TAR)\n"
            "  3. Re-run:\n"
            "       python3 scripts/download_yelp_open_dataset.py "
            "--archive /path/to/yelp_dataset.tar\n",
            file=sys.stderr,
        )
        raise SystemExit(2)

    archive = Path(archive).expanduser().resolve()
    if not archive.is_file():
        raise SystemExit(f"Archive not found: {archive}")

    # Keep a copy under data/ for reproducibility if sourced elsewhere.
    local_copy = archives_dir / archive.name
    if archive.resolve() != local_copy.resolve():
        if not local_copy.exists() or local_copy.stat().st_size != archive.stat().st_size:
            print(f"Copying archive → {local_copy}")
            shutil.copy2(archive, local_copy)

    if args.skip_extract:
        print(f"Archive ready: {local_copy if local_copy.exists() else archive}")
        return

    _extract_tar(archive, extract_dir)
    _summarize(extract_dir)
    print(
        "\nNext:\n"
        "  export CONFIG=configs/categories/Yelp.yaml\n"
        "  python3 scripts/build_dataset.py --config \"$CONFIG\" --log-dir logs/Yelp\n"
        "  # optional filter export still available:\n"
        "  # python3 scripts/prepare_yelp_dataset.py --cities Philadelphia\n"
    )


if __name__ == "__main__":
    main()
