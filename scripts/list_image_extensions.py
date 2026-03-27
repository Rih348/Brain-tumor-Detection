#!/usr/bin/env python3
"""List file extensions in dataset folders."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan dataset folders recursively and report extension counts. "
            "Useful to detect unexpected formats (e.g. .dcm, .nii, files without extension)."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--class-dirs",
        nargs="+",
        default=["stroke", "normal"],
        help="Subdirectories under --data-dir to scan (default: stroke normal).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Maximum number of extensions to print per section.",
    )
    return parser.parse_args()


def normalized_extension(path: Path) -> str:
    # Keep common double extension visible.
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes[-2:] == [".nii", ".gz"]:
        return ".nii.gz"
    if not suffixes:
        return "<no_extension>"
    return suffixes[-1]


def scan_files(root: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for p in root.rglob("*"):
        if p.is_file():
            counts[normalized_extension(p)] += 1
    return counts


def print_counts(title: str, counts: Counter[str], limit: int) -> None:
    total = sum(counts.values())
    print(f"\n{title}")
    print(f"Total files: {total}")
    if total == 0:
        print("  (no files found)")
        return
    for ext, count in counts.most_common(limit):
        print(f"  {ext:>12} : {count}")


def summarize_supported(counts: Counter[str]) -> Dict[str, int]:
    supported = sum(count for ext, count in counts.items() if ext in SUPPORTED_IMAGE_EXTENSIONS)
    unsupported = sum(count for ext, count in counts.items() if ext not in SUPPORTED_IMAGE_EXTENSIONS)
    return {"supported": supported, "unsupported": unsupported, "total": supported + unsupported}


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    overall_counter: Counter[str] = Counter()
    for class_name in args.class_dirs:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"\n[warning] missing directory: {class_dir}")
            continue
        counts = scan_files(class_dir)
        overall_counter.update(counts)
        print_counts(f"{class_name} -> {class_dir}", counts, args.top)
        summary = summarize_supported(counts)
        print(
            "  Supported by train script: "
            f"{summary['supported']}/{summary['total']} "
            f"(unsupported: {summary['unsupported']})"
        )

    print_counts("Overall", overall_counter, args.top)
    overall_summary = summarize_supported(overall_counter)
    print(
        "\nOverall supported by train script: "
        f"{overall_summary['supported']}/{overall_summary['total']} "
        f"(unsupported: {overall_summary['unsupported']})"
    )
    print("\nSupported extensions:", ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS)))


if __name__ == "__main__":
    main()
