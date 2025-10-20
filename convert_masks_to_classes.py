#!/usr/bin/env python3
# convert_masks_to_classes_four_mt.py
#
# Create 4-class image-level labels (clean / transparent / semi_transparent / opaque)
# from WoodScape soiling masks, scanning:
#   soiling_dataset/<SPLIT>/{rgb_images, gt_labels}
# Uses multithreading to speed up I/O-bound PNG reads.
#
# Example:
#   python convert_masks_to_classes_four_mt.py \
#       --root . \
#       --split train \
#       --out soiling_labels_4class_train.csv \
#       --workers 8

import argparse
import csv
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
import numpy as np


def parse_args():
    p = argparse.ArgumentParser("Generate 4-class labels from WoodScape soiling masks (multithreaded)")
    p.add_argument("--root", type=str, default=".", help="Project root containing soiling_dataset/")
    p.add_argument("--soiling-dir", type=str, default="soiling_dataset", help="Soiling dataset folder")
    p.add_argument("--split", type=str, default="train", help="Split subfolder (e.g., train, test)")
    p.add_argument("--rgb-dirname", type=str, default="rgbImages", help="RGB folder inside split")
    p.add_argument("--mask-dirname", type=str, default="rgbLabels", help="Mask folder inside split")
    p.add_argument("--info-json", type=str, default="soiling_annotation_info.json",
                   help="JSON in soiling_dataset/ with class_names and class_colors")
    p.add_argument("--ext", type=str, default="png", help="Image extension (png/jpg/jpeg)")
    p.add_argument("--out", type=str, default="soiling_labels_4class.csv", help="Output CSV path")
    p.add_argument("--strict-colors", action="store_true",
                   help="Exact RGB match; otherwise allow tolerance per channel")
    p.add_argument("--tol", type=int, default=0, help="Per-channel tolerance for non-strict matching")
    p.add_argument("--workers", type=int, default=8, help="Thread pool size (2–16 usually best)")
    return p.parse_args()


def load_info(info_path: Path):
    if not info_path.exists():
        sys.exit(f"[ERR] info JSON not found: {info_path}")
    info = json.loads(info_path.read_text())

    class_names = info.get("class_names", [])
    class_colors = info.get("class_colors", [])
    if not class_names or not class_colors or len(class_names) != len(class_colors):
        sys.exit("[ERR] Invalid info JSON: class_names/class_colors mismatch or empty.")

    # Expected classes
    expected = ["clean", "transparent", "semi_transparent", "Opaque"]
    for name in expected:
        if name not in class_names:
            sys.exit(f"[ERR] JSON must contain class '{name}'. Found: {class_names}")

    # Preserve JSON order; convert to tuples
    class_colors = [tuple(int(c) for c in rgb) for rgb in class_colors]
    return class_names, class_colors


def color_ratio(mask_rgb: np.ndarray, target_rgb: tuple, strict: bool, tol: int) -> float:
    if strict or tol <= 0:
        match = np.all(mask_rgb == np.array(target_rgb, dtype=np.uint8), axis=2)
    else:
        diff = np.abs(mask_rgb.astype(np.int16) - np.array(target_rgb, dtype=np.int16))
        match = np.all(diff <= tol, axis=2)
    return float(match.sum()) / (mask_rgb.shape[0] * mask_rgb.shape[1])


def process_one(rgb_path: Path, mask_path: Path, class_names, class_colors, strict, tol):
    """Return (image_path, label, *ratios) or None if mask missing."""
    if not mask_path.exists():
        return None

    mask = np.array(Image.open(mask_path).convert("RGB"))
    ratios = [color_ratio(mask, class_colors[i], strict, tol) for i in range(len(class_names))]
    label = class_names[int(np.argmax(ratios))]
    return (str(rgb_path), label, *(f"{r:.6f}" for r in ratios))


def main():
    args = parse_args()

    root = Path(args.root).resolve()
    sdir = root / args.soiling_dir
    split_dir = sdir / args.split
    rgb_dir = split_dir / args.rgb_dirname
    mask_dir = split_dir / args.mask_dirname
    info_json = sdir / args.info_json
    out_csv = Path(args.out)

    # Checks
    for p in [sdir, split_dir, rgb_dir, mask_dir, info_json]:
        if not p.exists():
            sys.exit(f"[ERR] Path not found: {p}")

    class_names, class_colors = load_info(info_json)
    ext = args.ext.lower().lstrip(".")
    rgb_files = sorted(rgb_dir.glob(f"*.{ext}"))
    if not rgb_files:
        sys.exit(f"[ERR] No *.{ext} files in {rgb_dir}")

    print(f"[INFO] Split: {args.split}")
    print(f"[INFO] Images: {len(rgb_files)} | Classes: {class_names}")
    print(f"[INFO] Threads: {args.workers} | strict={args.strict_colors} tol={args.tol}")

    rows = [("image_path", "label") + tuple(f"{name}_pct" for name in class_names)]
    missing = 0
    processed = 0

    # Multithreaded processing (I/O-bound)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = []
        for rgb_path in rgb_files:
            mask_path = mask_dir / rgb_path.name
            futures.append(ex.submit(
                process_one, rgb_path, mask_path,
                class_names, class_colors, args.strict_colors, args.tol
            ))

        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                missing += 1
            else:
                rows.append(res)
                processed += 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"[OK] Wrote {processed} rows to {out_csv}. Missing pairs: {missing}")
    print(f"[INFO] Folders:\n  RGB:  {rgb_dir}\n  MASK: {mask_dir}\n  INFO: {info_json}")


if __name__ == "__main__":
    main()
