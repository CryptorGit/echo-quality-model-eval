from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd


def _resize_pad_square_bgr(img_bgr: np.ndarray, out_size: int, pad_value: int = 0) -> np.ndarray:
    if out_size <= 0:
        raise ValueError("out_size must be positive")
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("invalid image")

    scale = float(out_size) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((out_size, out_size, 3), pad_value, dtype=resized.dtype)
    y0 = (out_size - new_h) // 2
    x0 = (out_size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _resize_pad_to_size_bgr(img_bgr: np.ndarray, target_w: int, target_h: int, pad_value: int = 0) -> np.ndarray:
    if target_w <= 0 or target_h <= 0:
        raise ValueError("target size must be positive")
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("invalid image")

    scale = min(float(target_w) / float(w), float(target_h) / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_h, target_w, 3), pad_value, dtype=resized.dtype)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _load_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return mask


def _apply_mask_and_crop(img_bgr: np.ndarray, mask: np.ndarray, out_size: int, resize_mode: str) -> np.ndarray:
    h, w = mask.shape[:2]
    if resize_mode == "warp":
        img_aligned = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    elif resize_mode == "pad":
        img_aligned = _resize_pad_to_size_bgr(img_bgr, target_w=w, target_h=h, pad_value=0)
    else:
        raise ValueError(f"unknown resize_mode: {resize_mode}")
    masked = cv2.bitwise_and(img_aligned, img_aligned, mask=mask)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        crop = masked
    else:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        crop = masked[y0:y1 + 1, x0:x1 + 1]

    if resize_mode == "warp":
        final = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    elif resize_mode == "pad":
        final = _resize_pad_square_bgr(crop, out_size=out_size, pad_value=0)
    else:
        raise ValueError(f"unknown resize_mode: {resize_mode}")
    return final


def _out_path(root: Path, view: str, src_path: str) -> Path:
    src = Path(src_path)
    h = hashlib.sha1(src_path.encode("utf-8")).hexdigest()[:12]
    name = f"{src.stem}__{h}.jpg"
    return root / view / name


def process_manifest(
    in_csv: Path,
    out_csv: Path,
    out_root: Path,
    template_dir: Path,
    out_size: int,
    resize_mode: str,
    overwrite: bool,
) -> Dict[str, int]:
    df = pd.read_csv(in_csv)
    required = {"path", "view"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"{in_csv} missing columns: {sorted(miss)}")

    out_paths = []
    masks: Dict[str, np.ndarray] = {}

    n_total = len(df)
    n_written = 0
    n_skipped = 0
    n_missing = 0

    for i, row in df.iterrows():
        src = str(row["path"])
        view = str(row["view"])
        if view not in masks:
            masks[view] = _load_mask(template_dir / view / "mask.png")
        mask = masks[view]

        dst = _out_path(out_root, view, src)
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not overwrite:
            n_skipped += 1
            out_paths.append(str(dst))
            continue

        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            n_missing += 1
            out_paths.append(src)
            continue

        final = _apply_mask_and_crop(img, mask, out_size=out_size, resize_mode=resize_mode)
        cv2.imwrite(str(dst), final, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        n_written += 1
        out_paths.append(str(dst))

        if (i + 1) % 2000 == 0:
            print(f"  - {in_csv.name}: {i + 1}/{n_total}")

    out_df = df.copy()
    out_df["orig_path"] = out_df["path"]
    out_df["path"] = out_paths
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    stats = {
        "total": n_total,
        "written": n_written,
        "skipped": n_skipped,
        "missing": n_missing,
    }
    print(f"[OK] {in_csv.name} -> {out_csv.name} | {stats}")
    return stats


def parse_args():
    p = argparse.ArgumentParser(description="One-time CACTUS consensus preprocessing and manifest rewrite")
    root = Path(__file__).resolve().parents[3]
    p.add_argument("--template_dir", default=str(root / "phases" / "01_quality" / "outputs" / "sector_templates"))
    p.add_argument("--out_root", default=str(root / "datasets" / "CACTUS" / "processed" / "consensus224"))
    p.add_argument("--out_size", type=int, default=224)
    p.add_argument("--resize_mode", choices=["pad", "warp"], default="pad", help="pad=keep aspect ratio, warp=force square")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--group_in_dir", default=str(root / "datasets" / "CACTUS" / "manifests_group"))
    p.add_argument("--group_out_dir", default=str(root / "datasets" / "CACTUS" / "manifests_group_consensus"))

    p.add_argument("--base_in_dir", default=str(root / "datasets" / "CACTUS" / "manifests"))
    p.add_argument("--base_out_dir", default=str(root / "datasets" / "CACTUS" / "manifests_consensus"))
    return p.parse_args()


def main():
    args = parse_args()
    template_dir = Path(args.template_dir)
    out_root = Path(args.out_root)

    group_in_dir = Path(args.group_in_dir)
    group_out_dir = Path(args.group_out_dir)
    base_in_dir = Path(args.base_in_dir)
    base_out_dir = Path(args.base_out_dir)

    group_files = ["phase1_train.csv", "phase1_val.csv", "phase1_test.csv"]
    base_files = ["cactus_manifest.csv", "cactus_train.csv", "cactus_val.csv", "cactus_test.csv"]

    for name in group_files:
        in_csv = group_in_dir / name
        if not in_csv.exists():
            print(f"[WARN] missing: {in_csv}")
            continue
        process_manifest(
            in_csv=in_csv,
            out_csv=group_out_dir / name,
            out_root=out_root,
            template_dir=template_dir,
            out_size=int(args.out_size),
            resize_mode=str(args.resize_mode),
            overwrite=bool(args.overwrite),
        )

    for name in base_files:
        in_csv = base_in_dir / name
        if not in_csv.exists():
            print(f"[WARN] missing: {in_csv}")
            continue
        process_manifest(
            in_csv=in_csv,
            out_csv=base_out_dir / name,
            out_root=out_root,
            template_dir=template_dir,
            out_size=int(args.out_size),
            resize_mode=str(args.resize_mode),
            overwrite=bool(args.overwrite),
        )

    print("[DONE] CACTUS consensus preprocessing completed.")
    print(f"  images : {out_root}")
    print(f"  group manifests : {group_out_dir}")
    print(f"  base manifests  : {base_out_dir}")


if __name__ == "__main__":
    main()
