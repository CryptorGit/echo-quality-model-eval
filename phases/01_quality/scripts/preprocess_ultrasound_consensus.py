from __future__ import annotations

import argparse
from pathlib import Path

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


def _largest_component(bin_img: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    if num_labels <= 1:
        return bin_img
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    out = np.zeros_like(bin_img)
    out[labels == idx] = 255
    return out


def _postprocess_mask(mask: np.ndarray, erode_px: int = 3) -> np.ndarray:
    kernel3 = np.ones((3, 3), np.uint8)
    kernel7 = np.ones((7, 7), np.uint8)
    proc = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, kernel7, iterations=2)
    proc = _largest_component(proc)

    cnts, _ = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        hull_mask = np.zeros_like(proc)
        cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)
        proc = hull_mask

    if erode_px > 0:
        k = np.ones((erode_px, erode_px), np.uint8)
        proc = cv2.erode(proc, k, iterations=1)
    return proc


def _resize_pad_to_size_gray(img: np.ndarray, target_w: int, target_h: int, pad_value: int = 0, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    if target_w <= 0 or target_h <= 0:
        raise ValueError("target size must be positive")
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("invalid image")
    scale = min(float(target_w) / float(w), float(target_h) / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    canvas = np.full((target_h, target_w), pad_value, dtype=resized.dtype)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
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


def _read_mask(path: str, size: int, threshold: int) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Threshold first, then keep aspect ratio while letterboxing to the template size.
    bin_img = ((img > threshold).astype(np.uint8) * 255)
    if bin_img.shape[0] != size or bin_img.shape[1] != size:
        bin_img = _resize_pad_to_size_gray(bin_img, target_w=size, target_h=size, pad_value=0, interpolation=cv2.INTER_NEAREST)
    mask = (bin_img > 0).astype(np.uint8)
    return mask


def build_templates(
    manifest_csv: Path,
    out_dir: Path,
    size: int,
    threshold: int,
    freq_thresh: float,
    max_images: int,
    by_view: bool,
    views: set[str] | None,
    erode_px: int,
) -> None:
    df = pd.read_csv(manifest_csv)
    if "path" not in df.columns:
        raise ValueError("manifest must contain 'path' column")
    if "view" not in df.columns:
        raise ValueError("manifest must contain 'view' column")

    if views:
        df = df[df["view"].isin(views)].copy()
    if len(df) == 0:
        raise ValueError("no rows after filtering by view")

    key_series = df["view"] if by_view else pd.Series(["ALL"] * len(df), index=df.index)
    groups: dict[str, list[str]] = {}
    for key, sub in df.groupby(key_series):
        paths = sub["path"].astype(str).tolist()
        if max_images > 0 and len(paths) > max_images:
            step = max(1, len(paths) // max_images)
            paths = paths[::step][:max_images]
        groups[str(key)] = paths

    out_dir.mkdir(parents=True, exist_ok=True)
    for key, paths in groups.items():
        acc = np.zeros((size, size), dtype=np.float32)
        used = 0
        for p in paths:
            m = _read_mask(p, size=size, threshold=threshold)
            if m is None:
                continue
            acc += m
            used += 1

        if used == 0:
            print(f"[WARN] {key}: no readable image")
            continue

        freq = acc / float(used)
        raw = (freq >= freq_thresh).astype(np.uint8) * 255
        mask = _postprocess_mask(raw, erode_px=erode_px)

        key_dir = out_dir / key
        key_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(key_dir / "freq.png"), np.clip(freq * 255.0, 0, 255).astype(np.uint8))
        cv2.imwrite(str(key_dir / "mask.png"), mask)
        np.save(str(key_dir / "freq.npy"), freq)

        coverage = float((mask > 0).mean())
        print(f"[OK] {key}: used={used}, coverage={coverage:.3f}, saved={key_dir}")


def apply_template(
    image_path: Path,
    mask_path: Path,
    out_dir: Path,
    out_size: int,
    resize_mode: str,
) -> None:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"cannot read mask: {mask_path}")

    h, w = mask.shape[:2]
    img_rs = _resize_pad_to_size_bgr(img, target_w=w, target_h=h, pad_value=0)
    masked = cv2.bitwise_and(img_rs, img_rs, mask=mask)

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

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    cv2.imwrite(str(out_dir / f"{stem}_01_resized.png"), img_rs)
    cv2.imwrite(str(out_dir / f"{stem}_02_template_mask.png"), mask)
    cv2.imwrite(str(out_dir / f"{stem}_03_masked.png"), masked)
    cv2.imwrite(str(out_dir / f"{stem}_04_final_{out_size}.png"), final)
    print(f"[OK] wrote demo to: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Consensus sector template by overlaying non-black pixels")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--manifest", required=True)
    p_build.add_argument("--outdir", required=True)
    p_build.add_argument("--size", type=int, default=512)
    p_build.add_argument("--threshold", type=int, default=10)
    p_build.add_argument("--freq_thresh", type=float, default=0.10)
    p_build.add_argument("--max_images", type=int, default=0, help="0=all")
    p_build.add_argument("--by_view", action="store_true")
    p_build.add_argument("--views", default="")
    p_build.add_argument("--erode_px", type=int, default=3)

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--input", required=True)
    p_apply.add_argument("--mask", required=True)
    p_apply.add_argument("--outdir", required=True)
    p_apply.add_argument("--out_size", type=int, default=224)
    p_apply.add_argument("--resize_mode", choices=["pad", "warp"], default="pad")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.cmd == "build":
        views = {v.strip() for v in args.views.split(",") if v.strip()}
        build_templates(
            manifest_csv=Path(args.manifest),
            out_dir=Path(args.outdir),
            size=int(args.size),
            threshold=int(args.threshold),
            freq_thresh=float(args.freq_thresh),
            max_images=int(args.max_images),
            by_view=bool(args.by_view),
            views=views if views else None,
            erode_px=int(args.erode_px),
        )
        return

    if args.cmd == "apply":
        apply_template(
            image_path=Path(args.input),
            mask_path=Path(args.mask),
            out_dir=Path(args.outdir),
            out_size=int(args.out_size),
            resize_mode=str(args.resize_mode),
        )


if __name__ == "__main__":
    main()
