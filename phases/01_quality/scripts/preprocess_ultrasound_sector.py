from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


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


def _largest_component_bbox(bin_img: np.ndarray) -> Tuple[int, int, int, int]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    if num_labels <= 1:
        h, w = bin_img.shape
        return 0, 0, w, h
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = int(np.argmax(areas)) + 1
    x = int(stats[k, cv2.CC_STAT_LEFT])
    y = int(stats[k, cv2.CC_STAT_TOP])
    w = int(stats[k, cv2.CC_STAT_WIDTH])
    h = int(stats[k, cv2.CC_STAT_HEIGHT])
    return x, y, w, h


def _line_to_abc(x1, y1, x2, y2):
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return float(a), float(b), float(c)


def _intersect(l1, l2) -> Optional[Tuple[float, float]]:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    d = a1 * b2 - a2 * b1
    if abs(d) < 1e-8:
        return None
    x = (b1 * c2 - b2 * c1) / d
    y = (c1 * a2 - c2 * a1) / d
    return x, y


def _pick_boundary_lines(lines: np.ndarray, h: int, w: int):
    if lines is None or len(lines) == 0:
        return None, None

    left_candidates = []
    right_candidates = []
    cx = w * 0.5

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in line]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < min(h, w) * 0.20:
            continue
        if abs(dx) < 1e-6:
            continue

        slope = dy / dx
        ang = abs(math.degrees(math.atan2(dy, dx)))
        if ang < 20 or ang > 80:
            continue

        x_top = x1 + (0 - y1) * dx / dy if abs(dy) > 1e-6 else x1

        # image coords: left boundary tends to have positive slope, right negative
        score = length
        if slope > 0 and x_top < cx:
            left_candidates.append((score, (x1, y1, x2, y2)))
        elif slope < 0 and x_top > cx:
            right_candidates.append((score, (x1, y1, x2, y2)))

    left = max(left_candidates, key=lambda t: t[0])[1] if left_candidates else None
    right = max(right_candidates, key=lambda t: t[0])[1] if right_candidates else None
    return left, right


def _angles_between(theta1: float, theta2: float, n: int = 240) -> np.ndarray:
    # unwrap so sweep is <= pi
    d = theta2 - theta1
    while d <= -math.pi:
        d += 2 * math.pi
    while d > math.pi:
        d -= 2 * math.pi
    return np.linspace(theta1, theta1 + d, n)


def extract_sector(img_bgr: np.ndarray, thresh: int = 10, out_size: int = 224):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Step1: rough ROI (largest non-black component)
    bin0 = (gray > thresh).astype(np.uint8) * 255
    bin0 = cv2.morphologyEx(bin0, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
    x, y, w, h = _largest_component_bbox(bin0)
    crop = img_bgr[y:y + h, x:x + w].copy()
    crop_gray = gray[y:y + h, x:x + w].copy()

    # Step2: Canny + Hough lines
    edges = cv2.Canny(crop_gray, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=int(min(h, w) * 0.20), maxLineGap=15)
    left, right = _pick_boundary_lines(lines, h, w)

    # fallback mask: thresholded largest component in crop
    fallback = (crop_gray > thresh).astype(np.uint8) * 255
    fallback = cv2.morphologyEx(fallback, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)

    debug_lines = crop.copy()
    if left is not None:
        cv2.line(debug_lines, (int(left[0]), int(left[1])), (int(left[2]), int(left[3])), (0, 255, 0), 2)
    if right is not None:
        cv2.line(debug_lines, (int(right[0]), int(right[1])), (int(right[2]), int(right[3])), (0, 255, 255), 2)

    if left is None or right is None:
        mask = fallback
    else:
        l1 = _line_to_abc(*left)
        l2 = _line_to_abc(*right)
        apex = _intersect(l1, l2)
        if apex is None:
            mask = fallback
        else:
            ax, ay = apex
            cv2.circle(debug_lines, (int(round(ax)), int(round(ay))), 5, (0, 0, 255), -1)

            # choose ray directions by far endpoint from apex
            lp = np.array([[left[0], left[1]], [left[2], left[3]]], dtype=np.float32)
            rp = np.array([[right[0], right[1]], [right[2], right[3]]], dtype=np.float32)
            la = lp[np.argmax(np.sum((lp - np.array([ax, ay])) ** 2, axis=1))]
            ra = rp[np.argmax(np.sum((rp - np.array([ax, ay])) ** 2, axis=1))]

            theta_l = math.atan2(float(la[1] - ay), float(la[0] - ax))
            theta_r = math.atan2(float(ra[1] - ay), float(ra[0] - ax))
            thetas = _angles_between(theta_l, theta_r, n=220)

            r_max = int(math.hypot(w, h))
            arc_points = []
            for th in thetas:
                rs = np.arange(8, r_max, 1)
                xs = np.round(ax + rs * np.cos(th)).astype(np.int32)
                ys = np.round(ay + rs * np.sin(th)).astype(np.int32)
                valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                xs = xs[valid]
                ys = ys[valid]
                if len(xs) == 0:
                    continue
                vals = crop_gray[ys, xs]
                fg = np.where(vals > thresh)[0]
                if len(fg) == 0:
                    continue
                j = int(fg[-1])
                arc_points.append((int(xs[j]), int(ys[j])))

            if len(arc_points) < 16:
                mask = fallback
            else:
                # smooth arc
                arc = np.array(arc_points, dtype=np.int32)
                k = 7
                if len(arc) > k:
                    pad = k // 2
                    arc_pad = np.pad(arc, ((pad, pad), (0, 0)), mode="edge")
                    arc_sm = []
                    for i in range(len(arc)):
                        arc_sm.append(np.median(arc_pad[i:i + k], axis=0))
                    arc = np.array(arc_sm, dtype=np.int32)

                poly = np.vstack([
                    np.array([[int(round(ax)), int(round(ay))]], dtype=np.int32),
                    arc,
                ])

                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)

    # remove boundary text/ticks near edge
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)

    masked = cv2.bitwise_and(crop, crop, mask=mask)

    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        final_crop = masked[y0:y1 + 1, x0:x1 + 1]
    else:
        final_crop = masked

    final = _resize_pad_square_bgr(final_crop, out_size=out_size, pad_value=0)

    debug = {
        "gray": gray,
        "roi_crop": crop,
        "edges": edges,
        "debug_lines": debug_lines,
        "mask": mask,
        "masked": masked,
        "final": final,
    }
    return debug


def save_debug(debug: dict, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{stem}_01_roi.png"), debug["roi_crop"])
    cv2.imwrite(str(out_dir / f"{stem}_02_edges.png"), debug["edges"])
    cv2.imwrite(str(out_dir / f"{stem}_03_lines_apex.png"), debug["debug_lines"])
    cv2.imwrite(str(out_dir / f"{stem}_04_mask.png"), debug["mask"])
    cv2.imwrite(str(out_dir / f"{stem}_05_masked.png"), debug["masked"])
    cv2.imwrite(str(out_dir / f"{stem}_06_final_224.png"), debug["final"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--out_size", type=int, default=224)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.outdir)

    img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {in_path}")

    debug = extract_sector(img, thresh=args.threshold, out_size=args.out_size)
    save_debug(debug, out_dir, in_path.stem)

    print(f"[OK] wrote debug outputs to: {out_dir}")


if __name__ == "__main__":
    main()
