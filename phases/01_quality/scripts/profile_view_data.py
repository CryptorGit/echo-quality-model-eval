from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def summarize_series(x: pd.Series) -> dict:
    return {
        "count": int(len(x)),
        "min": float(np.min(x)) if len(x) else 0.0,
        "p10": float(np.percentile(x, 10)) if len(x) else 0.0,
        "median": float(np.median(x)) if len(x) else 0.0,
        "p90": float(np.percentile(x, 90)) if len(x) else 0.0,
        "max": float(np.max(x)) if len(x) else 0.0,
        "mean": float(np.mean(x)) if len(x) else 0.0,
    }


def coverage_ratio(path: str, threshold: int = 10) -> float:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return float("nan")
    return float((img > threshold).mean())


def main():
    p = argparse.ArgumentParser(description="Profile PSAV/SC label and sample distributions")
    p.add_argument("--split_csv", required=True)
    p.add_argument("--views", default="PSAV,SC")
    p.add_argument("--out_json", required=True)
    p.add_argument("--max_images_for_coverage", type=int, default=1200)
    args = p.parse_args()

    df = pd.read_csv(args.split_csv)
    views = [v.strip() for v in args.views.split(",") if v.strip()]

    report = {"split_csv": args.split_csv, "views": {}}
    rng = np.random.default_rng(42)

    for view in views:
        gv = df[df["view"] == view].copy()
        if len(gv) == 0:
            continue

        q_counts = gv["quality"].round().astype(int).value_counts().sort_index()
        group_sizes = gv.groupby("group").size().astype(int)

        n_cov = min(int(args.max_images_for_coverage), len(gv))
        sample_idx = rng.choice(len(gv), size=n_cov, replace=False)
        sample_paths = gv.iloc[sample_idx]["path"].astype(str).tolist()
        cov = [coverage_ratio(p) for p in sample_paths]
        cov = np.array([c for c in cov if np.isfinite(c)], dtype=np.float32)

        cov_summary = summarize_series(pd.Series(cov)) if len(cov) else {}
        low_paths = []
        if len(cov):
            thr = float(np.percentile(cov, 2))
            by_cov = []
            for pth in sample_paths:
                c = coverage_ratio(pth)
                if np.isfinite(c) and c <= thr:
                    by_cov.append((c, pth))
            by_cov.sort(key=lambda x: x[0])
            low_paths = [{"coverage": float(c), "path": p} for c, p in by_cov[:10]]

        report["views"][view] = {
            "n_rows": int(len(gv)),
            "quality_counts": {str(int(k)): int(v) for k, v in q_counts.items()},
            "group_size_summary": summarize_series(group_sizes),
            "coverage_ratio_summary_sampled": cov_summary,
            "lowest_coverage_examples": low_paths,
        }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
