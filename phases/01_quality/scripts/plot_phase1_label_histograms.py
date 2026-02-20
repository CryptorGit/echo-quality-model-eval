from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_phase1_full_dataset(splits_root: Path, fold: str) -> pd.DataFrame:
    fold_dir = splits_root / fold
    csv_paths = [
        fold_dir / "phase1_train.csv",
        fold_dir / "phase1_val.csv",
        fold_dir / "phase1_test.csv",
    ]

    missing = [str(p) for p in csv_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CSV files: {missing}")

    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    if "path" in df.columns:
        df = df.drop_duplicates(subset=["path"]).reset_index(drop=True)
    return df


def plot_single_histogram(series: pd.Series, title: str, out_path: Path) -> None:
    q = series.dropna().astype(float).round().astype(int)
    if q.empty:
        return

    min_q = int(q.min())
    max_q = int(q.max())
    bins = list(range(min_q, max_q + 2))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(q.values, bins=bins, align="left", rwidth=0.85)
    ax.set_xticks(list(range(min_q, max_q + 1)))
    ax.set_xlabel("Quality Label")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot phase1 label histograms (overall + per-view)")
    parser.add_argument("--splits_root", default="splits/kfold_v1")
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument(
        "--out_dir",
        default="phases/01_quality/outputs/data_profile/label_histograms",
    )
    args = parser.parse_args()

    splits_root = Path(args.splits_root)
    out_dir = Path(args.out_dir)
    df = load_phase1_full_dataset(splits_root, args.fold)

    if "quality" not in df.columns or "view" not in df.columns:
        raise ValueError("Input CSV must contain 'quality' and 'view' columns")

    plot_single_histogram(
        df["quality"],
        f"All Views Label Histogram (n={len(df)})",
        out_dir / "label_hist_all_views.png",
    )

    for view in sorted(df["view"].dropna().astype(str).unique().tolist()):
        gv = df[df["view"].astype(str) == view]
        plot_single_histogram(
            gv["quality"],
            f"{view} Label Histogram (n={len(gv)})",
            out_dir / f"label_hist_{view}.png",
        )

    counts = (
        df.assign(quality_int=df["quality"].round().astype(int))
        .groupby(["view", "quality_int"])
        .size()
        .reset_index(name="count")
        .sort_values(["view", "quality_int"])
    )
    counts.to_csv(out_dir / "label_counts_by_view.csv", index=False)
    print(f"[OK] wrote histograms and counts to: {out_dir}")


if __name__ == "__main__":
    main()
