from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_raw_grades(grades_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(grades_dir.glob("*_grades.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No *_grades.csv found in: {grades_dir}")

    frames = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        required = {"Image Name", "Subfolder Name", "Grade"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"Unexpected schema in {csv_path}. required={required}")
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.rename(
        columns={"Image Name": "image_name", "Subfolder Name": "view", "Grade": "quality"}
    )
    return all_df


def plot_histogram(quality: pd.Series, title: str, out_path: Path) -> None:
    q = quality.dropna().astype(float).round().astype(int)
    if q.empty:
        return

    min_q, max_q = int(q.min()), int(q.max())
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
    parser = argparse.ArgumentParser(description="Plot raw CACTUS label histograms from Grades CSVs")
    parser.add_argument(
        "--grades_dir",
        default="datasets/CACTUS/extracted/Cactus Dataset/Grades",
    )
    parser.add_argument(
        "--out_dir",
        default="phases/01_quality/outputs/data_profile/cactus_raw_label_histograms",
    )
    args = parser.parse_args()

    grades_dir = Path(args.grades_dir)
    out_dir = Path(args.out_dir)

    df = load_raw_grades(grades_dir)
    df["view"] = df["view"].astype(str)
    df["quality"] = pd.to_numeric(df["quality"], errors="coerce")

    plot_histogram(
        df["quality"],
        f"CACTUS Raw Label Histogram (All Views, n={len(df)})",
        out_dir / "label_hist_all_views_raw.png",
    )

    for view in sorted(df["view"].dropna().unique().tolist()):
        gv = df[df["view"] == view]
        plot_histogram(
            gv["quality"],
            f"CACTUS Raw Label Histogram ({view}, n={len(gv)})",
            out_dir / f"label_hist_{view}_raw.png",
        )

    counts = (
        df.assign(quality_int=df["quality"].round().astype("Int64"))
        .dropna(subset=["quality_int"])
        .groupby(["view", "quality_int"])
        .size()
        .reset_index(name="count")
        .sort_values(["view", "quality_int"])
    )
    counts.to_csv(out_dir / "label_counts_by_view_raw.csv", index=False)

    print(f"[OK] wrote: {out_dir}")


if __name__ == "__main__":
    main()
