from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def load_phase1_full_df(splits_root: Path, fold: str) -> pd.DataFrame:
    fold_dir = splits_root / fold
    csv_paths = [
        fold_dir / "phase1_train.csv",
        fold_dir / "phase1_val.csv",
        fold_dir / "phase1_test.csv",
    ]
    missing = [str(path) for path in csv_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CSV files: {missing}")

    df = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
    if "path" in df.columns:
        df = df.drop_duplicates(subset=["path"]).reset_index(drop=True)
    return df


def ensure_path(path_value: str, workspace_root: Path) -> Path:
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return workspace_root / path_obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export sample images per (view, quality) for both raw and padded datasets"
    )
    parser.add_argument("--workspace_root", default=".")
    parser.add_argument("--splits_root", default="splits/kfold_v1")
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--samples_per_pair", type=int, default=2)
    parser.add_argument(
        "--out_dir",
        default="phases/01_quality/outputs/data_profile/quality_samples",
    )
    parser.add_argument("--include_random", action="store_true")
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    splits_root = workspace_root / args.splits_root
    out_dir = workspace_root / args.out_dir

    df = load_phase1_full_df(splits_root=splits_root, fold=args.fold)
    required_cols = {"view", "quality", "path", "orig_path"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV must include columns: {required_cols}")

    df["view"] = df["view"].astype(str)
    if not args.include_random:
        df = df[df["view"] != "Random"].copy()

    df["quality_int"] = pd.to_numeric(df["quality"], errors="coerce").round().astype("Int64")
    df = df.dropna(subset=["quality_int"]).copy()
    df["quality_int"] = df["quality_int"].astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for (view, quality_int), group_df in df.groupby(["view", "quality_int"], sort=True):
        selected = group_df.head(args.samples_per_pair).copy()
        for sample_idx, row in enumerate(selected.itertuples(index=False), start=1):
            raw_src = ensure_path(str(row.orig_path), workspace_root)
            padded_src = ensure_path(str(row.path), workspace_root)

            sample_dir = out_dir / view / f"q{quality_int}" / f"sample_{sample_idx:02d}"
            raw_dst_dir = sample_dir / "raw"
            padded_dst_dir = sample_dir / "padded"
            raw_dst_dir.mkdir(parents=True, exist_ok=True)
            padded_dst_dir.mkdir(parents=True, exist_ok=True)

            raw_dst = raw_dst_dir / raw_src.name
            padded_dst = padded_dst_dir / padded_src.name

            raw_ok = raw_src.exists()
            padded_ok = padded_src.exists()

            if raw_ok:
                shutil.copy2(raw_src, raw_dst)
            if padded_ok:
                shutil.copy2(padded_src, padded_dst)

            rows.append(
                {
                    "view": view,
                    "quality": quality_int,
                    "sample_idx": sample_idx,
                    "raw_src": str(raw_src),
                    "padded_src": str(padded_src),
                    "raw_dst": str(raw_dst) if raw_ok else "",
                    "padded_dst": str(padded_dst) if padded_ok else "",
                    "raw_exists": raw_ok,
                    "padded_exists": padded_ok,
                }
            )

    manifest_df = pd.DataFrame(rows)
    manifest_path = out_dir / "sample_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    summary = (
        manifest_df.groupby(["view", "quality"], as_index=False)
        .agg(
            samples=("sample_idx", "count"),
            raw_ok=("raw_exists", "sum"),
            padded_ok=("padded_exists", "sum"),
        )
        .sort_values(["view", "quality"])
    )
    summary_path = out_dir / "sample_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"[OK] exported samples to: {out_dir}")
    print(f"[OK] manifest: {manifest_path}")
    print(f"[OK] summary: {summary_path}")


if __name__ == "__main__":
    main()
