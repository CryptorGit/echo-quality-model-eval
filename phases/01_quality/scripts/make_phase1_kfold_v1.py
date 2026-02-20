from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def quality_band(q: float) -> str:
    qi = int(np.clip(round(float(q)), 0, 9))
    if qi <= 2:
        return "low"
    if qi <= 5:
        return "mid"
    return "high"


def read_phase1_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["path", "view", "quality", "group"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns {miss} in {path}")
    return df


def group_level_strata(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for g, gv in df.groupby("group"):
        v_mode = gv["view"].astype(str).value_counts().index[0]
        q_mean = float(gv["quality"].astype(float).mean())
        b = quality_band(q_mean)
        rows.append({
            "group": str(g),
            "view_mode": str(v_mode),
            "q_mean": q_mean,
            "band": b,
            "stratum": f"{v_mode}__{b}",
            "n": int(len(gv)),
        })
    return pd.DataFrame(rows)


def add_band(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["band"] = out["quality"].astype(float).map(quality_band)
    return out


def sanity_groups(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, int]:
    tr = set(train_df["group"].astype(str))
    va = set(val_df["group"].astype(str))
    te = set(test_df["group"].astype(str))
    return {
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_train_groups": int(len(tr)),
        "n_val_groups": int(len(va)),
        "n_test_groups": int(len(te)),
        "overlap_train_val": int(len(tr & va)),
        "overlap_train_test": int(len(tr & te)),
        "overlap_val_test": int(len(va & te)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create group-aware phase1 K-fold splits (v1 layout).")
    parser.add_argument("--base_split_dir", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, default=Path("splits/kfold_v1"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude_random", action="store_true")
    args = parser.parse_args()

    train_csv = args.base_split_dir / "phase1_train.csv"
    val_csv = args.base_split_dir / "phase1_val.csv"
    test_csv = args.base_split_dir / "phase1_test.csv"
    if not train_csv.exists() or not val_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Missing phase1 split CSVs under {args.base_split_dir}")

    tr_df = read_phase1_csv(train_csv)
    va_df = read_phase1_csv(val_csv)
    te_df = read_phase1_csv(test_csv)

    if args.exclude_random:
        tr_df = tr_df[tr_df["view"].astype(str) != "Random"].reset_index(drop=True)
        va_df = va_df[va_df["view"].astype(str) != "Random"].reset_index(drop=True)
        te_df = te_df[te_df["view"].astype(str) != "Random"].reset_index(drop=True)

    pool_df = pd.concat([tr_df, va_df], ignore_index=True)
    groups_df = group_level_strata(pool_df)

    skf = StratifiedKFold(n_splits=int(args.k), shuffle=True, random_state=int(args.seed))
    grp_arr = groups_df["group"].astype(str).to_numpy()
    y = groups_df["stratum"].astype(str).to_numpy()

    args.out_root.mkdir(parents=True, exist_ok=True)
    meta_dir = args.out_root / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    assignments = []
    fold_stats: Dict[str, dict] = {}

    for fold_i, (_, val_idx) in enumerate(skf.split(np.zeros(len(groups_df)), y)):
        fold_name = f"fold_{fold_i}"
        fold_dir = args.out_root / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        val_groups = set(grp_arr[val_idx].tolist())
        fold_train = pool_df[~pool_df["group"].astype(str).isin(val_groups)].reset_index(drop=True)
        fold_val = pool_df[pool_df["group"].astype(str).isin(val_groups)].reset_index(drop=True)
        fold_test = te_df.copy().reset_index(drop=True)

        fold_train = add_band(fold_train)
        fold_val = add_band(fold_val)
        fold_test = add_band(fold_test)

        stat = sanity_groups(fold_train, fold_val, fold_test)
        if stat["overlap_train_val"] or stat["overlap_train_test"] or stat["overlap_val_test"]:
            raise RuntimeError(f"Group overlap detected in {fold_name}: {stat}")

        for g in sorted(val_groups):
            assignments.append({"fold": int(fold_i), "group": str(g)})

        fold_train.to_csv(fold_dir / "train.csv", index=False)
        fold_val.to_csv(fold_dir / "val.csv", index=False)
        fold_test.to_csv(fold_dir / "test.csv", index=False)

        # Compatibility files for existing scripts.
        fold_train.to_csv(fold_dir / "phase1_train.csv", index=False)
        fold_val.to_csv(fold_dir / "phase1_val.csv", index=False)
        fold_test.to_csv(fold_dir / "phase1_test.csv", index=False)

        (fold_dir / "split_stats.json").write_text(json.dumps(stat, ensure_ascii=False, indent=2), encoding="utf-8")
        fold_stats[fold_name] = stat

    pd.DataFrame(assignments).to_csv(meta_dir / "fold_assignments.csv", index=False)
    (meta_dir / "kfold_meta.json").write_text(
        json.dumps(
            {
                "base_split_dir": str(args.base_split_dir),
                "k": int(args.k),
                "seed": int(args.seed),
                "exclude_random": bool(args.exclude_random),
                "fold_stats": fold_stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[OK] wrote kfold v1 splits: {args.out_root}")


if __name__ == "__main__":
    main()
