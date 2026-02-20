from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _quality_bin(q: float) -> str:
    # 0-3 low, 4-6 mid, 7-9 high
    q_i = int(np.clip(round(float(q)), 0, 9))
    if q_i <= 3:
        return "low"
    if q_i <= 6:
        return "mid"
    return "high"


def _dominant_view(df_g: pd.DataFrame) -> str:
    vc = df_g["view"].astype(str).value_counts()
    return str(vc.index[0])


def _read_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    required = ["path", "view", "quality", "group"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {p}")
    return df


def _group_strata(pool_df: pd.DataFrame) -> pd.DataFrame:
    # one row per group with a stratum label that balances view + quality band
    rows = []
    for g, df_g in pool_df.groupby("group"):
        q_mean = float(df_g["quality"].astype(float).mean())
        v_dom = _dominant_view(df_g)
        rows.append({
            "group": str(g),
            "q_mean": q_mean,
            "q_bin": _quality_bin(q_mean),
            "v_dom": v_dom,
            "stratum": f"{v_dom}__{_quality_bin(q_mean)}",
            "n": int(len(df_g)),
        })
    return pd.DataFrame(rows)


def _make_folds_stratified(groups_df: pd.DataFrame, k: int, seed: int) -> Dict[int, list[str]]:
    # StratifiedKFold on group-level strata (safe because groups are unique here)
    try:
        from sklearn.model_selection import StratifiedKFold
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for k-fold split generation") from e

    if k < 2:
        raise ValueError("k must be >= 2")
    if len(groups_df) < k:
        raise ValueError(f"Not enough groups ({len(groups_df)}) for k={k}")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(seed))
    y = groups_df["stratum"].astype(str).to_numpy()
    g = groups_df["group"].astype(str).to_numpy()

    folds: Dict[int, list[str]] = {i: [] for i in range(k)}
    for fold_i, (_, val_idx) in enumerate(skf.split(np.zeros(len(groups_df)), y)):
        folds[fold_i] = g[val_idx].tolist()
    return folds


def _sanity_check(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[bool, Dict[str, int]]:
    tr_g = set(train_df["group"].astype(str))
    va_g = set(val_df["group"].astype(str))
    te_g = set(test_df["group"].astype(str))

    ok = True
    if tr_g & va_g:
        ok = False
    if tr_g & te_g:
        ok = False
    if va_g & te_g:
        ok = False

    stats = {
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_train_groups": int(len(tr_g)),
        "n_val_groups": int(len(va_g)),
        "n_test_groups": int(len(te_g)),
        "overlap_train_val_groups": int(len(tr_g & va_g)),
        "overlap_train_test_groups": int(len(tr_g & te_g)),
        "overlap_val_test_groups": int(len(va_g & te_g)),
    }
    return ok, stats


@dataclass
class Args:
    base_split_dir: Path
    out_root: Path
    k: int
    seed: int


def main() -> None:
    p = argparse.ArgumentParser(description="Create group-aware K-fold manifests for Phase1.")
    p.add_argument(
        "--base_split_dir",
        type=Path,
        required=True,
        help="Directory containing phase1_train.csv/phase1_val.csv/phase1_test.csv",
    )
    p.add_argument(
        "--out_root",
        type=Path,
        default=None,
        help="Output root directory (default: sibling next to base_split_dir with suffix)",
    )
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args_ns = p.parse_args()

    base_split_dir = Path(args_ns.base_split_dir)
    if args_ns.out_root is None:
        out_root = base_split_dir.parent / f"{base_split_dir.name}_kfold{int(args_ns.k)}_seed{int(args_ns.seed)}"
    else:
        out_root = Path(args_ns.out_root)

    args = Args(base_split_dir=base_split_dir, out_root=out_root, k=int(args_ns.k), seed=int(args_ns.seed))

    train_csv = args.base_split_dir / "phase1_train.csv"
    val_csv = args.base_split_dir / "phase1_val.csv"
    test_csv = args.base_split_dir / "phase1_test.csv"
    if not train_csv.exists() or not val_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Missing phase1_* CSVs in {args.base_split_dir}")

    tr_df = _read_csv(train_csv)
    va_df = _read_csv(val_csv)
    te_df = _read_csv(test_csv)

    pool_df = pd.concat([tr_df, va_df], ignore_index=True)
    groups_df = _group_strata(pool_df)

    folds = _make_folds_stratified(groups_df, k=args.k, seed=args.seed)

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "_meta").mkdir(parents=True, exist_ok=True)

    # Save assignment table
    assign_rows = []
    for fold_i, val_groups in folds.items():
        for g in val_groups:
            assign_rows.append({"fold": int(fold_i), "group": str(g)})
    assign_df = pd.DataFrame(assign_rows).merge(groups_df, on="group", how="left")
    assign_df.to_csv(out_root / "_meta" / "fold_assignments.csv", index=False)

    meta = {
        "base_split_dir": str(args.base_split_dir),
        "k": int(args.k),
        "seed": int(args.seed),
        "n_pool": int(len(pool_df)),
        "n_test": int(len(te_df)),
        "n_groups_pool": int(pool_df["group"].nunique()),
        "n_groups_test": int(te_df["group"].nunique()),
    }

    # Write per-fold manifests as separate directories with informative names
    fold_stats = {}
    for fold_i in range(args.k):
        fold_name = f"f{fold_i:02d}of{args.k:02d}"
        fold_dir = out_root / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        val_groups = set(folds[fold_i])
        df_val = pool_df[pool_df["group"].astype(str).isin(val_groups)].reset_index(drop=True)
        df_train = pool_df[~pool_df["group"].astype(str).isin(val_groups)].reset_index(drop=True)

        ok, stats = _sanity_check(df_train, df_val, te_df)
        stats["ok"] = bool(ok)
        fold_stats[fold_name] = stats
        if not ok:
            raise RuntimeError(f"Group overlap detected in fold={fold_name}: {stats}")

        df_train.to_csv(fold_dir / "phase1_train.csv", index=False)
        df_val.to_csv(fold_dir / "phase1_val.csv", index=False)
        te_df.to_csv(fold_dir / "phase1_test.csv", index=False)

        # lightweight per-fold stats
        (fold_dir / "phase1_split_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    meta["folds"] = fold_stats
    (out_root / "_meta" / "kfold_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote k-fold manifests under: {out_root}")


if __name__ == "__main__":
    main()
