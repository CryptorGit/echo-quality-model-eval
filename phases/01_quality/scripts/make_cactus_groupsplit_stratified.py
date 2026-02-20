from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


GROUP_PAT = re.compile(r"(\d+_(?:D\d+|V\d+|DR\d+|Folder\d+))", re.IGNORECASE)


def extract_group(path_str: str) -> str:
    name = Path(path_str).name
    m = GROUP_PAT.search(name)
    if m:
        return m.group(1)
    return "UNMATCHED"


def grade_int(q: float) -> int:
    return int(np.clip(np.rint(q), 0, 9))


@dataclass
class SplitTargets:
    train_frac: float
    val_frac: float
    test_frac: float


def build_group_vectors(
    df: pd.DataFrame,
    views: List[str],
) -> Tuple[List[str], np.ndarray, Dict[Tuple[str, int], int], np.ndarray]:
    """Returns (groups, group_vectors, dim_index, group_totals)."""
    dim_index: Dict[Tuple[str, int], int] = {}
    dim = 0
    for v in views:
        for g in range(10):
            dim_index[(v, g)] = dim
            dim += 1

    groups = sorted(df["group"].astype(str).unique().tolist())
    g_to_i = {g: i for i, g in enumerate(groups)}

    X = np.zeros((len(groups), dim), dtype=np.float64)
    totals = np.zeros((len(groups),), dtype=np.float64)

    for _, r in df.iterrows():
        g = str(r["group"])
        v = str(r["view"])
        if v not in views:
            continue
        q = grade_int(float(r["quality"]))
        j = dim_index[(v, q)]
        i = g_to_i[g]
        X[i, j] += 1.0
        totals[i] += 1.0

    return groups, X, dim_index, totals


def greedy_assign_groups(
    groups: List[str],
    X: np.ndarray,
    totals: np.ndarray,
    targets: SplitTargets,
    seed: int,
) -> Dict[str, str]:
    """Assign each group to train/val/test to match multi-view bin counts."""
    rng = np.random.default_rng(seed)

    total_all = float(totals.sum())
    target_total = {
        "train": total_all * targets.train_frac,
        "val": total_all * targets.val_frac,
        "test": total_all * targets.test_frac,
    }

    n_groups_all = float(len(groups))
    target_groups = {
        "train": n_groups_all * targets.train_frac,
        "val": n_groups_all * targets.val_frac,
        "test": n_groups_all * targets.test_frac,
    }

    target_dim = {
        "train": X.sum(axis=0) * targets.train_frac,
        "val": X.sum(axis=0) * targets.val_frac,
        "test": X.sum(axis=0) * targets.test_frac,
    }

    # Order groups by size (desc), but keep deterministic tie-breaking.
    order = np.argsort(-totals, kind="mergesort")

    cur_total = {"train": 0.0, "val": 0.0, "test": 0.0}
    cur_groups = {"train": 0.0, "val": 0.0, "test": 0.0}
    cur_dim = {
        "train": np.zeros((X.shape[1],), dtype=np.float64),
        "val": np.zeros((X.shape[1],), dtype=np.float64),
        "test": np.zeros((X.shape[1],), dtype=np.float64),
    }

    splits = ["train", "val", "test"]

    assignment: Dict[str, str] = {}

    def split_error(split: str, dim_vec: np.ndarray, total: float, n_groups: float) -> float:
        td = target_dim[split]
        tt = target_total[split]
        tg = target_groups[split]

        rel_dim = (dim_vec - td) / (td + 1.0)
        rel_total = (total - tt) / (tt + 1.0)
        rel_groups = (n_groups - tg) / (tg + 1.0)

        # total/group terms are important to prevent degenerate assignments.
        w_total = 2.0
        w_groups = 0.5
        return float((rel_dim * rel_dim).mean() + w_total * (rel_total * rel_total) + w_groups * (rel_groups * rel_groups))

    def global_objective() -> float:
        return float(
            split_error("train", cur_dim["train"], cur_total["train"], cur_groups["train"])
            + split_error("val", cur_dim["val"], cur_total["val"], cur_groups["val"])
            + split_error("test", cur_dim["test"], cur_total["test"], cur_groups["test"])
        )

    # Process groups (largest first, deterministic).
    remaining = list(order.tolist())
    for idx in remaining:
        g = groups[int(idx)]
        x_g = X[int(idx)]
        t_g = float(totals[int(idx)])

        best_s: str | None = None
        best_obj: float | None = None

        # Try each possible split and pick the assignment that minimizes the GLOBAL objective.
        for s in splits:
            cur_total[s] += t_g
            cur_dim[s] += x_g
            cur_groups[s] += 1.0
            obj = global_objective()
            cur_total[s] -= t_g
            cur_dim[s] -= x_g
            cur_groups[s] -= 1.0

            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_s = s

        assert best_s is not None
        assignment[g] = best_s
        cur_total[best_s] += t_g
        cur_dim[best_s] += x_g
        cur_groups[best_s] += 1.0

    return assignment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Group split with stratification across views/quality bins")
    root = Path(__file__).resolve().parents[3]
    p.add_argument(
        "--manifest_csv",
        default=str(root / "datasets" / "CACTUS" / "manifests_consensus" / "cactus_manifest.csv"),
    )
    p.add_argument(
        "--out_dir",
        default=str(root / "datasets" / "CACTUS" / "manifests_group_consensus_stratified"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument(
        "--views",
        default="A4C,PL,PSAV,PSMV,SC",
        help="Comma-separated views to stratify on (others are kept but not in strat objective)",
    )
    p.add_argument(
        "--exclude_random",
        action="store_true",
        help="Exclude view == Random (recommended for Phase1)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    manifest_csv = Path(args.manifest_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_csv)
    for c in ["path", "view", "quality"]:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {manifest_csv}")

    # Prefer extracting group from orig_path if available (less likely to have hash suffix).
    group_src_col = "orig_path" if "orig_path" in df.columns else "path"

    df = df.copy()
    df["quality"] = df["quality"].astype(float)
    df["group"] = df[group_src_col].astype(str).map(extract_group)

    if args.exclude_random:
        df = df[df["view"].astype(str) != "Random"].reset_index(drop=True)

    df = df[df["group"].astype(str) != "UNMATCHED"].reset_index(drop=True)

    views = [v.strip() for v in str(args.views).split(",") if v.strip()]

    groups, X, _, totals = build_group_vectors(df, views=views)
    if len(groups) < 3:
        raise RuntimeError(f"Need at least 3 groups to split, got {len(groups)}")

    fr_train = float(args.train_frac)
    fr_val = float(args.val_frac)
    fr_test = float(args.test_frac)
    s = fr_train + fr_val + fr_test
    if not np.isclose(s, 1.0):
        raise ValueError(f"Fractions must sum to 1.0, got {s}")

    assignment = greedy_assign_groups(
        groups=groups,
        X=X,
        totals=totals,
        targets=SplitTargets(train_frac=fr_train, val_frac=fr_val, test_frac=fr_test),
        seed=int(args.seed),
    )

    tr_groups = {g for g, sp in assignment.items() if sp == "train"}
    va_groups = {g for g, sp in assignment.items() if sp == "val"}
    te_groups = {g for g, sp in assignment.items() if sp == "test"}

    tr = df[df["group"].isin(tr_groups)].sample(frac=1.0, random_state=int(args.seed)).reset_index(drop=True)
    va = df[df["group"].isin(va_groups)].sample(frac=1.0, random_state=int(args.seed)).reset_index(drop=True)
    te = df[df["group"].isin(te_groups)].sample(frac=1.0, random_state=int(args.seed)).reset_index(drop=True)

    overlap_tv = len(set(tr["group"]) & set(va["group"]))
    overlap_tt = len(set(tr["group"]) & set(te["group"]))
    overlap_vt = len(set(va["group"]) & set(te["group"]))
    if overlap_tv != 0 or overlap_tt != 0 or overlap_vt != 0:
        raise RuntimeError("Group overlap detected")

    tr_path = out_dir / "phase1_train.csv"
    va_path = out_dir / "phase1_val.csv"
    te_path = out_dir / "phase1_test.csv"

    tr.to_csv(tr_path, index=False)
    va.to_csv(va_path, index=False)
    te.to_csv(te_path, index=False)

    stats = {
        "manifest_csv": str(manifest_csv),
        "group_source_col": group_src_col,
        "rows": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "groups": {
            "train": int(tr["group"].nunique()),
            "val": int(va["group"].nunique()),
            "test": int(te["group"].nunique()),
        },
        "overlap": {"train_val": overlap_tv, "train_test": overlap_tt, "val_test": overlap_vt},
        "views": sorted(df["view"].astype(str).unique().tolist()),
        "exclude_random": bool(args.exclude_random),
        "fracs": {"train": fr_train, "val": fr_val, "test": fr_test},
        "seed": int(args.seed),
        "stratify_views": views,
    }

    stats_path = out_dir / "phase1_split_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {tr_path}")
    print(f"[OK] {va_path}")
    print(f"[OK] {te_path}")
    print(f"[OK] {stats_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
