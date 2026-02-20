from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLS = ["path", "view", "quality", "group"]


def _read_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    miss = [c for c in REQUIRED_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"{path} missing columns: {miss}")
    return df


def _views(df: pd.DataFrame) -> list[str]:
    return sorted(df["view"].astype(str).unique().tolist())


def main():
    p = argparse.ArgumentParser(description="Phase1 split sanity checks")
    p.add_argument("--split_dir", required=True)
    p.add_argument("--out_json", default="phase1_split_sanity.json")
    args = p.parse_args()

    split_dir = Path(args.split_dir)
    tr = _read_split(split_dir / "phase1_train.csv")
    va = _read_split(split_dir / "phase1_val.csv")
    te = _read_split(split_dir / "phase1_test.csv")

    g_tr = set(tr["group"].astype(str))
    g_va = set(va["group"].astype(str))
    g_te = set(te["group"].astype(str))

    overlap = {
        "train_val": len(g_tr & g_va),
        "train_test": len(g_tr & g_te),
        "val_test": len(g_va & g_te),
    }

    random_counts = {
        "train": int((tr["view"].astype(str) == "Random").sum()),
        "val": int((va["view"].astype(str) == "Random").sum()),
        "test": int((te["view"].astype(str) == "Random").sum()),
    }

    summary = {
        "split_dir": str(split_dir),
        "rows": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "groups": {
            "train": int(tr["group"].nunique()),
            "val": int(va["group"].nunique()),
            "test": int(te["group"].nunique()),
        },
        "group_overlap": overlap,
        "random_counts": random_counts,
        "views": {
            "train": _views(tr),
            "val": _views(va),
            "test": _views(te),
        },
        "pass": {
            "group_overlap_zero": all(v == 0 for v in overlap.values()),
            "random_excluded": all(v == 0 for v in random_counts.values()),
        },
    }

    out_path = split_dir / args.out_json
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
