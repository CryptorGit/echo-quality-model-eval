from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

MANIFEST = ROOT / "datasets" / "CACTUS" / "manifests" / "cactus_manifest.csv"
OUTDIR = ROOT / "datasets" / "CACTUS" / "manifests_group"
OUTDIR.mkdir(parents=True, exist_ok=True)

GROUP_PAT = re.compile(r"(\d+_(?:D\d+|V\d+|DR\d+|Folder\d+))", re.IGNORECASE)


def extract_group(path_str: str) -> str:
    name = Path(path_str).name
    m = GROUP_PAT.search(name)
    if m:
        return m.group(1)
    return "UNMATCHED"


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    df = pd.read_csv(MANIFEST)
    for c in ["path", "view", "quality"]:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {MANIFEST}")

    df = df.copy()
    df["quality"] = df["quality"].astype(float)
    df["group"] = df["path"].astype(str).map(extract_group)

    # Phase1対象: Random除外 + group未マッチ除外
    df = df[(df["view"] != "Random") & (df["group"] != "UNMATCHED")].reset_index(drop=True)

    keys = np.array(sorted(df["group"].unique()))
    rng = np.random.default_rng(42)
    rng.shuffle(keys)

    n = len(keys)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    g_train = set(keys[:n_train])
    g_val = set(keys[n_train:n_train + n_val])
    g_test = set(keys[n_train + n_val:])

    tr = df[df["group"].isin(g_train)].sample(frac=1.0, random_state=42).reset_index(drop=True)
    va = df[df["group"].isin(g_val)].sample(frac=1.0, random_state=42).reset_index(drop=True)
    te = df[df["group"].isin(g_test)].sample(frac=1.0, random_state=42).reset_index(drop=True)

    overlap_tv = len(set(tr["group"]) & set(va["group"]))
    overlap_tt = len(set(tr["group"]) & set(te["group"]))
    overlap_vt = len(set(va["group"]) & set(te["group"]))

    if overlap_tv != 0 or overlap_tt != 0 or overlap_vt != 0:
        raise RuntimeError("Group overlap detected")

    tr_path = OUTDIR / "phase1_train.csv"
    va_path = OUTDIR / "phase1_val.csv"
    te_path = OUTDIR / "phase1_test.csv"

    tr.to_csv(tr_path, index=False)
    va.to_csv(va_path, index=False)
    te.to_csv(te_path, index=False)

    stats = {
        "rows": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "groups": {
            "train": int(tr["group"].nunique()),
            "val": int(va["group"].nunique()),
            "test": int(te["group"].nunique()),
        },
        "overlap": {"train_val": overlap_tv, "train_test": overlap_tt, "val_test": overlap_vt},
        "views": sorted(df["view"].unique().tolist()),
    }

    stats_path = OUTDIR / "phase1_split_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[OK] {tr_path}")
    print(f"[OK] {va_path}")
    print(f"[OK] {te_path}")
    print(f"[OK] {stats_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
