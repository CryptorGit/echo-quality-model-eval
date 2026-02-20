# make_cactus_manifest.py
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def guess_quality_col(cols: List[str]) -> Optional[str]:
    cand = ["grade", "quality", "score", "rating", "Grade", "Quality", "Score", "Rating"]
    for c in cand:
        if c in cols:
            return c
    # それっぽい列名をゆるく探す
    for c in cols:
        if re.search(r"(grade|qual|score|rating)", c, re.IGNORECASE):
            return c
    return None


def guess_path_cols(cols: List[str]) -> List[str]:
    # 「ファイル名」や「フレーム」など、画像に繋がりそうな列を候補にする
    keys = []
    for c in cols:
        if re.search(r"(file|path|image|frame|png|jpg|name)", c, re.IGNORECASE):
            keys.append(c)
    return keys


def build_index_by_basename(root: Path, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")) -> Dict[str, Path]:
    # 大量ファイルでも耐えるように basename -> Path を一回だけ作る
    idx: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            # basename衝突があるかもしれないので、最初に見つかったものを採用
            idx.setdefault(p.name, p)
    return idx


def resolve_image_path(
    row: pd.Series,
    root: Path,
    basename_index: Dict[str, Path],
    path_cols: List[str]
) -> Optional[Path]:
    # 1) 直接パスが書かれている場合（相対/絶対）
    for c in path_cols:
        v = row.get(c)
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() == "nan":
            continue

        # パスっぽいなら試す
        p = Path(s)
        if p.is_absolute() and p.exists():
            return p
        # ルートからの相対
        rp = (root / s)
        if rp.exists():
            return rp

        # 2) ファイル名だけの場合 → basename index で引く
        # 拡張子が無い場合もあるので補完
        if "." in s:
            hit = basename_index.get(Path(s).name)
            if hit:
                return hit
        else:
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                hit = basename_index.get(s + ext)
                if hit:
                    return hit

    return None


def main():
    # あなたの固定パス
    root = Path(__file__).resolve().parents[3]
    cactus_root = root / "datasets" / "CACTUS" / "extracted" / "Cactus Dataset"
    out_dir = root / "datasets" / "CACTUS" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    grades_dir = cactus_root / "Grades"
    if not grades_dir.exists():
        raise FileNotFoundError(f"Grades folder not found: {grades_dir}")

    # 画像探索のインデックスを作る（最初だけ重いが、以後速い）
    print("[INFO] Building basename index (may take a bit)...")
    basename_index = build_index_by_basename(cactus_root)
    print(f"[INFO] Indexed {len(basename_index)} image basenames.")

    rows_out: List[Dict[str, str]] = []

    # *_grades.csv を全部処理
    for csv_path in sorted(grades_dir.glob("*_grades.csv")):
        view = csv_path.name.replace("_grades.csv", "")
        df = pd.read_csv(csv_path)

        cols = list(df.columns)
        qcol = guess_quality_col(cols)
        if qcol is None:
            raise ValueError(f"Cannot find quality/grade column in {csv_path}. cols={cols}")

        path_cols = guess_path_cols(cols)
        if not path_cols:
            # 最悪、全列を候補にする（雑だが止まらない）
            path_cols = cols

        for _, r in df.iterrows():
            img_path = resolve_image_path(r, cactus_root, basename_index, path_cols)
            if img_path is None:
                # 解決できない行はスキップ（後で未解決数で気付ける）
                continue

            quality = r[qcol]
            try:
                q = float(quality)
            except Exception:
                # 数値化できないならスキップ
                continue

            rows_out.append({
                "path": str(img_path),
                "view": view,
                "quality": str(q),
                "source_csv": str(csv_path)
            })

        print(f"[INFO] {view}: collected {sum(1 for x in rows_out if x['view']==view)} rows so far.")

    if not rows_out:
        raise RuntimeError("No rows collected. Likely path resolution failed. Check CSV columns / dataset structure.")

    manifest = pd.DataFrame(rows_out).drop_duplicates(subset=["path"])
    manifest_path = out_dir / "cactus_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"[OK] Wrote manifest: {manifest_path}  rows={len(manifest)}")

    # ざっくり分割（viewごとに 80/10/10）
    # ※品質の層化までやりたければ後で直す。まず回す。
    train, val, test = [], [], []
    for view, g in manifest.groupby("view"):
        g = g.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n = len(g)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        train.append(g.iloc[:n_train])
        val.append(g.iloc[n_train:n_train+n_val])
        test.append(g.iloc[n_train+n_val:])

    pd.concat(train).to_csv(out_dir / "cactus_train.csv", index=False)
    pd.concat(val).to_csv(out_dir / "cactus_val.csv", index=False)
    pd.concat(test).to_csv(out_dir / "cactus_test.csv", index=False)
    print("[OK] Wrote split manifests: cactus_train.csv / cactus_val.csv / cactus_test.csv")


if __name__ == "__main__":
    main()
