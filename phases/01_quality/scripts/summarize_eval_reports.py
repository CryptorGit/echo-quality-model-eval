from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--eval_dir",
        default=str(Path(__file__).resolve().parents[3] / "shared" / "runs" / "phase1" / "eval"),
    )
    p.add_argument("--out_csv", default="comparison_summary_consensus.csv")
    p.add_argument(
        "--glob",
        default="",
        help="Optional glob for report JSONs (e.g. '*pad2*_report.json'). If omitted, uses the legacy fixed file list.",
    )
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)
    if args.glob:
        files = sorted(eval_dir.glob(str(args.glob)))
        if not files:
            raise FileNotFoundError(f"No report JSONs matched: {args.glob} in {eval_dir}")
    else:
        files = [
            eval_dir / "multihead_all_report.json",
            eval_dir / "perview_A4C_report.json",
            eval_dir / "perview_PL_report.json",
            eval_dir / "perview_PSAV_report.json",
            eval_dir / "perview_PSMV_report.json",
            eval_dir / "perview_SC_report.json",
        ]

    rows = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        model_name = data["mode"] if data["mode"] == "multihead" else f"perview_{data['view']}"
        if args.glob:
            model_name = f"{model_name}:{path.stem}"
        rows.append(
            {
                "model": model_name,
                "report": path.name,
                "val_rmse": data["val"]["rmse"],
                "val_mae": data["val"]["mae"],
                "val_qwk": data["val"]["qwk"],
                "test_rmse": data["test"]["rmse"],
                "test_mae": data["test"]["mae"],
                "test_qwk": data["test"]["qwk"],
            }
        )

    df = pd.DataFrame(rows)
    if args.glob:
        df = df.sort_values(["test_qwk", "test_rmse"], ascending=[False, True])
    else:
        df = df.sort_values(["test_rmse", "test_mae"])
    out_path = eval_dir / args.out_csv
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\n[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
