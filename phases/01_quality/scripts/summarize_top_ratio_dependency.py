from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Summarize top10/20/30 dependency for direction stability")
    p.add_argument("--report_json", required=True)
    p.add_argument("--views", default="PSAV,SC")
    p.add_argument("--out_csv", required=True)
    args = p.parse_args()

    report = json.loads(Path(args.report_json).read_text(encoding="utf-8"))
    views = [v.strip() for v in args.views.split(",") if v.strip()]

    rows = []
    for view in views:
        rep = report.get("stability", {}).get(view, {})
        if not rep:
            continue

        vals = {}
        for t in [10, 20, 30]:
            key = f"top{t}"
            vals[t] = rep.get(key, {}).get("mean", {}).get("mean_cos", float("nan"))

        best_t = max(vals, key=lambda k: vals[k])
        top10_only = vals[10] >= 0.7 and vals[20] < 0.7 and vals[30] < 0.7

        rows.append(
            {
                "view": view,
                "top10_mean_cos": vals[10],
                "top20_mean_cos": vals[20],
                "top30_mean_cos": vals[30],
                "best_top": int(best_t),
                "top10_only_good": bool(top10_only),
                "max_gap_top10_vs_top20": float(vals[10] - vals[20]),
                "max_gap_top20_vs_top30": float(vals[20] - vals[30]),
            }
        )

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(df.to_string(index=False))
    print(f"\n[OK] wrote: {out_csv}")


if __name__ == "__main__":
    main()
