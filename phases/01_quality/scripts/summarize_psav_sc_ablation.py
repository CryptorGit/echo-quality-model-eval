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
    p.add_argument("--out_json", default="psav_sc_ablation_summary.json")
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)

    base_rep = json.loads((eval_dir / "multihead_all_report.json").read_text(encoding="utf-8"))
    dq2_rep = json.loads((eval_dir / "multihead_all_dq2e30_report.json").read_text(encoding="utf-8"))

    base_band = pd.read_csv(eval_dir / "band_analysis" / "baseline_multihead" / "band_metrics.csv")
    dq2_band = pd.read_csv(eval_dir / "band_analysis" / "dq2e30_multihead" / "band_metrics.csv")

    base_top = pd.read_csv(eval_dir / "top_ratio_dependency_baseline_psav_sc.csv")
    dq2_top = pd.read_csv(eval_dir / "top_ratio_dependency_dq2e30_psav_sc.csv")

    views = ["PSAV", "SC"]
    summary = {"per_view_test": {}, "band_metrics": {}, "top_ratio_dependency": {}}

    for v in views:
        b = base_rep.get("per_view_test", {}).get(v, {})
        d = dq2_rep.get("per_view_test", {}).get(v, {})
        summary["per_view_test"][v] = {
            "baseline": b,
            "dq2e30": d,
            "delta_qwk": float(d.get("qwk", 0.0) - b.get("qwk", 0.0)),
            "delta_mae": float(d.get("mae", 0.0) - b.get("mae", 0.0)),
            "delta_rmse": float(d.get("rmse", 0.0) - b.get("rmse", 0.0)),
        }

        bb = base_band[base_band["view"] == v].copy()
        dd = dq2_band[dq2_band["view"] == v].copy()
        rows = {}
        for band in ["0-2", "3-5", "6-9"]:
            r1 = bb[bb["band"] == band]
            r2 = dd[dd["band"] == band]
            rows[band] = {
                "baseline": r1.iloc[0].to_dict() if len(r1) else {},
                "dq2e30": r2.iloc[0].to_dict() if len(r2) else {},
            }
        summary["band_metrics"][v] = rows

        bt = base_top[base_top["view"] == v]
        dt = dq2_top[dq2_top["view"] == v]
        summary["top_ratio_dependency"][v] = {
            "baseline": bt.iloc[0].to_dict() if len(bt) else {},
            "dq2e30": dt.iloc[0].to_dict() if len(dt) else {},
        }

    summary["overall_test"] = {
        "baseline": base_rep.get("test", {}),
        "dq2e30": dq2_rep.get("test", {}),
        "delta_qwk": float(dq2_rep.get("test", {}).get("qwk", 0.0) - base_rep.get("test", {}).get("qwk", 0.0)),
    }

    out_path = eval_dir / args.out_json
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
