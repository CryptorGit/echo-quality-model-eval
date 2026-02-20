from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[3]
    p.add_argument("--eval_dir", default=str(root / "shared" / "runs" / "phase1" / "eval"))
    p.add_argument(
        "--sanity_json",
        default=str(root / "datasets" / "CACTUS" / "manifests_group_consensus" / "phase1_split_sanity.json"),
    )
    p.add_argument("--multihead_report", default="multihead_all_report.json")
    p.add_argument("--direction_report", default="direction_stability_multihead_consensus.json")
    p.add_argument("--summary_csv", default="comparison_summary_consensus.csv")
    args = p.parse_args()

    base = Path(args.eval_dir)
    sanity_path = Path(args.sanity_json)
    mh_path = base / args.multihead_report
    dir_path = base / args.direction_report

    sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
    mh = json.loads(mh_path.read_text(encoding="utf-8"))
    dr = json.loads(dir_path.read_text(encoding="utf-8"))

    print("SANITY_PASS", sanity["pass"])
    print("MULTIHEAD_TEST", mh["test"])

    print("PER_VIEW_TEST")
    for key, val in mh.get("per_view_test", {}).items():
        print(key, val)

    print("DIR_TOP20_MEAN_GLOBAL", dr["global_mean_cos_top20_mean_method"])
    print("DIR_TOP20_PER_VIEW_MEAN")
    for view, rep in dr.get("stability", {}).items():
        m = rep.get("top20", {}).get("mean", {})
        print(view, m.get("mean_cos"), m.get("ratio_cos_gt_0.7"))

    print("MONOTONICITY")
    for view, val in dr.get("monotonicity", {}).items():
        print(view, val.get("guided_improve_rate"), val.get("random_improve_rate"), val.get("delta_guided_minus_random"))

    csv = base / args.summary_csv
    if csv.exists():
        df = pd.read_csv(csv)
        print("BEST_BY_TEST_RMSE", df.iloc[0].to_dict())


if __name__ == "__main__":
    main()
