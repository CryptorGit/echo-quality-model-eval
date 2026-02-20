from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--eval_dir",
        default=str(Path(__file__).resolve().parents[3] / "shared" / "runs" / "phase1" / "eval"),
    )
    p.add_argument("--out_csv", default="multihead_run_compare.csv")
    p.add_argument("--out_json", default="multihead_run_compare.json")
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)
    runs = {
        "baseline": eval_dir / "multihead_all_report.json",
        "dq2e30": eval_dir / "multihead_all_dq2e30_report.json",
        "dq2e30_w18": eval_dir / "multihead_all_dq2e30_w18_report.json",
    }

    rows = []
    full = {}
    for name, rp in runs.items():
        d = load_report(rp)
        full[name] = d
        rows.append(
            {
                "run": name,
                "test_qwk": d["test"]["qwk"],
                "test_rmse": d["test"]["rmse"],
                "test_mae": d["test"]["mae"],
                "psav_qwk": d.get("per_view_test", {}).get("PSAV", {}).get("qwk", float("nan")),
                "sc_qwk": d.get("per_view_test", {}).get("SC", {}).get("qwk", float("nan")),
            }
        )

    df = pd.DataFrame(rows).sort_values("test_qwk", ascending=False)
    out_csv = eval_dir / args.out_csv
    out_json = eval_dir / args.out_json
    df.to_csv(out_csv, index=False)

    summary = {
        "runs": rows,
        "best_test_qwk_run": str(df.iloc[0]["run"]),
        "phase2_gate_hint": {
            "criterion_overall_qwk_ge_0_85": bool(df["test_qwk"].max() >= 0.85),
            "criterion_psav_sc_qwk_ge_0_6": bool((df["psav_qwk"].max() >= 0.6) and (df["sc_qwk"].max() >= 0.6)),
        },
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(df.to_string(index=False))
    print(f"\n[OK] csv: {out_csv}")
    print(f"[OK] json: {out_json}")


if __name__ == "__main__":
    main()
