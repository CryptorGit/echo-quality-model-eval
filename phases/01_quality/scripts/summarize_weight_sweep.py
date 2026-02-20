from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List


VIEWS = ["A4C", "PL", "PSAV", "PSMV", "SC"]


def ms(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": float(mean(vals)),
        "std": float(stdev(vals)) if len(vals) > 1 else 0.0,
        "n": int(len(vals)),
    }


def load_fold(exp_root: Path, fold: int) -> tuple[dict, dict]:
    m = json.loads((exp_root / f"fold_{fold}" / "metrics.json").read_text(encoding="utf-8"))
    d = json.loads((exp_root / f"fold_{fold}" / "direction_stability.json").read_text(encoding="utf-8"))
    return m, d


def summarize_exp(exp_root: Path) -> dict:
    rows = {"All": {"qwk": [], "mae": [], "rmse": [], "top20_cos": [], "global_top20_cos": []}}
    for v in VIEWS:
        rows[v] = {"qwk": [], "mae": [], "rmse": [], "top20_cos": [], "global_top20_cos": []}

    for fold in range(5):
        m, d = load_fold(exp_root, fold)
        rows["All"]["qwk"].append(float(m["test"]["qwk"]))
        rows["All"]["mae"].append(float(m["test"]["mae"]))
        rows["All"]["rmse"].append(float(m["test"]["rmse"]))

        global_cos = float(d.get("global_mean_cos_top20_mean_method", 0.0))
        rows["All"]["top20_cos"].append(global_cos)
        rows["All"]["global_top20_cos"].append(global_cos)

        pv_test = m.get("per_view_test", {})
        stab = d.get("stability", {})
        for v in VIEWS:
            pm = pv_test.get(v, {"qwk": 0.0, "mae": 0.0, "rmse": 0.0})
            rows[v]["qwk"].append(float(pm.get("qwk", 0.0)))
            rows[v]["mae"].append(float(pm.get("mae", 0.0)))
            rows[v]["rmse"].append(float(pm.get("rmse", 0.0)))

            # view内 top20 定義（既存direction_stability_testの top20.mean.mean_cos）
            top20 = float(stab.get(v, {}).get("top20", {}).get("mean", {}).get("mean_cos", 0.0))
            rows[v]["top20_cos"].append(top20)

            # 全体 top20 定義（既存global_mean_cos_top20_mean_method）を各view行にも併記
            rows[v]["global_top20_cos"].append(global_cos)

    summary = {"rows": {}, "meta": {"exp": exp_root.name}}
    for row_name, cols in rows.items():
        summary["rows"][row_name] = {k: ms(v) for k, v in cols.items()}

    return summary


def write_summary_files(exp_root: Path, summary: dict) -> None:
    (exp_root / "kfold_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append(f"# kfold_summary: {exp_root.name}")
    lines.append("")
    lines.append("| Row | QWK | MAE | RMSE | top20_cos | global_top20_cos |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in ["All"] + VIEWS:
        r = summary["rows"][row]
        lines.append(
            f"| {row} | {r['qwk']['mean']:.4f} ± {r['qwk']['std']:.4f} | {r['mae']['mean']:.4f} ± {r['mae']['std']:.4f} | {r['rmse']['mean']:.4f} ± {r['rmse']['std']:.4f} | {r['top20_cos']['mean']:.4f} ± {r['top20_cos']['std']:.4f} | {r['global_top20_cos']['mean']:.4f} ± {r['global_top20_cos']['std']:.4f} |"
        )
    lines.append("")
    lines.append("定義固定:")
    lines.append("- top20_cos: 各view内 top20%（direction_stability の top20.mean.mean_cos）")
    lines.append("- global_top20_cos: 全体 top20%（direction_stability の global_mean_cos_top20_mean_method）")
    (exp_root / "kfold_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ablation(exp_root: Path, base_summary: dict, cur_summary: dict) -> None:
    lines = []
    lines.append(f"# ablation_delta vs {base_summary['meta']['exp']}")
    lines.append("")
    lines.append("| Row | ΔQWK | ΔMAE | ΔRMSE | Δtop20_cos | Δglobal_top20_cos |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in ["All"] + VIEWS:
        b = base_summary["rows"][row]
        c = cur_summary["rows"][row]
        lines.append(
            f"| {row} | {c['qwk']['mean']-b['qwk']['mean']:+.4f} | {c['mae']['mean']-b['mae']['mean']:+.4f} | {c['rmse']['mean']-b['rmse']['mean']:+.4f} | {c['top20_cos']['mean']-b['top20_cos']['mean']:+.4f} | {c['global_top20_cos']['mean']-b['global_top20_cos']['mean']:+.4f} |"
        )
    (exp_root / "ablation_delta.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_best(sum12: dict, sum14: dict) -> str:
    q12 = sum12["rows"]["All"]["qwk"]["mean"]
    q14 = sum14["rows"]["All"]["qwk"]["mean"]
    if q12 != q14:
        return "w12" if q12 > q14 else "w14"

    weak12 = mean([sum12["rows"][v]["qwk"]["mean"] for v in ["A4C", "SC", "PSAV"]])
    weak14 = mean([sum14["rows"][v]["qwk"]["mean"] for v in ["A4C", "SC", "PSAV"]])
    if weak12 != weak14:
        return "w12" if weak12 > weak14 else "w14"

    g12 = sum12["rows"]["All"]["global_top20_cos"]["mean"]
    g14 = sum14["rows"]["All"]["global_top20_cos"]["mean"]
    if g12 != g14:
        return "w12" if g12 > g14 else "w14"

    s12 = sum12["rows"]["All"]["qwk"]["std"]
    s14 = sum14["rows"]["All"]["qwk"]["std"]
    return "w12" if s12 < s14 else "w14"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kfold_root", type=Path, default=Path("shared/runs/phase1/kfold"))
    p.add_argument("--base_exp", type=str, default="loss_reweight_view_band")
    p.add_argument("--exp_w12", type=str, default="loss_reweight_view_band_w12")
    p.add_argument("--exp_w14", type=str, default="loss_reweight_view_band_w14")
    args = p.parse_args()

    base_root = args.kfold_root / args.base_exp
    r12 = args.kfold_root / args.exp_w12
    r14 = args.kfold_root / args.exp_w14

    base_sum = summarize_exp(base_root)
    s12 = summarize_exp(r12)
    s14 = summarize_exp(r14)

    write_summary_files(r12, s12)
    write_summary_files(r14, s14)

    write_ablation(r12, base_sum, s12)
    write_ablation(r14, base_sum, s14)

    winner = choose_best(s12, s14)
    wroot = r12 if winner == "w12" else r14

    lines = []
    lines.append("# weight_sweep_decision")
    lines.append("")
    lines.append(f"- winner: {wroot.name}")
    lines.append("- rule order:")
    lines.append("  1) All QWK mean")
    lines.append("  2) weak-view mean QWK (A4C/SC/PSAV)")
    lines.append("  3) global_top20_cos")
    lines.append("  4) smaller std")
    lines.append("")
    lines.append(f"- w12 All QWK: {s12['rows']['All']['qwk']['mean']:.4f} ± {s12['rows']['All']['qwk']['std']:.4f}")
    lines.append(f"- w14 All QWK: {s14['rows']['All']['qwk']['mean']:.4f} ± {s14['rows']['All']['qwk']['std']:.4f}")
    lines.append(f"- w12 weak avg QWK: {mean([s12['rows'][v]['qwk']['mean'] for v in ['A4C','SC','PSAV']]):.4f}")
    lines.append(f"- w14 weak avg QWK: {mean([s14['rows'][v]['qwk']['mean'] for v in ['A4C','SC','PSAV']]):.4f}")
    lines.append(f"- w12 global_top20_cos: {s12['rows']['All']['global_top20_cos']['mean']:.4f}")
    lines.append(f"- w14 global_top20_cos: {s14['rows']['All']['global_top20_cos']['mean']:.4f}")

    out = args.kfold_root / "weight_sweep_decision.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] {r12 / 'kfold_summary.md'}")
    print(f"[OK] {r14 / 'kfold_summary.md'}")
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
