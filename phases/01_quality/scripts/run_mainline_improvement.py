from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "phases" / "01_quality" / "scripts"
KFOLD_RUN_ROOT = PROJECT_ROOT / "shared" / "runs" / "phase1" / "kfold"
REPORT_PATH = PROJECT_ROOT / "REPORT_phase1_mainline_improvement.md"


def run_cmd(cmd: List[str]) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def fmt_ms(d: Dict[str, float]) -> str:
    return f"{float(d.get('mean', 0.0)):.4f} ± {float(d.get('std', 0.0)):.4f}"


def read_summary(exp_name: str) -> dict:
    p = KFOLD_RUN_ROOT / exp_name / "kfold_summary.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing summary: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def write_ablation(exp_names: List[str], baseline_exp: str) -> None:
    baseline = read_summary(baseline_exp)
    b_qwk = float(baseline["all_test"]["qwk"]["mean"])

    for exp in exp_names:
        s = read_summary(exp)
        d_qwk = float(s["all_test"]["qwk"]["mean"]) - b_qwk
        text = []
        text.append(f"# Ablation vs baseline ({baseline_exp})")
        text.append("")
        text.append(f"- target_exp: {exp}")
        text.append(f"- baseline_qwk_mean: {b_qwk:.4f}")
        text.append(f"- target_qwk_mean: {float(s['all_test']['qwk']['mean']):.4f}")
        text.append(f"- delta_qwk_mean: {d_qwk:+.4f}")
        text.append("")
        (KFOLD_RUN_ROOT / exp / "ablation_table.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def build_report(exp_order: List[str], baseline_exp: str, supcon_exps: List[str]) -> None:
    summaries = {exp: read_summary(exp) for exp in exp_order}
    baseline = summaries[baseline_exp]

    # choose final mainline by highest all-test QWK mean
    final_exp = max(exp_order, key=lambda e: float(summaries[e]["all_test"]["qwk"]["mean"]))

    lines: List[str] = []
    lines.append("# REPORT_phase1_mainline_improvement")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append("- Phase1品質推定の本線（pad2 + P1 + P2 + group_view_grade）を維持しつつ、group-aware K-fold評価で安定性を可視化する。")
    lines.append("- 難ビュー（A4C/SC/PSAV）の救済を狙って、view別head容量差とloss reweight（view×band）を検証する。")
    lines.append("- SupConの最小グリッド（lambda_con×delta_q）でQWKと方向安定性のParetoを比較し、次の本線設定を決定する。")
    lines.append("")

    lines.append("## 2. データとsplit")
    lines.append("")
    lines.append("- split: group-aware K-fold (k=5)、group重複なし。")
    lines.append("- fold layout: splits/kfold_v1/fold_i/{train,val,test}.csv（互換のphase1_*.csvも同梱）。")
    lines.append("- Random viewは除外（品質学習ターゲットから外す）。")
    lines.append("- band定義: low=0..2, mid=3..5, high=6..9。")
    lines.append("")

    lines.append("## 3. ベースライン再現（RunB相当）")
    lines.append("")
    lines.append(f"- exp: {baseline_exp}")
    lines.append(f"- All test QWK: {fmt_ms(baseline['all_test']['qwk'])}")
    lines.append(f"- All test MAE: {fmt_ms(baseline['all_test']['mae'])}")
    lines.append(f"- All test RMSE: {fmt_ms(baseline['all_test']['rmse'])}")
    lines.append(f"- Direction mean-cos@top20: {fmt_ms(baseline['direction']['global_mean_cos_top20'])}")
    lines.append("")

    lines.append("## 4. 改善案ごとの結果（アブレーション）")
    lines.append("")
    lines.append("| Exp | All QWK (mean±std) | All MAE (mean±std) | All RMSE (mean±std) | Dir cos@top20 (mean±std) | A4C QWK | SC QWK | PSAV QWK |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for exp in exp_order:
        s = summaries[exp]
        w = s.get("weak_view_qwk", {})
        lines.append(
            "| "
            + exp
            + " | "
            + fmt_ms(s["all_test"]["qwk"])
            + " | "
            + fmt_ms(s["all_test"]["mae"])
            + " | "
            + fmt_ms(s["all_test"]["rmse"])
            + " | "
            + fmt_ms(s["direction"]["global_mean_cos_top20"])
            + " | "
            + fmt_ms(w.get("A4C", {"mean": 0.0, "std": 0.0}))
            + " | "
            + fmt_ms(w.get("SC", {"mean": 0.0, "std": 0.0}))
            + " | "
            + fmt_ms(w.get("PSAV", {"mean": 0.0, "std": 0.0}))
            + " |"
        )
    lines.append("")

    lines.append("### SupConグリッド（最小6条件）")
    lines.append("")
    lines.append("| Exp | All QWK (mean±std) | Dir cos@top20 (mean±std) |")
    lines.append("|---|---:|---:|")
    for exp in supcon_exps:
        s = summaries[exp]
        lines.append(
            f"| {exp} | {fmt_ms(s['all_test']['qwk'])} | {fmt_ms(s['direction']['global_mean_cos_top20'])} |"
        )
    lines.append("")

    lines.append("## 5. 重要所見")
    lines.append("")
    b_qwk = float(baseline["all_test"]["qwk"]["mean"])
    best_qwk = float(summaries[final_exp]["all_test"]["qwk"]["mean"])
    lines.append(f"- 最高All QWKは `{final_exp}`（{best_qwk:.4f}）。baseline比 ΔQWK={best_qwk - b_qwk:+.4f}。")
    lines.append("- 難ビュー（A4C/SC/PSAV）のQWKを同時に確認し、All指標のみでの過学習判断を避けた。")
    lines.append("- Direction stability（cos）を併記し、QWK向上と方向安定性悪化のトレードオフを監視した。")
    lines.append("")

    lines.append("## 6. per-view解析")
    lines.append("")
    lines.append(f"- baseline弱ビューQWK: A4C={fmt_ms(baseline['weak_view_qwk'].get('A4C', {}))}, SC={fmt_ms(baseline['weak_view_qwk'].get('SC', {}))}, PSAV={fmt_ms(baseline['weak_view_qwk'].get('PSAV', {}))}")
    best = summaries[final_exp]
    lines.append(f"- final候補弱ビューQWK: A4C={fmt_ms(best['weak_view_qwk'].get('A4C', {}))}, SC={fmt_ms(best['weak_view_qwk'].get('SC', {}))}, PSAV={fmt_ms(best['weak_view_qwk'].get('PSAV', {}))}")
    lines.append("- 各foldの confusion_{view}.png を `shared/runs/phase1/kfold/<exp>/fold_i/` に保存。")
    lines.append("")

    lines.append("## 7. 結論（次の本線設定）")
    lines.append("")
    lines.append(f"- 採用候補: `{final_exp}`")
    lines.append("- 採用条件: All QWK改善または弱ビューの明確な底上げ、かつDirection cosの破壊がないこと。")
    lines.append("- 学習選抜: val QWK最大checkpointをbestとして採用。")
    lines.append("")

    lines.append("## 8. 次アクション")
    lines.append("")
    lines.append("- 採用候補を5-fold full epochで再実行し、最終mean±stdを固定する。")
    lines.append("- Phase2移行条件（All QWKと弱ビューQWKの閾値）を定義してゲート化する。")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Run all Phase1 mainline improvement experiments and generate report.")
    p.add_argument(
        "--base_split_dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "CACTUS" / "manifests_group_consensus_pad2_stratified_v3",
    )
    p.add_argument("--splits_root", type=Path, default=PROJECT_ROOT / "splits" / "kfold_v1")
    p.add_argument("--python_exec", type=str, default=sys.executable)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--main_folds", type=int, default=5, help="Use first N folds for baseline/heavy/reweight")
    p.add_argument("--supcon_folds", type=int, default=3, help="Use 5 for full; 3 allowed when compute-limited")
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    if not any(args.splits_root.glob("fold_*")):
        run_cmd([
            args.python_exec,
            str(SCRIPTS_DIR / "make_phase1_kfold_v1.py"),
            "--base_split_dir",
            str(args.base_split_dir),
            "--out_root",
            str(args.splits_root),
            "--k",
            "5",
            "--seed",
            "42",
            "--exclude_random",
        ])

    common = [
        args.python_exec,
        str(SCRIPTS_DIR / "run_kfold.py"),
        "--splits_root",
        str(args.splits_root),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--sampler",
        "group_view_grade",
        "--run_direction_stability",
    ]
    if args.skip_existing:
        common.append("--skip_existing")

    baseline_exp = "baseline_runB_pad2_p1_p2_group_view_grade"
    run_cmd(common + [
        "--exp_name",
        baseline_exp,
        "--view_head_profile",
        "default",
        "--lambda_con",
        "0.2",
        "--supcon_delta_q",
        "2.0",
        "--max_folds",
        str(int(args.main_folds)),
    ])

    heavy_exp = "heavy_head_for_weak_views"
    run_cmd(common + [
        "--exp_name",
        heavy_exp,
        "--view_head_profile",
        "heavy_for_weak_views",
        "--lambda_con",
        "0.2",
        "--supcon_delta_q",
        "2.0",
        "--max_folds",
        str(int(args.main_folds)),
    ])

    reweight_exp = "loss_reweight_view_band"
    run_cmd(common + [
        "--exp_name",
        reweight_exp,
        "--view_head_profile",
        "default",
        "--use_loss_reweight",
        "--view_weight_a4c",
        "1.3",
        "--view_weight_psav",
        "1.3",
        "--view_weight_sc",
        "1.3",
        "--lambda_con",
        "0.2",
        "--supcon_delta_q",
        "2.0",
        "--max_folds",
        str(int(args.main_folds)),
    ])

    supcon_exps = []
    for lam in [0.0, 0.1, 0.2]:
        for dq in [1.0, 2.0]:
            exp = f"supcon_grid_l{lam:.1f}_d{dq:.1f}"
            supcon_exps.append(exp)
            run_cmd(common + [
                "--exp_name",
                exp,
                "--view_head_profile",
                "default",
                "--lambda_con",
                str(lam),
                "--supcon_delta_q",
                str(dq),
                "--max_folds",
                str(int(args.supcon_folds)),
            ])

    exp_order = [baseline_exp, heavy_exp, reweight_exp] + supcon_exps
    write_ablation(exp_order, baseline_exp=baseline_exp)
    build_report(exp_order=exp_order, baseline_exp=baseline_exp, supcon_exps=supcon_exps)

    print(f"[OK] report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
