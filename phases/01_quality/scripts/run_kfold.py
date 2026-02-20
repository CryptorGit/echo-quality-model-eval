from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "phases" / "01_quality" / "scripts"
RUNS_ROOT = PROJECT_ROOT / "shared" / "runs" / "phase1"
EVAL_ROOT = RUNS_ROOT / "eval"
KFOLD_OUT_ROOT = RUNS_ROOT / "kfold"


def run_cmd(cmd: List[str]) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def fold_dirs(splits_root: Path) -> List[Path]:
    out = [p for p in sorted(splits_root.glob("fold_*")) if p.is_dir()]
    if not out:
        raise FileNotFoundError(f"No fold directories found under {splits_root}")
    return out


def run_train_eval_for_fold(
    py: str,
    fold_dir: Path,
    exp_name: str,
    fold_idx: int,
    train_args: List[str],
    run_direction_stability: bool,
    skip_existing: bool,
) -> Tuple[Path, Path, Path | None]:
    fold_out = KFOLD_OUT_ROOT / exp_name / f"fold_{fold_idx}"
    fold_out.mkdir(parents=True, exist_ok=True)

    run_tag = f"{exp_name}_fold{fold_idx:02d}"
    ckpt_dir = RUNS_ROOT / fold_dir.name / f"multihead_{run_tag}"
    ckpt_path = ckpt_dir / "best.pt"

    report_name = f"multihead_all_{run_tag}_report.json"
    report_path = EVAL_ROOT / report_name

    if not (skip_existing and ckpt_path.exists()):
        train_cmd = [
            py,
            str(SCRIPTS_DIR / "train_phase1.py"),
            "--mode",
            "multihead",
            "--split_dir",
            str(fold_dir),
            "--run_tag",
            run_tag,
            *train_args,
        ]
        run_cmd(train_cmd)

    if not (skip_existing and report_path.exists()):
        eval_cmd = [
            py,
            str(SCRIPTS_DIR / "eval_phase1.py"),
            "--mode",
            "multihead",
            "--ckpt",
            str(ckpt_path),
            "--split_dir",
            str(fold_dir),
            "--save_confusion",
            "--out_suffix",
            run_tag,
        ]
        run_cmd(eval_cmd)

    ds_path: Path | None = None
    if run_direction_stability:
        ds_name = f"direction_stability_{run_tag}.json"
        ds_path = EVAL_ROOT / ds_name
        if not (skip_existing and ds_path.exists()):
            ds_cmd = [
                py,
                str(SCRIPTS_DIR / "direction_stability_test.py"),
                "--mode",
                "multihead",
                "--ckpt",
                str(ckpt_path),
                "--split_dir",
                str(fold_dir),
                "--out_json",
                ds_name,
            ]
            run_cmd(ds_cmd)

    # copy canonical artifacts to fold output
    shutil.copy2(ckpt_dir / "config.json", fold_out / "config.json")
    shutil.copy2(report_path, fold_out / "metrics.json")
    if ds_path is not None and ds_path.exists():
        shutil.copy2(ds_path, fold_out / "direction_stability.json")

    return fold_out, report_path, ds_path


def _mean_std(vals: List[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "std": float(np.std(arr)) if arr.size else 0.0,
        "n": int(arr.size),
    }


def try_write_confusion_png(fold_out: Path, report: dict) -> None:
    cm_by_view = report.get("confusion_test", {})
    if not cm_by_view:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    for view, cm in cm_by_view.items():
        arr = np.array(cm, dtype=np.float32)
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        im = ax.imshow(arr, cmap="Blues")
        ax.set_title(f"Confusion ({view})")
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(fold_out / f"confusion_{view}.png", dpi=140)
        plt.close(fig)


def summarize_kfold(exp_root: Path, weak_views: List[str]) -> Tuple[dict, str]:
    fold_reports = []
    for fold_dir in sorted(exp_root.glob("fold_*")):
        metrics_path = fold_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        rep = json.loads(metrics_path.read_text(encoding="utf-8"))
        ds_path = fold_dir / "direction_stability.json"
        if ds_path.exists():
            ds = json.loads(ds_path.read_text(encoding="utf-8"))
            rep["direction_global_mean_cos_top20"] = float(ds.get("global_mean_cos_top20_mean_method", 0.0))
        else:
            rep["direction_global_mean_cos_top20"] = 0.0
        fold_reports.append((fold_dir.name, rep))

    all_qwk = [r["test"]["qwk"] for _, r in fold_reports]
    all_mae = [r["test"]["mae"] for _, r in fold_reports]
    all_rmse = [r["test"]["rmse"] for _, r in fold_reports]
    all_cos = [r.get("direction_global_mean_cos_top20", 0.0) for _, r in fold_reports]

    per_view_keys = sorted({k for _, r in fold_reports for k in r.get("per_view_test", {}).keys()})
    per_view = {}
    for v in per_view_keys:
        q = []
        m = []
        r_ = []
        for _, rep in fold_reports:
            pv = rep.get("per_view_test", {}).get(v)
            if pv:
                q.append(float(pv["qwk"]))
                m.append(float(pv["mae"]))
                r_.append(float(pv["rmse"]))
        per_view[v] = {
            "qwk": _mean_std(q),
            "mae": _mean_std(m),
            "rmse": _mean_std(r_),
        }

    summary = {
        "folds": [name for name, _ in fold_reports],
        "all_test": {
            "qwk": _mean_std(all_qwk),
            "mae": _mean_std(all_mae),
            "rmse": _mean_std(all_rmse),
        },
        "direction": {
            "global_mean_cos_top20": _mean_std(all_cos),
        },
        "per_view_test": per_view,
        "weak_view_qwk": {
            v: per_view.get(v, {}).get("qwk", {"mean": 0.0, "std": 0.0, "n": 0}) for v in weak_views
        },
    }

    lines = []
    lines.append(f"# K-fold Summary: {exp_root.name}")
    lines.append("")
    lines.append("## All (test)")
    lines.append("")
    lines.append("| Metric | Mean | Std |")
    lines.append("|---|---:|---:|")
    lines.append(f"| QWK | {summary['all_test']['qwk']['mean']:.4f} | {summary['all_test']['qwk']['std']:.4f} |")
    lines.append(f"| MAE | {summary['all_test']['mae']['mean']:.4f} | {summary['all_test']['mae']['std']:.4f} |")
    lines.append(f"| RMSE | {summary['all_test']['rmse']['mean']:.4f} | {summary['all_test']['rmse']['std']:.4f} |")
    lines.append(f"| Direction mean-cos@top20 | {summary['direction']['global_mean_cos_top20']['mean']:.4f} | {summary['direction']['global_mean_cos_top20']['std']:.4f} |")
    lines.append("")
    lines.append("## Per-view (test)")
    lines.append("")
    lines.append("| View | QWK mean | QWK std | MAE mean | RMSE mean |")
    lines.append("|---|---:|---:|---:|---:|")
    for v in per_view_keys:
        lines.append(
            f"| {v} | {summary['per_view_test'][v]['qwk']['mean']:.4f} | {summary['per_view_test'][v]['qwk']['std']:.4f} | {summary['per_view_test'][v]['mae']['mean']:.4f} | {summary['per_view_test'][v]['rmse']['mean']:.4f} |"
        )

    return summary, "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Run train/eval over k-fold splits and summarize.")
    p.add_argument("--splits_root", type=Path, required=True)
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--python_exec", type=str, default=sys.executable)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--sampler", type=str, default="group_view_grade")
    p.add_argument("--lambda_con", type=float, default=0.2)
    p.add_argument("--supcon_delta_q", type=float, default=2.0)
    p.add_argument("--view_head_profile", type=str, default="default", choices=["default", "heavy_for_weak_views"])
    p.add_argument("--weak_views", type=str, default="A4C,SC,PSAV")
    p.add_argument("--use_loss_reweight", action="store_true")
    p.add_argument("--view_weight_a4c", type=float, default=1.3)
    p.add_argument("--view_weight_pl", type=float, default=1.0)
    p.add_argument("--view_weight_psav", type=float, default=1.3)
    p.add_argument("--view_weight_psmv", type=float, default=1.0)
    p.add_argument("--view_weight_sc", type=float, default=1.3)
    p.add_argument("--band_weight_scheme", type=str, default="invfreq_clip", choices=["invfreq_clip", "none"])
    p.add_argument("--band_weight_clip_min", type=float, default=0.5)
    p.add_argument("--band_weight_clip_max", type=float, default=2.0)
    p.add_argument("--run_direction_stability", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--max_folds", type=int, default=0, help="Use first N folds only when >0")
    args = p.parse_args()

    exp_root = KFOLD_OUT_ROOT / args.exp_name
    exp_root.mkdir(parents=True, exist_ok=True)

    weak_views = [v.strip() for v in args.weak_views.split(",") if v.strip()]

    train_args = [
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--img_size",
        str(args.img_size),
        "--lr",
        str(args.lr),
        "--amp",
        "--sampler",
        str(args.sampler),
        "--lambda_con",
        str(args.lambda_con),
        "--supcon_delta_q",
        str(args.supcon_delta_q),
        "--view_head_profile",
        str(args.view_head_profile),
        "--weak_views",
        ",".join(weak_views),
        "--exclude_random",
    ]

    if args.use_loss_reweight:
        train_args += [
            "--use_loss_reweight",
            "--view_weight_a4c",
            str(args.view_weight_a4c),
            "--view_weight_pl",
            str(args.view_weight_pl),
            "--view_weight_psav",
            str(args.view_weight_psav),
            "--view_weight_psmv",
            str(args.view_weight_psmv),
            "--view_weight_sc",
            str(args.view_weight_sc),
            "--band_weight_scheme",
            str(args.band_weight_scheme),
            "--band_weight_clip_min",
            str(args.band_weight_clip_min),
            "--band_weight_clip_max",
            str(args.band_weight_clip_max),
        ]

    all_folds = fold_dirs(args.splits_root)
    if int(args.max_folds) > 0:
        all_folds = all_folds[: int(args.max_folds)]

    for i, fd in enumerate(all_folds):
        fold_out, report_path, _ = run_train_eval_for_fold(
            py=args.python_exec,
            fold_dir=fd,
            exp_name=args.exp_name,
            fold_idx=i,
            train_args=train_args,
            run_direction_stability=bool(args.run_direction_stability),
            skip_existing=bool(args.skip_existing),
        )
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        try_write_confusion_png(fold_out, rep)

    summary, summary_md = summarize_kfold(exp_root, weak_views=weak_views)
    (exp_root / "kfold_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (exp_root / "kfold_summary.md").write_text(summary_md, encoding="utf-8")

    print(f"[OK] k-fold summary json: {exp_root / 'kfold_summary.json'}")
    print(f"[OK] k-fold summary md: {exp_root / 'kfold_summary.md'}")


if __name__ == "__main__":
    main()
