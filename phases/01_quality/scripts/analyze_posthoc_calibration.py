from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error


def qwk_round(y: np.ndarray, p: np.ndarray) -> float:
    y_i = np.clip(np.rint(y), 0, 9).astype(int)
    p_i = np.clip(np.rint(p), 0, 9).astype(int)
    return float(cohen_kappa_score(y_i, p_i, weights="quadratic"))


def metrics_round(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    rmse = float(mean_squared_error(y, p) ** 0.5)
    mae = float(mean_absolute_error(y, p))
    qwk = qwk_round(y, p)
    return {"rmse": rmse, "mae": mae, "qwk": qwk}


def fit_affine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit y ~= a*x + b by least squares."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    xm = float(x.mean())
    ym = float(y.mean())
    xv = float(((x - xm) ** 2).mean())
    if xv < 1e-12:
        return 0.0, ym
    cov = float(((x - xm) * (y - ym)).mean())
    a = cov / xv
    b = ym - a * xm
    return float(a), float(b)


def apply_affine(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x + b


def per_view_fit_apply(
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    method: str,
) -> Tuple[pd.Series, pd.Series, Dict[str, Dict[str, float]]]:
    """Return calibrated predictions for val/test and fitted parameters."""
    params: Dict[str, Dict[str, float]] = {}
    p_val = pd.Series(index=df_val.index, dtype=np.float32)
    p_test = pd.Series(index=df_test.index, dtype=np.float32)

    for view, g in df_val.groupby("view"):
        x = g["y_hat"].to_numpy(dtype=np.float32)
        y = g["quality"].to_numpy(dtype=np.float32)

        if method == "affine":
            a, b = fit_affine(x, y)
            params[str(view)] = {"a": a, "b": b}
            p_val.loc[g.index] = apply_affine(x, a, b)

            gt = df_test[df_test["view"] == view]
            if len(gt) > 0:
                xt = gt["y_hat"].to_numpy(dtype=np.float32)
                p_test.loc[gt.index] = apply_affine(xt, a, b)
        elif method == "shift_mean":
            # shift so that mean(p) matches mean(y) on val
            b = float(y.mean() - x.mean())
            params[str(view)] = {"b": b}
            p_val.loc[g.index] = x + b

            gt = df_test[df_test["view"] == view]
            if len(gt) > 0:
                xt = gt["y_hat"].to_numpy(dtype=np.float32)
                p_test.loc[gt.index] = xt + b
        else:
            raise ValueError(f"Unknown method: {method}")

    return p_val, p_test, params


def collapse_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for view, g in df.groupby("view"):
        y = g["quality"].to_numpy(dtype=np.float32)
        p = g["y_hat"].to_numpy(dtype=np.float32)
        pr = np.clip(np.rint(p), 0, 9).astype(int)
        bc = np.bincount(pr, minlength=10)
        top = int(bc.argmax())
        top_frac = float(bc[top] / max(1, bc.sum()))
        corr = float("nan")
        if np.std(p) > 1e-9 and np.std(y) > 1e-9:
            corr = float(np.corrcoef(y, p)[0, 1])

        out[str(view)] = {
            "n": int(len(g)),
            "y_mean": float(np.mean(y)),
            "p_mean": float(np.mean(p)),
            "p_std": float(np.std(p)),
            "p_min": float(np.min(p)),
            "p_max": float(np.max(p)),
            "corr_y_p": corr,
            "rounded_top": top,
            "rounded_top_frac": top_frac,
            "rounded_counts": bc.astype(int).tolist(),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", type=Path, required=True)
    ap.add_argument("--test_csv", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, required=True)
    args = ap.parse_args()

    df_val = pd.read_csv(args.val_csv)
    df_test = pd.read_csv(args.test_csv)

    for df, name in [(df_val, "val"), (df_test, "test")]:
        missing = {"view", "quality", "y_hat"} - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {name}: {sorted(missing)}")

    report: Dict[str, Any] = {
        "val_csv": str(args.val_csv),
        "test_csv": str(args.test_csv),
        "raw": {
            "val": metrics_round(df_val["quality"].to_numpy(np.float32), df_val["y_hat"].to_numpy(np.float32)),
            "test": metrics_round(df_test["quality"].to_numpy(np.float32), df_test["y_hat"].to_numpy(np.float32)),
        },
        "collapse": {
            "val": collapse_indicators(df_val),
            "test": collapse_indicators(df_test),
        },
        "methods": {},
    }

    for method in ["shift_mean", "affine"]:
        p_val, p_test, params = per_view_fit_apply(df_val, df_test, method=method)
        # fill any NaNs (views absent from val) with identity
        p_val = p_val.fillna(df_val["y_hat"])
        p_test = p_test.fillna(df_test["y_hat"])

        report["methods"][method] = {
            "params": params,
            "val": metrics_round(df_val["quality"].to_numpy(np.float32), p_val.to_numpy(np.float32)),
            "test": metrics_round(df_test["quality"].to_numpy(np.float32), p_test.to_numpy(np.float32)),
        }

        # per-view test QWK
        pv = {}
        for view, g in df_test.groupby("view"):
            y = g["quality"].to_numpy(np.float32)
            p_raw = g["y_hat"].to_numpy(np.float32)
            p_cal = p_test.loc[g.index].to_numpy(np.float32)
            pv[str(view)] = {
                "qwk_raw": qwk_round(y, p_raw),
                "qwk_cal": qwk_round(y, p_cal),
                "mae_raw": float(mean_absolute_error(y, p_raw)),
                "mae_cal": float(mean_absolute_error(y, p_cal)),
            }
        report["methods"][method]["per_view_test"] = pv

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote: {args.out_json}")
    print("RAW test QWK:", report["raw"]["test"]["qwk"])
    for method in ["shift_mean", "affine"]:
        print(method, "test QWK:", report["methods"][method]["test"]["qwk"])


if __name__ == "__main__":
    main()
