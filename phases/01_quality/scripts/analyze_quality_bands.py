from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error


BANDS = {
    "0-2": (0.0, 2.999),
    "3-5": (3.0, 5.999),
    "6-9": (6.0, 9.0),
}


def band_metrics(df: pd.DataFrame) -> dict[str, float]:
    y = df["quality"].to_numpy(dtype=np.float32)
    p = df["y_hat"].to_numpy(dtype=np.float32)
    mae = float(mean_absolute_error(y, p)) if len(df) > 0 else float("nan")

    y_r = np.clip(np.rint(y), 0, 9).astype(int)
    p_r = np.clip(np.rint(p), 0, 9).astype(int)

    if len(np.unique(y_r)) < 2 and len(np.unique(p_r)) < 2:
        qwk = 1.0 if np.array_equal(y_r, p_r) else 0.0
    else:
        qwk = float(cohen_kappa_score(y_r, p_r, weights="quadratic"))
    return {"qwk": qwk, "mae": mae}


def main():
    parser = argparse.ArgumentParser(description="Band-wise metrics and prediction histograms")
    parser.add_argument("--emb_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--views", default="PSAV,SC")
    args = parser.parse_args()

    emb_csv = Path(args.emb_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(emb_csv)
    views = [v.strip() for v in args.views.split(",") if v.strip()]

    metric_rows = []
    hist_rows = []

    for view in views:
        gv = df[df["view"] == view].copy()
        if len(gv) == 0:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
        for ax_i, (band_name, (lo, hi)) in enumerate(BANDS.items()):
            gb = gv[(gv["quality"] >= lo) & (gv["quality"] <= hi)].copy()
            m = band_metrics(gb)
            metric_rows.append(
                {
                    "view": view,
                    "band": band_name,
                    "n": int(len(gb)),
                    "qwk": m["qwk"],
                    "mae": m["mae"],
                }
            )

            pred_round = np.clip(np.rint(gb["y_hat"].to_numpy(dtype=np.float32)), 0, 9).astype(int)
            counts = np.bincount(pred_round, minlength=10)
            for score, cnt in enumerate(counts.tolist()):
                hist_rows.append(
                    {
                        "view": view,
                        "band": band_name,
                        "pred_score": int(score),
                        "count": int(cnt),
                    }
                )

            ax = axes[ax_i]
            ax.bar(np.arange(10), counts)
            ax.set_title(f"{view} {band_name} (n={len(gb)})")
            ax.set_xlabel("pred(round)")
            ax.set_ylabel("count")
            ax.set_xticks(np.arange(10))

        fig.suptitle(f"Prediction Histograms by Quality Band: {view}")
        fig.savefig(out_dir / f"{view}_pred_hist_by_band.png", dpi=140)
        plt.close(fig)

    metrics_df = pd.DataFrame(metric_rows)
    hist_df = pd.DataFrame(hist_rows)

    metrics_path = out_dir / "band_metrics.csv"
    hist_path = out_dir / "band_pred_hist.csv"
    metrics_df.to_csv(metrics_path, index=False)
    hist_df.to_csv(hist_path, index=False)

    print(metrics_df.to_string(index=False))
    print(f"\n[OK] metrics: {metrics_path}")
    print(f"[OK] hist: {hist_path}")


if __name__ == "__main__":
    main()
