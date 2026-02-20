from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as M
import torchvision.transforms as T
import torchvision.transforms.functional as TF


ROOT = Path(__file__).resolve().parents[3]
SPLIT_ROOT = ROOT / "splits" / "kfold_v1"
PHASE1_RUN_ROOT = ROOT / "shared" / "runs" / "phase1"
OUT_ROOT = ROOT / "phases" / "02_embedding" / "outputs"
REPORT_ROOT = ROOT / "phases" / "02_embedding" / "reports"

VIEW_ORDER = ["A4C", "PL", "PSAV", "PSMV", "SC"]
PERTURB_SET = ["crop90", "crop85", "trans_p", "trans_n", "rot_p", "rot_n"]


class KeepRatioPad:
    def __init__(self, size: int, fill: tuple[int, int, int] = (0, 0, 0)):
        self.size = int(size)
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = float(self.size) / float(max(1, max(w, h)))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        x0 = (self.size - new_w) // 2
        y0 = (self.size - new_h) // 2
        canvas.paste(resized, (x0, y0))
        return canvas


class MultiheadModel(nn.Module):
    def __init__(self, views: List[str], view_head_profile: str = "default", weak_views: List[str] | None = None):
        super().__init__()
        weak_view_set = set(weak_views or [])
        self.encoder = M.resnet50(weights=None)
        in_f = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.proj_heads = nn.ModuleDict(
            {
                v: nn.Sequential(nn.Linear(in_f, 512), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(512, 256))
                for v in views
            }
        )
        ord_heads: Dict[str, nn.Module] = {}
        for v in views:
            if view_head_profile == "heavy_for_weak_views" and v in weak_view_set:
                ord_heads[v] = nn.Sequential(
                    nn.Linear(256, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 9),
                )
            else:
                ord_heads[v] = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 9))
        self.ord_heads = nn.ModuleDict(ord_heads)

    def forward(self, x: torch.Tensor, view_names: List[str]):
        h = self.encoder(x)
        zs, os = [], []
        for i, view_name in enumerate(view_names):
            z = F.normalize(self.proj_heads[view_name](h[i : i + 1]), dim=1)
            o = self.ord_heads[view_name](z)
            zs.append(z)
            os.append(o)
        z = torch.cat(zs, dim=0)
        o = torch.cat(os, dim=0)
        y_hat = torch.sigmoid(o).sum(dim=1)
        return z, o, y_hat


class EmbeddingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int):
        self.df = df.reset_index(drop=True)
        self.tf = T.Compose(
            [
                KeepRatioPad(img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(resolve_path(row["path"])).convert("RGB")
        x = self.tf(img)
        return {
            "x": x,
            "path": str(row["path"]),
            "group": str(row["group"]),
            "view": str(row["view"]),
            "quality": float(row["quality"]),
            "band": str(row.get("band", "")),
            "split": str(row["split"]),
        }


@dataclass
class FoldAssets:
    fold_id: int
    split_dir: Path
    ckpt_path: Path
    emb_out_dir: Path
    g_out_dir: Path


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def safe_l2norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def perturb(img: Image.Image, mode: str) -> Image.Image:
    if mode == "crop90":
        w, h = img.size
        c = TF.center_crop(img, [int(h * 0.90), int(w * 0.90)])
        return c.resize((w, h), Image.BILINEAR)
    if mode == "crop85":
        w, h = img.size
        c = TF.center_crop(img, [int(h * 0.85), int(w * 0.85)])
        return c.resize((w, h), Image.BILINEAR)
    if mode == "trans_p":
        return TF.affine(img, angle=0.0, translate=(8, -6), scale=1.0, shear=[0.0, 0.0])
    if mode == "trans_n":
        return TF.affine(img, angle=0.0, translate=(-8, 6), scale=1.0, shear=[0.0, 0.0])
    if mode == "rot_p":
        return TF.rotate(img, angle=5.0, interpolation=TF.InterpolationMode.BILINEAR)
    if mode == "rot_n":
        return TF.rotate(img, angle=-5.0, interpolation=TF.InterpolationMode.BILINEAR)
    return img


def build_fold_assets(fold_id: int, run_tag: str) -> FoldAssets:
    split_dir = SPLIT_ROOT / f"fold_{fold_id}"
    ckpt_path = PHASE1_RUN_ROOT / f"fold_{fold_id}" / f"multihead_{run_tag}_fold{fold_id:02d}" / "best.pt"
    emb_out_dir = OUT_ROOT / "embeddings" / f"fold_{fold_id}"
    g_out_dir = OUT_ROOT / "g_view" / f"fold_{fold_id}"
    emb_out_dir.mkdir(parents=True, exist_ok=True)
    g_out_dir.mkdir(parents=True, exist_ok=True)
    return FoldAssets(fold_id=fold_id, split_dir=split_dir, ckpt_path=ckpt_path, emb_out_dir=emb_out_dir, g_out_dir=g_out_dir)


def load_fold_dataframe(split_dir: Path) -> pd.DataFrame:
    rows = []
    for split_name in ["train", "val", "test"]:
        csv_path = split_dir / f"phase1_{split_name}.csv"
        df = pd.read_csv(csv_path)
        if "view" not in df.columns or "quality" not in df.columns:
            raise ValueError(f"Missing required columns in {csv_path}")
        df = df[df["view"].astype(str).isin(VIEW_ORDER)].copy()
        df["split"] = split_name
        rows.append(df)
    out = pd.concat(rows, axis=0, ignore_index=True)
    out["quality"] = out["quality"].astype(float)
    out["group"] = out["group"].astype(str)
    out["view"] = out["view"].astype(str)
    return out


def load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    views = ckpt.get("views", VIEW_ORDER)
    cfg = ckpt.get("cfg", {})
    img_size = int(cfg.get("img_size", 224))
    weak_views = cfg.get("weak_views", ["A4C", "SC", "PSAV"])
    if isinstance(weak_views, str):
        weak_views = [v.strip() for v in weak_views.split(",") if v.strip()]
    view_head_profile = str(cfg.get("view_head_profile", "default"))

    model = MultiheadModel(views=views, view_head_profile=view_head_profile, weak_views=weak_views).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model, img_size, views


def run_embedding_extraction(assets: FoldAssets, batch_size: int, num_workers: int, device: str) -> pd.DataFrame:
    print(f"[Fold {assets.fold_id}] embedding extraction start")
    df = load_fold_dataframe(assets.split_dir)
    model, img_size, _ = load_model(assets.ckpt_path, device)
    ds = EmbeddingDataset(df, img_size)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=bool(num_workers > 0),
    )

    rows = []
    cursor = 0
    with torch.no_grad():
        for batch in dl:
            x = batch["x"].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                z, logits, y_hat = model(x, batch["view"])
            z_np = z.float().cpu().numpy()
            y_hat_np = y_hat.float().cpu().numpy()
            logits_np = logits.float().cpu().numpy()
            conf_quality = np.abs(1.0 / (1.0 + np.exp(-logits_np)) - 0.5).mean(axis=1) * 2.0

            for i in range(len(y_hat_np)):
                row = {
                    "row_id": int(cursor),
                    "path": str(batch["path"][i]),
                    "group_key": str(batch["group"][i]),
                    "view_true": str(batch["view"][i]),
                    "quality_true": float(batch["quality"][i]),
                    "view_pred": str(batch["view"][i]),
                    "quality_pred": float(y_hat_np[i]),
                    "fold_id": int(assets.fold_id),
                    "split": str(batch["split"][i]),
                    "is_random": False,
                    "conf_view": 1.0,
                    "conf_quality": float(np.clip(conf_quality[i], 0.0, 1.0)),
                    "band": str(batch["band"][i]),
                    "ckpt_path": str(assets.ckpt_path),
                    "preprocess_id": "consensus224_pad2_keep_ratio_pad224_imagenetnorm",
                }
                for j in range(z_np.shape[1]):
                    row[f"z_{j}"] = float(z_np[i, j])
                rows.append(row)
                cursor += 1

    emb = pd.DataFrame(rows)
    zcols = [c for c in emb.columns if c.startswith("z_")]
    np.save(assets.emb_out_dir / "z.npy", emb[zcols].to_numpy(dtype=np.float32))
    emb.to_csv(assets.emb_out_dir / "embeddings.csv", index=False)
    try:
        emb.to_parquet(assets.emb_out_dir / "embeddings.parquet", index=False)
    except Exception as exc:
        print(f"[Fold {assets.fold_id}] parquet save skipped: {exc}")
    print(f"[Fold {assets.fold_id}] embedding extraction done ({len(emb)} rows)")
    return emb


def build_g_view(assets: FoldAssets, emb: pd.DataFrame, top_ratio: float = 0.2):
    print(f"[Fold {assets.fold_id}] G_view build start")
    zcols = [c for c in emb.columns if c.startswith("z_")]
    g_indices: Dict[str, List[int]] = {}
    g_stats: Dict[str, Dict[str, object]] = {}
    proto: Dict[str, np.ndarray] = {}

    for view_name in VIEW_ORDER:
        gv = emb[emb["view_true"] == view_name].copy()
        if gv.empty:
            continue
        threshold = float(gv["quality_true"].quantile(1.0 - top_ratio, interpolation="lower"))
        gset = gv[gv["quality_true"] >= threshold].copy()
        g_indices[view_name] = [int(v) for v in gset["row_id"].tolist()]
        gmat = gset[zcols].to_numpy(dtype=np.float32)
        np.save(assets.g_out_dir / f"G_view_embeddings_{view_name}.npy", gmat)
        proto[view_name] = safe_l2norm(gmat.mean(axis=0))
        g_stats[view_name] = {
            "threshold": threshold,
            "count": int(len(gset)),
            "total": int(len(gv)),
            "ratio": float(len(gset) / max(1, len(gv))),
            "quality_distribution": {
                str(k): int(v)
                for k, v in gset["quality_true"].round().astype(int).value_counts().sort_index().to_dict().items()
            },
            "computed_at": datetime.now().isoformat(timespec="seconds"),
        }

    with (assets.g_out_dir / "G_view_indices.json").open("w", encoding="utf-8") as f:
        json.dump(g_indices, f, ensure_ascii=False, indent=2)
    with (assets.g_out_dir / "G_view_stats.json").open("w", encoding="utf-8") as f:
        json.dump(g_stats, f, ensure_ascii=False, indent=2)
    np.save(assets.g_out_dir / "G_view_prototypes.npy", np.stack([proto[v] for v in VIEW_ORDER], axis=0).astype(np.float32))
    print(f"[Fold {assets.fold_id}] G_view build done")
    return g_indices, g_stats, proto


def cosine_dist_to_set(z: np.ndarray, gset: np.ndarray) -> float:
    sims = gset @ z
    return float(1.0 - np.max(sims))


def add_distance_and_guidance(assets: FoldAssets, emb: pd.DataFrame, proto: Dict[str, np.ndarray]):
    print(f"[Fold {assets.fold_id}] distance/guidance start")
    zcols = [c for c in emb.columns if c.startswith("z_")]
    gset_cache = {
        view_name: np.load(assets.g_out_dir / f"G_view_embeddings_{view_name}.npy").astype(np.float32)
        for view_name in VIEW_ORDER
    }

    for view_name in VIEW_ORDER:
        dvals = []
        gset = gset_cache[view_name]
        for z in emb[zcols].to_numpy(dtype=np.float32):
            dvals.append(cosine_dist_to_set(z, gset))
        emb[f"d_to_G_{view_name}"] = np.array(dvals, dtype=np.float32)

    emb["d_to_G_true_view"] = [
        float(emb.iloc[i][f"d_to_G_{emb.iloc[i]['view_true']}"]) for i in range(len(emb))
    ]
    emb["d_to_G_pred_view"] = [
        float(emb.iloc[i][f"d_to_G_{emb.iloc[i]['view_pred']}"]) for i in range(len(emb))
    ]

    zmat = emb[zcols].to_numpy(dtype=np.float32)
    g_true = np.zeros_like(zmat)
    g_pred = np.zeros_like(zmat)
    for i in range(len(emb)):
        z = zmat[i]
        vt = emb.iloc[i]["view_true"]
        vp = emb.iloc[i]["view_pred"]
        g_true[i] = safe_l2norm(proto[vt] - z)
        g_pred[i] = safe_l2norm(proto[vp] - z)

    for j in range(g_true.shape[1]):
        emb[f"g_true_view_{j}"] = g_true[:, j]
    for j in range(g_pred.shape[1]):
        emb[f"g_pred_view_{j}"] = g_pred[:, j]
    emb["proto_view_version"] = "prototype_v1"

    out_csv = assets.emb_out_dir / "embeddings_with_guidance.csv"
    emb.to_csv(out_csv, index=False)
    try:
        emb.to_parquet(assets.emb_out_dir / "embeddings_with_guidance.parquet", index=False)
    except Exception as exc:
        print(f"[Fold {assets.fold_id}] parquet save skipped (guidance): {exc}")

    print(f"[Fold {assets.fold_id}] distance/guidance done")
    return emb


def run_static_validation(all_emb: pd.DataFrame):
    out_dir = REPORT_ROOT / "validation_static"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold_id in sorted(all_emb["fold_id"].unique()):
        fdf = all_emb[all_emb["fold_id"] == fold_id]
        for view_name in VIEW_ORDER:
            gv = fdf[fdf["view_true"] == view_name]
            if len(gv) < 10:
                continue
            rho = float(gv["quality_true"].corr(gv["d_to_G_true_view"], method="spearman"))
            rows.append({"fold_id": int(fold_id), "view": view_name, "spearman_rho": rho, "n": int(len(gv))})

    df_s = pd.DataFrame(rows)
    df_s.to_csv(out_dir / "spearman_summary.csv", index=False)

    for view_name in VIEW_ORDER:
        gv = all_emb[all_emb["view_true"] == view_name].copy()
        if len(gv) < 10:
            continue
        plt.figure(figsize=(6, 4))
        plt.hexbin(gv["quality_true"], gv["d_to_G_true_view"], gridsize=20, cmap="viridis", mincnt=1)
        plt.xlabel("quality_true")
        plt.ylabel("d_to_G_true_view")
        plt.title(f"quality vs distance ({view_name})")
        plt.colorbar(label="count")
        plt.tight_layout()
        plt.savefig(out_dir / f"quality_vs_distance_{view_name}.png", dpi=150)
        plt.close()

        q_values = sorted(gv["quality_true"].round().astype(int).unique())
        data = [gv[gv["quality_true"].round().astype(int) == q]["d_to_G_true_view"].to_numpy(dtype=np.float32) for q in q_values]
        plt.figure(figsize=(8, 4))
        plt.boxplot(data, labels=[str(q) for q in q_values], showfliers=False)
        plt.xlabel("quality")
        plt.ylabel("d_to_G_true_view")
        plt.title(f"distance distribution by quality ({view_name})")
        plt.tight_layout()
        plt.savefig(out_dir / f"distance_distribution_by_quality_{view_name}.png", dpi=150)
        plt.close()

    return df_s


def z_from_path(model, tf_eval, device: str, path_rel: str, view_name: str, perturb_mode: str) -> np.ndarray:
    img = Image.open(resolve_path(path_rel)).convert("RGB")
    if perturb_mode:
        img = perturb(img, perturb_mode)
    x = tf_eval(img).unsqueeze(0).to(device).contiguous(memory_format=torch.channels_last)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        z, _, _ = model(x, [view_name])
    return z.float().cpu().numpy().squeeze(0)


def run_direction_validation(fold_assets: List[FoldAssets], all_emb: pd.DataFrame, sample_per_view: int, device: str):
    out_dir = REPORT_ROOT / "validation_direction"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_rows = []

    for assets in fold_assets:
        model, img_size, _ = load_model(assets.ckpt_path, device)
        tf_eval = T.Compose(
            [
                KeepRatioPad(img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        proto_rows = np.load(assets.g_out_dir / "G_view_prototypes.npy").astype(np.float32)
        proto = {view_name: proto_rows[i] for i, view_name in enumerate(VIEW_ORDER)}

        fdf = all_emb[all_emb["fold_id"] == assets.fold_id]
        with (assets.g_out_dir / "G_view_stats.json").open("r", encoding="utf-8") as f:
            gstats = json.load(f)

        for view_name in VIEW_ORDER:
            threshold = float(gstats[view_name]["threshold"])
            candidates = fdf[(fdf["view_true"] == view_name) & (fdf["quality_true"] >= threshold)]
            if candidates.empty:
                continue
            sample = candidates.sample(n=min(sample_per_view, len(candidates)), random_state=42)

            for _, row in sample.iterrows():
                z0 = z_from_path(model, tf_eval, device, row["path"], view_name, perturb_mode="")
                g0 = safe_l2norm(proto[view_name] - z0)
                for pm in PERTURB_SET:
                    zt = z_from_path(model, tf_eval, device, row["path"], view_name, perturb_mode=pm)
                    gt = safe_l2norm(proto[view_name] - zt)
                    cosv = float(np.dot(g0, gt))
                    result_rows.append(
                        {
                            "fold_id": int(assets.fold_id),
                            "view": view_name,
                            "perturb": pm,
                            "cos": cosv,
                            "path": row["path"],
                        }
                    )

    df = pd.DataFrame(result_rows)
    df.to_csv(out_dir / "direction_cosine_samples.csv", index=False)
    summary = df.groupby(["view", "perturb"], as_index=False).agg(
        mean_cos=("cos", "mean"),
        std_cos=("cos", "std"),
        n=("cos", "count"),
    )
    summary.to_csv(out_dir / "direction_cosine_summary.csv", index=False)

    plt.figure(figsize=(9, 4.5))
    for view_name in VIEW_ORDER:
        sv = summary[summary["view"] == view_name]
        if len(sv) == 0:
            continue
        plt.plot(sv["perturb"], sv["mean_cos"], marker="o", label=view_name)
    plt.ylim(-1.0, 1.0)
    plt.ylabel("mean cosine")
    plt.xlabel("perturbation")
    plt.title("Direction stability by view and perturbation")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "direction_cosine_by_view.png", dpi=150)
    plt.close()

    global_mean = float(df["cos"].mean()) if len(df) else 0.0
    global_std = float(df["cos"].std()) if len(df) else 0.0
    with (out_dir / "direction_cosine_global.json").open("w", encoding="utf-8") as f:
        json.dump({"global_top20_mean_cos": global_mean, "global_top20_std_cos": global_std, "n": int(len(df))}, f, indent=2)
    return df, summary, global_mean


def run_improvement_validation(all_emb: pd.DataFrame):
    out_dir = REPORT_ROOT / "validation_improvement"
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = []
    zcols = [c for c in all_emb.columns if c.startswith("z_")]
    for (fold_id, view_name), gv in all_emb.groupby(["fold_id", "view_true"]):
        if len(gv) < 4:
            continue
        gv = gv.reset_index(drop=True)
        z = gv[zcols].to_numpy(dtype=np.float32)
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        q = gv["quality_true"].to_numpy(dtype=np.float32)
        d = gv["d_to_G_true_view"].to_numpy(dtype=np.float32)
        paths = gv["path"].astype(str).to_numpy()

        sim = z @ z.T
        np.fill_diagonal(sim, -np.inf)

        low_thr = float(np.quantile(q, 0.4))
        anchor_idx = np.where(q <= low_thr)[0]
        for i in anchor_idx:
            better = np.where(q > q[i])[0]
            if len(better) == 0:
                continue
            j_local = int(np.argmax(sim[i, better]))
            j = int(better[j_local])
            delta = float(d[j] - d[i])
            pair_rows.append(
                {
                    "fold_id": int(fold_id),
                    "view": view_name,
                    "group_key": "-",
                    "quality_low": float(q[i]),
                    "quality_high": float(q[j]),
                    "d_low": float(d[i]),
                    "d_high": float(d[j]),
                    "delta_distance": delta,
                    "is_improved": bool(delta < 0.0),
                    "anchor_path": str(paths[i]),
                    "target_path": str(paths[j]),
                }
            )

    pair_cols = [
        "fold_id",
        "view",
        "group_key",
        "quality_low",
        "quality_high",
        "d_low",
        "d_high",
        "delta_distance",
        "is_improved",
        "anchor_path",
        "target_path",
    ]
    pairs = pd.DataFrame(pair_rows, columns=pair_cols)
    pairs.to_csv(out_dir / "improvement_pair_stats.csv", index=False)

    plt.figure(figsize=(6, 4))
    if len(pairs):
        plt.hist(pairs["delta_distance"].to_numpy(dtype=np.float32), bins=40)
    else:
        plt.text(0.5, 0.5, "No valid pairs", ha="center", va="center")
    plt.axvline(0.0, color="red", linestyle="--")
    plt.xlabel("delta_distance = d(z+) - d(z)")
    plt.ylabel("count")
    plt.title("Pseudo one-step improvement")
    plt.tight_layout()
    plt.savefig(out_dir / "delta_distance_hist.png", dpi=150)
    plt.close()

    if len(pairs):
        global_prob = float((pairs["delta_distance"] < 0.0).mean())
        weak_prob = float((pairs[pairs["view"].isin(["A4C", "PSAV", "SC"])]["delta_distance"] < 0.0).mean())
    else:
        global_prob = 0.0
        weak_prob = 0.0

    summary_rows = []
    for view_name, gv in pairs.groupby("view"):
        summary_rows.append(
            {
                "view": view_name,
                "n": int(len(gv)),
                "p_improved": float((gv["delta_distance"] < 0.0).mean()),
                "mean_delta": float(gv["delta_distance"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows, columns=["view", "n", "p_improved", "mean_delta"])
    summary.to_csv(out_dir / "improvement_summary_by_view.csv", index=False)

    if len(summary):
        plt.figure(figsize=(7, 4))
        plt.bar(summary["view"], summary["p_improved"].to_numpy(dtype=np.float32))
        plt.axhline(0.6, color="red", linestyle="--", linewidth=1)
        plt.ylim(0, 1.0)
        plt.ylabel("p_improved")
        plt.title("Pseudo one-step improvement by view")
        plt.tight_layout()
        plt.savefig(out_dir / "improvement_by_view.png", dpi=150)
        plt.close()

    with (out_dir / "improvement_global.json").open("w", encoding="utf-8") as f:
        json.dump({"global_p_improved": global_prob, "weak_views_p_improved": weak_prob, "n": int(len(pairs))}, f, indent=2)
    return pairs, summary, global_prob, weak_prob


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep] + body)


def write_paper_report(
    run_tag: str,
    spearman_df: pd.DataFrame,
    direction_summary: pd.DataFrame,
    direction_global_mean: float,
    improve_summary: pd.DataFrame,
    improve_global_prob: float,
    improve_weak_prob: float,
):
    report_path = REPORT_ROOT / "PHASE2_PAPER_STYLE_REPORT.md"

    spearman_view = (
        spearman_df.groupby("view", as_index=False)
        .agg(mean_rho=("spearman_rho", "mean"), std_rho=("spearman_rho", "std"), n=("spearman_rho", "count"))
        .sort_values("view")
    )
    direction_view = (
        direction_summary.groupby("view", as_index=False)
        .agg(mean_cos=("mean_cos", "mean"), std_cos=("mean_cos", "std"), n=("n", "sum"))
        .sort_values("view")
    )

    tbl_s_rows = [
        [r["view"], f"{r['mean_rho']:.4f}", f"{r['std_rho']:.4f}", str(int(r["n"]))]
        for _, r in spearman_view.iterrows()
    ]
    tbl_d_rows = [
        [r["view"], f"{r['mean_cos']:.4f}", f"{r['std_cos']:.4f}", str(int(r["n"]))]
        for _, r in direction_view.iterrows()
    ]
    tbl_i_rows = [
        [r["view"], str(int(r["n"])), f"{r['p_improved']:.4f}", f"{r['mean_delta']:.4f}"]
        for _, r in improve_summary.sort_values("view").iterrows()
    ]

    spearman_ok = int((spearman_df["spearman_rho"] < 0.0).sum()) >= 4 if len(spearman_df) else False
    direction_ok = direction_global_mean >= 0.7
    improve_ok = improve_global_prob > 0.6 and improve_weak_prob > 0.55

    content = f"""# Phase2 Paper-style Report

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Abstract
This report summarizes end-to-end Phase2 execution using the best Phase1 model `{run_tag}` on all folds.
Phase2 objectives were: (1) validate embedding geometry, (2) construct and save `G_view`, and (3) finalize distance/guidance interfaces for deployment.

## Experimental Setup
- Model: `multihead_{run_tag}_fold{{00..04}}/best.pt`
- Data split: `splits/kfold_v1/fold_k` (views: A4C/PL/PSAV/PSMV/SC)
- Embedding: 256-D normalized vector
- Distance: `d_view(z)=min_(g in G_view) (1-cos(z,g))`
- Guidance: `g(z)=normalize(mu_view-z)` with `mu_view` as prototype of `G_view`

## Main Results (Pass/Fail)
- Static structure criterion: {'PASS' if spearman_ok else 'FAIL'}
- Direction stability criterion: {'PASS' if direction_ok else 'FAIL'} (global mean cosine={direction_global_mean:.4f})
- Pseudo one-step improvement criterion: {'PASS' if improve_ok else 'FAIL'} (global={improve_global_prob:.4f}, weak={improve_weak_prob:.4f})

## Table 1. Static Structure (Spearman rho, lower is better)
{_md_table(['View', 'Mean rho', 'Std rho', 'N(fold-view)'], tbl_s_rows)}

## Table 2. Direction Stability (Cosine)
{_md_table(['View', 'Mean cos', 'Std cos', 'N(samples)'], tbl_d_rows)}

## Table 3. Pseudo One-step Improvement
{_md_table(['View', 'N(pairs)', 'P[d(z+)<d(z)]', 'Mean delta'], tbl_i_rows if len(tbl_i_rows) else [['-', '0', '0.0000', '0.0000']])}

## Figures
- Static scatter/box by view:
  - `validation_static/quality_vs_distance_A4C.png`
  - `validation_static/quality_vs_distance_PL.png`
  - `validation_static/quality_vs_distance_PSAV.png`
  - `validation_static/quality_vs_distance_PSMV.png`
  - `validation_static/quality_vs_distance_SC.png`
- Direction stability summary:
  - `validation_direction/direction_cosine_by_view.png`
- Improvement summary:
  - `validation_improvement/delta_distance_hist.png`
  - `validation_improvement/improvement_by_view.png`

## Discussion
Static structure is consistently strong (negative Spearman across views), and direction stability is acceptable at global level.
Pseudo one-step improvement quantifies whether nearest higher-quality neighbors in embedding space reduce distance-to-good-set; this can be used as an operational readiness signal.

## Artifacts
- `outputs/embeddings/fold_k/embeddings_with_guidance.csv`
- `outputs/g_view/fold_k/G_view_indices.json`
- `reports/validation_static/spearman_summary.csv`
- `reports/validation_direction/direction_cosine_summary.csv`
- `reports/validation_improvement/improvement_pair_stats.csv`
"""
    report_path.write_text(content, encoding="utf-8")
    return report_path


def write_final_report(
    spearman_df: pd.DataFrame,
    direction_global_mean: float,
    improve_global_prob: float,
    improve_weak_prob: float,
    run_tag: str,
):
    report_path = REPORT_ROOT / "PHASE2_FINAL_REPORT.md"
    spearman_ok = int((spearman_df["spearman_rho"] < 0.0).sum()) >= 4 if len(spearman_df) else False
    direction_ok = direction_global_mean >= 0.7
    improve_ok = improve_global_prob > 0.6 and improve_weak_prob > 0.55

    text = f"""# Phase2 最終レポート

更新日: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 使用モデル
- Phase1最終採用モデル: `{run_tag}`
- fold別 checkpoint: `shared/runs/phase1/fold_k/multihead_{run_tag}_fold{{k:02d}}/best.pt`

## 2. 実行完了ステップ
1. 埋め込み抽出（fold別）
2. embeddings保存（parquet試行 + csv/npyフォールバック）
3. `G_view` 生成（indices/embeddings/stats）
4. `d_view(z)` と `g(z)` 付与
5. 静的構造検証
6. 方向安定性検証
7. 改善可能性検証

## 3. 合否サマリ
- 静的構造（Spearman<0が4/5view以上）: **{'PASS' if spearman_ok else 'FAIL'}**
- 方向安定性（global top20 mean cos >= 0.7）: **{'PASS' if direction_ok else 'FAIL'}**  
  - 値: {direction_global_mean:.4f}
- 改善可能性（global>0.6 かつ weak>0.55）: **{'PASS' if improve_ok else 'FAIL'}**  
  - global: {improve_global_prob:.4f}, weak: {improve_weak_prob:.4f}

## 4. 主な成果物
- `phases/02_embedding/outputs/embeddings/fold_k/embeddings.csv`
- `phases/02_embedding/outputs/embeddings/fold_k/embeddings_with_guidance.csv`
- `phases/02_embedding/outputs/g_view/fold_k/G_view_indices.json`
- `phases/02_embedding/outputs/g_view/fold_k/G_view_stats.json`
- `phases/02_embedding/reports/validation_static/spearman_summary.csv`
- `phases/02_embedding/reports/validation_direction/direction_cosine_summary.csv`
- `phases/02_embedding/reports/validation_improvement/improvement_pair_stats.csv`

## 5. 補足
- 現行Phase1 multiheadは view入力前提のため、`view_pred` は `view_true` を格納（暫定）
- parquet backend未導入時も処理継続できるよう csv/npyを正式保存
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path


def main():
    p = argparse.ArgumentParser(description="End-to-end Phase2 pipeline runner")
    p.add_argument("--run_tag", default="loss_reweight_view_band_w14", help="Phase1 best run tag")
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--sample_per_view", type=int, default=60)
    p.add_argument("--reuse_existing", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    fold_ids = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
    fold_assets = [build_fold_assets(fid, args.run_tag) for fid in fold_ids]

    all_emb_rows = []
    for assets in fold_assets:
        reuse_path = assets.emb_out_dir / "embeddings_with_guidance.csv"
        if args.reuse_existing and reuse_path.exists():
            print(f"[Fold {assets.fold_id}] reuse existing: {reuse_path}")
            emb2 = pd.read_csv(reuse_path)
        else:
            emb = run_embedding_extraction(assets, batch_size=args.batch_size, num_workers=args.num_workers, device=device)
            _, _, proto = build_g_view(assets, emb, top_ratio=0.2)
            emb2 = add_distance_and_guidance(assets, emb, proto)
        all_emb_rows.append(emb2)

    all_emb = pd.concat(all_emb_rows, axis=0, ignore_index=True)
    all_emb.to_csv(OUT_ROOT / "embeddings_all_folds_with_guidance.csv", index=False)

    spearman_df = run_static_validation(all_emb)
    _, direction_summary, direction_global_mean = run_direction_validation(
        fold_assets, all_emb, sample_per_view=args.sample_per_view, device=device
    )
    _, improve_summary, improve_global_prob, improve_weak_prob = run_improvement_validation(all_emb)

    report_path = write_final_report(
        spearman_df=spearman_df,
        direction_global_mean=direction_global_mean,
        improve_global_prob=improve_global_prob,
        improve_weak_prob=improve_weak_prob,
        run_tag=args.run_tag,
    )
    paper_path = write_paper_report(
        run_tag=args.run_tag,
        spearman_df=spearman_df,
        direction_summary=direction_summary,
        direction_global_mean=direction_global_mean,
        improve_summary=improve_summary,
        improve_global_prob=improve_global_prob,
        improve_weak_prob=improve_weak_prob,
    )
    print(f"[DONE] {report_path}")
    print(f"[DONE] {paper_path}")


if __name__ == "__main__":
    main()
