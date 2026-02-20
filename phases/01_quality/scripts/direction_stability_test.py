from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

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

DEFAULT_SPLIT_DIR = ROOT / "datasets" / "CACTUS" / "manifests_group_consensus"
OUT_DIR = ROOT / "shared" / "runs" / "phase1" / "eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class KeepRatioPad:
    def __init__(self, size: int, fill: int = 0):
        self.size = int(size)
        self.fill = int(fill)

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == 0 or h == 0:
            return Image.new("RGB", (self.size, self.size), (self.fill, self.fill, self.fill))
        scale = self.size / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new("RGB", (self.size, self.size), (self.fill, self.fill, self.fill))
        left = (self.size - new_w) // 2
        top = (self.size - new_h) // 2
        canvas.paste(img, (left, top))
        return canvas


class EvalDataset(Dataset):
    def __init__(self, csv_path: Path, img_size: int, view_to_idx: Dict[str, int]):
        df = pd.read_csv(csv_path)
        if "view" not in df.columns:
            raise ValueError(f"Missing view in {csv_path}")
        self.df = df[df["view"] != "Random"].reset_index(drop=True)
        self.paths = self.df["path"].astype(str).tolist()
        self.views = self.df["view"].astype(str).tolist()
        self.y = self.df["quality"].astype(float).tolist()
        self.groups = self.df["group"].astype(str).tolist()
        self.view_to_idx = view_to_idx

        self.tf = T.Compose(
            [
                KeepRatioPad(img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        x = self.tf(img)
        return {
            "x": x,
            "path": self.paths[i],
            "view": self.views[i],
            "quality": float(self.y[i]),
            "group": self.groups[i],
        }


class MultiheadModel(nn.Module):
    def __init__(self, views: List[str], view_head_profile: str = "default", weak_views: List[str] | None = None):
        super().__init__()
        weak_view_set = set(weak_views or [])
        self.encoder = M.resnet50(weights=None)
        in_f = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.proj_heads = nn.ModuleDict({
            v: nn.Sequential(nn.Linear(in_f, 512), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(512, 256))
            for v in views
        })
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
        for i, v in enumerate(view_names):
            z = F.normalize(self.proj_heads[v](h[i:i + 1]), dim=1)
            o = self.ord_heads[v](z)
            zs.append(z)
            os.append(o)
        z = torch.cat(zs, dim=0)
        o = torch.cat(os, dim=0)
        y_hat = torch.sigmoid(o).sum(dim=1)
        return z, o, y_hat


class PerViewModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = M.resnet50(weights=None)
        in_f = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.proj = nn.Sequential(nn.Linear(in_f, 512), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(512, 256))
        self.ord = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 9))

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        z = F.normalize(self.proj(h), dim=1)
        o = self.ord(z)
        y_hat = torch.sigmoid(o).sum(dim=1)
        return z, o, y_hat


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


def l2norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def infer_embeddings(model, mode: str, loader: DataLoader, device: str) -> pd.DataFrame:
    rows = []
    with torch.no_grad():
        for b in loader:
            x = b["x"].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                if mode == "multihead":
                    z, _, yhat = model(x, b["view"])
                else:
                    z, _, yhat = model(x)

            zc = z.float().cpu().numpy()
            yc = np.array(b["quality"], dtype=np.float32)
            pc = yhat.float().cpu().numpy()
            for i in range(len(pc)):
                rows.append({
                    "path": b["path"][i],
                    "view": b["view"][i],
                    "quality": float(yc[i]),
                    "y_hat": float(pc[i]),
                    **{f"z_{j}": float(zc[i, j]) for j in range(zc.shape[1])},
                })
    return pd.DataFrame(rows)


def z_from_image(model, mode: str, tf: T.Compose, device: str, path: str, view: str, perturb_mode: str = "") -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if perturb_mode:
        img = perturb(img, perturb_mode)
    x = tf(img).unsqueeze(0).to(device).contiguous(memory_format=torch.channels_last)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        if mode == "multihead":
            z, _, _ = model(x, [view])
        else:
            z, _, _ = model(x)
    return z.float().cpu().numpy().squeeze(0)


def build_g(df_view: pd.DataFrame, ratio: int) -> np.ndarray:
    k = max(1, int(np.ceil(len(df_view) * (ratio / 100.0))))
    zcols = [c for c in df_view.columns if c.startswith("z_")]
    g = df_view.sort_values("quality", ascending=False).head(k)[zcols].to_numpy(dtype=np.float32).mean(axis=0)
    return g


def nearest_mean(good_z: np.ndarray, z0: np.ndarray, k: int) -> np.ndarray:
    d = np.linalg.norm(good_z - z0[None, :], axis=1)
    idx = np.argsort(d)[: max(1, min(k, len(good_z)))]
    return good_z[idx].mean(axis=0)


def direction_stats(vals: List[float]) -> Dict[str, float]:
    arr = np.array(vals, dtype=np.float32)
    if len(arr) == 0:
        return {"mean_cos": 0.0, "std_cos": 0.0, "ratio_cos_gt_0.7": 0.0}
    return {
        "mean_cos": float(arr.mean()),
        "std_cos": float(arr.std()),
        "ratio_cos_gt_0.7": float((arr > 0.7).mean()),
    }


def main():
    p = argparse.ArgumentParser(description="Direction stability and improvement-potential test")
    p.add_argument("--mode", choices=["multihead", "perview"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--view", default="")
    p.add_argument("--split_dir", default=str(DEFAULT_SPLIT_DIR))
    p.add_argument("--top_ratios", default="10,20,30")
    p.add_argument("--sample_per_view", type=int, default=80)
    p.add_argument("--lowq_ratio", type=float, default=0.30)
    p.add_argument("--knn_k", type=int, default=20)
    p.add_argument("--out_json", default="direction_stability_report.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(Path(args.ckpt), map_location="cpu")
    views = ckpt.get("views", [args.view] if args.view else ["A4C", "PL", "PSAV", "PSMV", "SC"])
    view_to_idx = ckpt.get("view_to_idx", {v: i for i, v in enumerate(views)})
    img_size = int(ckpt.get("cfg", {}).get("img_size", 224))
    view_head_profile = str(ckpt.get("cfg", {}).get("view_head_profile", ckpt.get("view_head_profile", "default")))
    weak_views = ckpt.get("cfg", {}).get("weak_views", ckpt.get("weak_views", ["A4C", "SC", "PSAV"]))
    if isinstance(weak_views, str):
        weak_views = [v.strip() for v in weak_views.split(",") if v.strip()]

    if args.mode == "multihead":
        model = MultiheadModel(views=views, view_head_profile=view_head_profile, weak_views=weak_views).to(device)
    else:
        model = PerViewModel().to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(memory_format=torch.channels_last)
    model.eval()

    split_dir = Path(args.split_dir)
    ds_val = EvalDataset(split_dir / "phase1_val.csv", img_size, view_to_idx)
    loader = DataLoader(ds_val, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    emb = infer_embeddings(model, args.mode, loader, device)

    if args.mode == "perview" and args.view:
        emb = emb[emb["view"] == args.view].reset_index(drop=True)

    ratios = [int(x.strip()) for x in args.top_ratios.split(",") if x.strip()]
    zcols = [c for c in emb.columns if c.startswith("z_")]
    perturb_set = ["crop90", "crop85", "trans_p", "trans_n", "rot_p", "rot_n"]

    stability: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    monotonicity: Dict[str, Dict[str, float]] = {}

    rng = np.random.default_rng(42)

    for view, gv in emb.groupby("view"):
        if len(gv) < 20:
            continue

        sample_n = min(args.sample_per_view, len(gv))
        sample = gv.sample(n=sample_n, random_state=42).reset_index(drop=True)
        good_cache = {}

        stability[view] = {}
        for ratio in ratios:
            g_mean = build_g(gv, ratio)
            good_df = gv.sort_values("quality", ascending=False).head(max(1, int(np.ceil(len(gv) * (ratio / 100.0)))))
            good_z = good_df[zcols].to_numpy(dtype=np.float32)
            good_cache[ratio] = (g_mean, good_z)

            cos_a, cos_b = [], []
            for _, row in sample.iterrows():
                z0 = z_from_image(model, args.mode, ds_val.tf, device, row["path"], view, perturb_mode="")

                v0_a = l2norm(g_mean - z0)
                g_loc = nearest_mean(good_z, z0, args.knn_k)
                v0_b = l2norm(g_loc - z0)

                for pm in perturb_set:
                    zt = z_from_image(model, args.mode, ds_val.tf, device, row["path"], view, perturb_mode=pm)

                    vt_a = l2norm(g_mean - zt)
                    cos_a.append(float(np.dot(v0_a, vt_a)))

                    g_loc_t = nearest_mean(good_z, zt, args.knn_k)
                    vt_b = l2norm(g_loc_t - zt)
                    cos_b.append(float(np.dot(v0_b, vt_b)))

            stability[view][f"top{ratio}"] = {
                "mean": direction_stats(cos_a),
                "knn": direction_stats(cos_b),
            }

        # monotonicity with top20 default
        ratio = 20 if 20 in ratios else ratios[0]
        g_mean, good_z = good_cache[ratio]

        q_thr = float(gv["quality"].quantile(args.lowq_ratio))
        low_df = gv[gv["quality"] <= q_thr].copy()
        if len(low_df) == 0:
            continue
        low_df = low_df.sample(n=min(args.sample_per_view, len(low_df)), random_state=43)

        guided_hits = 0
        random_hits = 0
        total = 0

        for _, row in low_df.iterrows():
            z0 = z_from_image(model, args.mode, ds_val.tf, device, row["path"], view, perturb_mode="")
            d0 = float(np.linalg.norm(z0 - g_mean))
            v0 = l2norm(g_mean - z0)

            candidates = []
            for pm in perturb_set:
                zt = z_from_image(model, args.mode, ds_val.tf, device, row["path"], view, perturb_mode=pm)
                delta = l2norm(zt - z0)
                align = float(np.dot(delta, v0))
                dt = float(np.linalg.norm(zt - g_mean))
                candidates.append((pm, align, dt))

            candidates.sort(key=lambda x: x[1], reverse=True)
            guided_dt = candidates[0][2]
            rand_dt = candidates[int(rng.integers(0, len(candidates)))][2]

            guided_hits += int(guided_dt < d0)
            random_hits += int(rand_dt < d0)
            total += 1

        monotonicity[view] = {
            "lowq_threshold": q_thr,
            "n_samples": int(total),
            "guided_improve_rate": float(guided_hits / max(total, 1)),
            "random_improve_rate": float(random_hits / max(total, 1)),
            "delta_guided_minus_random": float((guided_hits - random_hits) / max(total, 1)),
        }

    # aggregate top20 mean-method gate
    all_top20 = []
    for view, rep in stability.items():
        if "top20" in rep:
            all_top20.append(rep["top20"]["mean"]["mean_cos"])
    global_mean_cos_top20 = float(np.mean(all_top20)) if all_top20 else 0.0

    out = {
        "mode": args.mode,
        "view": args.view,
        "ckpt": args.ckpt,
        "split_dir": str(split_dir),
        "top_ratios": ratios,
        "stability": stability,
        "monotonicity": monotonicity,
        "global_mean_cos_top20_mean_method": global_mean_cos_top20,
        "phase2_gate_hint": {
            "target_mean_cos_top20": 0.7,
            "pass_top20_mean_cos": bool(global_mean_cos_top20 >= 0.7),
        },
    }

    out_path = OUT_DIR / args.out_json
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] report: {out_path}")
    print(f"[OK] global_mean_cos_top20_mean_method={global_mean_cos_top20:.4f}")


if __name__ == "__main__":
    main()
