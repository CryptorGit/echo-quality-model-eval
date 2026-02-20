from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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

from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error, confusion_matrix


_PIL_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)


class KeepRatioPad:
    def __init__(self, size: int, fill: tuple[int, int, int] = (0, 0, 0)):
        self.size = int(size)
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.size <= 0:
            raise ValueError("size must be positive")
        w, h = img.size
        if w <= 0 or h <= 0:
            raise ValueError("invalid image")

        scale = float(self.size) / float(max(w, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = img.resize((new_w, new_h), resample=_PIL_BILINEAR)

        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        x0 = (self.size - new_w) // 2
        y0 = (self.size - new_h) // 2
        canvas.paste(resized, (x0, y0))
        return canvas


ROOT = Path(__file__).resolve().parents[3]

DEFAULT_SPLIT_DIR = ROOT / "datasets" / "CACTUS" / "manifests_group"
RUNS_DIR = ROOT / "shared" / "runs" / "phase1"
OUT_DIR = RUNS_DIR / "eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class EvalDataset(Dataset):
    def __init__(self, csv_path: Path, img_size: int, view_to_idx: Dict[str, int], view_filter: str = ""):
        df = pd.read_csv(csv_path)
        if view_filter:
            df = df[df["view"] == view_filter].reset_index(drop=True)
        df = df[df["view"].astype(str).isin(set(view_to_idx.keys()))].reset_index(drop=True)

        self.df = df.reset_index(drop=True)
        self.paths = self.df["path"].astype(str).tolist()
        self.views = self.df["view"].astype(str).tolist()
        self.y = self.df["quality"].astype(float).tolist()
        self.groups = self.df["group"].astype(str).tolist()
        self.view_to_idx = view_to_idx
        self.view_ids = [view_to_idx[v] for v in self.views]

        self.tf = T.Compose([
            KeepRatioPad(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        x = self.tf(img)
        return {
            "x": x,
            "y": torch.tensor(self.y[i], dtype=torch.float32),
            "view": self.views[i],
            "view_id": torch.tensor(self.view_ids[i], dtype=torch.long),
            "path": self.paths[i],
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


def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    rmse = float(mean_squared_error(y, p) ** 0.5)
    mae = float(mean_absolute_error(y, p))
    y_r = np.clip(np.rint(y), 0, 9).astype(int)
    p_r = np.clip(np.rint(p), 0, 9).astype(int)
    qwk = float(cohen_kappa_score(y_r, p_r, weights="quadratic"))
    return {"rmse": rmse, "mae": mae, "qwk": qwk}


def per_view_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out = {}
    for view, gv in df.groupby("view"):
        y = gv["quality"].to_numpy(dtype=np.float32)
        p = gv["y_hat"].to_numpy(dtype=np.float32)
        out[str(view)] = metrics(y, p)
    return out


def per_view_confusion(df: pd.DataFrame) -> Dict[str, List[List[int]]]:
    out: Dict[str, List[List[int]]] = {}
    labels = list(range(10))
    for view, gv in df.groupby("view"):
        y = np.clip(np.rint(gv["quality"].to_numpy(dtype=np.float32)), 0, 9).astype(int)
        p = np.clip(np.rint(gv["y_hat"].to_numpy(dtype=np.float32)), 0, 9).astype(int)
        cm = confusion_matrix(y, p, labels=labels)
        out[str(view)] = cm.astype(int).tolist()
    return out


def perturb(img: Image.Image, mode: str) -> Image.Image:
    if mode == "crop":
        w, h = img.size
        c = TF.center_crop(img, [int(h * 0.9), int(w * 0.9)])
        return c.resize((w, h), Image.BILINEAR)
    if mode == "translate":
        return TF.affine(img, angle=0.0, translate=(8, -6), scale=1.0, shear=[0.0, 0.0])
    if mode == "rotate":
        return TF.rotate(img, angle=5.0, interpolation=TF.InterpolationMode.BILINEAR)
    return img


def run_eval(
    ckpt_path: Path,
    mode: str,
    view: str = "",
    split_dir: Path = DEFAULT_SPLIT_DIR,
    save_confusion: bool = False,
    out_suffix: str = "",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    views = ckpt.get("views", [view] if view else ["A4C", "PL", "PSAV", "PSMV", "SC"])
    view_to_idx = ckpt.get("view_to_idx", {v: i for i, v in enumerate(views)})
    img_size = int(ckpt.get("cfg", {}).get("img_size", 224))

    view_head_profile = str(ckpt.get("cfg", {}).get("view_head_profile", ckpt.get("view_head_profile", "default")))
    weak_views = ckpt.get("cfg", {}).get("weak_views", ckpt.get("weak_views", ["A4C", "SC", "PSAV"]))
    if isinstance(weak_views, str):
        weak_views = [v.strip() for v in weak_views.split(",") if v.strip()]

    if mode == "multihead":
        model = MultiheadModel(views=views, view_head_profile=view_head_profile, weak_views=weak_views).to(device)
    else:
        model = PerViewModel().to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(memory_format=torch.channels_last)
    model.eval()

    val_csv = split_dir / "phase1_val.csv"
    test_csv = split_dir / "phase1_test.csv"
    if not val_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Missing split CSVs in: {split_dir}")

    ds_val = EvalDataset(val_csv, img_size, view_to_idx, view_filter=view)
    ds_test = EvalDataset(test_csv, img_size, view_to_idx, view_filter=view)
    dl_val = DataLoader(ds_val, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    dl_test = DataLoader(ds_test, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    def infer(loader: DataLoader):
        ys, ps = [], []
        emb_rows = []
        with torch.no_grad():
            for b in loader:
                x = b["x"].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                y = b["y"].to(device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                    if mode == "multihead":
                        z, _, y_hat = model(x, b["view"])
                    else:
                        z, _, y_hat = model(x)

                ys.append(y.float().cpu())
                ps.append(y_hat.float().cpu())

                zc = z.float().cpu().numpy()
                yc = y.float().cpu().numpy()
                pc = y_hat.float().cpu().numpy()
                for i in range(len(pc)):
                    emb_rows.append({
                        "path": b["path"][i],
                        "view": b["view"][i],
                        "quality": float(yc[i]),
                        "group": b["group"][i],
                        "y_hat": float(pc[i]),
                        **{f"z_{j}": float(zc[i, j]) for j in range(zc.shape[1])},
                    })

        y_all = torch.cat(ys).numpy()
        p_all = torch.cat(ps).numpy()
        return y_all, p_all, pd.DataFrame(emb_rows)

    yv, pv, emb_val = infer(dl_val)
    yt, pt, emb_test = infer(dl_test)

    m_val = metrics(yv, pv)
    m_test = metrics(yt, pt)
    pv_val = per_view_metrics(emb_val)
    pv_test = per_view_metrics(emb_test)

    # per-view G(top20%) direction stability on val
    stab = {}
    for v, gv in emb_val.groupby("view"):
        if len(gv) < 10:
            continue
        k = max(1, int(np.ceil(len(gv) * 0.2)))
        g_good = gv.sort_values("quality", ascending=False).head(k)
        gvec = g_good[[c for c in g_good.columns if c.startswith("z_")]].to_numpy(dtype=np.float32).mean(axis=0)

        tf = ds_val.tf
        rows = gv.sample(n=min(120, len(gv)), random_state=42)
        cos_scores = {"crop": [], "translate": [], "rotate": []}
        for _, r in rows.iterrows():
            img = Image.open(r["path"]).convert("RGB")
            x0 = tf(img).unsqueeze(0).to(device).contiguous(memory_format=torch.channels_last)
            if mode == "multihead":
                z0, _, _ = model(x0, [v])
            else:
                z0, _, _ = model(x0)
            z0 = z0.detach().float().cpu().numpy().squeeze(0)
            d0 = gvec - z0
            d0 = d0 / (np.linalg.norm(d0) + 1e-12)

            for tmode in ["crop", "translate", "rotate"]:
                imgp = perturb(img, tmode)
                xp = tf(imgp).unsqueeze(0).to(device).contiguous(memory_format=torch.channels_last)
                if mode == "multihead":
                    zp, _, _ = model(xp, [v])
                else:
                    zp, _, _ = model(xp)
                zp = zp.detach().float().cpu().numpy().squeeze(0)
                dp = gvec - zp
                dp = dp / (np.linalg.norm(dp) + 1e-12)
                cos_scores[tmode].append(float((d0 * dp).sum()))

        stab[v] = {
            m: {
                "mean_cos": float(np.mean(vals)),
                "std_cos": float(np.std(vals)),
                "ratio_cos_gt_0.7": float(np.mean(np.array(vals) > 0.7)),
            }
            for m, vals in cos_scores.items()
        }

    out_name = f"{mode}_{view if view else 'all'}"
    if out_suffix:
        out_name = f"{out_name}_{out_suffix}"
    out = {
        "mode": mode,
        "view": view,
        "ckpt": str(ckpt_path),
        "val": m_val,
        "test": m_test,
        "per_view_val": pv_val,
        "per_view_test": pv_test,
        "direction_stability": stab,
    }

    if save_confusion:
        out["confusion_test"] = per_view_confusion(emb_test)

    with (OUT_DIR / f"{out_name}_report.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    emb_dir = OUT_DIR / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    emb_val.to_csv(emb_dir / f"{out_name}_val_embeddings.csv", index=False)
    emb_test.to_csv(emb_dir / f"{out_name}_test_embeddings.csv", index=False)

    print(f"[OK] report: {OUT_DIR / (out_name + '_report.json')}")
    print(f"[OK] val={m_val} test={m_test}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["multihead", "perview"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--view", default="")
    p.add_argument("--split_dir", default=str(DEFAULT_SPLIT_DIR))
    p.add_argument("--save_confusion", action="store_true")
    p.add_argument("--out_suffix", default="")
    args = p.parse_args()

    run_eval(
        Path(args.ckpt),
        mode=args.mode,
        view=args.view,
        split_dir=Path(args.split_dir),
        save_confusion=bool(args.save_confusion),
        out_suffix=args.out_suffix,
    )


if __name__ == "__main__":
    main()
