from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as M
import torchvision.transforms as T

from sklearn.metrics import mean_absolute_error, mean_squared_error


_PIL_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)


class KeepRatioPad:
    def __init__(self, size: int, fill: tuple[int, int, int] = (0, 0, 0)):
        self.size = int(size)
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
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

VAL_CSV = ROOT / "datasets" / "CACTUS" / "manifests" / "cactus_val.csv"
RUN_DIR = ROOT / "shared" / "runs" / "phase1_quality"
OUT_CSV = RUN_DIR / "checkpoint_val_summary.csv"
OUT_PNG = RUN_DIR / "checkpoint_val_summary.png"


@dataclass
class EvalCfg:
    batch_size: int = 128
    num_workers: int = 8
    img_size: int = 224


class CactusValDataset(Dataset):
    def __init__(self, csv_path: Path, img_size: int):
        df = pd.read_csv(csv_path)
        for c in ["path", "quality"]:
            if c not in df.columns:
                raise ValueError(f"Missing column {c} in {csv_path}")

        self.paths = df["path"].astype(str).tolist()
        self.y = df["quality"].astype(float).tolist()
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
        y = torch.tensor(self.y[i], dtype=torch.float32)
        return x, y


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        m = M.resnet18(weights=None)
        in_f = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        self.head = nn.Sequential(
            nn.Linear(in_f, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        z = self.backbone(x)
        y = self.head(z).squeeze(1)
        return y, z


def parse_epoch(path: Path, ckpt: dict) -> int:
    if "epoch" in ckpt and isinstance(ckpt["epoch"], int):
        return ckpt["epoch"]
    name = path.stem
    if name.startswith("regressor_ep"):
        try:
            return int(name.replace("regressor_ep", ""))
        except Exception:
            return -1
    return -1


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            if device == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                p, _ = model(x)
            ys.append(y.detach().float().cpu())
            ps.append(p.detach().float().cpu())

    y_all = torch.cat(ys).numpy()
    p_all = torch.cat(ps).numpy()
    mae = mean_absolute_error(y_all, p_all)
    rmse = mean_squared_error(y_all, p_all) ** 0.5
    return mae, rmse


def list_checkpoints(run_dir: Path) -> List[Path]:
    files = sorted(run_dir.glob("regressor_ep*.pt"))
    best = run_dir / "best_regressor.pt"
    if best.exists():
        files.append(best)
    return files


def main():
    cfg = EvalCfg()
    if not VAL_CSV.exists():
        raise FileNotFoundError(f"Validation CSV not found: {VAL_CSV}")

    ckpts = list_checkpoints(RUN_DIR)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in: {RUN_DIR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] evaluate device={device}")

    ds = CactusValDataset(VAL_CSV, img_size=cfg.img_size)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4,
    )

    rows = []
    for path in ckpts:
        print(f"[INFO] Evaluating {path.name}")
        ckpt = torch.load(path, map_location="cpu")

        model = Regressor().to(device)
        model = model.to(memory_format=torch.channels_last)
        model.load_state_dict(ckpt["model"], strict=True)

        mae, rmse = evaluate(model, dl, device)
        rows.append(
            {
                "checkpoint": path.name,
                "epoch": parse_epoch(path, ckpt),
                "mae": float(mae),
                "rmse": float(rmse),
                "is_best_file": path.name == "best_regressor.pt",
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["epoch", "checkpoint"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)

    df_ep = df[(df["epoch"] > 0) & (~df["is_best_file"])].copy().sort_values("epoch")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=140)
    axes[0].plot(df_ep["epoch"], df_ep["mae"], marker="o")
    axes[0].set_title("Validation MAE vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MAE")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df_ep["epoch"], df_ep["rmse"], marker="o")
    axes[1].set_title("Validation RMSE vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(True, alpha=0.3)

    if len(df_ep) > 0:
        best_i = df_ep["rmse"].idxmin()
        best_ep = int(df_ep.loc[best_i, "epoch"])
        best_rmse = float(df_ep.loc[best_i, "rmse"])
        axes[1].scatter([best_ep], [best_rmse], s=80, marker="*", zorder=5)
        axes[1].annotate(
            f"best ep={best_ep}\nrmse={best_rmse:.3f}",
            (best_ep, best_rmse),
            textcoords="offset points",
            xytext=(8, 8),
        )

    fig.tight_layout()
    fig.savefig(OUT_PNG)

    print(f"[OK] wrote: {OUT_CSV}")
    print(f"[OK] wrote: {OUT_PNG}")
    print("\nTop-5 by RMSE:")
    print(df.sort_values("rmse").head(5).to_string(index=False))


if __name__ == "__main__":
    main()
