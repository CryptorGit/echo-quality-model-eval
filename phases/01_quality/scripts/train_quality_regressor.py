from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as M

from sklearn.metrics import mean_absolute_error, mean_squared_error


# ===== Paths (repo-root relative by default) =====
ROOT = Path(__file__).resolve().parents[3]
TRAIN_CSV = str(ROOT / "datasets" / "CACTUS" / "manifests" / "cactus_train.csv")
VAL_CSV = str(ROOT / "datasets" / "CACTUS" / "manifests" / "cactus_val.csv")
OUTDIR = ROOT / "shared" / "runs" / "phase1_quality"
OUTDIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Cfg:
    seed: int = 42
    batch_size: int = 128
    num_workers: int = 8
    epochs: int = 200
    lr: float = 3e-4
    img_size: int = 224
    loss: str = "huber"  # "mse" or "huber"
    view_as_feature: bool = False  # まずはFalse（シンプルに）
    amp: bool = True
    compile: bool = False
    resume_from: str = str(OUTDIR / "regressor_ep100.pt")


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_gpu(cfg: Cfg) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device is not available. Install CUDA-enabled PyTorch and ensure NVIDIA driver is active."
        )

    device = "cuda"
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    name = torch.cuda.get_device_name(0)
    print(f"[INFO] device={device}  gpu={name}")
    if "5090" not in name:
        print(f"[WARN] Target GPU expected 5090, but detected: {name}")

    return device


class CactusImageDataset(Dataset):
    def __init__(self, csv_path: str, img_size: int, train: bool):
        df = pd.read_csv(csv_path)
        # 必須列チェック
        for c in ["path", "view", "quality"]:
            if c not in df.columns:
                raise ValueError(f"Missing column {c} in {csv_path}")

        self.paths = df["path"].astype(str).tolist()
        self.views = df["view"].astype(str).tolist()
        self.y = df["quality"].astype(float).tolist()

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

        # 画像変換
        if train:
            self.tf = T.Compose(
                [
                    KeepRatioPad(img_size),
                    T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1)], p=0.3),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
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
        p = self.paths[i]
        y = torch.tensor(self.y[i], dtype=torch.float32)
        # 画像読み込み（壊れファイル対策：例外は上に投げる）
        img = Image.open(p).convert("RGB")
        x = self.tf(img)
        return x, y


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        # 軽くて強い：resnet18（まずはベースライン）
        m = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
        # 最終fcを回帰に
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
        return y, z  # zは後で埋め込みに使える


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            if device == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            y = y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                p, _ = model(x)
            ys.append(y.detach().float().cpu())
            ps.append(p.detach().float().cpu())

    y_all = torch.cat(ys).numpy()
    p_all = torch.cat(ps).numpy()
    mae = mean_absolute_error(y_all, p_all)
    rmse = mean_squared_error(y_all, p_all) ** 0.5
    return mae, rmse


def main():
    cfg = Cfg()
    set_seed(cfg.seed)

    device = setup_gpu(cfg)

    ds_tr = CactusImageDataset(TRAIN_CSV, cfg.img_size, train=True)
    ds_va = CactusImageDataset(VAL_CSV, cfg.img_size, train=False)

    pin_memory = device == "cuda"

    dl_tr = DataLoader(
        ds_tr,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4,
    )

    model = Regressor().to(device)
    model = model.to(memory_format=torch.channels_last)
    if cfg.compile:
        try:
            import triton  # noqa: F401
            model = torch.compile(model)
            print("[INFO] torch.compile enabled")
        except Exception as e:
            print(f"[WARN] torch.compile disabled: {e}")

    if cfg.loss == "mse":
        crit = nn.MSELoss()
    else:
        crit = nn.HuberLoss(delta=1.0)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    start_epoch = 1
    best_rmse = 1e9

    if cfg.resume_from:
        resume_path = Path(cfg.resume_from)
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=True)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_rmse = float(ckpt.get("best_rmse", best_rmse))
            print(
                f"[INFO] Resumed from {resume_path}  start_epoch={start_epoch}  best_rmse={best_rmse:.3f}"
            )
        else:
            print(f"[WARN] resume checkpoint not found: {resume_path}. Start from scratch.")

    for ep in range(start_epoch, cfg.epochs + 1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"epoch {ep}/{cfg.epochs}")
        total = 0.0
        n = 0

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            x = x.contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.amp):
                pred, _ = model(x)
                loss = crit(pred, y)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total += float(loss.detach().cpu()) * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=total / n)

        mae, rmse = evaluate(model, dl_va, device)
        print(f"[VAL] ep={ep}  MAE={mae:.3f}  RMSE={rmse:.3f}")

        # save best
        if rmse < best_rmse:
            best_rmse = rmse
            ckpt = {
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
                "epoch": ep,
                "mae": mae,
                "best_rmse": best_rmse,
            }
            out = OUTDIR / "best_regressor.pt"
            torch.save(ckpt, out)
            print(f"[SAVE] {out}  best_rmse={best_rmse:.3f}")

        if ep % 10 == 0:
            out_ep = OUTDIR / f"regressor_ep{ep:03d}.pt"
            ckpt_ep = {
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
                "epoch": ep,
                "mae": mae,
                "rmse": rmse,
                "best_rmse": best_rmse,
            }
            torch.save(ckpt_ep, out_ep)
            print(f"[SAVE] {out_ep}")

    print("[DONE]")


if __name__ == "__main__":
    # WindowsでDataLoaderが詰まる場合の保険
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
