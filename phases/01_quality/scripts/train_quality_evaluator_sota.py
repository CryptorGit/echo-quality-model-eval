from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset
import torchvision.models as M
import torchvision.transforms as T

from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[3]
TRAIN_CSV = str(ROOT / "datasets" / "CACTUS" / "manifests" / "cactus_train.csv")
VAL_CSV = str(ROOT / "datasets" / "CACTUS" / "manifests" / "cactus_val.csv")
OUTDIR = ROOT / "shared" / "runs" / "phase1_quality_sota"
OUTDIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Cfg:
    seed: int = 42
    img_size: int = 288
    batch_size: int = 96
    num_workers: int = 8
    epochs: int = 60
    lr: float = 3e-4
    wd: float = 1e-4
    amp: bool = True
    ema_decay: float = 0.999
    save_every: int = 10
    loss_reg_w: float = 1.0
    loss_cls_w: float = 0.7
    loss_consistency_w: float = 0.15


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = "cuda"
    print(f"[INFO] device={device}  gpu={torch.cuda.get_device_name(0)}")
    return device


class CactusDataset(Dataset):
    def __init__(self, csv_path: str, img_size: int, train: bool, view_to_idx: Dict[str, int] | None = None):
        df = pd.read_csv(csv_path)
        for c in ["path", "view", "quality"]:
            if c not in df.columns:
                raise ValueError(f"Missing column {c} in {csv_path}")

        self.paths = df["path"].astype(str).tolist()
        self.views = df["view"].astype(str).tolist()
        self.y_float = df["quality"].astype(float).tolist()
        self.y_cls = [int(np.clip(round(x), 0, 9)) for x in self.y_float]

        if view_to_idx is None:
            uniq = sorted(set(self.views))
            self.view_to_idx = {v: i for i, v in enumerate(uniq)}
        else:
            self.view_to_idx = view_to_idx

        self.view_ids = [self.view_to_idx[v] for v in self.views]

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

        if train:
            self.tf = T.Compose(
                [
                    T.Resize((img_size, img_size)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomApply([T.ColorJitter(0.15, 0.15, 0.10, 0.05)], p=0.5),
                    T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.tf = T.Compose(
                [
                    T.Resize((img_size, img_size)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(self.y_float[i], dtype=torch.float32)
        y_cls = torch.tensor(self.y_cls[i], dtype=torch.long)
        view_id = torch.tensor(self.view_ids[i], dtype=torch.long)
        return x, view_id, y, y_cls


class QualityEvaluatorModel(nn.Module):
    def __init__(self, n_views: int):
        super().__init__()

        backbone = M.efficientnet_v2_s(weights=M.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_f = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.view_emb = nn.Embedding(n_views, 16)

        fusion_dim = in_f + 16
        self.trunk = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.20),
        )
        self.head_reg = nn.Linear(256, 1)
        self.head_cls = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor, view_id: torch.Tensor):
        feat = self.backbone(x)
        v = self.view_emb(view_id)
        z = torch.cat([feat, v], dim=1)
        h = self.trunk(z)

        reg = self.head_reg(h).squeeze(1)
        logits = self.head_cls(h)

        prob = F.softmax(logits, dim=1)
        bins = torch.arange(10, device=logits.device, dtype=logits.dtype).unsqueeze(0)
        exp_score = (prob * bins).sum(dim=1)

        pred = 0.7 * exp_score + 0.3 * reg
        return pred, reg, logits


def compute_class_weights(y_cls: List[int]) -> torch.Tensor:
    counts = np.bincount(np.array(y_cls, dtype=np.int64), minlength=10).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    w = inv / inv.sum() * len(inv)
    return torch.tensor(w, dtype=torch.float32)


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    ys, ps = [], []

    with torch.no_grad():
        for x, view_id, y, _ in loader:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            view_id = view_id.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                pred, _, _ = model(x, view_id)

            ys.append(y.detach().float().cpu())
            ps.append(pred.detach().float().cpu())

    y_all = torch.cat(ys).numpy()
    p_all = torch.cat(ps).numpy()
    mae = mean_absolute_error(y_all, p_all)
    rmse = mean_squared_error(y_all, p_all) ** 0.5
    return mae, rmse


def save_ckpt(path: Path, model: nn.Module, cfg: Cfg, epoch: int, mae: float, rmse: float, best_rmse: float):
    ckpt = {
        "model": model.state_dict(),
        "cfg": cfg.__dict__,
        "epoch": epoch,
        "mae": mae,
        "rmse": rmse,
        "best_rmse": best_rmse,
    }
    torch.save(ckpt, path)


def main():
    cfg = Cfg()
    set_seed(cfg.seed)

    ds_tr = CactusDataset(TRAIN_CSV, cfg.img_size, train=True)
    ds_va = CactusDataset(VAL_CSV, cfg.img_size, train=False, view_to_idx=ds_tr.view_to_idx)

    print(f"[INFO] views={ds_tr.view_to_idx}")

    dl_tr = DataLoader(
        ds_tr,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4,
    )

    model = QualityEvaluatorModel(n_views=len(ds_tr.view_to_idx)).to(device)
    model = model.to(memory_format=torch.channels_last)

    ema_model = AveragedModel(model, avg_fn=lambda avg, cur, n: cfg.ema_decay * avg + (1 - cfg.ema_decay) * cur)
    ema_model = ema_model.to(device)

    class_w = compute_class_weights(ds_tr.y_cls).to(device)
    ce = nn.CrossEntropyLoss(weight=class_w)
    huber = nn.HuberLoss(delta=1.0)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    best_rmse = 1e9

    for ep in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"epoch {ep}/{cfg.epochs}")
        total = 0.0
        n = 0

        for x, view_id, y, y_cls in pbar:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            view_id = view_id.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_cls = y_cls.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.amp):
                pred, reg, logits = model(x, view_id)
                loss_reg = huber(pred, y)
                loss_cls = ce(logits, y_cls)
                exp_score = (F.softmax(logits, dim=1) * torch.arange(10, device=device).view(1, -1)).sum(dim=1)
                loss_consistency = F.mse_loss(reg, exp_score)

                loss = (
                    cfg.loss_reg_w * loss_reg
                    + cfg.loss_cls_w * loss_cls
                    + cfg.loss_consistency_w * loss_consistency
                )

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            ema_model.update_parameters(model)

            total += float(loss.detach().cpu()) * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=total / n, lr=opt.param_groups[0]["lr"])

        sched.step()

        mae, rmse = evaluate(ema_model, dl_va, device)
        print(f"[VAL] ep={ep}  MAE={mae:.4f}  RMSE={rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            out_best = OUTDIR / "best_quality_evaluator.pt"
            save_ckpt(out_best, ema_model, cfg, ep, mae, rmse, best_rmse)
            print(f"[SAVE] {out_best} best_rmse={best_rmse:.4f}")

        if ep % cfg.save_every == 0:
            out_ep = OUTDIR / f"quality_evaluator_ep{ep:03d}.pt"
            save_ckpt(out_ep, ema_model, cfg, ep, mae, rmse, best_rmse)
            print(f"[SAVE] {out_ep}")

    print("[DONE]")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
