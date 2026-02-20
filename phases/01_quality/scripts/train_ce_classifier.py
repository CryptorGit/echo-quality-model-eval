from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as M
import torchvision.transforms as T

from sklearn.metrics import cohen_kappa_score, confusion_matrix, mean_absolute_error, mean_squared_error


_PIL_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)

ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = ROOT / "shared" / "runs" / "phase1" / "upperbound"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SPLIT_DIR = ROOT / "datasets" / "CACTUS" / "manifests_group_consensus_pad2_stratified_v3"


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


@dataclass
class Cfg:
    split_dir: str = str(DEFAULT_SPLIT_DIR)
    seed: int = 42
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 8
    epochs: int = 30
    lr: float = 3e-4
    wd: float = 1e-4
    amp: bool = True
    warmup_frac: float = 0.05
    patience: int = 5

    # light augs only (no RRC)
    affine_degrees: float = 3.0
    affine_translate: float = 0.02
    jitter_p: float = 0.25
    jitter_brightness: float = 0.06
    jitter_contrast: float = 0.06
    jitter_saturation: float = 0.03
    jitter_hue: float = 0.02

    # weighted CE
    use_weighted_ce: bool = True

    # view conditioning
    view_emb_dim: int = 16

    run_tag: str = "ce_resnet50"


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    return "cuda"


class Phase1ClsDataset(Dataset):
    def __init__(self, csv_path: Path, cfg: Cfg, train: bool, view_to_idx: Dict[str, int] | None = None):
        df = pd.read_csv(csv_path)
        for c in ["path", "view", "quality"]:
            if c not in df.columns:
                raise ValueError(f"Missing column {c} in {csv_path}")

        self.df = df.reset_index(drop=True)
        self.paths = self.df["path"].astype(str).tolist()
        self.views = self.df["view"].astype(str).tolist()
        self.y_float = self.df["quality"].astype(float).tolist()
        self.y_cls = [int(np.clip(round(v), 0, 9)) for v in self.y_float]

        if view_to_idx is None:
            uniq = sorted(set(self.views))
            self.view_to_idx = {v: i for i, v in enumerate(uniq)}
        else:
            self.view_to_idx = view_to_idx

        self.view_ids = [self.view_to_idx[v] for v in self.views]

        jitter = (cfg.jitter_brightness, cfg.jitter_contrast, cfg.jitter_saturation, cfg.jitter_hue)
        if train:
            parts: list = [
                KeepRatioPad(cfg.img_size),
                T.RandomAffine(degrees=float(cfg.affine_degrees), translate=(float(cfg.affine_translate), float(cfg.affine_translate))),
                T.RandomApply([T.ColorJitter(*jitter)], p=float(cfg.jitter_p)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        else:
            parts = [
                KeepRatioPad(cfg.img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        self.tf = T.Compose(parts)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        x = self.tf(img)
        return {
            "x": x,
            "view": self.views[i],
            "view_id": torch.tensor(self.view_ids[i], dtype=torch.long),
            "y": torch.tensor(self.y_cls[i], dtype=torch.long),
            "y_float": torch.tensor(self.y_float[i], dtype=torch.float32),
        }


class ResNet50CE(nn.Module):
    def __init__(self, n_views: int, view_emb_dim: int):
        super().__init__()
        enc = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2)
        in_f = enc.fc.in_features
        enc.fc = nn.Identity()
        self.encoder = enc

        self.view_emb = nn.Embedding(n_views, int(view_emb_dim))

        self.head = nn.Sequential(
            nn.Linear(in_f + int(view_emb_dim), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor, view_id: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        v = self.view_emb(view_id)
        z = torch.cat([h, v], dim=1)
        return self.head(z)


def qwk_from_int(y_int: np.ndarray, p_int: np.ndarray) -> float:
    return float(cohen_kappa_score(y_int, p_int, weights="quadratic"))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, amp: bool) -> dict:
    model.eval()
    ys, ps = [], []
    views: List[str] = []

    for b in loader:
        x = b["x"].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        vid = b["view_id"].to(device, non_blocking=True)
        y = b["y"].cpu().numpy()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(amp and device == "cuda")):
            logits = model(x, vid)
        pred = logits.argmax(dim=1).detach().cpu().numpy()

        ys.append(y)
        ps.append(pred)
        views.extend(b["view"])

    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)

    out = {
        "qwk": qwk_from_int(y_all, p_all),
        "mae": float(mean_absolute_error(y_all, p_all)),
        "rmse": float(mean_squared_error(y_all, p_all) ** 0.5),
        "confusion": confusion_matrix(y_all, p_all, labels=list(range(10))).tolist(),
        "per_view": {},
    }

    views_arr = np.array(views)
    for v in sorted(set(views)):
        idx = views_arr == v
        out["per_view"][v] = {
            "qwk": qwk_from_int(y_all[idx], p_all[idx]),
            "mae": float(mean_absolute_error(y_all[idx], p_all[idx])),
            "rmse": float(mean_squared_error(y_all[idx], p_all[idx]) ** 0.5),
        }

    return out


def class_weights(y: List[int]) -> torch.Tensor:
    c = np.bincount(np.array(y, dtype=np.int64), minlength=10).astype(np.float64)
    c[c == 0] = 1.0
    inv = 1.0 / c
    w = inv / inv.sum() * len(inv)
    return torch.tensor(w, dtype=torch.float32)


def lr_at(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if total_steps <= 0:
        return float(base_lr)
    if warmup_steps > 0 and step < warmup_steps:
        return float(base_lr) * float(step + 1) / float(warmup_steps)
    t = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return float(base_lr) * 0.5 * (1.0 + np.cos(np.pi * t))


def train(cfg: Cfg):
    device = get_device()
    set_seed(cfg.seed)

    split_dir = Path(cfg.split_dir)
    tr_csv = split_dir / "phase1_train.csv"
    va_csv = split_dir / "phase1_val.csv"
    te_csv = split_dir / "phase1_test.csv"

    out_dir = RUNS_DIR / split_dir.name / cfg.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    ds_tr = Phase1ClsDataset(tr_csv, cfg, train=True)
    ds_va = Phase1ClsDataset(va_csv, cfg, train=False, view_to_idx=ds_tr.view_to_idx)
    ds_te = Phase1ClsDataset(te_csv, cfg, train=False, view_to_idx=ds_tr.view_to_idx)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)
    dl_te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)

    model = ResNet50CE(n_views=len(ds_tr.view_to_idx), view_emb_dim=cfg.view_emb_dim).to(device).to(memory_format=torch.channels_last)

    if cfg.use_weighted_ce:
        w = class_weights(ds_tr.y_cls).to(device)
        ce = nn.CrossEntropyLoss(weight=w)
    else:
        ce = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    steps_per_epoch = len(dl_tr)
    total_steps = int(cfg.epochs * steps_per_epoch)
    warmup_steps = int(cfg.warmup_frac * total_steps)

    best_qwk = -1e9
    best_epoch = 0
    bad = 0
    history = []

    global_step = 0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"ce ep {ep}/{cfg.epochs}")
        total_loss = 0.0
        n = 0

        for b in pbar:
            lr = lr_at(global_step, total_steps, cfg.lr, warmup_steps)
            for g in opt.param_groups:
                g["lr"] = lr

            x = b["x"].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            vid = b["view_id"].to(device, non_blocking=True)
            y = b["y"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.amp):
                logits = model(x, vid)
                loss = ce(logits, y)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += float(loss.detach().cpu()) * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=total_loss / max(n, 1), lr=lr)
            global_step += 1

        m_val = evaluate(model, dl_va, device, amp=cfg.amp)
        history.append({"epoch": ep, "loss": total_loss / max(n, 1), **{f"val_{k}": v for k, v in m_val.items() if k != "per_view"}})
        print(f"[VAL] ep={ep} QWK={m_val['qwk']:.4f} RMSE={m_val['rmse']:.4f} MAE={m_val['mae']:.4f}")

        ckpt = {
            "model": model.state_dict(),
            "cfg": asdict(cfg),
            "view_to_idx": ds_tr.view_to_idx,
            "epoch": ep,
            "val": m_val,
        }

        if m_val["qwk"] > best_qwk:
            best_qwk = m_val["qwk"]
            best_epoch = ep
            bad = 0
            torch.save(ckpt, out_dir / "best.pt")
            print(f"[SAVE] best.pt (by QWK) ep={ep} qwk={best_qwk:.4f}")
        else:
            bad += 1

        if ep % 5 == 0:
            torch.save(ckpt, out_dir / f"ep{ep:03d}.pt")

        if cfg.patience > 0 and bad >= cfg.patience:
            print(f"[EARLY STOP] no improvement for {bad} epochs (best_ep={best_epoch} best_qwk={best_qwk:.4f})")
            break

    best = torch.load(out_dir / "best.pt", map_location="cpu")
    model.load_state_dict(best["model"], strict=True)

    m_val = evaluate(model, dl_va, device, amp=cfg.amp)
    m_test = evaluate(model, dl_te, device, amp=cfg.amp)

    report = {
        "mode": "upperbound_ce",
        "run_tag": cfg.run_tag,
        "split_dir": str(split_dir),
        "out_dir": str(out_dir),
        "best_epoch": int(best.get("epoch", -1)),
        "val": {k: v for k, v in m_val.items() if k != "per_view"},
        "test": {k: v for k, v in m_test.items() if k != "per_view"},
        "per_view_val": m_val["per_view"],
        "per_view_test": m_test["per_view"],
        "confusion_val": m_val["confusion"],
        "confusion_test": m_test["confusion"],
    }

    (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] report: {report_path}")
    print(f"[OK] best_ep={report['best_epoch']} val={report['val']} test={report['test']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", default=str(DEFAULT_SPLIT_DIR))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=5)

    p.add_argument("--affine_degrees", type=float, default=3.0)
    p.add_argument("--affine_translate", type=float, default=0.02)
    p.add_argument("--jitter_p", type=float, default=0.25)
    p.add_argument("--jitter_brightness", type=float, default=0.06)
    p.add_argument("--jitter_contrast", type=float, default=0.06)
    p.add_argument("--jitter_saturation", type=float, default=0.03)
    p.add_argument("--jitter_hue", type=float, default=0.02)

    p.add_argument("--use_weighted_ce", action="store_true")
    p.add_argument("--view_emb_dim", type=int, default=16)
    p.add_argument("--run_tag", default="ce_resnet50")

    args = p.parse_args()

    cfg = Cfg(
        split_dir=args.split_dir,
        seed=args.seed,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.wd,
        amp=bool(args.amp),
        warmup_frac=args.warmup_frac,
        patience=args.patience,
        affine_degrees=args.affine_degrees,
        affine_translate=args.affine_translate,
        jitter_p=args.jitter_p,
        jitter_brightness=args.jitter_brightness,
        jitter_contrast=args.jitter_contrast,
        jitter_saturation=args.jitter_saturation,
        jitter_hue=args.jitter_hue,
        use_weighted_ce=bool(args.use_weighted_ce),
        view_emb_dim=args.view_emb_dim,
        run_tag=args.run_tag,
    )

    train(cfg)


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
