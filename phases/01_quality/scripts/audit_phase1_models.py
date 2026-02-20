from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
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

from sklearn.metrics import cohen_kappa_score, confusion_matrix, mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[3]
MANIFEST_DIR = ROOT / "datasets" / "CACTUS" / "manifests"
TRAIN_CSV = MANIFEST_DIR / "cactus_train.csv"
VAL_CSV = MANIFEST_DIR / "cactus_val.csv"
TEST_CSV = MANIFEST_DIR / "cactus_test.csv"
FULL_CSV = MANIFEST_DIR / "cactus_manifest.csv"

BASE_RUN = ROOT / "shared" / "runs" / "phase1_quality"
SOTA_RUN = ROOT / "shared" / "runs" / "phase1_quality_sota"

OUT_DIR = ROOT / "shared" / "runs" / "phase1_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AuditCfg:
    seed: int = 42
    img_size_base: int = 224
    img_size_sota: int = 288
    batch_size: int = 96
    num_workers: int = 8
    group_train_epochs_base: int = 12
    group_train_epochs_sota: int = 12
    good_top_pct_per_view: float = 0.2
    stability_samples: int = 240


GROUP_PAT = re.compile(r"(\d+_[A-Z]\d+)")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    return "cuda"


def extract_group_key(path_str: str) -> str:
    name = Path(path_str).name
    m = GROUP_PAT.search(name)
    if m:
        return m.group(1)
    return "UNK_" + name.split("_")[0]


def add_group_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["group_key"] = out["path"].astype(str).map(extract_group_key)
    return out


def check_leakage() -> dict:
    tr = add_group_col(pd.read_csv(TRAIN_CSV))
    va = add_group_col(pd.read_csv(VAL_CSV))
    te = add_group_col(pd.read_csv(TEST_CSV))

    gtr, gva, gte = set(tr.group_key), set(va.group_key), set(te.group_key)
    leak_tr_va = sorted(gtr & gva)
    leak_tr_te = sorted(gtr & gte)
    leak_va_te = sorted(gva & gte)

    return {
        "n_groups_train": len(gtr),
        "n_groups_val": len(gva),
        "n_groups_test": len(gte),
        "overlap_train_val": len(leak_tr_va),
        "overlap_train_test": len(leak_tr_te),
        "overlap_val_test": len(leak_va_te),
        "examples_train_val": leak_tr_va[:15],
        "examples_train_test": leak_tr_te[:15],
        "examples_val_test": leak_va_te[:15],
    }


def make_group_split(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = add_group_col(pd.read_csv(FULL_CSV))

    rng = np.random.default_rng(seed)
    keys = np.array(sorted(df["group_key"].unique()))
    rng.shuffle(keys)

    n = len(keys)
    n_tr = int(n * 0.8)
    n_va = int(n * 0.1)

    tr_keys = set(keys[:n_tr])
    va_keys = set(keys[n_tr:n_tr + n_va])
    te_keys = set(keys[n_tr + n_va:])

    tr = df[df["group_key"].isin(tr_keys)].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    va = df[df["group_key"].isin(va_keys)].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    te = df[df["group_key"].isin(te_keys)].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return tr, va, te


class DatasetBase(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int, train: bool, view_to_idx: Dict[str, int] | None = None):
        self.df = df.reset_index(drop=True)
        self.paths = self.df["path"].astype(str).tolist()
        self.views = self.df["view"].astype(str).tolist()
        self.y = self.df["quality"].astype(float).tolist()
        self.y_cls = [int(np.clip(round(v), 0, 9)) for v in self.y]

        if view_to_idx is None:
            uniq = sorted(set(self.views))
            self.view_to_idx = {v: i for i, v in enumerate(uniq)}
        else:
            self.view_to_idx = view_to_idx
        self.view_ids = [self.view_to_idx[v] for v in self.views]

        if train:
            self.tf = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(0.1, 0.1, 0.05, 0.03)], p=0.4),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        x = self.tf(img)
        return (
            x,
            torch.tensor(self.view_ids[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.float32),
            torch.tensor(self.y_cls[i], dtype=torch.long),
            self.views[i],
        )


class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        m = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
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


class SotaModel(nn.Module):
    def __init__(self, n_views: int):
        super().__init__()
        backbone = M.efficientnet_v2_s(weights=M.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_f = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.view_emb = nn.Embedding(n_views, 16)
        self.trunk = nn.Sequential(
            nn.Linear(in_f + 16, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.20),
        )
        self.head_reg = nn.Linear(256, 1)
        self.head_cls = nn.Linear(256, 10)

    def forward(self, x, view_id):
        feat = self.backbone(x)
        v = self.view_emb(view_id)
        h = self.trunk(torch.cat([feat, v], dim=1))
        reg = self.head_reg(h).squeeze(1)
        logits = self.head_cls(h)
        prob = F.softmax(logits, dim=1)
        bins = torch.arange(10, device=logits.device, dtype=logits.dtype).unsqueeze(0)
        exp_score = (prob * bins).sum(dim=1)
        pred = 0.7 * exp_score + 0.3 * reg
        return pred, reg, logits, h


@torch.no_grad()
def predict_baseline(model: BaselineModel, loader: DataLoader, device: str):
    model.eval()
    ys, ps, views = [], [], []
    for x, _, y, _, view in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            p, _ = model(x)
        ys.append(y.float().cpu())
        ps.append(p.float().cpu())
        views.extend(view)
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    return y, p, views


@torch.no_grad()
def predict_sota(model: SotaModel, loader: DataLoader, device: str):
    model.eval()
    ys, ps, views = [], [], []
    for x, view_id, y, _, view in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        view_id = view_id.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            p, _, _, _ = model(x, view_id)
        ys.append(y.float().cpu())
        ps.append(p.float().cpu())
        views.extend(view)
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    return y, p, views


def calc_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    rmse = float(mean_squared_error(y, p) ** 0.5)
    mae = float(mean_absolute_error(y, p))
    y_round = np.clip(np.rint(y), 0, 9).astype(int)
    p_round = np.clip(np.rint(p), 0, 9).astype(int)
    qwk = float(cohen_kappa_score(y_round, p_round, weights="quadratic"))
    return {"rmse": rmse, "mae": mae, "qwk": qwk}


def load_state_dict_flexible(model: nn.Module, state: dict):
    state = {k: v for k, v in state.items() if k != "n_averaged"}

    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError:
        pass

    if any(k.startswith("module.") for k in state.keys()):
        stripped = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(stripped, strict=True)
        return

    prefixed = {"module." + k: v for k, v in state.items()}
    model.load_state_dict(prefixed, strict=True)


def random_confusion(views: List[str], p: np.ndarray) -> dict:
    true_random = np.array([1 if v == "Random" else 0 for v in views], dtype=int)
    pred_random = (np.clip(np.rint(p), 0, 9).astype(int) == 0).astype(int)
    cm = confusion_matrix(true_random, pred_random, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "confusion_matrix_rows_true_cols_pred": cm.tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision_random": prec,
        "recall_random": rec,
    }


def train_group_baseline(df_tr: pd.DataFrame, df_va: pd.DataFrame, cfg: AuditCfg, device: str) -> dict:
    ds_tr = DatasetBase(df_tr, cfg.img_size_base, train=True)
    ds_va = DatasetBase(df_va, cfg.img_size_base, train=False, view_to_idx=ds_tr.view_to_idx)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)

    model = BaselineModel().to(device).to(memory_format=torch.channels_last)
    crit = nn.HuberLoss(delta=1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    best = {"rmse": 1e9, "mae": None, "qwk": None, "epoch": -1}

    for ep in range(1, cfg.group_train_epochs_base + 1):
        model.train()
        for x, _, y, _, _ in dl_tr:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                p, _ = model(x)
                loss = crit(p, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        yv, pv, _ = predict_baseline(model, dl_va, device)
        m = calc_metrics(yv, pv)
        if m["rmse"] < best["rmse"]:
            best = {**m, "epoch": ep}

    return best


def train_group_sota(df_tr: pd.DataFrame, df_va: pd.DataFrame, cfg: AuditCfg, device: str) -> dict:
    ds_tr = DatasetBase(df_tr, cfg.img_size_sota, train=True)
    ds_va = DatasetBase(df_va, cfg.img_size_sota, train=False, view_to_idx=ds_tr.view_to_idx)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)

    model = SotaModel(n_views=len(ds_tr.view_to_idx)).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    counts = np.bincount(np.array(ds_tr.y_cls), minlength=10).astype(np.float64)
    counts[counts == 0] = 1
    inv = 1.0 / counts
    w = torch.tensor(inv / inv.sum() * len(inv), dtype=torch.float32, device=device)

    ce = nn.CrossEntropyLoss(weight=w)
    huber = nn.HuberLoss(delta=1.0)

    best = {"rmse": 1e9, "mae": None, "qwk": None, "epoch": -1}

    for ep in range(1, cfg.group_train_epochs_sota + 1):
        model.train()
        for x, view_id, y, y_cls, _ in dl_tr:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            view_id = view_id.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_cls = y_cls.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                pred, reg, logits, _ = model(x, view_id)
                exp_score = (F.softmax(logits, dim=1) * torch.arange(10, device=device).view(1, -1)).sum(dim=1)
                loss = huber(pred, y) + 0.7 * ce(logits, y_cls) + 0.15 * F.mse_loss(reg, exp_score)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        yv, pv, _ = predict_sota(model, dl_va, device)
        m = calc_metrics(yv, pv)
        if m["rmse"] < best["rmse"]:
            best = {**m, "epoch": ep}

    return best


@torch.no_grad()
def embed_baseline(model: BaselineModel, x: torch.Tensor) -> torch.Tensor:
    _, z = model(x)
    return z


@torch.no_grad()
def embed_sota(model: SotaModel, x: torch.Tensor, view_id: torch.Tensor) -> torch.Tensor:
    _, _, _, h = model(x, view_id)
    return h


def make_perturb(img: Image.Image, mode: str) -> Image.Image:
    if mode == "crop":
        w, h = img.size
        c = TF.center_crop(img, [int(h * 0.9), int(w * 0.9)])
        return c.resize((w, h), Image.BILINEAR)
    if mode == "translate":
        return TF.affine(img, angle=0.0, translate=(8, -6), scale=1.0, shear=[0.0, 0.0])
    if mode == "rotate":
        return TF.rotate(img, angle=5.0, interpolation=TF.InterpolationMode.BILINEAR)
    return img


def select_good_indices(df: pd.DataFrame, per_view_top_pct: float) -> Tuple[np.ndarray, np.ndarray]:
    q = df["quality"].astype(float).values
    global_good = np.where(q >= 8.0)[0]

    per_view_good: List[int] = []
    for view, g in df.groupby("view"):
        n = len(g)
        k = max(1, int(math.ceil(n * per_view_top_pct)))
        idx = g.sort_values("quality", ascending=False).head(k).index.tolist()
        per_view_good.extend(idx)
    per_view_good = np.array(sorted(set(per_view_good)), dtype=int)

    return global_good, per_view_good


def direction_stability_for_model(
    model_name: str,
    df: pd.DataFrame,
    view_to_idx: Dict[str, int],
    device: str,
    cfg: AuditCfg,
):
    tf_base = T.Compose([
        T.Resize((cfg.img_size_base if model_name == "baseline" else cfg.img_size_sota,
                  cfg.img_size_base if model_name == "baseline" else cfg.img_size_sota)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if model_name == "baseline":
        model = BaselineModel().to(device).to(memory_format=torch.channels_last)
        ckpt = torch.load(BASE_RUN / "best_regressor.pt", map_location="cpu")
        load_state_dict_flexible(model, ckpt["model"])
        embed_fn = lambda x, vid: embed_baseline(model, x)
    else:
        model = SotaModel(n_views=len(view_to_idx)).to(device).to(memory_format=torch.channels_last)
        ckpt = torch.load(SOTA_RUN / "best_quality_evaluator.pt", map_location="cpu")
        load_state_dict_flexible(model, ckpt["model"])
        embed_fn = lambda x, vid: embed_sota(model, x, vid)

    model.eval()

    global_good, per_view_good = select_good_indices(df, cfg.good_top_pct_per_view)

    rng = np.random.default_rng(cfg.seed)
    all_idx = np.arange(len(df))
    rng.shuffle(all_idx)
    sample_idx = all_idx[: min(cfg.stability_samples, len(all_idx))]

    # precompute clean embeddings for all val samples for building G
    emb_all = []
    for i in range(len(df)):
        p = df.iloc[i]["path"]
        v = df.iloc[i]["view"]
        img = Image.open(p).convert("RGB")
        x = tf_base(img).unsqueeze(0).to(device).contiguous(memory_format=torch.channels_last)
        vid = torch.tensor([view_to_idx[v]], device=device)
        z = embed_fn(x, vid).float().squeeze(0)
        emb_all.append(z)
    emb_all = torch.stack(emb_all, dim=0)

    def score_for_good_set(good_idx: np.ndarray) -> dict:
        G = emb_all[torch.tensor(good_idx, device=device)]

        cos_by_mode = {"crop": [], "translate": [], "rotate": []}

        for i in sample_idx:
            row = df.iloc[i]
            p = row["path"]
            v = row["view"]
            img = Image.open(p).convert("RGB")

            x0 = tf_base(img).unsqueeze(0).to(device).contiguous(memory_format=torch.channels_last)
            vid0 = torch.tensor([view_to_idx[v]], device=device)
            z0 = embed_fn(x0, vid0).float().squeeze(0)

            d0_all = G - z0.unsqueeze(0)
            d0_norm = torch.norm(d0_all, dim=1)
            nn0 = torch.argmin(d0_norm)
            d0 = d0_all[nn0]
            d0 = d0 / (torch.norm(d0) + 1e-8)

            for mode in ["crop", "translate", "rotate"]:
                imgp = make_perturb(img, mode)
                xp = tf_base(imgp).unsqueeze(0).to(device).contiguous(memory_format=torch.channels_last)
                zp = embed_fn(xp, vid0).float().squeeze(0)

                dp_all = G - zp.unsqueeze(0)
                dp_norm = torch.norm(dp_all, dim=1)
                nnp = torch.argmin(dp_norm)
                dp = dp_all[nnp]
                dp = dp / (torch.norm(dp) + 1e-8)

                cos = torch.sum(d0 * dp).item()
                cos_by_mode[mode].append(cos)

        out = {}
        for mode, vals in cos_by_mode.items():
            a = np.array(vals, dtype=np.float64)
            out[mode] = {
                "mean_cos": float(a.mean()),
                "std_cos": float(a.std()),
                "pct_cos_lt_0.5": float((a < 0.5).mean()),
                "pct_cos_lt_0": float((a < 0.0).mean()),
            }
        return out

    return {
        "global_quality_ge_8": score_for_good_set(global_good),
        f"per_view_top_{int(cfg.good_top_pct_per_view * 100)}pct": score_for_good_set(per_view_good),
        "n_samples": int(len(sample_idx)),
        "n_global_good": int(len(global_good)),
        "n_per_view_good": int(len(per_view_good)),
    }


def main():
    cfg = AuditCfg()
    set_seed(cfg.seed)
    device = get_device()

    leak = check_leakage()

    # existing split performance for both trained models
    df_val = pd.read_csv(VAL_CSV)
    ds_val_base = DatasetBase(df_val, cfg.img_size_base, train=False)
    ds_val_sota = DatasetBase(df_val, cfg.img_size_sota, train=False, view_to_idx=ds_val_base.view_to_idx)

    dl_val_base = DataLoader(ds_val_base, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)
    dl_val_sota = DataLoader(ds_val_sota, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)

    base_model = BaselineModel().to(device).to(memory_format=torch.channels_last)
    load_state_dict_flexible(base_model, torch.load(BASE_RUN / "best_regressor.pt", map_location="cpu")["model"])
    yb, pb, vb = predict_baseline(base_model, dl_val_base, device)
    base_existing = calc_metrics(yb, pb)

    sota_model = SotaModel(n_views=len(ds_val_base.view_to_idx)).to(device).to(memory_format=torch.channels_last)
    load_state_dict_flexible(sota_model, torch.load(SOTA_RUN / "best_quality_evaluator.pt", map_location="cpu")["model"])
    ys, ps, vs = predict_sota(sota_model, dl_val_sota, device)
    sota_existing = calc_metrics(ys, ps)

    base_random = random_confusion(vb, pb)
    sota_random = random_confusion(vs, ps)

    # group split retrain
    gtr, gva, gte = make_group_split(cfg.seed)
    group_info = {
        "train_rows": int(len(gtr)),
        "val_rows": int(len(gva)),
        "test_rows": int(len(gte)),
        "train_groups": int(gtr.group_key.nunique()),
        "val_groups": int(gva.group_key.nunique()),
        "test_groups": int(gte.group_key.nunique()),
        "overlap_train_val": int(len(set(gtr.group_key) & set(gva.group_key))),
        "overlap_train_test": int(len(set(gtr.group_key) & set(gte.group_key))),
        "overlap_val_test": int(len(set(gva.group_key) & set(gte.group_key))),
    }

    base_group = train_group_baseline(gtr, gva, cfg, device)
    sota_group = train_group_sota(gtr, gva, cfg, device)

    # direction stability on trained best models
    # use shared view mapping from val
    view_to_idx = ds_val_base.view_to_idx
    stab_base = direction_stability_for_model("baseline", df_val, view_to_idx, device, cfg)
    stab_sota = direction_stability_for_model("sota", df_val, view_to_idx, device, cfg)

    report = {
        "leakage_check_original_split": leak,
        "group_split_info": group_info,
        "existing_split_metrics": {
            "baseline": base_existing,
            "sota": sota_existing,
        },
        "group_split_metrics_retrained": {
            "baseline": base_group,
            "sota": sota_group,
        },
        "drop_existing_to_group": {
            "baseline_rmse_delta": float(base_group["rmse"] - base_existing["rmse"]),
            "baseline_qwk_delta": float(base_group["qwk"] - base_existing["qwk"]),
            "sota_rmse_delta": float(sota_group["rmse"] - sota_existing["rmse"]),
            "sota_qwk_delta": float(sota_group["qwk"] - sota_existing["qwk"]),
        },
        "random_detector_confusion": {
            "baseline": base_random,
            "sota": sota_random,
        },
        "good_set_definition_candidates": {
            "global_quality_ge_8": "quality>=8 を G とする。ビュー偏りが入る可能性あり。",
            f"per_view_top_{int(cfg.good_top_pct_per_view * 100)}pct": "viewごとに上位x%を G とする。ビュー間バランスが良く安定しやすい。",
        },
        "direction_stability": {
            "baseline": stab_base,
            "sota": stab_sota,
        },
    }

    out_json = OUT_DIR / "phase1_audit_report.json"
    out_md = OUT_DIR / "phase1_audit_report.md"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md = []
    md.append("# Phase1 Audit Report\n")
    md.append("## 1) 分割リーク\n")
    md.append(f"- train-val overlap groups: {leak['overlap_train_val']}\n")
    md.append(f"- train-test overlap groups: {leak['overlap_train_test']}\n")
    md.append(f"- val-test overlap groups: {leak['overlap_val_test']}\n")

    md.append("\n## 2) Group Splitでの性能\n")
    md.append(f"- Baseline existing RMSE/QWK: {base_existing['rmse']:.4f} / {base_existing['qwk']:.4f}\n")
    md.append(f"- Baseline group RMSE/QWK: {base_group['rmse']:.4f} / {base_group['qwk']:.4f}\n")
    md.append(f"- SOTA existing RMSE/QWK: {sota_existing['rmse']:.4f} / {sota_existing['qwk']:.4f}\n")
    md.append(f"- SOTA group RMSE/QWK: {sota_group['rmse']:.4f} / {sota_group['qwk']:.4f}\n")

    md.append("\n## 3) Random検出器チェック\n")
    for k, v in {"baseline": base_random, "sota": sota_random}.items():
        md.append(
            f"- {k}: CM={v['confusion_matrix_rows_true_cols_pred']}, "
            f"precision={v['precision_random']:.3f}, recall={v['recall_random']:.3f}\n"
        )

    md.append("\n## 4) G定義と方向安定性\n")
    md.append("- 推奨: viewごと上位20%をG（ビュー偏り低減）\n")
    for model_name, st in {"baseline": stab_base, "sota": stab_sota}.items():
        md.append(f"\n### {model_name}\n")
        for gname in ["global_quality_ge_8", f"per_view_top_{int(cfg.good_top_pct_per_view * 100)}pct"]:
            md.append(f"- {gname}\n")
            for mode, met in st[gname].items():
                md.append(
                    f"  - {mode}: mean_cos={met['mean_cos']:.4f}, "
                    f"pct<0.5={met['pct_cos_lt_0.5']:.3f}, pct<0={met['pct_cos_lt_0']:.3f}\n"
                )

    out_md.write_text("".join(md), encoding="utf-8")

    print(f"[OK] {out_json}")
    print(f"[OK] {out_md}")


if __name__ == "__main__":
    main()
