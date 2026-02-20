from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.models as M
import torchvision.transforms as T

from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error


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
RUNS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Cfg:
    mode: str = "multihead"  # multihead | perview
    view: str = ""           # required when mode=perview
    views: str = "A4C,PL,PSAV,PSMV,SC"
    seed: int = 42
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 8
    epochs: int = 20
    lr: float = 3e-4
    wd: float = 1e-4
    amp: bool = True
    lambda_con: float = 0.2
    save_every: int = 5
    split_dir: str = str(DEFAULT_SPLIT_DIR)
    supcon_delta_q: float = 1.0
    ord_view_weights: str = ""
    run_tag: str = ""
    sampler: str = "shuffle"  # shuffle | group | group_view | group_view_grade | view_band
    sampler_clip_min: float = 0.2
    sampler_clip_max: float = 5.0
    view_head_profile: str = "default"  # default | heavy_for_weak_views
    weak_views: str = "A4C,SC,PSAV"
    exclude_random: bool = True

    # Loss reweighting (view x band)
    use_loss_reweight: bool = False
    band_weight_scheme: str = "invfreq_clip"  # invfreq_clip | none
    band_weight_clip_min: float = 0.5
    band_weight_clip_max: float = 2.0
    view_weight_a4c: float = 1.0
    view_weight_pl: float = 1.0
    view_weight_psav: float = 1.0
    view_weight_psmv: float = 1.0
    view_weight_sc: float = 1.0

    # Augmentations (P1): keep geometry stable by default
    use_rrc: bool = False
    rrc_scale_min: float = 0.85
    rrc_scale_max: float = 1.0
    affine_degrees: float = 3.0
    affine_translate: float = 0.02
    jitter_p: float = 0.25
    jitter_brightness: float = 0.06
    jitter_contrast: float = 0.06
    jitter_saturation: float = 0.03
    jitter_hue: float = 0.02


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


def ordinal_targets(y: torch.Tensor) -> torch.Tensor:
    bins = torch.arange(1, 10, device=y.device, dtype=y.dtype).unsqueeze(0)
    return (y.unsqueeze(1) >= bins).float()


def expected_score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits).sum(dim=1)


def quality_to_band(q: torch.Tensor) -> torch.Tensor:
    q_i = torch.clamp(torch.round(q), 0, 9).long()
    out = torch.full_like(q_i, 2)
    out = torch.where(q_i <= 2, torch.zeros_like(out), out)
    out = torch.where((q_i >= 3) & (q_i <= 5), torch.ones_like(out), out)
    return out


def supcon_loss(features: torch.Tensor, pos_mask: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    # features: [N, d], pos_mask: [N, N] bool (diag False)
    feat = F.normalize(features, dim=1)
    sim = torch.mm(feat, feat.t()) / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    exp_sim = torch.exp(sim)
    eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    exp_sim = exp_sim.masked_fill(eye, 0.0)

    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
    pos_mask_f = pos_mask.float()

    pos_count = pos_mask_f.sum(dim=1)
    valid = pos_count > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=features.device)

    mean_log_prob_pos = (pos_mask_f * log_prob).sum(dim=1) / (pos_count + 1e-12)
    loss = -mean_log_prob_pos[valid].mean()
    return loss


class Phase1Dataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        img_size: int,
        train: bool,
        view_to_idx: Dict[str, int] | None = None,
        *,
        use_rrc: bool = False,
        rrc_scale: tuple[float, float] = (0.85, 1.0),
        affine_degrees: float = 3.0,
        affine_translate: float = 0.02,
        jitter_p: float = 0.25,
        jitter: tuple[float, float, float, float] = (0.06, 0.06, 0.03, 0.02),
    ):
        df = pd.read_csv(csv_path)
        for c in ["path", "view", "quality", "group"]:
            if c not in df.columns:
                raise ValueError(f"Missing column {c} in {csv_path}")

        self.df = df.reset_index(drop=True)
        self.paths = self.df["path"].astype(str).tolist()
        self.views = self.df["view"].astype(str).tolist()
        self.groups = self.df["group"].astype(str).tolist()
        self.quality = self.df["quality"].astype(float).tolist()
        self.sample_ids = [
            int(hashlib.sha1(p.encode("utf-8")).hexdigest()[:16], 16) % (2**63 - 1)
            for p in self.paths
        ]

        if view_to_idx is None:
            uniq = sorted(set(self.views))
            self.view_to_idx = {v: i for i, v in enumerate(uniq)}
        else:
            self.view_to_idx = view_to_idx

        self.view_ids = [self.view_to_idx[v] for v in self.views]

        tf_train_parts: list = [KeepRatioPad(img_size)]
        if use_rrc:
            tf_train_parts.append(
                T.RandomResizedCrop(img_size, scale=(float(rrc_scale[0]), float(rrc_scale[1])), ratio=(1.0, 1.0))
            )
        if float(affine_degrees) > 0.0 or float(affine_translate) > 0.0:
            tf_train_parts.append(
                T.RandomAffine(degrees=float(affine_degrees), translate=(float(affine_translate), float(affine_translate)))
            )
        if float(jitter_p) > 0.0:
            tf_train_parts.append(T.RandomApply([T.ColorJitter(*jitter)], p=float(jitter_p)))
        tf_train_parts.extend(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.tf_train = T.Compose(tf_train_parts)
        self.tf_eval = T.Compose([
            KeepRatioPad(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.train:
            x1 = self.tf_train(img)
            x2 = self.tf_train(img)
        else:
            x1 = self.tf_eval(img)
            x2 = x1

        return {
            "x1": x1,
            "x2": x2,
            "view_id": torch.tensor(self.view_ids[i], dtype=torch.long),
            "view": self.views[i],
            "group": self.groups[i],
            "y": torch.tensor(self.quality[i], dtype=torch.float32),
            "sample_id": torch.tensor(self.sample_ids[i], dtype=torch.long),
            "path": self.paths[i],
        }


class MultiheadModel(nn.Module):
    def __init__(self, views: List[str], view_head_profile: str = "default", weak_views: List[str] | None = None):
        super().__init__()
        self.views = views
        self.id_to_view = {i: v for i, v in enumerate(views)}
        self.view_head_profile = str(view_head_profile)
        weak_view_set = set(weak_views or [])
        self.encoder = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2)
        in_f = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        self.proj_heads = nn.ModuleDict({
            v: nn.Sequential(
                nn.Linear(in_f, 512), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(512, 256)
            )
            for v in views
        })
        ord_heads: Dict[str, nn.Module] = {}
        for v in views:
            if self.view_head_profile == "heavy_for_weak_views" and v in weak_view_set:
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

    def forward(self, x: torch.Tensor, view_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = torch.zeros((h.size(0), 256), device=h.device, dtype=h.dtype)
        o = torch.zeros((h.size(0), 9), device=h.device, dtype=h.dtype)

        for vid in view_ids.unique().tolist():
            idx = torch.nonzero(view_ids == vid, as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            v = self.id_to_view[int(vid)]
            hv = h.index_select(0, idx)
            zv = F.normalize(self.proj_heads[v](hv), dim=1)
            ov = self.ord_heads[v](zv)
            zv = zv.to(dtype=z.dtype)
            ov = ov.to(dtype=o.dtype)
            z.index_copy_(0, idx, zv)
            o.index_copy_(0, idx, ov)

        y_hat = expected_score_from_logits(o)
        return z, o, y_hat


class PerViewModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2)
        in_f = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.proj = nn.Sequential(
            nn.Linear(in_f, 512), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(512, 256)
        )
        self.ord = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 9))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = F.normalize(self.proj(h), dim=1)
        o = self.ord(z)
        y_hat = expected_score_from_logits(o)
        return z, o, y_hat


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, mode: str, amp: bool) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    for b in loader:
        x = b["x1"].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        y = b["y"].to(device, non_blocking=True)

        view_ids = b["view_id"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(amp and device == "cuda")):
            if mode == "multihead":
                _, _, y_hat = model(x, view_ids)
            else:
                _, _, y_hat = model(x)

        ys.append(y.float().cpu())
        ps.append(y_hat.float().cpu())

    y_all = torch.cat(ys).numpy()
    p_all = torch.cat(ps).numpy()
    mae = float(mean_absolute_error(y_all, p_all))
    rmse = float(mean_squared_error(y_all, p_all) ** 0.5)
    y_r = np.clip(y_all, 0, 9).astype(int)
    p_r = np.clip(np.rint(p_all), 0, 9).astype(int)
    qwk = float(cohen_kappa_score(y_r, p_r, weights="quadratic"))
    return {"mae": mae, "rmse": rmse, "qwk": qwk}


def build_supcon_mask(view_ids: torch.Tensor, y: torch.Tensor, sample_ids: torch.Tensor, max_delta_q: float) -> torch.Tensor:
    # base B samples -> duplicate as [x1;x2] => N=2B
    B = view_ids.size(0)
    view2 = torch.cat([view_ids, view_ids], dim=0)
    y2 = torch.cat([y, y], dim=0)
    sid2 = torch.cat([sample_ids, sample_ids], dim=0)

    same_view = view2.unsqueeze(0) == view2.unsqueeze(1)
    close_q = (y2.unsqueeze(0) - y2.unsqueeze(1)).abs() <= float(max_delta_q)
    same_sample = sid2.unsqueeze(0) == sid2.unsqueeze(1)

    pos = (same_view & close_q) | same_sample
    eye = torch.eye(2 * B, device=view_ids.device, dtype=torch.bool)
    pos = pos & (~eye)
    return pos


def train(cfg: Cfg):
    device = get_device()
    set_seed(cfg.seed)

    split_dir = Path(cfg.split_dir)
    train_csv = split_dir / "phase1_train.csv"
    val_csv = split_dir / "phase1_val.csv"
    test_csv = split_dir / "phase1_test.csv"

    if not train_csv.exists() or not val_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Missing split CSVs in: {split_dir}")

    views = [v.strip() for v in cfg.views.split(",") if v.strip()]
    mode = cfg.mode

    tr_df = pd.read_csv(train_csv)
    va_df = pd.read_csv(val_csv)
    te_df = pd.read_csv(test_csv)

    if mode == "perview":
        if not cfg.view:
            raise ValueError("--view is required for perview mode")
        tr_df = tr_df[tr_df["view"] == cfg.view].reset_index(drop=True)
        va_df = va_df[va_df["view"] == cfg.view].reset_index(drop=True)
        te_df = te_df[te_df["view"] == cfg.view].reset_index(drop=True)
        if len(tr_df) == 0:
            raise ValueError(f"No rows for view={cfg.view}")
        views = [cfg.view]
    else:
        allowed_views = set(views)
        if cfg.exclude_random:
            allowed_views.discard("Random")
        tr_df = tr_df[tr_df["view"].astype(str).isin(allowed_views)].reset_index(drop=True)
        va_df = va_df[va_df["view"].astype(str).isin(allowed_views)].reset_index(drop=True)
        te_df = te_df[te_df["view"].astype(str).isin(allowed_views)].reset_index(drop=True)
        present_views = sorted(set(tr_df["view"].astype(str)))
        views = [v for v in views if v in present_views]
        if not views:
            raise ValueError("No training rows after view filtering")

    run_root = RUNS_DIR / split_dir.name
    sub_dir = "multihead" if mode == "multihead" else f"perview/{cfg.view}"
    if cfg.run_tag:
        sub_dir = f"{sub_dir}_{cfg.run_tag}"
    out_dir = run_root / sub_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_tmp = out_dir / "_train_tmp.csv"
    va_tmp = out_dir / "_val_tmp.csv"
    te_tmp = out_dir / "_test_tmp.csv"
    tr_df.to_csv(tr_tmp, index=False)
    va_df.to_csv(va_tmp, index=False)
    te_df.to_csv(te_tmp, index=False)

    ds_tr = Phase1Dataset(
        tr_tmp,
        cfg.img_size,
        train=True,
        use_rrc=cfg.use_rrc,
        rrc_scale=(cfg.rrc_scale_min, cfg.rrc_scale_max),
        affine_degrees=cfg.affine_degrees,
        affine_translate=cfg.affine_translate,
        jitter_p=cfg.jitter_p,
        jitter=(cfg.jitter_brightness, cfg.jitter_contrast, cfg.jitter_saturation, cfg.jitter_hue),
    )
    ds_va = Phase1Dataset(va_tmp, cfg.img_size, train=False, view_to_idx=ds_tr.view_to_idx)
    ds_te = Phase1Dataset(te_tmp, cfg.img_size, train=False, view_to_idx=ds_tr.view_to_idx)

    sampler = None
    shuffle = True
    if cfg.sampler != "shuffle":
        # Compute sample weights from the *training* dataframe.
        gcount = tr_df.groupby("group").size().to_dict()
        vcount = tr_df.groupby("view").size().to_dict()
        q_int = tr_df["quality"].round().clip(0, 9).astype(int)
        vgcount = tr_df.assign(q_int=q_int).groupby(["view", "q_int"]).size().to_dict()

        vbcount = {}
        if cfg.sampler == "view_band":
            q_band = pd.cut(q_int, bins=[-1, 3, 6, 9], labels=["low", "mid", "high"]).astype(str)
            vbcount = tr_df.assign(q_band=q_band).groupby(["view", "q_band"]).size().to_dict()

        w = np.ones((len(tr_df),), dtype=np.float64)
        if cfg.sampler in {"group", "group_view", "group_view_grade"}:
            w *= 1.0 / tr_df["group"].astype(str).map(lambda g: float(gcount[str(g)])).to_numpy()
        if cfg.sampler in {"group_view", "group_view_grade"}:
            w *= 1.0 / tr_df["view"].astype(str).map(lambda v: float(vcount[str(v)])).to_numpy()
        if cfg.sampler == "group_view_grade":
            w *= 1.0 / np.array(
                [float(vgcount[(str(v), int(q))]) for v, q in zip(tr_df["view"].astype(str), q_int)],
                dtype=np.float64,
            )
        if cfg.sampler == "view_band":
            q_band = pd.cut(q_int, bins=[-1, 3, 6, 9], labels=["low", "mid", "high"]).astype(str)
            w *= 1.0 / np.array(
                [float(vbcount[(str(v), str(b))]) for v, b in zip(tr_df["view"].astype(str), q_band)],
                dtype=np.float64,
            )

        # Normalize for numerical stability.
        w = w / (w.mean() + 1e-12)
        if cfg.sampler_clip_min > 0 and cfg.sampler_clip_max > 0 and cfg.sampler_clip_max >= cfg.sampler_clip_min:
            w = np.clip(w, float(cfg.sampler_clip_min), float(cfg.sampler_clip_max))
            w = w / (w.mean() + 1e-12)
        weights = torch.as_tensor(w, dtype=torch.double)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False

    dl_tr = DataLoader(
        ds_tr,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)
    dl_te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)

    if mode == "multihead":
        weak_views = [v.strip() for v in cfg.weak_views.split(",") if v.strip()]
        model = MultiheadModel(views=views, view_head_profile=cfg.view_head_profile, weak_views=weak_views).to(device)
    else:
        model = PerViewModel().to(device)

    model = model.to(memory_format=torch.channels_last)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    bce = nn.BCEWithLogitsLoss(reduction="none")

    view_weight_map: Dict[str, float] = {}
    if cfg.ord_view_weights:
        for token in cfg.ord_view_weights.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(f"Invalid --ord_view_weights token: {token}")
            k, v = token.split(":", 1)
            view_weight_map[k.strip()] = float(v.strip())

    view_weight_map_cfg: Dict[str, float] = {
        "A4C": float(cfg.view_weight_a4c),
        "PL": float(cfg.view_weight_pl),
        "PSAV": float(cfg.view_weight_psav),
        "PSMV": float(cfg.view_weight_psmv),
        "SC": float(cfg.view_weight_sc),
    }

    band_weight_map: Dict[int, float] = {0: 1.0, 1: 1.0, 2: 1.0}
    if cfg.use_loss_reweight and cfg.band_weight_scheme == "invfreq_clip":
        q_int = tr_df["quality"].round().clip(0, 9).astype(int)
        b_ser = pd.cut(q_int, bins=[-1, 2, 5, 9], labels=[0, 1, 2]).astype(int)
        b_counts = b_ser.value_counts().to_dict()
        raw = {
            0: 1.0 / float(b_counts.get(0, 1)),
            1: 1.0 / float(b_counts.get(1, 1)),
            2: 1.0 / float(b_counts.get(2, 1)),
        }
        mean_raw = (raw[0] + raw[1] + raw[2]) / 3.0
        for k in [0, 1, 2]:
            w_k = raw[k] / (mean_raw + 1e-12)
            w_k = float(np.clip(w_k, float(cfg.band_weight_clip_min), float(cfg.band_weight_clip_max)))
            band_weight_map[k] = w_k

    # P2: pick best checkpoint by validation QWK (still keep best RMSE for comparison).
    best_qwk = -1e9
    best_rmse = 1e9
    best_qwk_rec: Dict[str, float] | None = None
    best_rmse_rec: Dict[str, float] | None = None
    history = []

    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    for ep in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"{mode} ep {ep}/{cfg.epochs}")
        total = 0.0
        n = 0

        for b in pbar:
            x1 = b["x1"].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            x2 = b["x2"].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = b["y"].to(device, non_blocking=True)
            sample_ids = b["sample_id"].to(device, non_blocking=True)
            view_ids = b["view_id"].to(device, non_blocking=True)

            x = torch.cat([x1, x2], dim=0)
            y2 = torch.cat([y, y], dim=0)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.amp):
                if mode == "multihead":
                    view_ids2 = torch.cat([view_ids, view_ids], dim=0)
                    z, o, y_hat = model(x, view_ids2)
                else:
                    z, o, y_hat = model(x)

                t = ordinal_targets(y2)
                l_ord_raw = bce(o, t).mean(dim=1)

                if mode == "multihead" and (view_weight_map or cfg.use_loss_reweight):
                    w_base = torch.ones_like(y)
                    for view_name, weight in view_weight_map.items():
                        if view_name not in ds_tr.view_to_idx:
                            continue
                        vid = ds_tr.view_to_idx[view_name]
                        w_base = torch.where(view_ids == vid, torch.full_like(w_base, float(weight)), w_base)

                    if cfg.use_loss_reweight:
                        for view_name, weight in view_weight_map_cfg.items():
                            if view_name not in ds_tr.view_to_idx:
                                continue
                            vid = ds_tr.view_to_idx[view_name]
                            w_base = torch.where(view_ids == vid, w_base * float(weight), w_base)

                        if cfg.band_weight_scheme != "none":
                            b_idx = quality_to_band(y)
                            bw = torch.ones_like(w_base)
                            bw = torch.where(b_idx == 0, torch.full_like(bw, float(band_weight_map[0])), bw)
                            bw = torch.where(b_idx == 1, torch.full_like(bw, float(band_weight_map[1])), bw)
                            bw = torch.where(b_idx == 2, torch.full_like(bw, float(band_weight_map[2])), bw)
                            w_base = w_base * bw

                    w2 = torch.cat([w_base, w_base], dim=0)
                    l_ord = (l_ord_raw * w2).sum() / (w2.sum() + 1e-12)
                else:
                    l_ord = l_ord_raw.mean()

                pos_mask = build_supcon_mask(view_ids, y, sample_ids, max_delta_q=cfg.supcon_delta_q)
                l_con = supcon_loss(z, pos_mask)

                loss = l_ord + cfg.lambda_con * l_con

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total += float(loss.detach().cpu()) * x1.size(0)
            n += x1.size(0)
            pbar.set_postfix(loss=total / max(n, 1), lr=opt.param_groups[0]["lr"])

        sched.step()

        m_val = evaluate(model, dl_va, device, mode, cfg.amp)
        rec = {"epoch": ep, **m_val}
        history.append(rec)
        print(f"[VAL] ep={ep} RMSE={m_val['rmse']:.4f} MAE={m_val['mae']:.4f} QWK={m_val['qwk']:.4f}")

        ckpt = {
            "model": model.state_dict(),
            "cfg": asdict(cfg),
            "views": views,
            "view_to_idx": ds_tr.view_to_idx,
            "view_head_profile": cfg.view_head_profile,
            "weak_views": [v.strip() for v in cfg.weak_views.split(",") if v.strip()],
            "epoch": ep,
            "val": m_val,
            "mode": mode,
            "view": cfg.view,
        }

        if m_val["qwk"] > best_qwk:
            best_qwk = m_val["qwk"]
            best_qwk_rec = rec
            torch.save(ckpt, out_dir / "best.pt")
            torch.save(ckpt, out_dir / "best_qwk.pt")
            print(f"[SAVE] {out_dir / 'best.pt'} (by QWK)")

        if m_val["rmse"] < best_rmse:
            best_rmse = m_val["rmse"]
            best_rmse_rec = rec
            torch.save(ckpt, out_dir / "best_rmse.pt")
            print(f"[SAVE] {out_dir / 'best_rmse.pt'}")

        if ep % cfg.save_every == 0:
            torch.save(ckpt, out_dir / f"ep{ep:03d}.pt")
            print(f"[SAVE] {out_dir / f'ep{ep:03d}.pt'}")

    best_ckpt = torch.load(out_dir / "best.pt", map_location="cpu")
    model.load_state_dict(best_ckpt["model"], strict=True)
    m_test = evaluate(model, dl_te, device, mode, cfg.amp)

    with (out_dir / "best_selection.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_by": "qwk",
                "best_qwk": best_qwk_rec,
                "best_rmse": best_rmse_rec,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    out_metrics = {
        "mode": mode,
        "view": cfg.view,
        "best_val": best_ckpt["val"],
        "test": m_test,
        "epochs": cfg.epochs,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(out_metrics, f, ensure_ascii=False, indent=2)

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    print(f"[OK] metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["multihead", "perview"], required=True)
    p.add_argument("--view", type=str, default="")
    p.add_argument("--views", type=str, default="A4C,PL,PSAV,PSMV,SC")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--lambda_con", type=float, default=0.2)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--split_dir", type=str, default=str(DEFAULT_SPLIT_DIR))
    p.add_argument("--supcon_delta_q", type=float, default=1.0)
    p.add_argument("--ord_view_weights", type=str, default="")
    p.add_argument("--run_tag", type=str, default="")
    p.add_argument(
        "--sampler",
        type=str,
        default="shuffle",
        choices=["shuffle", "group", "group_view", "group_view_grade", "view_band"],
        help="Training sampler (default: shuffle).",
    )

    p.add_argument("--sampler_clip_min", type=float, default=0.2)
    p.add_argument("--sampler_clip_max", type=float, default=5.0)
    p.add_argument("--view_head_profile", type=str, default="default", choices=["default", "heavy_for_weak_views"])
    p.add_argument("--weak_views", type=str, default="A4C,SC,PSAV")
    p.add_argument("--exclude_random", dest="exclude_random", action="store_true")
    p.add_argument("--include_random", dest="exclude_random", action="store_false")
    p.set_defaults(exclude_random=True)

    p.add_argument("--use_loss_reweight", action="store_true")
    p.add_argument("--band_weight_scheme", type=str, default="invfreq_clip", choices=["invfreq_clip", "none"])
    p.add_argument("--band_weight_clip_min", type=float, default=0.5)
    p.add_argument("--band_weight_clip_max", type=float, default=2.0)
    p.add_argument("--view_weight_a4c", type=float, default=1.0)
    p.add_argument("--view_weight_pl", type=float, default=1.0)
    p.add_argument("--view_weight_psav", type=float, default=1.0)
    p.add_argument("--view_weight_psmv", type=float, default=1.0)
    p.add_argument("--view_weight_sc", type=float, default=1.0)

    p.add_argument("--use_rrc", action="store_true", help="Enable RandomResizedCrop (default: off).")
    p.add_argument("--rrc_scale_min", type=float, default=0.85)
    p.add_argument("--rrc_scale_max", type=float, default=1.0)
    p.add_argument("--affine_degrees", type=float, default=3.0)
    p.add_argument("--affine_translate", type=float, default=0.02)
    p.add_argument("--jitter_p", type=float, default=0.25)
    p.add_argument("--jitter_brightness", type=float, default=0.06)
    p.add_argument("--jitter_contrast", type=float, default=0.06)
    p.add_argument("--jitter_saturation", type=float, default=0.03)
    p.add_argument("--jitter_hue", type=float, default=0.02)
    args = p.parse_args()

    cfg = Cfg(
        mode=args.mode,
        view=args.view,
        views=args.views,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        lr=args.lr,
        amp=bool(args.amp),
        lambda_con=args.lambda_con,
        save_every=args.save_every,
        split_dir=args.split_dir,
        supcon_delta_q=args.supcon_delta_q,
        ord_view_weights=args.ord_view_weights,
        run_tag=args.run_tag,
        sampler=args.sampler,
        sampler_clip_min=args.sampler_clip_min,
        sampler_clip_max=args.sampler_clip_max,
        view_head_profile=args.view_head_profile,
        weak_views=args.weak_views,
        exclude_random=bool(args.exclude_random),
        use_loss_reweight=bool(args.use_loss_reweight),
        band_weight_scheme=args.band_weight_scheme,
        band_weight_clip_min=args.band_weight_clip_min,
        band_weight_clip_max=args.band_weight_clip_max,
        view_weight_a4c=args.view_weight_a4c,
        view_weight_pl=args.view_weight_pl,
        view_weight_psav=args.view_weight_psav,
        view_weight_psmv=args.view_weight_psmv,
        view_weight_sc=args.view_weight_sc,
        use_rrc=args.use_rrc,
        rrc_scale_min=args.rrc_scale_min,
        rrc_scale_max=args.rrc_scale_max,
        affine_degrees=args.affine_degrees,
        affine_translate=args.affine_translate,
        jitter_p=args.jitter_p,
        jitter_brightness=args.jitter_brightness,
        jitter_contrast=args.jitter_contrast,
        jitter_saturation=args.jitter_saturation,
        jitter_hue=args.jitter_hue,
    )
    train(cfg)
