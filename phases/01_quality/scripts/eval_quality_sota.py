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

from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CKPT = ROOT / "shared" / "runs" / "phase1_quality_sota" / "best_quality_evaluator.pt"
DEFAULT_SPLIT_DIR = ROOT / "datasets" / "CACTUS" / "manifests_group_consensus_pad2_stratified_v3"
OUT_DIR = ROOT / "shared" / "runs" / "phase1" / "eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)


_PIL_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)


class EvalDataset(Dataset):
    def __init__(self, csv_path: Path, img_size: int, view_to_idx: Dict[str, int]):
        df = pd.read_csv(csv_path)
        for c in ["path", "view", "quality"]:
            if c not in df.columns:
                raise ValueError(f"Missing column {c} in {csv_path}")

        self.paths = df["path"].astype(str).tolist()
        self.views = df["view"].astype(str).tolist()
        self.y = df["quality"].astype(float).tolist()
        self.view_ids = [view_to_idx[v] for v in self.views]

        self.tf = T.Compose(
            [
                T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        x = self.tf(img)
        return {
            "x": x,
            "view": self.views[i],
            "view_id": torch.tensor(self.view_ids[i], dtype=torch.long),
            "y": torch.tensor(self.y[i], dtype=torch.float32),
        }


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

    def forward(self, x: torch.Tensor, view_id: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        v = self.view_emb(view_id)
        h = self.trunk(torch.cat([feat, v], dim=1))
        reg = self.head_reg(h).squeeze(1)
        logits = self.head_cls(h)
        prob = F.softmax(logits, dim=1)
        bins = torch.arange(10, device=logits.device, dtype=logits.dtype).unsqueeze(0)
        exp_score = (prob * bins).sum(dim=1)
        pred = 0.7 * exp_score + 0.3 * reg
        return pred


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    # AveragedModel (SWA/EMA) adds bookkeeping buffers.
    out.pop("n_averaged", None)
    return out


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str, amp: bool) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    ys, ps, views = [], [], []

    for b in loader:
        x = b["x"].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        vid = b["view_id"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(amp and device == "cuda")):
            pred = model(x, vid)

        ys.append(b["y"].float().cpu())
        ps.append(pred.float().cpu())
        views.extend(b["view"])

    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    return y, p, views


def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y, p))
    rmse = float(mean_squared_error(y, p) ** 0.5)
    y_r = np.clip(y, 0, 9).astype(int)
    p_r = np.clip(np.rint(p), 0, 9).astype(int)
    qwk = float(cohen_kappa_score(y_r, p_r, weights="quadratic"))
    return {"mae": mae, "rmse": rmse, "qwk": qwk}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--split_dir", default=str(DEFAULT_SPLIT_DIR))
    p.add_argument("--img_size", type=int, default=288)
    p.add_argument(
        "--views",
        default="A4C,PL,PSAV,PSMV,Random,SC",
        help="Comma-separated view list used to map view->id (must match SOTA checkpoint embedding size).",
    )
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out_name", default="quality_sota_pad2")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required")

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    view_to_idx = {v: i for i, v in enumerate(views)}

    split_dir = Path(args.split_dir)
    val_csv = split_dir / "phase1_val.csv"
    test_csv = split_dir / "phase1_test.csv"

    ckpt = torch.load(Path(args.ckpt), map_location="cpu")
    state = strip_module_prefix(ckpt["model"])

    emb_w = state.get("view_emb.weight")
    if emb_w is None:
        raise KeyError("view_emb.weight not found in checkpoint")
    n_views = int(emb_w.shape[0])
    if n_views != len(view_to_idx):
        raise ValueError(f"Checkpoint expects n_views={n_views} but --views has {len(view_to_idx)}")

    model = SotaModel(n_views=n_views).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(state, strict=True)

    ds_val = EvalDataset(val_csv, args.img_size, view_to_idx)
    ds_test = EvalDataset(test_csv, args.img_size, view_to_idx)

    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    yv, pv, vv = predict(model, dl_val, device, amp=bool(args.amp))
    yt, pt, vt = predict(model, dl_test, device, amp=bool(args.amp))

    out = {
        "mode": "quality_sota",
        "ckpt": str(args.ckpt),
        "split_dir": str(split_dir),
        "img_size": int(args.img_size),
        "views": views,
        "val": metrics(yv, pv),
        "test": metrics(yt, pt),
        "per_view_val": {},
        "per_view_test": {},
    }

    for name in sorted(set(vv)):
        idx = np.array([v == name for v in vv], dtype=bool)
        out["per_view_val"][name] = metrics(yv[idx], pv[idx])

    for name in sorted(set(vt)):
        idx = np.array([v == name for v in vt], dtype=bool)
        out["per_view_test"][name] = metrics(yt[idx], pt[idx])

    out_path = OUT_DIR / f"{args.out_name}_report.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] report: {out_path}")
    print(f"[OK] val={out['val']} test={out['test']}")


if __name__ == "__main__":
    main()
