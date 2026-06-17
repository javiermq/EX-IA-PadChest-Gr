from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from region_signature.config import load_config
from region_signature.dataset import collate_region_batch, make_datasets
from region_signature.losses import compute_loss
from region_signature.metrics import MetricsAccumulator
from region_signature.model import RegionSignatureQwenPrototype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--dummy_data", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.dummy_data:
        cfg["model"]["densenet_pretrained"] = False
        cfg["model"]["load_qwen_weights"] = False
        cfg["model"]["require_qwen"] = False
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else ("cpu" if args.device == "auto" else args.device))
    _, val_ds, class_names = make_datasets(cfg, args.dummy_data, cfg["outputs"]["logs_dir"])
    loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=collate_region_batch)
    model = RegionSignatureQwenPrototype(cfg["model"], len(class_names))
    if cfg["model"].get("load_in_4bit", False) and cfg["model"].get("load_qwen_weights", False):
        model.move_task_modules(device)
    else:
        model.to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    metrics = MetricsAccumulator(class_names, cfg["model"]["max_rois"])
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            output = model(batch["image"], batch["boxes"], batch["roi_valid"], qwen_images=batch.get("pil_image"))
            loss, _ = compute_loss(output, batch, cfg["loss"])
            total_loss += float(loss.cpu())
            metrics.update(output, batch)
    print({"val_loss": total_loss / max(1, len(loader)), **metrics.compute()})


if __name__ == "__main__":
    main()
