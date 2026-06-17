from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from region_signature.config import load_config
from region_signature.dataset import collate_region_batch, make_datasets
from region_signature.model import RegionSignatureQwenPrototype
from region_signature.visualization import save_epoch_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--dummy_data", action="store_true")
    parser.add_argument("--epoch", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.dummy_data:
        cfg["model"]["densenet_pretrained"] = False
        cfg["model"]["load_qwen_weights"] = False
        cfg["model"]["require_qwen"] = False
    _, val_ds, class_names = make_datasets(cfg, args.dummy_data, cfg["outputs"]["logs_dir"])
    loader = DataLoader(val_ds, batch_size=3, shuffle=False, collate_fn=collate_region_batch)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegionSignatureQwenPrototype(cfg["model"], len(class_names))
    if cfg["model"].get("load_in_4bit", False) and cfg["model"].get("load_qwen_weights", False):
        model.move_task_modules(device)
    else:
        model.to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    batch = next(iter(loader))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        output = model(batch["image"], batch["boxes"], batch["roi_valid"], qwen_images=batch.get("pil_image"))
    epoch = args.epoch or int(ckpt.get("epoch", 0))
    save_epoch_examples(output, batch, class_names, Path(cfg["outputs"]["outputs_dir"]) / f"epoch_{epoch:03d}")


if __name__ == "__main__":
    main()
