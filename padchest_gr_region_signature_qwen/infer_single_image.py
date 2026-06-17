from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from region_signature.config import load_config
from region_signature.boxes import heatmaps_to_boxes
from region_signature.dataset import build_transforms
from region_signature.model import RegionSignatureQwenPrototype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/infer_single_image.json")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else ("cpu" if args.device == "auto" else args.device))
    ckpt = torch.load(args.checkpoint, map_location=device)
    class_names = ckpt["class_names"]
    model = RegionSignatureQwenPrototype(cfg["model"], len(class_names))
    if cfg["model"].get("load_in_4bit", False) and cfg["model"].get("load_qwen_weights", False):
        model.move_task_modules(device)
    else:
        model.to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    transform = build_transforms(cfg["model"]["image_size"], train=False)
    pil_image = Image.open(args.image).convert("RGB")
    image = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image, qwen_images=[pil_image])
    boxes, labels, scores = heatmaps_to_boxes(output.heatmap_logits[0].cpu(), cfg["model"]["max_rois"])
    roi_pred = output.roi_label_logits[0].argmax(dim=-1).cpu()
    payload = {
        "image": args.image,
        "global_labels": [
            {"label": class_names[i], "score": float(score)}
            for i, score in enumerate(output.global_logits[0].sigmoid().cpu())
            if float(score) >= 0.5
        ],
        "rois": [
            {
                "box": boxes[i].tolist(),
                "heatmap_label": class_names[int(labels[i])] if labels[i] >= 0 else None,
                "roi_label": class_names[int(roi_pred[i])],
                "score": float(scores[i]),
            }
            for i in range(len(boxes))
        ],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
