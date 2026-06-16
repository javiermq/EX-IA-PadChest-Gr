from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .box_dataset import PadChestBoxDataset
from .gradcam import DenseNetGradCAM, heatmap_iou, pointing_game
from .models import DenseNetMultilabel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DenseNet GradCAM maps against PadChest-GR box heatmaps.")
    parser.add_argument("--boxes-tsv", type=Path, required=True)
    parser.add_argument("--box-vocab-json", type=Path, required=True)
    parser.add_argument("--category-vocab-json", type=Path, required=True)
    parser.add_argument("--densenet-checkpoint", type=Path, required=True)
    parser.add_argument("--out-tsv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    category_vocab = json.loads(args.category_vocab_json.read_text(encoding="utf-8"))
    box_vocab = json.loads(args.box_vocab_json.read_text(encoding="utf-8"))
    dataset = PadChestBoxDataset(args.boxes_tsv, args.box_vocab_json, args.image_root, args.image_size, train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    num_outputs = len(category_vocab["label_columns"])
    model = DenseNetMultilabel(num_classes=num_outputs, pretrained=False).to(args.device)
    checkpoint = torch.load(args.densenet_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    model.eval()
    gradcam = DenseNetGradCAM(model)

    category_ids = [entry["id"] for entry in category_vocab["categories"]]
    box_ids = [entry["id"] for entry in box_vocab["labels"]]
    shared = [(category_ids.index(label_id), box_ids.index(label_id), label_id) for label_id in box_ids if label_id in category_ids]
    if not shared:
        print("No overlapping label ids between category vocab and box vocab; output will be empty.")

    rows: list[dict[str, object]] = []
    for batch in tqdm(loader, desc="GradCAM vs boxes"):
        image = batch["image"].to(args.device)
        heatmaps = batch["box_heatmaps"].to(args.device)
        for category_idx, box_idx, label_id in shared:
            target = heatmaps[:, box_idx]
            if target.sum().item() == 0:
                continue
            cam = gradcam(image, category_idx, grid_size=target.size(-1))
            iou = heatmap_iou(cam, target, threshold=args.threshold)
            hit = pointing_game(cam, target)
            for row_idx, image_id in enumerate(batch["image_id"]):
                rows.append(
                    {
                        "image_id": image_id,
                        "label_id": label_id,
                        "gradcam_iou": float(iou[row_idx].item()),
                        "pointing_hit": float(hit[row_idx].item()),
                        "threshold": args.threshold,
                    }
                )

    out_df = pd.DataFrame(rows)
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"Wrote {len(out_df)} GradCAM-box comparisons to {args.out_tsv}")


if __name__ == "__main__":
    main()
