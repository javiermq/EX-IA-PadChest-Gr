from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont

from .boxes import heatmaps_to_boxes, pairwise_iou


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    img = (image.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    arr = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(arr)


def _draw_box(draw: ImageDraw.ImageDraw, box: torch.Tensor, color: str, text: str, w: int, h: int) -> None:
    x1, y1, x2, y2 = box.tolist()
    xy = [x1 * w, y1 * h, x2 * w, y2 * h]
    draw.rectangle(xy, outline=color, width=3)
    draw.text((xy[0] + 2, max(0, xy[1] - 14)), text, fill=color)


def save_epoch_examples(
    output,
    batch: dict[str, Any],
    class_names: list[str],
    out_dir: str | Path,
    limit: int = 3,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pred_hm = output.heatmap_logits.detach().cpu()
    roi_logits = output.roi_label_logits.detach().cpu()
    for i in range(min(limit, batch["image"].shape[0])):
        image = tensor_to_pil(batch["image"][i])
        draw = ImageDraw.Draw(image)
        width, height = image.size
        gt_boxes = batch["boxes"][i][batch["roi_valid"][i]].detach().cpu()
        gt_labels = batch["box_labels"][i][batch["roi_valid"][i]].detach().cpu()
        pred_boxes, pred_labels, pred_scores = heatmaps_to_boxes(pred_hm[i], max_rois=batch["boxes"].shape[1])
        pred_roi_labels = roi_logits[i].argmax(dim=-1)
        for box, label in zip(gt_boxes, gt_labels):
            _draw_box(draw, box, "lime", f"GT {class_names[int(label)]}", width, height)
        for r, box in enumerate(pred_boxes):
            if pred_labels[r] < 0:
                continue
            label_id = int(pred_roi_labels[r]) if r < len(pred_roi_labels) else int(pred_labels[r])
            _draw_box(draw, box, "red", f"P {class_names[label_id]} {float(pred_scores[r]):.2f}", width, height)
        stem = f"example_{i:02d}"
        image.save(out_path / f"{stem}_boxes.png")
        gt_sum = batch["heatmaps_gt"][i].detach().cpu().amax(dim=0)
        pred_sum = pred_hm[i].sigmoid().amax(dim=0)
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        axes[0].imshow(gt_sum, cmap="magma", vmin=0, vmax=1)
        axes[0].set_title("GT heatmap")
        axes[1].imshow(pred_sum, cmap="magma", vmin=0, vmax=1)
        axes[1].set_title("Pred heatmap")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path / f"{stem}_heatmaps.png", dpi=130)
        plt.close(fig)
        ious = pairwise_iou(pred_boxes, gt_boxes).max(dim=1).values.tolist() if len(gt_boxes) else []
        payload = {
            "labels_gt": [class_names[int(x)] for x in gt_labels],
            "labels_pred": [class_names[int(x)] for x in pred_roi_labels[: len(pred_boxes)]],
            "boxes_gt": gt_boxes.tolist(),
            "boxes_pred": pred_boxes.tolist(),
            "iou_per_pred_roi": ious,
            "scores": pred_scores.tolist(),
        }
        (out_path / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
