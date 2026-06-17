from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import RegionSignatureOutput


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = logits.sigmoid()
    dims = tuple(range(1, probs.ndim))
    inter = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def make_roi_targets(class_ids: torch.Tensor, valid: torch.Tensor, num_classes: int) -> torch.Tensor:
    targets = torch.zeros(*class_ids.shape, num_classes, device=class_ids.device)
    safe_ids = class_ids.clamp(min=0)
    targets.scatter_(-1, safe_ids.unsqueeze(-1), 1.0)
    return targets * valid.unsqueeze(-1).float()


def compute_loss(
    output: RegionSignatureOutput,
    batch: dict[str, torch.Tensor],
    lambdas: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    labels = batch["labels"]
    heatmaps_gt = batch["heatmaps_gt"]
    roi_valid = batch["roi_valid"]
    box_labels = batch["box_labels"]
    boxes = batch["boxes"]
    global_loss = F.binary_cross_entropy_with_logits(output.global_logits, labels)
    global_loss = global_loss + 0.5 * F.binary_cross_entropy_with_logits(output.densenet_global_logits, labels)
    hm_bce = F.binary_cross_entropy_with_logits(output.heatmap_logits, heatmaps_gt)
    hm_dice = dice_loss_with_logits(output.heatmap_logits, heatmaps_gt)
    heatmap_loss = hm_bce + hm_dice
    roi_targets = make_roi_targets(box_labels, roi_valid, output.roi_label_logits.shape[-1])
    roi_raw = F.binary_cross_entropy_with_logits(output.roi_label_logits, roi_targets, reduction="none")
    roi_loss = (roi_raw * roi_valid.unsqueeze(-1).float()).sum() / (roi_valid.sum().clamp(min=1) * roi_targets.shape[-1])
    box_raw = F.l1_loss(output.bbox_pred, boxes, reduction="none").mean(dim=-1)
    box_loss = (box_raw * roi_valid.float()).sum() / roi_valid.sum().clamp(min=1)
    total = (
        global_loss
        + float(lambdas.get("lambda_hm", 1.0)) * heatmap_loss
        + float(lambdas.get("lambda_box", 1.0)) * box_loss
        + float(lambdas.get("lambda_roi", 1.0)) * roi_loss
    )
    return total, {
        "global_loss": float(global_loss.detach().cpu()),
        "heatmap_loss": float(heatmap_loss.detach().cpu()),
        "roi_loss": float(roi_loss.detach().cpu()),
        "box_loss": float(box_loss.detach().cpu()),
    }
