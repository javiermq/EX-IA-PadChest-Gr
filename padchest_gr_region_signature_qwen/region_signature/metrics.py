from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .boxes import heatmaps_to_boxes, pairwise_iou


class MetricsAccumulator:
    def __init__(self, class_names: list[str], max_rois: int) -> None:
        self.class_names = class_names
        self.max_rois = max_rois
        self.global_true: list[np.ndarray] = []
        self.global_pred: list[np.ndarray] = []
        self.roi_true: list[int] = []
        self.roi_pred: list[int] = []
        self.ious: list[float] = []
        self.hm_dice_values: list[float] = []
        self.counts = defaultdict(float)

    def update(self, output, batch: dict[str, torch.Tensor]) -> None:
        global_probs = output.global_logits.detach().sigmoid().cpu()
        labels = batch["labels"].detach().cpu()
        self.global_true.extend(labels.numpy())
        self.global_pred.extend((global_probs >= 0.5).numpy())
        pred_hm = output.heatmap_logits.detach().cpu()
        gt_hm = batch["heatmaps_gt"].detach().cpu()
        dice = (2 * (pred_hm.sigmoid() * gt_hm).sum(dim=(1, 2, 3)) + 1e-6) / (
            pred_hm.sigmoid().sum(dim=(1, 2, 3)) + gt_hm.sum(dim=(1, 2, 3)) + 1e-6
        )
        self.hm_dice_values.extend(dice.tolist())
        roi_logits = output.roi_label_logits.detach().cpu()
        roi_pred = roi_logits.argmax(dim=-1)
        for bi in range(labels.shape[0]):
            gt_boxes = batch["boxes"][bi][batch["roi_valid"][bi]].detach().cpu()
            gt_labels = batch["box_labels"][bi][batch["roi_valid"][bi]].detach().cpu()
            pred_boxes, pred_labels, _ = heatmaps_to_boxes(pred_hm[bi], self.max_rois)
            valid_pred = pred_labels >= 0
            pred_boxes = pred_boxes[valid_pred]
            pred_labels = pred_labels[valid_pred]
            if len(gt_boxes) == 0:
                continue
            ious = pairwise_iou(pred_boxes, gt_boxes)
            matched_gt: set[int] = set()
            matched_pred: set[int] = set()
            if ious.numel() > 0:
                pairs = []
                for pi in range(ious.shape[0]):
                    for gi in range(ious.shape[1]):
                        label_match = int(pred_labels[pi]) == int(gt_labels[gi])
                        pairs.append((float(ious[pi, gi]), pi, gi, label_match))
                pairs.sort(reverse=True, key=lambda x: x[0])
                for iou, pi, gi, label_match in pairs:
                    if pi in matched_pred or gi in matched_gt:
                        continue
                    matched_pred.add(pi)
                    matched_gt.add(gi)
                    self.ious.append(iou)
                    if label_match:
                        self.counts["label_matched"] += 1
                    for thr in (0.25, 0.5):
                        if iou >= thr and label_match:
                            self.counts[f"tp_{thr}"] += 1
            self.counts["pred"] += max(1, len(pred_boxes))
            self.counts["gt"] += len(gt_boxes)
            self.counts["roi_recalled"] += len(matched_gt)
            self.counts["roi_total"] += len(gt_boxes)
            for ri, valid in enumerate(batch["roi_valid"][bi]):
                if bool(valid):
                    self.roi_true.append(int(batch["box_labels"][bi, ri]))
                    self.roi_pred.append(int(roi_pred[bi, ri]))

    def compute(self) -> dict[str, float]:
        y_true = np.array(self.global_true)
        y_pred = np.array(self.global_pred)
        out = {
            "global_micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)) if len(y_true) else 0.0,
            "global_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else 0.0,
            "roi_recall_at_k": float(self.counts["roi_recalled"] / max(1.0, self.counts["roi_total"])),
            "mean_iou": float(np.mean(self.ious)) if self.ious else 0.0,
            "heatmap_dice": float(np.mean(self.hm_dice_values)) if self.hm_dice_values else 0.0,
            "roi_cls_accuracy": float(accuracy_score(self.roi_true, self.roi_pred)) if self.roi_true else 0.0,
            "roi_cls_f1": float(f1_score(self.roi_true, self.roi_pred, average="macro", zero_division=0)) if self.roi_true else 0.0,
        }
        for thr in (0.25, 0.5):
            tp = self.counts[f"tp_{thr}"]
            out[f"precision_iou_{thr}"] = float(tp / max(1.0, self.counts["pred"]))
            out[f"recall_iou_{thr}"] = float(tp / max(1.0, self.counts["gt"]))
        return out
