from __future__ import annotations

import torch
import torch.nn.functional as F

from .models import DenseNetMultilabel


class DenseNetGradCAM:
    def __init__(self, model: DenseNetMultilabel) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target = self.model.backbone.features.denseblock4
        target.register_forward_hook(self._forward_hook)
        target.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, output) -> None:
        self.activations = output

    def _backward_hook(self, _module, _grad_inputs, grad_output) -> None:
        self.gradients = grad_output[0]

    def __call__(self, images: torch.Tensor, class_index: int, grid_size: int = 49) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images)
        score = logits[:, class_index].sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(1), size=(grid_size, grid_size), mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        flat = cam.flatten(start_dim=1)
        cam_min = flat.min(dim=1).values.view(-1, 1, 1)
        cam_max = flat.max(dim=1).values.view(-1, 1, 1)
        return (cam - cam_min) / (cam_max - cam_min + 1e-6)


def heatmap_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    pred_bin = pred >= threshold
    target_bin = target >= 0.5
    intersection = (pred_bin & target_bin).sum(dim=(-2, -1)).float()
    union = (pred_bin | target_bin).sum(dim=(-2, -1)).float()
    return intersection / (union + 1e-6)


def pointing_game(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch = pred.size(0)
    flat_idx = pred.flatten(start_dim=1).argmax(dim=1)
    y = flat_idx // pred.size(-1)
    x = flat_idx % pred.size(-1)
    return target[torch.arange(batch, device=target.device), y, x].float()
