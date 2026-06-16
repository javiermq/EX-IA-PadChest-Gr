from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet121_Weights, densenet121


class DenseNetMultilabel(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = densenet121(weights=weights)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Linear(in_features, num_classes)
        self.feature_dim = in_features

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(images))


class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CategoryProjectors(nn.Module):
    def __init__(self, image_dim: int, qwen_dim: int, num_categories: int = 10, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.projectors = nn.ModuleList(
            [TwoLayerMLP(image_dim, hidden_dim, qwen_dim) for _ in range(num_categories)]
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        slots = [projector(image_features) for projector in self.projectors]
        return torch.stack(slots, dim=1)


class QwenHiddenClassifier(nn.Module):
    def __init__(self, qwen_dim: int, num_outputs: int, hidden_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = hidden_dim or qwen_dim
        self.net = nn.Sequential(
            nn.Linear(qwen_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, category_hidden: torch.Tensor) -> torch.Tensor:
        pooled = category_hidden.mean(dim=1)
        return self.net(pooled)


def multilabel_clip_loss(
    visual_slots: torch.Tensor,
    token_embeddings: torch.Tensor,
    labels_top10: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    visual = F.normalize(visual_slots.float(), dim=-1)
    tokens = F.normalize(token_embeddings.float(), dim=-1).unsqueeze(0)
    cosine = (visual * tokens).sum(dim=-1)
    positive = labels_top10.float() * (1.0 - cosine)
    negative = (1.0 - labels_top10.float()) * F.relu(cosine - margin)
    return (positive + negative).mean()
