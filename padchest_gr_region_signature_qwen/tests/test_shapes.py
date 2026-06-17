from __future__ import annotations

import torch

from region_signature.dataset import DummyPadChestGRDataset, collate_region_batch
from region_signature.losses import compute_loss
from region_signature.model import RegionSignatureQwenPrototype


def test_model_shapes_and_loss() -> None:
    cfg = {
        "image_size": 64,
        "grid_size": 14,
        "max_rois": 5,
        "hidden_dim": 64,
        "use_qwen": True,
        "load_qwen_weights": False,
        "require_qwen": False,
        "densenet_pretrained": False,
        "qwen_name": "Qwen/Qwen2.5-VL-3B-Instruct",
    }
    ds = DummyPadChestGRDataset("train", 64, 14, 5, [f"c{i}" for i in range(10)], length=2)
    batch = collate_region_batch([ds[0], ds[1]])
    model = RegionSignatureQwenPrototype(cfg, 10)
    out = model(batch["image"], batch["boxes"], batch["roi_valid"])
    assert out.heatmap_logits.shape == (2, 10, 14, 14)
    assert out.roi_label_logits.shape == (2, 5, 10)
    assert out.bbox_pred.shape == (2, 5, 4)
    loss, parts = compute_loss(out, batch, {"lambda_hm": 1.0, "lambda_box": 1.0, "lambda_roi": 1.0})
    assert torch.isfinite(loss)
    assert "heatmap_loss" in parts
