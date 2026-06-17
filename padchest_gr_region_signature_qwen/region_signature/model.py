from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet121_Weights, densenet121


class DenseNetSignatureEncoder(nn.Module):
    def __init__(self, num_classes: int, grid_size: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = densenet121(weights=weights)
        self.features = backbone.features
        self.feature_dim = backbone.classifier.in_features
        self.global_head = nn.Linear(self.feature_dim, num_classes)
        self.heatmap_head = nn.Conv2d(self.feature_dim, num_classes, kernel_size=1)
        self.grid_size = grid_size

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        fmap = F.relu(self.features(images), inplace=False)
        local = F.interpolate(fmap, size=(self.grid_size, self.grid_size), mode="bilinear", align_corners=False)
        pooled = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
        return {
            "local_features": local,
            "global_logits": self.global_head(pooled),
            "local_heatmap_logits": self.heatmap_head(local),
        }


class SimpleVisualEncoder(nn.Module):
    def __init__(self, hidden_dim: int, grid_size: int) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, hidden_dim, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        fmap = self.net(images)
        fmap = F.interpolate(fmap, size=(self.grid_size, self.grid_size), mode="bilinear", align_corners=False)
        return fmap.flatten(2).transpose(1, 2)


class RegionSignatureProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, local_features: torch.Tensor, local_scores: torch.Tensor) -> torch.Tensor:
        b, _, g, _ = local_features.shape
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, g, device=local_features.device),
            torch.linspace(0, 1, g, device=local_features.device),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
        x = torch.cat([local_features, local_scores.sigmoid(), coords], dim=1)
        return self.net(x.flatten(2).transpose(1, 2))


class FusionModule(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, visual_tokens: torch.Tensor, signature_tokens: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([visual_tokens, signature_tokens], dim=-1))


class ROIClassifier(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.label_queries = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
        self.roi_proj = nn.Linear(hidden_dim, hidden_dim)
        self.confidence = nn.Linear(hidden_dim, 1)

    def forward(self, roi_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        roi_hidden = self.roi_proj(roi_tokens)
        queries = F.normalize(self.label_queries, dim=-1)
        logits = torch.einsum("bkh,ch->bkc", F.normalize(roi_hidden, dim=-1), queries) * 10.0
        confidence = self.confidence(roi_hidden).squeeze(-1)
        return logits, confidence


class OptionalQwenWrapper(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.enabled = False
        self.load_error: str | None = None
        self.processor = None
        self.hidden_dim = int(cfg.get("qwen_hidden_dim", cfg["hidden_dim"]))
        self.grid_size = int(cfg["grid_size"])
        self.qwen_visual_no_grad = bool(cfg.get("qwen_visual_no_grad", True))
        if not cfg.get("use_qwen", False) or not cfg.get("load_qwen_weights", False):
            return
        try:
            import transformers
            from transformers import AutoProcessor, BitsAndBytesConfig

            quantization_config = None
            if cfg.get("load_in_4bit", False):
                compute_dtype_name = str(cfg.get("bnb_4bit_compute_dtype", "bf16")).lower()
                compute_dtype = torch.bfloat16 if compute_dtype_name == "bf16" else torch.float16
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=str(cfg.get("bnb_4bit_quant_type", "nf4")),
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
                )

            self.processor = AutoProcessor.from_pretrained(cfg["qwen_name"])
            model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
            if model_cls is None:
                model_cls = getattr(transformers, "AutoModelForVision2Seq")
            self.qwen = model_cls.from_pretrained(
                cfg["qwen_name"],
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if cfg.get("load_in_4bit", False) else None,
                quantization_config=quantization_config,
            )
            if cfg.get("use_lora", False):
                try:
                    from peft import prepare_model_for_kbit_training
                    from peft import LoraConfig, get_peft_model

                    if cfg.get("load_in_4bit", False):
                        self.qwen = prepare_model_for_kbit_training(self.qwen)
                    lora_cfg = LoraConfig(
                        r=int(cfg.get("lora_r", 8)),
                        lora_alpha=int(cfg.get("lora_alpha", 16)),
                        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        bias="none",
                        task_type="CAUSAL_LM",
                    )
                    self.qwen = get_peft_model(self.qwen, lora_cfg)
                except Exception as exc:
                    if cfg.get("require_qwen", False):
                        raise RuntimeError(f"Qwen loaded, but LoRA setup failed: {exc}") from exc
            if cfg.get("freeze_qwen", True):
                for param in self.qwen.parameters():
                    param.requires_grad = False
                self.qwen.eval()
            self.enabled = True
            # TODO: Inject Region Signature Tokens directly into Qwen's multimodal sequence
            # before selected language layers. For now, we consume Qwen visual hidden states
            # and fuse them with the region signatures in task-specific heads.
        except Exception as exc:  # pragma: no cover - depends on local model availability.
            self.qwen = None
            self.load_error = str(exc)
            if cfg.get("require_qwen", False):
                raise RuntimeError(f"Could not load required Qwen model {cfg['qwen_name']}: {exc}") from exc

    def _base_model(self) -> nn.Module:
        return self.qwen.get_base_model() if hasattr(self.qwen, "get_base_model") else self.qwen

    def _find_visual_module(self) -> nn.Module:
        candidates = [
            self.qwen,
            self._base_model(),
            getattr(self.qwen, "base_model", None),
            getattr(getattr(self.qwen, "base_model", None), "model", None),
            getattr(getattr(getattr(self.qwen, "base_model", None), "model", None), "model", None),
            getattr(self._base_model(), "model", None),
            getattr(getattr(self._base_model(), "model", None), "model", None),
        ]
        for candidate in candidates:
            if candidate is not None and hasattr(candidate, "visual"):
                return getattr(candidate, "visual")
        for name, module in self.qwen.named_modules():
            if name.endswith("visual") and hasattr(module, "patch_embed"):
                return module
        for name, module in self.qwen.named_modules():
            if "visual" in name and hasattr(module, "patch_embed"):
                return module
        raise RuntimeError(
            "Loaded Qwen model does not expose a discoverable visual module. "
            "Run `python - <<'PY' ... named_modules ... PY` to inspect the wrapper layout."
        )

    def forward(self, pil_images: list[Any], device: torch.device) -> torch.Tensor:
        if not self.enabled or self.processor is None or any(image is None for image in pil_images):
            raise RuntimeError("Qwen visual tokens requested but Qwen processor/images are unavailable.")
        image_prompt = "<|vision_start|><|image_pad|><|vision_end|>"
        inputs = self.processor(text=[image_prompt] * len(pil_images), images=pil_images, return_tensors="pt")
        base = self._base_model()
        visual = self._find_visual_module()
        visual_device = next(visual.parameters()).device
        pixel_values = inputs["pixel_values"].to(visual_device)
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(visual_device)
        dtype = next(visual.parameters()).dtype
        pixel_values = pixel_values.to(dtype=dtype)
        if self.qwen_visual_no_grad:
            with torch.no_grad():
                visual_tokens = self._run_visual(visual, pixel_values, image_grid_thw)
        else:
            visual_tokens = self._run_visual(visual, pixel_values, image_grid_thw)
        visual_tokens = self._unwrap_visual_output(visual_tokens)
        if visual_tokens.ndim == 2:
            lengths = self._visual_lengths(image_grid_thw, visual_tokens.shape[0], len(pil_images), base)
            visual_tokens = torch.split(visual_tokens, lengths, dim=0)
            visual_tokens = [self._resample_sequence(tokens) for tokens in visual_tokens]
            return torch.stack(visual_tokens, dim=0).float()
        if visual_tokens.ndim == 3:
            return torch.stack([self._resample_sequence(tokens) for tokens in visual_tokens], dim=0).float()
        raise RuntimeError(f"Unexpected Qwen visual token shape: {tuple(visual_tokens.shape)}")

    def _run_visual(self, visual: nn.Module, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor | None) -> Any:
        try:
            return visual(pixel_values, grid_thw=image_grid_thw)
        except TypeError:
            return visual(pixel_values, image_grid_thw)

    def _unwrap_visual_output(self, visual_output: Any) -> torch.Tensor:
        if torch.is_tensor(visual_output):
            return visual_output
        if hasattr(visual_output, "last_hidden_state") and visual_output.last_hidden_state is not None:
            return visual_output.last_hidden_state
        if hasattr(visual_output, "hidden_states") and visual_output.hidden_states:
            return visual_output.hidden_states[-1]
        if hasattr(visual_output, "pooler_output") and visual_output.pooler_output is not None:
            pooled = visual_output.pooler_output
            return pooled.unsqueeze(1) if pooled.ndim == 2 else pooled
        if isinstance(visual_output, tuple) and visual_output:
            return self._unwrap_visual_output(visual_output[0])
        raise RuntimeError(f"Could not extract tensor from Qwen visual output type: {type(visual_output)}")

    def _visual_lengths(self, image_grid_thw: torch.Tensor | None, total: int, batch: int, base: nn.Module) -> list[int]:
        if image_grid_thw is None:
            return [total // batch] * batch
        spatial_merge_size = getattr(getattr(base, "config", None), "spatial_merge_size", 2)
        lengths = []
        for row in image_grid_thw:
            t, h, w = [int(x) for x in row.tolist()]
            lengths.append(max(1, t * (h // spatial_merge_size) * (w // spatial_merge_size)))
        if sum(lengths) != total:
            return [total // batch] * batch
        return lengths

    def _resample_sequence(self, tokens: torch.Tensor) -> torch.Tensor:
        target = self.grid_size * self.grid_size
        seq = tokens.transpose(0, 1).unsqueeze(0)
        seq = F.interpolate(seq, size=target, mode="linear", align_corners=False)
        return seq.squeeze(0).transpose(0, 1)


@dataclass
class RegionSignatureOutput:
    global_logits: torch.Tensor
    heatmap_logits: torch.Tensor
    densenet_global_logits: torch.Tensor
    densenet_heatmap_logits: torch.Tensor
    roi_label_logits: torch.Tensor
    roi_confidence: torch.Tensor
    bbox_pred: torch.Tensor
    fused_tokens: torch.Tensor


class RegionSignatureQwenPrototype(nn.Module):
    def __init__(self, cfg: dict[str, Any], num_classes: int) -> None:
        super().__init__()
        hidden_dim = int(cfg["hidden_dim"])
        grid_size = int(cfg["grid_size"])
        self.grid_size = grid_size
        self.max_rois = int(cfg["max_rois"])
        self.num_classes = num_classes
        self.signature_encoder = DenseNetSignatureEncoder(
            num_classes=num_classes,
            grid_size=grid_size,
            pretrained=bool(cfg.get("densenet_pretrained", True)),
        )
        self.visual_encoder = SimpleVisualEncoder(hidden_dim=hidden_dim, grid_size=grid_size)
        self.qwen = OptionalQwenWrapper(cfg)
        self.qwen_projector = nn.LazyLinear(hidden_dim)
        self.signature_projector = RegionSignatureProjector(self.signature_encoder.feature_dim + num_classes + 2, hidden_dim)
        self.fusion = FusionModule(hidden_dim)
        self.heatmap_head = nn.Linear(hidden_dim, num_classes)
        self.global_head = nn.Linear(hidden_dim, num_classes)
        self.roi_head = ROIClassifier(hidden_dim, num_classes)
        self.bbox_head = nn.Linear(hidden_dim, 4)

    def move_task_modules(self, device: torch.device) -> "RegionSignatureQwenPrototype":
        self.signature_encoder.to(device)
        self.visual_encoder.to(device)
        self.qwen_projector.to(device)
        self.signature_projector.to(device)
        self.fusion.to(device)
        self.heatmap_head.to(device)
        self.global_head.to(device)
        self.roi_head.to(device)
        self.bbox_head.to(device)
        return self

    def _roi_pool_from_boxes(self, tokens: torch.Tensor, boxes: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        b, _, h = tokens.shape
        grid_tokens = tokens.view(b, self.grid_size, self.grid_size, h)
        roi_tokens = torch.zeros(b, self.max_rois, h, device=tokens.device, dtype=tokens.dtype)
        for bi in range(b):
            for ri in range(self.max_rois):
                if not bool(valid[bi, ri]):
                    continue
                x1, y1, x2, y2 = boxes[bi, ri].tolist()
                gx1 = max(0, min(self.grid_size - 1, int(x1 * self.grid_size)))
                gy1 = max(0, min(self.grid_size - 1, int(y1 * self.grid_size)))
                gx2 = max(gx1, min(self.grid_size - 1, int(torch.ceil(torch.tensor(x2 * self.grid_size)).item()) - 1))
                gy2 = max(gy1, min(self.grid_size - 1, int(torch.ceil(torch.tensor(y2 * self.grid_size)).item()) - 1))
                roi_tokens[bi, ri] = grid_tokens[bi, gy1 : gy2 + 1, gx1 : gx2 + 1].reshape(-1, h).mean(0)
        return roi_tokens

    def forward(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor | None = None,
        roi_valid: torch.Tensor | None = None,
        qwen_images: list[Any] | None = None,
    ) -> RegionSignatureOutput:
        dense = self.signature_encoder(images)
        if self.qwen.enabled:
            if qwen_images is None:
                raise RuntimeError("Qwen is enabled, but qwen_images were not passed to the model.")
            visual_tokens = self.qwen_projector(self.qwen(qwen_images, images.device).to(images.device))
        else:
            visual_tokens = self.visual_encoder(images)
        signatures = self.signature_projector(dense["local_features"], dense["local_heatmap_logits"])
        fused = self.fusion(visual_tokens, signatures)
        heatmap_logits = self.heatmap_head(fused).transpose(1, 2).view(-1, self.num_classes, self.grid_size, self.grid_size)
        global_logits = self.global_head(fused.mean(dim=1))
        if boxes is None or roi_valid is None:
            probs = heatmap_logits.detach().sigmoid()
            boxes = torch.zeros(images.shape[0], self.max_rois, 4, device=images.device)
            roi_valid = torch.ones(images.shape[0], self.max_rois, dtype=torch.bool, device=images.device)
            for bi in range(images.shape[0]):
                flat = probs[bi].flatten()
                top = torch.topk(flat, k=self.max_rois).indices
                for ri, idx in enumerate(top):
                    cell = int(idx.item()) % (self.grid_size * self.grid_size)
                    y, x = cell // self.grid_size, cell % self.grid_size
                    boxes[bi, ri] = torch.tensor(
                        [x / self.grid_size, y / self.grid_size, (x + 1) / self.grid_size, (y + 1) / self.grid_size],
                        device=images.device,
                    )
        roi_tokens = self._roi_pool_from_boxes(fused, boxes, roi_valid)
        roi_label_logits, roi_confidence = self.roi_head(roi_tokens)
        bbox_pred = self.bbox_head(roi_tokens).sigmoid()
        return RegionSignatureOutput(
            global_logits=global_logits,
            heatmap_logits=heatmap_logits,
            densenet_global_logits=dense["global_logits"],
            densenet_heatmap_logits=dense["local_heatmap_logits"],
            roi_label_logits=roi_label_logits,
            roi_confidence=roi_confidence,
            bbox_pred=bbox_pred,
            fused_tokens=fused,
        )
