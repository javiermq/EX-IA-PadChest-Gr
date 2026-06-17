from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .boxes import box_area, boxes_to_heatmaps, normalize_box


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    steps: list[Any] = []
    if train:
        steps.extend(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
    else:
        steps.append(transforms.Resize((image_size, image_size)))
    steps.extend(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(steps)


def _read_records(annotation_path: str | Path) -> list[dict[str, Any]]:
    path = Path(annotation_path)
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("annotations", raw.get("data", []))
        return list(raw)
    df = pd.read_csv(path)
    return df.fillna("").to_dict("records")


def _coerce_image_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    per_box = {"x1", "y1", "x2", "y2"}.issubset(records[0].keys()) or "box_label" in records[0]
    if not per_box:
        return records
    grouped: dict[str, dict[str, Any]] = {}
    for row in records:
        image_path = str(row.get("image_path") or row.get("path") or row.get("filename") or row.get("image_id"))
        item = grouped.setdefault(
            image_path,
            {
                "image_path": image_path,
                "image_id": row.get("image_id", image_path),
                "split": row.get("split", "train"),
                "labels": set(),
                "boxes": [],
                "box_labels": [],
                "width": row.get("width") or row.get("image_width"),
                "height": row.get("height") or row.get("image_height"),
            },
        )
        label = str(row.get("box_label") or row.get("label") or row.get("category") or "").strip()
        if not label:
            continue
        box = [row["x1"], row["y1"], row["x2"], row["y2"]]
        item["boxes"].append(box)
        item["box_labels"].append(label)
        item["labels"].add(label)
    result = []
    for item in grouped.values():
        item["labels"] = sorted(item["labels"])
        result.append(item)
    return result


def load_annotation_records(annotation_path: str | Path) -> list[dict[str, Any]]:
    return _coerce_image_records(_read_records(annotation_path))


def compute_top_roi_labels(records: list[dict[str, Any]], split: str, top_n: int) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for item in records:
        if str(item.get("split", split)) != split:
            continue
        counter.update([str(label) for label in item.get("box_labels", [])])
    return [{"label": label, "frequency": int(freq)} for label, freq in counter.most_common(top_n)]


class DummyPadChestGRDataset(Dataset):
    def __init__(
        self,
        split: str,
        image_size: int,
        grid_size: int,
        max_rois: int,
        class_names: list[str] | None = None,
        length: int = 12,
    ) -> None:
        self.split = split
        self.image_size = image_size
        self.grid_size = grid_size
        self.max_rois = max_rois
        self.class_names = class_names or [f"roi_{i}" for i in range(10)]
        self.length = length if split == "train" else max(4, length // 3)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(idx + (0 if self.split == "train" else 1000))
        image = torch.rand(3, self.image_size, self.image_size, generator=generator)
        n = 1 + idx % min(3, self.max_rois)
        boxes = torch.zeros(self.max_rois, 4)
        class_ids = torch.full((self.max_rois,), -1, dtype=torch.long)
        valid = torch.zeros(self.max_rois, dtype=torch.bool)
        for r in range(n):
            x1 = 0.05 + 0.18 * ((idx + r) % 4)
            y1 = 0.08 + 0.20 * ((idx + 2 * r) % 4)
            x2 = min(0.98, x1 + 0.18 + 0.02 * r)
            y2 = min(0.98, y1 + 0.16 + 0.03 * r)
            boxes[r] = torch.tensor([x1, y1, x2, y2])
            class_ids[r] = (idx + r) % len(self.class_names)
            valid[r] = True
        global_labels = torch.zeros(len(self.class_names))
        global_labels[class_ids[valid]] = 1.0
        heatmaps = boxes_to_heatmaps(boxes, class_ids, valid, len(self.class_names), self.grid_size)
        return {
            "image": image,
            "pil_image": None,
            "image_path": f"dummy_{idx}.png",
            "image_id": f"dummy_{idx}",
            "labels": global_labels,
            "boxes": boxes,
            "box_labels": class_ids,
            "roi_valid": valid,
            "heatmaps_gt": heatmaps,
            "metadata": {"dummy": True},
        }


class PadChestGRRegionDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        image_dir: str | Path,
        split: str,
        class_names: list[str],
        image_size: int,
        grid_size: int,
        max_rois: int,
        train: bool,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.split = split
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.image_size = image_size
        self.grid_size = grid_size
        self.max_rois = max_rois
        self.transform = build_transforms(image_size, train=train)
        self.items = self._filter(records)

    def _filter(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items = []
        for item in records:
            if str(item.get("split", self.split)) != self.split:
                continue
            pairs = []
            width = item.get("width")
            height = item.get("height")
            for box, label in zip(item.get("boxes", []), item.get("box_labels", [])):
                label = str(label)
                if label not in self.class_to_idx:
                    continue
                norm = normalize_box(list(box), int(width) if width else None, int(height) if height else None)
                if box_area(norm) > 0:
                    pairs.append((norm, label))
            if not pairs:
                continue
            pairs.sort(key=lambda pair: box_area(pair[0]), reverse=True)
            item = dict(item)
            item["boxes"] = [p[0] for p in pairs[: self.max_rois]]
            item["box_labels"] = [p[1] for p in pairs[: self.max_rois]]
            items.append(item)
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        image_path = Path(str(item.get("image_path") or item.get("path") or item.get("filename")))
        if not image_path.is_absolute():
            image_path = self.image_dir / image_path
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        boxes = torch.zeros(self.max_rois, 4)
        class_ids = torch.full((self.max_rois,), -1, dtype=torch.long)
        valid = torch.zeros(self.max_rois, dtype=torch.bool)
        for i, (box, label) in enumerate(zip(item["boxes"], item["box_labels"])):
            boxes[i] = torch.tensor(box, dtype=torch.float32)
            class_ids[i] = self.class_to_idx[str(label)]
            valid[i] = True
        global_labels = torch.zeros(len(self.class_names), dtype=torch.float32)
        global_labels[class_ids[valid]] = 1.0
        heatmaps = boxes_to_heatmaps(boxes, class_ids, valid, len(self.class_names), self.grid_size)
        return {
            "image": image_tensor,
            "pil_image": image,
            "image_path": str(image_path),
            "image_id": str(item.get("image_id", image_path.stem)),
            "labels": global_labels,
            "boxes": boxes,
            "box_labels": class_ids,
            "roi_valid": valid,
            "heatmaps_gt": heatmaps,
            "metadata": {k: v for k, v in item.items() if k not in {"boxes", "box_labels"}},
        }


def collate_region_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = ["image", "labels", "boxes", "box_labels", "roi_valid", "heatmaps_gt"]
    out: dict[str, Any] = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    for key in ["image_path", "image_id", "metadata", "pil_image"]:
        out[key] = [item[key] for item in batch]
    return out


def make_datasets(cfg: dict[str, Any], dummy_data: bool, logs_dir: str | Path) -> tuple[Dataset, Dataset, list[str]]:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    top_path = Path(logs_dir) / "top10_roi_labels.json"
    if dummy_data:
        class_names = [f"roi_{i}" for i in range(data_cfg.get("num_roi_classes", 10))]
        top_path.parent.mkdir(parents=True, exist_ok=True)
        top_path.write_text(
            json.dumps({"classes": [{"label": c, "frequency": 1} for c in class_names]}, indent=2),
            encoding="utf-8",
        )
        train = DummyPadChestGRDataset("train", model_cfg["image_size"], model_cfg["grid_size"], model_cfg["max_rois"], class_names)
        val = DummyPadChestGRDataset("val", model_cfg["image_size"], model_cfg["grid_size"], model_cfg["max_rois"], class_names)
        return train, val, class_names
    records = load_annotation_records(data_cfg["annotations"])
    top = compute_top_roi_labels(records, data_cfg.get("train_split", "train"), data_cfg.get("num_roi_classes", 10))
    if not top:
        raise ValueError("No ROI labels found in the training split.")
    class_names = [entry["label"] for entry in top]
    top_path.parent.mkdir(parents=True, exist_ok=True)
    top_path.write_text(json.dumps({"classes": top}, indent=2, ensure_ascii=False), encoding="utf-8")
    train = PadChestGRRegionDataset(
        records,
        data_cfg["image_dir"],
        data_cfg.get("train_split", "train"),
        class_names,
        model_cfg["image_size"],
        model_cfg["grid_size"],
        model_cfg["max_rois"],
        train=True,
    )
    val = PadChestGRRegionDataset(
        records,
        data_cfg["image_dir"],
        data_cfg.get("val_split", "val"),
        class_names,
        model_cfg["image_size"],
        model_cfg["grid_size"],
        model_cfg["max_rois"],
        train=False,
    )
    if len(train) == 0:
        raise ValueError("No training images remain after top-10 ROI filtering.")
    return train, val, class_names
