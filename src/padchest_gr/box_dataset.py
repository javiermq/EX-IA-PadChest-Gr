from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .dataset import build_transforms


BOX_QUESTION_VARIANTS = (
    "Where are the localized findings?",
    "Which regions contain annotated findings?",
    "Locate the radiological box labels.",
    "What boxed findings are visible and where?",
)


class PadChestBoxDataset(Dataset):
    def __init__(
        self,
        boxes_tsv: str | Path,
        vocab_json: str | Path,
        image_root: str | Path | None = None,
        image_size: int = 384,
        train: bool = False,
    ) -> None:
        self.boxes = pd.read_csv(boxes_tsv, sep="\t").fillna("")
        self.vocab = json.loads(Path(vocab_json).read_text(encoding="utf-8"))
        self.image_root = Path(image_root) if image_root else None
        self.image_size = image_size
        self.grid_size = int(self.vocab["grid_size"])
        self.labels = list(self.vocab["labels"]) + [self.vocab["other"]]
        self.label_ids = [entry["id"] for entry in self.labels]
        self.tokens = [entry["token"] for entry in self.labels]
        self.transform = build_transforms(image_size=image_size, train=train)
        self.train = train

        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in self.boxes.to_dict("records"):
            grouped[str(row["image_id"])].append(row)
        self.image_ids = sorted(grouped)
        self.grouped = grouped

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, object]:
        image_id = self.image_ids[idx]
        rows = self.grouped[image_id]
        first = rows[0]
        image_path_value = str(first.get("image_path", "") or "")
        if image_path_value:
            image_path = Path(image_path_value)
            if self.image_root and not image_path.is_absolute():
                image_path = self.image_root / image_path
            image = self.transform(Image.open(image_path).convert("RGB"))
        else:
            image = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)

        labels = torch.zeros(len(self.label_ids), dtype=torch.float32)
        heatmaps = torch.zeros(len(self.label_ids), self.grid_size, self.grid_size, dtype=torch.float32)
        tokens: list[str] = []
        for row in rows:
            label_id = str(row["box_label_id"])
            if label_id not in self.label_ids:
                label_id = "other"
            label_idx = self.label_ids.index(label_id)
            labels[label_idx] = 1.0
            y0, y1 = int(row["grid_y_min"]), int(row["grid_y_max"])
            x0, x1 = int(row["grid_x_min"]), int(row["grid_x_max"])
            heatmaps[label_idx, y0 : y1 + 1, x0 : x1 + 1] = 1.0
            token = self.tokens[label_idx]
            if label_id != "other" and token not in tokens:
                tokens.append(token)

        question = random.choice(BOX_QUESTION_VARIANTS) if self.train else BOX_QUESTION_VARIANTS[0]
        answer = " ".join(tokens) if tokens else "none"
        return {
            "image": image,
            "box_labels": labels,
            "box_heatmaps": heatmaps,
            "text": f"Q: {question}\nA: {answer}",
            "prompt": f"Q: {question}\nA:",
            "answer": answer,
            "image_id": image_id,
        }
