from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


QUESTION_VARIANTS = (
    "What does the chest x-ray show?",
    "Which findings are visible?",
    "What abnormalities are present?",
    "Name the radiological findings.",
    "What is present in the image?",
)


class PadChestCategoryDataset(Dataset):
    def __init__(
        self,
        manifest_tsv: str | Path,
        vocab_json: str | Path,
        image_root: str | Path | None = None,
        image_size: int = 384,
        train: bool = False,
    ) -> None:
        self.df = pd.read_csv(manifest_tsv, sep="\t").fillna("")
        self.vocab = json.loads(Path(vocab_json).read_text(encoding="utf-8"))
        self.image_root = Path(image_root) if image_root else None
        self.label_cols = list(self.vocab["label_columns"])
        self.top10_ids = [entry["id"] for entry in self.vocab["categories"]]
        self.top10_tokens = [entry["token"] for entry in self.vocab["categories"]]
        self.other_token = self.vocab["other"]["token"]

        missing = [column for column in self.label_cols if column not in self.df.columns]
        if missing:
            raise ValueError(f"Missing label columns: {missing}")
        if "image_path" not in self.df.columns:
            raise ValueError("Manifest must contain image_path for image training.")

        self.transform = build_transforms(image_size=image_size, train=train)
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.df.iloc[idx]
        image_path = Path(str(row["image_path"]))
        if self.image_root and not image_path.is_absolute():
            image_path = self.image_root / image_path
        image = Image.open(image_path).convert("RGB")
        labels = torch.tensor(row[self.label_cols].astype(float).to_numpy(), dtype=torch.float32)
        target_tokens = [token for token in str(row.get("target_tokens", "")).split("|") if token]
        if not target_tokens:
            target_tokens = [self.other_token]

        question = random.choice(QUESTION_VARIANTS) if self.train else QUESTION_VARIANTS[0]
        answer = " ".join(target_tokens)
        text = f"Q: {question}\nA: {answer}"
        prompt = f"Q: {question}\nA:"

        return {
            "image": self.transform(image),
            "labels": labels,
            "text": text,
            "prompt": prompt,
            "answer": answer,
            "image_id": str(row.get("image_id", idx)),
        }


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    steps = []
    if train:
        steps.extend(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
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
