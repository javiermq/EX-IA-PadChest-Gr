from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from .dataset import PadChestCategoryDataset
from .models import DenseNetMultilabel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DenseNet121 baseline for PadChest-GR top-10 + Other.")
    parser.add_argument("--manifest-tsv", type=Path, required=True)
    parser.add_argument("--vocab-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def make_splits(dataset: PadChestCategoryDataset, val_fraction: float) -> tuple[Subset, Subset]:
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_score = 1.0 / (1.0 + np.exp(-np.concatenate(all_logits, axis=0)))
    y_pred = (y_score >= 0.5).astype(np.int64)
    metrics = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }
    try:
        metrics["macro_auroc"] = float(roc_auc_score(y_true, y_score, average="macro"))
    except ValueError:
        metrics["macro_auroc"] = float("nan")
    return metrics


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    full_train = PadChestCategoryDataset(
        args.manifest_tsv,
        args.vocab_json,
        image_root=args.image_root,
        image_size=args.image_size,
        train=True,
    )
    full_eval = PadChestCategoryDataset(
        args.manifest_tsv,
        args.vocab_json,
        image_root=args.image_root,
        image_size=args.image_size,
        train=False,
    )
    train_subset, val_subset = make_splits(full_train, args.val_fraction)
    val_indices = list(val_subset.indices)
    val_subset = Subset(full_eval, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    num_classes = len(full_train.label_cols)
    model = DenseNetMultilabel(num_classes=num_classes, pretrained=not args.no_pretrained).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    best_macro_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images = batch["image"].to(args.device)
            labels = batch["labels"].to(args.device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        metrics = evaluate(model, val_loader, args.device)
        metrics["train_loss"] = float(np.mean(losses))
        print(json.dumps({"epoch": epoch, **metrics}, indent=2))

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "label_cols": full_train.label_cols,
                    "metrics": metrics,
                    "epoch": epoch,
                },
                args.out_dir / "best.pt",
            )


if __name__ == "__main__":
    main()
