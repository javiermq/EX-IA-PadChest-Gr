from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dataset import PadChestCategoryDataset
from .models import CategoryProjectors, DenseNetMultilabel, QwenHiddenClassifier, multilabel_clip_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PadChest-GR Qwen category-token model with 10 visual MLP projectors."
    )
    parser.add_argument("--manifest-tsv", type=Path, required=True)
    parser.add_argument("--vocab-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--densenet-checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--token-lr", type=float, default=1e-5)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--projector-hidden-dim", type=int, default=1024)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-densenet", action="store_true")
    parser.add_argument("--no-pretrained-densenet", action="store_true")
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--lambda-lm", type=float, default=1.0)
    parser.add_argument("--lambda-clip", type=float, default=1.0)
    return parser.parse_args()


def make_splits(dataset: PadChestCategoryDataset, val_fraction: float) -> tuple[Subset, Subset]:
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))


def add_special_tokens(tokenizer: AutoTokenizer, qwen: AutoModelForCausalLM, tokens: list[str]) -> list[int]:
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "right"
    n_added = tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    if n_added > 0:
        qwen.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
    if any(token_id < 0 for token_id in token_ids):
        raise RuntimeError("At least one special category token was not added to the tokenizer.")
    return token_ids


def enable_only_token_rows(qwen: AutoModelForCausalLM, special_ids: list[int]) -> list[nn.Parameter]:
    qwen.requires_grad_(False)
    ids = torch.tensor(special_ids, dtype=torch.long)
    params: list[nn.Parameter] = []

    def make_hook(weight: torch.Tensor):
        row_ids = ids.to(weight.device)

        def hook(grad: torch.Tensor) -> torch.Tensor:
            mask = torch.zeros_like(grad)
            mask.index_fill_(0, row_ids, 1)
            return grad * mask

        return hook

    for embedding in [qwen.get_input_embeddings(), qwen.get_output_embeddings()]:
        if embedding is None:
            continue
        weight = embedding.weight
        if any(weight is param for param in params):
            continue
        weight.requires_grad_(True)
        weight.register_hook(make_hook(weight))
        params.append(weight)
    return params


def load_densenet(args: argparse.Namespace, num_classes: int, feature_dim_only: bool = False) -> DenseNetMultilabel:
    model = DenseNetMultilabel(num_classes=num_classes, pretrained=not args.no_pretrained_densenet)
    if args.densenet_checkpoint:
        checkpoint = torch.load(args.densenet_checkpoint, map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        model.load_state_dict(state, strict=False)
    if args.freeze_densenet:
        model.requires_grad_(False)
    return model


def build_sequence(
    qwen: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    visual_slots: torch.Tensor,
    category_token_ids: list[int],
    texts: list[str],
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    device = visual_slots.device
    tokenized = tokenizer(
        [text + tokenizer.eos_token for text in texts],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    token_embed = qwen.get_input_embeddings()
    class_ids = torch.tensor(category_token_ids, device=device, dtype=torch.long)
    class_embeds = token_embed(class_ids).unsqueeze(0).expand(visual_slots.size(0), -1, -1)

    prefix_parts = []
    for idx in range(visual_slots.size(1)):
        prefix_parts.append(visual_slots[:, idx : idx + 1, :])
        prefix_parts.append(class_embeds[:, idx : idx + 1, :])
    prefix = torch.cat(prefix_parts, dim=1)
    text_embeds = token_embed(tokenized.input_ids)
    embeds = torch.cat([prefix, text_embeds], dim=1)

    prefix_mask = torch.ones(prefix.size()[:2], dtype=tokenized.attention_mask.dtype, device=device)
    attention_mask = torch.cat([prefix_mask, tokenized.attention_mask], dim=1)
    return embeds, attention_mask, tokenized.input_ids, prefix.size(1), tokenized.attention_mask


def answer_only_lm_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    text_attention_mask: torch.Tensor,
    tokenizer: AutoTokenizer,
    prefix_len: int,
) -> torch.Tensor:
    batch_size, text_len = input_ids.shape
    full_len = logits.size(1)
    labels_full = torch.full((batch_size, full_len), -100, dtype=torch.long, device=logits.device)
    labels_full[:, prefix_len : prefix_len + text_len] = input_ids
    labels_full[:, prefix_len : prefix_len + text_len][text_attention_mask == 0] = -100

    answer_marker = tokenizer.encode("A:", add_special_tokens=False)
    for row_idx in range(batch_size):
        ids = input_ids[row_idx].tolist()
        answer_start = find_subsequence(ids, answer_marker)
        if answer_start >= 0:
            cut = prefix_len + answer_start + len(answer_marker)
            labels_full[row_idx, :cut] = -100

    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels_full[:, 1:].contiguous()
    valid = shift_labels.ne(-100).sum()
    if valid.item() == 0:
        return torch.zeros((), device=logits.device)

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss / valid / math.log(shift_logits.size(-1))


def find_subsequence(values: list[int], pattern: list[int]) -> int:
    for idx in range(0, len(values) - len(pattern) + 1):
        if values[idx : idx + len(pattern)] == pattern:
            return idx
    return -1


def evaluate(
    densenet: DenseNetMultilabel,
    projectors: CategoryProjectors,
    classifier: QwenHiddenClassifier,
    qwen: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    loader: DataLoader,
    category_token_ids: list[int],
    args: argparse.Namespace,
) -> dict[str, float]:
    densenet.eval()
    projectors.eval()
    classifier.eval()
    qwen.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(args.device)
            y = batch["labels"].to(args.device)
            features = densenet.forward_features(images)
            visual_slots = projectors(features)
            embeds, mask, _, prefix_len, _ = build_sequence(
                qwen, tokenizer, visual_slots, category_token_ids, batch["prompt"], args.max_length
            )
            out = qwen(inputs_embeds=embeds, attention_mask=mask, output_hidden_states=True)
            category_hidden = out.hidden_states[-1][:, :prefix_len:2, :]
            logits = classifier(category_hidden)
            scores.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.cpu().numpy())

    y_true = np.concatenate(labels, axis=0)
    y_score = np.concatenate(scores, axis=0)
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

    train_dataset = PadChestCategoryDataset(
        args.manifest_tsv, args.vocab_json, args.image_root, args.image_size, train=True
    )
    eval_dataset = PadChestCategoryDataset(
        args.manifest_tsv, args.vocab_json, args.image_root, args.image_size, train=False
    )
    train_subset, val_subset = make_splits(train_dataset, args.val_fraction)
    val_subset = Subset(eval_dataset, list(val_subset.indices))
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model_kwargs = {}
    if args.device == "cuda":
        model_kwargs["torch_dtype"] = torch.bfloat16
    qwen = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    qwen.to(args.device)
    qwen.config.use_cache = False

    all_tokens = train_dataset.top10_tokens + [train_dataset.other_token]
    all_token_ids = add_special_tokens(tokenizer, qwen, all_tokens)
    top10_token_ids = all_token_ids[:10]
    special_params = enable_only_token_rows(qwen, all_token_ids)

    densenet = load_densenet(args, num_classes=len(train_dataset.label_cols)).to(args.device)
    projectors = CategoryProjectors(
        densenet.feature_dim,
        qwen.config.hidden_size,
        num_categories=10,
        hidden_dim=args.projector_hidden_dim,
    ).to(args.device)
    classifier = QwenHiddenClassifier(qwen.config.hidden_size, len(train_dataset.label_cols)).to(args.device)

    params = [
        {"params": projectors.parameters(), "lr": args.lr},
        {"params": classifier.parameters(), "lr": args.lr},
        {"params": special_params, "lr": args.token_lr},
    ]
    if not args.freeze_densenet:
        params.append({"params": densenet.parameters(), "lr": args.lr})
    optimizer = torch.optim.AdamW(params)
    criterion_cls = nn.BCEWithLogitsLoss()
    best_macro_f1 = -1.0

    category_ids = torch.tensor(top10_token_ids, device=args.device, dtype=torch.long)
    for epoch in range(1, args.epochs + 1):
        densenet.train(not args.freeze_densenet)
        projectors.train()
        classifier.train()
        qwen.train()
        losses: list[float] = []

        for batch in tqdm(train_loader, desc=f"Qwen epoch {epoch}"):
            images = batch["image"].to(args.device)
            labels = batch["labels"].to(args.device)
            labels_top10 = labels[:, :10]

            optimizer.zero_grad()
            features = densenet.forward_features(images)
            visual_slots = projectors(features)
            embeds, mask, input_ids, prefix_len, text_mask = build_sequence(
                qwen, tokenizer, visual_slots, top10_token_ids, batch["text"], args.max_length
            )
            out = qwen(inputs_embeds=embeds, attention_mask=mask, output_hidden_states=True)

            category_hidden = out.hidden_states[-1][:, :prefix_len:2, :]
            logits_cls = classifier(category_hidden)
            loss_cls = criterion_cls(logits_cls.float(), labels)

            token_embeds = qwen.get_input_embeddings().weight[category_ids]
            loss_clip = multilabel_clip_loss(visual_slots, token_embeds, labels_top10)
            loss_lm = answer_only_lm_loss(out.logits, input_ids, text_mask, tokenizer, prefix_len)

            loss = args.lambda_cls * loss_cls + args.lambda_clip * loss_clip + args.lambda_lm * loss_lm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projectors.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            if not args.freeze_densenet:
                torch.nn.utils.clip_grad_norm_(densenet.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        metrics = evaluate(
            densenet, projectors, classifier, qwen, tokenizer, val_loader, top10_token_ids, args
        )
        metrics["train_loss"] = float(np.mean(losses))
        print(json.dumps({"epoch": epoch, **metrics}, indent=2))
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            torch.save(
                {
                    "densenet": densenet.state_dict(),
                    "projectors": projectors.state_dict(),
                    "classifier": classifier.state_dict(),
                    "label_cols": train_dataset.label_cols,
                    "top10_tokens": train_dataset.top10_tokens,
                    "other_token": train_dataset.other_token,
                    "metrics": metrics,
                    "epoch": epoch,
                },
                args.out_dir / "best.pt",
            )


if __name__ == "__main__":
    main()
