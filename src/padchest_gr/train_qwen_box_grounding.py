from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .box_dataset import PadChestBoxDataset
from .gradcam import heatmap_iou, pointing_game
from .models import HeatmapProjectors, QwenHeatmapDecoder, QwenHiddenClassifier, multilabel_clip_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Qwen grounding model from 49x49 box/GradCAM heatmaps to box labels and heatmaps."
    )
    parser.add_argument("--boxes-tsv", type=Path, required=True)
    parser.add_argument("--box-vocab-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
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
    parser.add_argument("--lambda-label", type=float, default=1.0)
    parser.add_argument("--lambda-heatmap", type=float, default=1.0)
    parser.add_argument("--lambda-lm", type=float, default=1.0)
    parser.add_argument("--lambda-clip", type=float, default=1.0)
    return parser.parse_args()


def make_splits(dataset: PadChestBoxDataset, val_fraction: float) -> tuple[Subset, Subset]:
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
    return [tokenizer.convert_tokens_to_ids(token) for token in tokens]


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


def build_sequence(
    qwen: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    slots: torch.Tensor,
    label_token_ids: list[int],
    texts: list[str],
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    device = slots.device
    tokenized = tokenizer(
        [text + tokenizer.eos_token for text in texts],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    token_embed = qwen.get_input_embeddings()
    label_ids = torch.tensor(label_token_ids, device=device, dtype=torch.long)
    label_embeds = token_embed(label_ids).unsqueeze(0).expand(slots.size(0), -1, -1)
    prefix_parts = []
    for idx in range(slots.size(1)):
        prefix_parts.append(slots[:, idx : idx + 1, :])
        prefix_parts.append(label_embeds[:, idx : idx + 1, :])
    prefix = torch.cat(prefix_parts, dim=1)
    text_embeds = token_embed(tokenized.input_ids)
    embeds = torch.cat([prefix, text_embeds], dim=1)
    prefix_mask = torch.ones(prefix.size()[:2], dtype=tokenized.attention_mask.dtype, device=device)
    mask = torch.cat([prefix_mask, tokenized.attention_mask], dim=1)
    return embeds, mask, tokenized.input_ids, prefix.size(1), tokenized.attention_mask


def answer_only_lm_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    text_attention_mask: torch.Tensor,
    tokenizer: AutoTokenizer,
    prefix_len: int,
) -> torch.Tensor:
    batch_size, text_len = input_ids.shape
    labels_full = torch.full((batch_size, logits.size(1)), -100, dtype=torch.long, device=logits.device)
    labels_full[:, prefix_len : prefix_len + text_len] = input_ids
    labels_full[:, prefix_len : prefix_len + text_len][text_attention_mask == 0] = -100
    answer_marker = tokenizer.encode("A:", add_special_tokens=False)
    for row_idx in range(batch_size):
        start = find_subsequence(input_ids[row_idx].tolist(), answer_marker)
        if start >= 0:
            labels_full[row_idx, : prefix_len + start + len(answer_marker)] = -100
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
    qwen: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    projectors: HeatmapProjectors,
    classifier: QwenHiddenClassifier,
    decoder: QwenHeatmapDecoder,
    loader: DataLoader,
    label_token_ids: list[int],
    args: argparse.Namespace,
) -> dict[str, float]:
    qwen.eval()
    projectors.eval()
    classifier.eval()
    decoder.eval()
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    ious: list[float] = []
    hits: list[float] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["box_labels"].to(args.device)
            heatmaps = batch["box_heatmaps"].to(args.device)
            slots = projectors(heatmaps[:, :10])
            embeds, mask, _, prefix_len, _ = build_sequence(
                qwen, tokenizer, slots, label_token_ids, batch["prompt"], args.max_length
            )
            out = qwen(inputs_embeds=embeds, attention_mask=mask, output_hidden_states=True)
            hidden = out.hidden_states[-1][:, :prefix_len:2, :]
            logits_label = classifier(hidden)
            logits_heatmap = decoder(hidden)
            probs = torch.sigmoid(logits_label)
            pred_heatmaps = torch.sigmoid(logits_heatmap)
            all_scores.append(probs.cpu().numpy())
            all_labels.append(labels[:, :10].cpu().numpy())
            target = heatmaps[:, :10]
            positive = target.flatten(start_dim=2).sum(dim=-1) > 0
            if positive.any():
                iou = heatmap_iou(pred_heatmaps[positive], target[positive], threshold=0.5)
                hit = pointing_game(pred_heatmaps[positive], target[positive])
                ious.extend(iou.cpu().tolist())
                hits.extend(hit.cpu().tolist())

    y_true = np.concatenate(all_labels, axis=0)
    y_score = np.concatenate(all_scores, axis=0)
    y_pred = (y_score >= 0.5).astype(np.int64)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "mean_heatmap_iou": float(np.mean(ious)) if ious else 0.0,
        "pointing_hit": float(np.mean(hits)) if hits else 0.0,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = PadChestBoxDataset(args.boxes_tsv, args.box_vocab_json, args.image_root, args.image_size, train=True)
    eval_dataset = PadChestBoxDataset(args.boxes_tsv, args.box_vocab_json, args.image_root, args.image_size, train=False)
    train_subset, val_subset = make_splits(train_dataset, args.val_fraction)
    val_subset = Subset(eval_dataset, list(val_subset.indices))
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model_kwargs = {"torch_dtype": torch.bfloat16} if args.device == "cuda" else {}
    qwen = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs).to(args.device)
    qwen.config.use_cache = False
    label_token_ids = add_special_tokens(tokenizer, qwen, train_dataset.tokens[:10])
    special_params = enable_only_token_rows(qwen, label_token_ids)

    projectors = HeatmapProjectors(
        grid_size=train_dataset.grid_size,
        qwen_dim=qwen.config.hidden_size,
        num_labels=10,
        hidden_dim=args.projector_hidden_dim,
    ).to(args.device)
    classifier = QwenHiddenClassifier(qwen.config.hidden_size, num_outputs=10).to(args.device)
    decoder = QwenHeatmapDecoder(qwen.config.hidden_size, grid_size=train_dataset.grid_size).to(args.device)
    optimizer = torch.optim.AdamW(
        [
            {"params": projectors.parameters(), "lr": args.lr},
            {"params": classifier.parameters(), "lr": args.lr},
            {"params": decoder.parameters(), "lr": args.lr},
            {"params": special_params, "lr": args.token_lr},
        ]
    )
    label_loss = nn.BCEWithLogitsLoss()
    heatmap_loss = nn.BCEWithLogitsLoss()
    token_ids_tensor = torch.tensor(label_token_ids, device=args.device, dtype=torch.long)
    best_score = -1.0

    for epoch in range(1, args.epochs + 1):
        qwen.train()
        projectors.train()
        classifier.train()
        decoder.train()
        losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"Qwen boxes epoch {epoch}"):
            labels = batch["box_labels"].to(args.device)[:, :10]
            heatmaps = batch["box_heatmaps"].to(args.device)[:, :10]
            optimizer.zero_grad()
            slots = projectors(heatmaps)
            embeds, mask, input_ids, prefix_len, text_mask = build_sequence(
                qwen, tokenizer, slots, label_token_ids, batch["text"], args.max_length
            )
            out = qwen(inputs_embeds=embeds, attention_mask=mask, output_hidden_states=True)
            hidden = out.hidden_states[-1][:, :prefix_len:2, :]
            logits_label = classifier(hidden)
            logits_heatmap = decoder(hidden)

            loss_label = label_loss(logits_label.float(), labels)
            loss_heatmap = heatmap_loss(logits_heatmap.float(), heatmaps)
            token_embeds = qwen.get_input_embeddings().weight[token_ids_tensor]
            loss_clip = multilabel_clip_loss(slots, token_embeds, labels)
            loss_lm = answer_only_lm_loss(out.logits, input_ids, text_mask, tokenizer, prefix_len)
            loss = (
                args.lambda_label * loss_label
                + args.lambda_heatmap * loss_heatmap
                + args.lambda_clip * loss_clip
                + args.lambda_lm * loss_lm
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projectors.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        metrics = evaluate(qwen, tokenizer, projectors, classifier, decoder, val_loader, label_token_ids, args)
        metrics["train_loss"] = float(np.mean(losses))
        print(json.dumps({"epoch": epoch, **metrics}, indent=2))
        score = metrics["macro_f1"] + metrics["pointing_hit"]
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "projectors": projectors.state_dict(),
                    "classifier": classifier.state_dict(),
                    "decoder": decoder.state_dict(),
                    "label_tokens": train_dataset.tokens[:10],
                    "metrics": metrics,
                    "epoch": epoch,
                },
                args.out_dir / "best.pt",
            )


if __name__ == "__main__":
    main()
