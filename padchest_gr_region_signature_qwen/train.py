from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from region_signature.config import load_config
from region_signature.dataset import collate_region_batch, make_datasets
from region_signature.losses import compute_loss
from region_signature.metrics import MetricsAccumulator
from region_signature.model import RegionSignatureQwenPrototype
from region_signature.visualization import save_epoch_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dummy_data", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--sample_every_n_batches", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def resolve_device(requested: str | None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detach_output(output):
    for name, value in output.__dict__.items():
        if torch.is_tensor(value):
            setattr(output, name, value.detach().cpu())
    return output


def quick_random_eval(
    model,
    dataset,
    cfg,
    device,
    class_names: list[str],
    epoch: int,
    global_step: int,
    train_loss_so_far: float,
) -> None:
    if len(dataset) == 0:
        return
    monitor_cfg = cfg.get("monitoring", {})
    sample_n = min(int(monitor_cfg.get("sample_num_examples", 3)), len(dataset))
    indices = random.sample(range(len(dataset)), k=sample_n)
    batch = collate_region_batch([dataset[i] for i in indices])
    was_training = model.training
    model.eval()
    with torch.no_grad():
        batch = move_batch(batch, device)
        output = model(batch["image"], batch["boxes"], batch["roi_valid"], qwen_images=batch.get("pil_image"))
        loss, _ = compute_loss(output, batch, cfg["loss"])
        metrics = MetricsAccumulator(class_names, cfg["model"]["max_rois"])
        metrics.update(output, batch)
    row = {
        "epoch": epoch,
        "global_step": global_step,
        "train_loss_so_far": train_loss_so_far,
        "sample_loss": float(loss.detach().cpu()),
        **metrics.compute(),
    }
    append_metrics(Path(cfg["outputs"]["logs_dir"]) / "batch_metrics.csv", row)
    if bool(monitor_cfg.get("save_batch_visuals", True)):
        cpu_batch = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}
        save_epoch_examples(
            detach_output(output),
            cpu_batch,
            class_names,
            Path(cfg["outputs"]["outputs_dir"]) / "batch_samples" / f"epoch_{epoch:03d}_step_{global_step:06d}",
            limit=sample_n,
        )
    model.train(was_training)


def run_epoch(
    model,
    loader,
    optimizer,
    cfg,
    device,
    train: bool,
    epoch: int = 0,
    val_dataset=None,
    class_names: list[str] | None = None,
) -> tuple[float, dict[str, float], dict | None, object | None]:
    model.train(train)
    total_loss = 0.0
    metrics = MetricsAccumulator(loader.dataset.class_names, cfg["model"]["max_rois"])
    first_batch = None
    first_output = None
    sample_every = int(cfg.get("monitoring", {}).get("sample_every_n_batches", 0) or 0)
    iterator = tqdm(loader, desc="train" if train else "val", leave=False)
    for step, batch in enumerate(iterator):
        global_step = step + 1
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(train):
            output = model(batch["image"], batch["boxes"], batch["roi_valid"], qwen_images=batch.get("pil_image"))
            loss, parts = compute_loss(output, batch, cfg["loss"])
            if train:
                loss = loss / int(cfg["training"]["grad_accum_steps"])
                loss.backward()
                if (step + 1) % int(cfg["training"]["grad_accum_steps"]) == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().cpu()) * (int(cfg["training"]["grad_accum_steps"]) if train else 1)
        metrics.update(output, batch)
        iterator.set_postfix(loss=f"{total_loss / (step + 1):.4f}")
        if first_batch is None:
            first_batch = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}
            first_output = detach_output(output)
        if (
            train
            and sample_every > 0
            and global_step % sample_every == 0
            and val_dataset is not None
            and class_names is not None
        ):
            quick_random_eval(
                model,
                val_dataset,
                cfg,
                device,
                class_names,
                epoch=epoch,
                global_step=global_step,
                train_loss_so_far=total_loss / (step + 1),
            )
    if train:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return total_loss / max(1, len(loader)), metrics.compute(), first_batch, first_output


def append_metrics(path: Path, row: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.sample_every_n_batches is not None:
        cfg.setdefault("monitoring", {})["sample_every_n_batches"] = args.sample_every_n_batches
    if args.device is not None:
        cfg["training"]["device"] = args.device
    if args.dummy_data:
        cfg["model"]["densenet_pretrained"] = False
        cfg["model"]["load_qwen_weights"] = False
        cfg["model"]["require_qwen"] = False
    seed = int(cfg["training"].get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logs_dir = Path(cfg["outputs"]["logs_dir"])
    out_root = Path(cfg["outputs"]["outputs_dir"])
    ckpt_dir = Path(cfg["outputs"]["checkpoints_dir"])
    train_ds, val_ds, class_names = make_datasets(cfg, args.dummy_data, logs_dir)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
        collate_fn=collate_region_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        collate_fn=collate_region_batch,
    )
    device = resolve_device(cfg["training"].get("device"))
    model = RegionSignatureQwenPrototype(cfg["model"], len(class_names))
    if cfg["model"].get("load_in_4bit", False) and cfg["model"].get("load_qwen_weights", False):
        model.move_task_modules(device)
    else:
        model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]))
    best_val = float("inf")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        train_loss, train_metrics, _, _ = run_epoch(
            model,
            train_loader,
            optimizer,
            cfg,
            device,
            train=True,
            epoch=epoch,
            val_dataset=val_ds,
            class_names=class_names,
        )
        val_loss, val_metrics, first_batch, first_output = run_epoch(model, val_loader, optimizer, cfg, device, train=False)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        append_metrics(logs_dir / "metrics.csv", row)
        if first_batch is not None and first_output is not None:
            save_epoch_examples(first_output, first_batch, class_names, out_root / f"epoch_{epoch:03d}")
        state = {"model": model.state_dict(), "cfg": cfg, "class_names": class_names, "epoch": epoch}
        torch.save(state, ckpt_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, ckpt_dir / "best.pt")
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_roi_recall@K={val_metrics['roi_recall_at_k']:.3f}")


if __name__ == "__main__":
    main()
