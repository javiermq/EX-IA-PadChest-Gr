from __future__ import annotations

from collections import deque

import torch


def box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def normalize_box(box: list[float], width: int | None = None, height: int | None = None) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    if max(x1, y1, x2, y2) > 1.5 and width and height:
        x1, x2 = x1 / width, x2 / width
        y1, y2 = y1 / height, y2 / height
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    return [x1, y1, x2, y2]


def boxes_to_heatmaps(
    boxes: torch.Tensor,
    class_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    num_classes: int,
    grid_size: int,
) -> torch.Tensor:
    heatmaps = torch.zeros(num_classes, grid_size, grid_size, dtype=torch.float32)
    for box, class_id, valid in zip(boxes, class_ids, valid_mask):
        if not bool(valid):
            continue
        c = int(class_id.item())
        x1, y1, x2, y2 = box.tolist()
        gx1 = max(0, min(grid_size - 1, int(torch.floor(torch.tensor(x1 * grid_size)).item())))
        gy1 = max(0, min(grid_size - 1, int(torch.floor(torch.tensor(y1 * grid_size)).item())))
        gx2 = max(0, min(grid_size - 1, int(torch.ceil(torch.tensor(x2 * grid_size)).item()) - 1))
        gy2 = max(0, min(grid_size - 1, int(torch.ceil(torch.tensor(y2 * grid_size)).item()) - 1))
        heatmaps[c, gy1 : gy2 + 1, gx1 : gx2 + 1] = 1.0
    return heatmaps


def heatmaps_to_boxes(
    heatmaps: torch.Tensor,
    max_rois: int,
    threshold: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert [C,G,G] logits/probabilities into top boxes, labels and scores."""
    if heatmaps.ndim != 3:
        raise ValueError("heatmaps must be [C,G,G]")
    probs = heatmaps.sigmoid() if heatmaps.min() < 0 or heatmaps.max() > 1 else heatmaps
    c_count, grid, _ = probs.shape
    candidates: list[tuple[float, int, list[float]]] = []
    for c in range(c_count):
        mask = probs[c] >= threshold
        visited = torch.zeros_like(mask, dtype=torch.bool)
        for y in range(grid):
            for x in range(grid):
                if not bool(mask[y, x]) or bool(visited[y, x]):
                    continue
                q: deque[tuple[int, int]] = deque([(y, x)])
                visited[y, x] = True
                cells: list[tuple[int, int]] = []
                while q:
                    cy, cx = q.popleft()
                    cells.append((cy, cx))
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < grid and 0 <= nx < grid and bool(mask[ny, nx]) and not bool(visited[ny, nx]):
                            visited[ny, nx] = True
                            q.append((ny, nx))
                ys = [p[0] for p in cells]
                xs = [p[1] for p in cells]
                score = float(probs[c, ys, xs].mean().item())
                box = [min(xs) / grid, min(ys) / grid, (max(xs) + 1) / grid, (max(ys) + 1) / grid]
                candidates.append((score, c, box))
    if not candidates:
        flat_idx = int(probs.argmax().item())
        c = flat_idx // (grid * grid)
        rem = flat_idx % (grid * grid)
        y, x = rem // grid, rem % grid
        candidates.append((float(probs[c, y, x].item()), c, [x / grid, y / grid, (x + 1) / grid, (y + 1) / grid]))
    candidates.sort(key=lambda item: item[0], reverse=True)
    candidates = candidates[:max_rois]
    boxes = torch.zeros(max_rois, 4)
    labels = torch.full((max_rois,), -1, dtype=torch.long)
    scores = torch.zeros(max_rois)
    for i, (score, label, box) in enumerate(candidates):
        boxes[i] = torch.tensor(box)
        labels[i] = label
        scores[i] = score
    return boxes, labels, scores


def pairwise_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros(a.shape[0], b.shape[0], device=a.device)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0))[:, None]
    area_b = ((b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0))[None, :]
    return inter / (area_a + area_b - inter + 1e-6)
