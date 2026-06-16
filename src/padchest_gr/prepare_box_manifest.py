from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .io_utils import (
    BOX_LABEL_COLUMN_CANDIDATES,
    HEIGHT_COLUMN_CANDIDATES,
    IMAGE_ID_COLUMN_CANDIDATES,
    IMAGE_PATH_COLUMN_CANDIDATES,
    WIDTH_COLUMN_CANDIDATES,
    X_MAX_COLUMN_CANDIDATES,
    X_MIN_COLUMN_CANDIDATES,
    Y_MAX_COLUMN_CANDIDATES,
    Y_MIN_COLUMN_CANDIDATES,
    detect_column,
    read_table,
    write_table,
)
from .taxonomy import category_token


GRID_SIZE = 49


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PadChest-GR box labels and 49x49 heatmap targets.")
    parser.add_argument("--boxes", type=Path, required=True, help="Input box annotation CSV/TSV/JSONL.")
    parser.add_argument("--out-tsv", type=Path, required=True)
    parser.add_argument("--vocab-out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--image-id-column", default=None)
    parser.add_argument("--image-path-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--x-min-column", default=None)
    parser.add_argument("--y-min-column", default=None)
    parser.add_argument("--x-max-column", default=None)
    parser.add_argument("--y-max-column", default=None)
    parser.add_argument("--width-column", default=None)
    parser.add_argument("--height-column", default=None)
    parser.add_argument("--image-width-column", default=None)
    parser.add_argument("--image-height-column", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_table(args.boxes).fillna("")

    image_id_col = args.image_id_column or detect_column(df, IMAGE_ID_COLUMN_CANDIDATES, "image id")
    image_path_col = args.image_path_column or detect_column(
        df, IMAGE_PATH_COLUMN_CANDIDATES, "image path", required=False
    )
    label_col = args.label_column or detect_column(df, BOX_LABEL_COLUMN_CANDIDATES, "box label")
    x_min_col = args.x_min_column or detect_column(df, X_MIN_COLUMN_CANDIDATES, "x min")
    y_min_col = args.y_min_column or detect_column(df, Y_MIN_COLUMN_CANDIDATES, "y min")

    x_max_col = args.x_max_column or detect_column(df, X_MAX_COLUMN_CANDIDATES, "x max", required=False)
    y_max_col = args.y_max_column or detect_column(df, Y_MAX_COLUMN_CANDIDATES, "y max", required=False)
    width_col = args.width_column or detect_column(df, WIDTH_COLUMN_CANDIDATES, "box width", required=False)
    height_col = args.height_column or detect_column(df, HEIGHT_COLUMN_CANDIDATES, "box height", required=False)

    if not x_max_col and not width_col:
        raise ValueError("Need either x_max or width column.")
    if not y_max_col and not height_col:
        raise ValueError("Need either y_max or height column.")

    labels = [str(value).strip() for value in df[label_col].tolist() if str(value).strip()]
    top_labels = [label for label, _ in Counter(labels).most_common(args.top_k)]
    top_label_ids = [make_label_id(label) for label in top_labels]
    top_label_set = set(top_labels)

    rows: list[dict[str, object]] = []
    for row in df.to_dict("records"):
        raw_label = str(row[label_col]).strip()
        label = raw_label if raw_label in top_label_set else "Other"
        label_id = make_label_id(label)

        x_min = float(row[x_min_col])
        y_min = float(row[y_min_col])
        x_max = float(row[x_max_col]) if x_max_col else x_min + float(row[width_col])
        y_max = float(row[y_max_col]) if y_max_col else y_min + float(row[height_col])

        image_width = get_optional_float(row, args.image_width_column)
        image_height = get_optional_float(row, args.image_height_column)
        gx_min, gy_min, gx_max, gy_max = box_to_grid(x_min, y_min, x_max, y_max, image_width, image_height)

        out = {
            "image_id": str(row[image_id_col]),
            "box_label_raw": raw_label,
            "box_label": label,
            "box_label_id": label_id,
            "box_token": category_token(f"box_{label_id}"),
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "grid_x_min": gx_min,
            "grid_y_min": gy_min,
            "grid_x_max": gx_max,
            "grid_y_max": gy_max,
        }
        if image_path_col:
            out["image_path"] = str(row.get(image_path_col, "") or "")
        rows.append(out)

    vocab = {
        "top_k": args.top_k,
        "grid_size": GRID_SIZE,
        "labels": [
            {"name": label, "id": label_id, "token": category_token(f"box_{label_id}")}
            for label, label_id in zip(top_labels, top_label_ids)
        ],
        "other": {"name": "Other", "id": "other", "token": category_token("box_other")},
        "label_columns": [f"label_box_{label_id}" for label_id in top_label_ids] + ["label_box_other"],
        "source_columns": {
            "image_id": image_id_col,
            "image_path": image_path_col,
            "label": label_col,
            "x_min": x_min_col,
            "y_min": y_min_col,
            "x_max": x_max_col,
            "y_max": y_max_col,
            "width": width_col,
            "height": height_col,
        },
    }

    out_df = pd.DataFrame(rows)
    write_table(out_df, args.out_tsv)
    args.vocab_out.parent.mkdir(parents=True, exist_ok=True)
    args.vocab_out.write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    print("Top box labels:")
    for label in top_labels:
        print(f"  {label}: {labels.count(label)}")
    print(f"Wrote {len(out_df)} boxes to {args.out_tsv}")
    print(f"Wrote vocabulary to {args.vocab_out}")


def get_optional_float(row: dict[str, object], column: str | None) -> float | None:
    if not column:
        return None
    value = row.get(column, "")
    if value == "":
        return None
    return float(value)


def box_to_grid(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    image_width: float | None,
    image_height: float | None,
) -> tuple[int, int, int, int]:
    if image_width and image_height and max(x_max, y_max) > 1.5:
        x_min /= image_width
        x_max /= image_width
        y_min /= image_height
        y_max /= image_height

    x0 = clamp_grid(int(x_min * GRID_SIZE))
    y0 = clamp_grid(int(y_min * GRID_SIZE))
    x1 = clamp_grid(int(x_max * GRID_SIZE))
    y1 = clamp_grid(int(y_max * GRID_SIZE))
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def clamp_grid(value: int) -> int:
    return max(0, min(GRID_SIZE - 1, value))


def make_label_id(label: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "_" for ch in label.strip()]
    out = "".join(chars)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "unknown"


if __name__ == "__main__":
    main()
