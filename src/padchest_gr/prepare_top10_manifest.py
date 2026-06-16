from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .io_utils import (
    CATEGORY_COLUMN_CANDIDATES,
    IMAGE_ID_COLUMN_CANDIDATES,
    IMAGE_PATH_COLUMN_CANDIDATES,
    detect_column,
    read_table,
    split_categories,
    write_table,
)
from .taxonomy import OTHER_CATEGORY, category_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a PadChest-GR manifest with the top-10 category annotations plus Other."
    )
    parser.add_argument("--annotations", type=Path, required=True, help="Input category annotation CSV/TSV/JSONL.")
    parser.add_argument("--out-tsv", type=Path, required=True, help="Output manifest TSV/CSV.")
    parser.add_argument("--vocab-out", type=Path, required=True, help="Output JSON vocabulary.")
    parser.add_argument("--image-id-column", default=None)
    parser.add_argument("--image-path-column", default=None)
    parser.add_argument("--category-column", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--other-name", default=OTHER_CATEGORY.name)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_table(args.annotations).fillna("")

    image_id_col = args.image_id_column or detect_column(df, IMAGE_ID_COLUMN_CANDIDATES, "image id")
    image_path_col = args.image_path_column or detect_column(
        df, IMAGE_PATH_COLUMN_CANDIDATES, "image path", required=False
    )
    category_col = args.category_column or detect_column(df, CATEGORY_COLUMN_CANDIDATES, "category")

    per_image_categories: dict[str, set[str]] = defaultdict(set)
    per_image_path: dict[str, str] = {}
    counts: Counter[str] = Counter()

    for row in df.to_dict("records"):
        image_id = str(row[image_id_col])
        categories = [category for category in split_categories(row[category_col]) if category]
        for category in categories:
            if category.lower() != args.other_name.lower():
                counts[category] += 1
                per_image_categories[image_id].add(category)
            else:
                per_image_categories[image_id].add(args.other_name)

        if image_path_col:
            image_path = str(row.get(image_path_col, "") or "")
            if image_path:
                per_image_path[image_id] = image_path

    top_categories = [category for category, _ in counts.most_common(args.top_k)]
    category_ids = [make_category_id(category) for category in top_categories]
    other_id = OTHER_CATEGORY.column
    all_ids = category_ids + [other_id]

    rows: list[dict[str, object]] = []
    for image_id, raw_categories in sorted(per_image_categories.items()):
        selected: list[str] = []
        has_other = False

        for category in sorted(raw_categories):
            if category in top_categories:
                selected.append(category)
            else:
                has_other = True

        out = {
            "image_id": image_id,
            "categories_top10": "|".join(selected),
            "has_other": int(has_other),
            "target_categories": "|".join(selected + ([args.other_name] if has_other else [])),
            "target_category_ids": "|".join(
                [make_category_id(category) for category in selected] + ([other_id] if has_other else [])
            ),
            "target_tokens": "|".join(
                [category_token(make_category_id(category)) for category in selected]
                + ([category_token(other_id)] if has_other else [])
            ),
        }
        if image_id in per_image_path:
            out["image_path"] = per_image_path[image_id]
        for category, category_id in zip(top_categories, category_ids):
            out[f"label_{category_id}"] = int(category in selected)
        out[f"label_{other_id}"] = int(has_other)
        rows.append(out)

    vocab = {
        "top_k": args.top_k,
        "categories": [
            {"name": category, "id": category_id, "token": category_token(category_id)}
            for category, category_id in zip(top_categories, category_ids)
        ],
        "other": {"name": args.other_name, "id": other_id, "token": category_token(other_id)},
        "label_columns": [f"label_{category_id}" for category_id in all_ids],
        "qwen_projector_category_ids": category_ids,
        "source_columns": {
            "image_id": image_id_col,
            "image_path": image_path_col,
            "category": category_col,
        },
    }

    out_df = pd.DataFrame(rows)
    write_table(out_df, args.out_tsv)
    args.vocab_out.parent.mkdir(parents=True, exist_ok=True)
    args.vocab_out.write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    print("Top categories:")
    for category in top_categories:
        print(f"  {category}: {counts[category]}")
    print(f"Wrote {len(out_df)} image rows to {args.out_tsv}")
    print(f"Wrote vocabulary to {args.vocab_out}")


def make_category_id(category: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "_" for ch in category.strip()]
    out = "".join(chars)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


if __name__ == "__main__":
    main()
