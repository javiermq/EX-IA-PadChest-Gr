from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .io_utils import CATEGORY_COLUMN_CANDIDATES, detect_column, read_table, split_categories, write_table
from .taxonomy import PADCHEST_GR_CATEGORIES, PADCHEST_GR_CATEGORY_SET, PADCHEST_GR_NAME_TO_COLUMN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter PadChest-GR rows to the selected category subset and drop Other."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV/TSV/JSONL file.")
    parser.add_argument("--output", type=Path, required=True, help="Filtered output CSV/TSV file.")
    parser.add_argument(
        "--category-column",
        default=None,
        help="Column containing the finding category or category list. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep rows without one of the selected categories. By default they are dropped.",
    )
    parser.add_argument(
        "--labels",
        action="store_true",
        help="Add one-hot columns named label_<category_id>.",
    )
    return parser.parse_args()


def detect_category_column(df: pd.DataFrame) -> str:
    return str(detect_column(df, CATEGORY_COLUMN_CANDIDATES, "category"))


def selected_categories(value: object) -> list[str]:
    seen: set[str] = set()
    selected: list[str] = []
    for category in split_categories(value):
        if category in PADCHEST_GR_CATEGORY_SET and category not in seen:
            selected.append(category)
            seen.add(category)
    return selected


def main() -> None:
    args = parse_args()
    df = read_table(args.input).fillna("")
    category_column = args.category_column or detect_category_column(df)
    if category_column not in df.columns:
        raise ValueError(f"Column not found: {category_column}")

    rows: list[dict[str, object]] = []
    for row in df.to_dict("records"):
        categories = selected_categories(row.get(category_column, ""))
        if not categories and not args.keep_empty:
            continue

        out = dict(row)
        out["padchest_gr_categories"] = "|".join(categories)
        out["padchest_gr_category_ids"] = "|".join(PADCHEST_GR_NAME_TO_COLUMN[category] for category in categories)

        if args.labels:
            selected_ids = {PADCHEST_GR_NAME_TO_COLUMN[category] for category in categories}
            for category in PADCHEST_GR_CATEGORIES:
                out[f"label_{category.column}"] = int(category.column in selected_ids)

        rows.append(out)

    out_df = pd.DataFrame(rows)
    write_table(out_df, args.output)
    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(out_df)}")
    print("Dropped category: Other")
    print(f"Category column: {category_column}")


if __name__ == "__main__":
    main()
