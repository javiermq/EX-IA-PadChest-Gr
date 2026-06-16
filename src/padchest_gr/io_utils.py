from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd


CATEGORY_COLUMN_CANDIDATES = (
    "category",
    "categories",
    "finding_category",
    "finding_categories",
    "label",
    "labels",
    "labels_available",
    "padchest_gr_categories",
)

IMAGE_ID_COLUMN_CANDIDATES = (
    "image_id",
    "imageid",
    "image",
    "filename",
    "file_name",
    "study_id",
    "dicom_id",
)

IMAGE_PATH_COLUMN_CANDIDATES = (
    "image_path",
    "path",
    "filepath",
    "file_path",
    "image_filename",
)

BOX_LABEL_COLUMN_CANDIDATES = (
    "box_label",
    "bbox_label",
    "label",
    "labels",
    "finding",
    "finding_label",
    "box_category",
)

X_MIN_COLUMN_CANDIDATES = ("x_min", "xmin", "x1", "left")
Y_MIN_COLUMN_CANDIDATES = ("y_min", "ymin", "y1", "top")
X_MAX_COLUMN_CANDIDATES = ("x_max", "xmax", "x2", "right")
Y_MAX_COLUMN_CANDIDATES = ("y_max", "ymax", "y2", "bottom")
WIDTH_COLUMN_CANDIDATES = ("width", "w")
HEIGHT_COLUMN_CANDIDATES = ("height", "h")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    sep = "\t" if suffix in {".tsv", ".tab"} else ","
    return pd.read_csv(path, sep=sep)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    df.to_csv(path, sep=sep, index=False)


def detect_column(df: pd.DataFrame, candidates: tuple[str, ...], kind: str, required: bool = True) -> str | None:
    lower_to_original = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    if required:
        raise ValueError(f"Could not detect {kind} column. Tried: {', '.join(candidates)}")
    return None


def normalize_category(value: object) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text)


def split_categories(value: object) -> list[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [normalize_category(item) for item in value if normalize_category(item)]

    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
            if isinstance(parsed, list):
                return [normalize_category(item) for item in parsed if normalize_category(item)]

    return [normalize_category(item) for item in re.split(r"\s*[|;,]\s*", text) if normalize_category(item)]
