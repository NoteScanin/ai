"""
Data quality utilities for CSV loading, schema normalization, and filtering.
Used by training scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin1")
REQUIRED_COLUMNS = {"image", "text"}
LABEL_DATA_COLUMNS = {"id", "label"}


@dataclass
class QualityFilterResult:
    dataframe: pd.DataFrame
    skipped_missing_images: int = 0
    skipped_empty_text: int = 0
    skipped_too_tall: int = 0
    skipped_too_long: int = 0


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def read_csv_with_fallback(csv_path: Path | str) -> pd.DataFrame:
    csv_path = Path(csv_path)
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return pd.read_csv(csv_path)


def label_id_to_image(value: object) -> str:
    raw = str(value).strip()
    if raw.lower().endswith(IMAGE_EXTENSIONS):
        return raw
    try:
        return f"{int(float(raw)):04d}.png"
    except ValueError:
        return f"{raw}.png"


def normalize_dataframe_schema(df: pd.DataFrame, csv_path: Path | str) -> pd.DataFrame:
    columns = set(df.columns)
    if REQUIRED_COLUMNS <= columns:
        return df.copy()
    if LABEL_DATA_COLUMNS <= columns:
        normalized = pd.DataFrame()
        normalized["image"] = df["id"].map(label_id_to_image)
        normalized["text"] = df["label"]
        normalized["status"] = "approved"
        return normalized
    missing = REQUIRED_COLUMNS - columns
    raise ValueError(f"{csv_path} missing columns: {', '.join(sorted(missing))}")


@lru_cache(maxsize=4096)
def image_size(image_path: str) -> tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size


def load_csv_dataframe(csv_path: Path | str) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found.")
    df = read_csv_with_fallback(csv_path)
    return normalize_dataframe_schema(df, csv_path)


def split_dataframe(
    df: pd.DataFrame,
    *,
    val_fraction: float = 0.1,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    total = len(shuffled)
    if total < 2:
        return shuffled.copy(), shuffled.iloc[0:0].copy(), shuffled.iloc[0:0].copy()

    val_n = max(1, int(round(total * val_fraction))) if val_fraction > 0 else 0
    test_n = max(1, int(round(total * test_fraction))) if test_fraction > 0 else 0

    if val_n + test_n >= total:
        overflow = val_n + test_n - total + 1
        test_n = max(0, test_n - min(test_n, overflow))
        overflow -= min(test_n, overflow)
        val_n = max(0, val_n - overflow)

    return (
        shuffled.iloc[val_n + test_n:].reset_index(drop=True),
        shuffled.iloc[:val_n].reset_index(drop=True),
        shuffled.iloc[val_n:val_n + test_n].reset_index(drop=True),
    )


def filter_line_level_rows(
    df: pd.DataFrame,
    image_dir: Path | str,
    *,
    max_image_height: int = 80,
    max_label_len: int = 150,
    require_approved: bool = True,
    limit: int | None = None,
) -> QualityFilterResult:
    image_dir = Path(image_dir)
    cleaned = df.dropna(subset=["image", "text"]).copy()
    cleaned["image"] = cleaned["image"].astype(str).str.strip()
    cleaned["text"] = cleaned["text"].astype(str).map(normalize_text)

    if "status" in cleaned.columns:
        cleaned["status"] = cleaned["status"].astype(str).str.strip().str.lower()
        if require_approved:
            cleaned = cleaned[cleaned["status"] == "approved"]

    cleaned = cleaned[cleaned["image"].str.lower().str.endswith(IMAGE_EXTENSIONS)]

    valid_rows, skip_missing, skip_empty, skip_tall, skip_long = [], 0, 0, 0, 0

    for row in cleaned.to_dict("records"):
        text = normalize_text(row["text"])
        if not text:
            skip_empty += 1
            continue
        img_path = image_dir / str(row["image"])
        if not img_path.exists():
            skip_missing += 1
            continue
        _, height = image_size(str(img_path))
        if height > max_image_height:
            skip_tall += 1
            continue
        if len(text) > max_label_len:
            skip_long += 1
            continue
        row["text"] = text
        valid_rows.append(row)

    valid_df = pd.DataFrame(valid_rows, columns=cleaned.columns).reset_index(drop=True)
    if limit and limit > 0:
        valid_df = valid_df.head(limit).copy()
    if valid_df.empty:
        raise ValueError("No usable rows found after filtering.")

    return QualityFilterResult(
        dataframe=valid_df,
        skipped_missing_images=skip_missing,
        skipped_empty_text=skip_empty,
        skipped_too_tall=skip_tall,
        skipped_too_long=skip_long,
    )
