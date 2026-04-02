"""Schema Detection Module - Classifies columns as numerical, categorical, date, or text."""
from __future__ import annotations

import pandas as pd
import numpy as np


COLUMN_TYPES = ("numerical", "categorical", "date", "text")


def detect_schema(df: pd.DataFrame) -> dict[str, str]:
    """Return a dict mapping column names to their detected type."""
    schema = {}
    for col in df.columns:
        schema[col] = _classify_column(df[col])
    return schema


def _classify_column(series: pd.Series) -> str:
    # Already datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    # Try to parse as date
    if series.dtype == object:
        sample = series.dropna().head(100)
        if len(sample) > 0:
            try:
                parsed = pd.to_datetime(sample, infer_datetime_format=True, format="mixed")
                success_rate = parsed.notna().sum() / len(sample)
                if success_rate >= 0.8:
                    return "date"
            except (ValueError, TypeError):
                pass

    # Numeric types
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"

    # Try numeric conversion for object columns
    if series.dtype == object:
        sample = series.dropna().head(200)
        if len(sample) > 0:
            converted = pd.to_numeric(sample, errors="coerce")
            success_rate = converted.notna().sum() / len(sample)
            if success_rate >= 0.8:
                return "numerical"

    # Categorical vs text heuristic
    if series.dtype == object:
        n_unique = series.nunique()
        n_total = len(series.dropna())
        if n_total == 0:
            return "text"
        avg_len = series.dropna().astype(str).str.len().mean()
        # Low cardinality or short strings -> categorical
        if n_unique <= 50 or (n_unique / max(n_total, 1)) < 0.3:
            return "categorical"
        if avg_len > 50:
            return "text"
        return "categorical"

    return "text"
