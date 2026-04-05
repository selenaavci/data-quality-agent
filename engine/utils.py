"""Shared utilities used across all detection modules."""
from __future__ import annotations

import unicodedata
import pandas as pd
import numpy as np


# ── Missing Value Check ────────────────────────────────────────────

def is_missing(val) -> bool:
    """Standardized missing value check: None, NaN, empty string, whitespace-only."""
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if pd.isna(val):
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False


def missing_mask(series: pd.Series) -> pd.Series:
    """Return a boolean mask for missing values in a Series."""
    return series.map(is_missing)


# ── Numeric / Date Helpers ─────────────────────────────────────────

def is_numeric(val) -> bool:
    if isinstance(val, (int, float, np.integer, np.floating)):
        return True
    try:
        float(str(val).replace(",", "."))
        return True
    except (ValueError, TypeError):
        return False


def is_parseable_date(val) -> bool:
    if isinstance(val, pd.Timestamp):
        return True
    try:
        pd.to_datetime(val)
        return True
    except (ValueError, TypeError):
        return False


# ── Turkish-Aware Semantic Normalizer ──────────────────────────────

# Turkish-specific character mapping
_TR_CHAR_MAP = str.maketrans({
    "ı": "i", "İ": "i",
    "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u",
    "ş": "s", "Ş": "s",
    "ö": "o", "Ö": "o",
    "ç": "c", "Ç": "c",
})

# Common Turkish abbreviations / aliases
_ABBREVIATION_MAP = {
    "ist": "istanbul",
    "ank": "ankara",
    "izm": "izmir",
    "dr": "doktor",
    "prof": "profesor",
    "mah": "mahalle",
    "cad": "cadde",
    "sok": "sokak",
    "apt": "apartman",
    "no": "numara",
    "tel": "telefon",
    "bl": "blok",
    "kat": "kat",
    "sk": "sokak",
    "cd": "cadde",
}


def normalize_semantic(text: str) -> str:
    """
    Deep normalization for semantic comparison:
    1. Unicode NFKD normalization
    2. Lowercase
    3. Turkish character folding
    4. Strip whitespace, dots, hyphens, slashes, underscores
    5. Abbreviation expansion
    """
    # Unicode normalize
    text = unicodedata.normalize("NFKD", text)
    # Lowercase (Turkish-aware: İ->i handled by char map)
    text = text.lower()
    # Turkish char folding
    text = text.translate(_TR_CHAR_MAP)
    # Remove accents (combining marks)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Remove punctuation: dots, hyphens, slashes, underscores, parens
    for ch in ".-/_()[]{}':;,\"":
        text = text.replace(ch, " ")
    # Collapse whitespace and strip
    tokens = text.split()
    # Expand abbreviations
    tokens = [_ABBREVIATION_MAP.get(t, t) for t in tokens]
    return " ".join(tokens)
