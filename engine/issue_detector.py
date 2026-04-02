"""Issue Detection Engine - Detects 8 types of data quality issues."""
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field


ISSUE_TYPES = {
    "missing_value": "Eksik Veriler",
    "format_issue": "Format Hataları",
    "type_drift": "Karışık Veri Tipleri",
    "semantic_inconsistency": "Tutarsız Yazımlar",
    "duplicate": "Tekrarlayan Kayıtlar",
    "range_violation": "Aralık Dışı Değerler",
    "sparse_column": "Çok Boş Kolonlar",
    "meaningless_feature": "Anlamsız Kolonlar",
}

ISSUE_COLORS = {
    "missing_value": "FFFF00",        # Yellow
    "format_issue": "FFA500",          # Orange
    "type_drift": "FF8C00",            # Dark Orange
    "semantic_inconsistency": "9370DB", # Purple
    "duplicate": "6495ED",             # Blue
    "range_violation": "008080",       # Teal
    "sparse_column": "FFB6C1",        # Pink
    "meaningless_feature": "FFD1DC",   # Light Pink
}


@dataclass
class Issue:
    row_idx: int | None  # None for column-level issues
    col: str
    issue_type: str
    detail: str
    value: object = None


def detect_all_issues(
    df: pd.DataFrame,
    schema: dict[str, str],
    user_rules: dict | None = None,
) -> list[Issue]:
    """Run all detectors and return a flat list of issues."""
    issues: list[Issue] = []
    issues.extend(_detect_missing_values(df))
    issues.extend(_detect_format_issues(df, schema))
    issues.extend(_detect_type_drift(df))
    issues.extend(_detect_semantic_inconsistency(df, schema))
    issues.extend(_detect_duplicates(df))
    issues.extend(_detect_range_violations(df, schema))
    issues.extend(_detect_sparse_columns(df))
    issues.extend(_detect_meaningless_features(df, schema))
    if user_rules:
        issues.extend(_detect_user_rule_violations(df, user_rules))
    return issues


# ── 1. Missing Values ──────────────────────────────────────────────

def _detect_missing_values(df: pd.DataFrame) -> list[Issue]:
    issues = []
    for col in df.columns:
        mask = df[col].isna() | df[col].astype(str).str.strip().eq("")
        for idx in df.index[mask]:
            issues.append(Issue(
                row_idx=idx, col=col,
                issue_type="missing_value",
                detail="Null, NaN, or empty string",
                value=df.at[idx, col],
            ))
    return issues


# ── 2. Format Issues ───────────────────────────────────────────────

def _detect_format_issues(df: pd.DataFrame, schema: dict[str, str]) -> list[Issue]:
    issues = []
    for col, col_type in schema.items():
        if col_type == "numerical":
            for idx in df.index:
                val = df.at[idx, col]
                if pd.isna(val):
                    continue
                if not _is_numeric(val):
                    issues.append(Issue(
                        row_idx=idx, col=col,
                        issue_type="format_issue",
                        detail=f"Non-numeric value in numeric column",
                        value=val,
                    ))
        elif col_type == "date":
            for idx in df.index:
                val = df.at[idx, col]
                if pd.isna(val):
                    continue
                if not _is_parseable_date(val):
                    issues.append(Issue(
                        row_idx=idx, col=col,
                        issue_type="format_issue",
                        detail="Unparseable date value",
                        value=val,
                    ))
    return issues


def _is_numeric(val) -> bool:
    if isinstance(val, (int, float, np.integer, np.floating)):
        return True
    try:
        float(str(val).replace(",", "."))
        return True
    except (ValueError, TypeError):
        return False


def _is_parseable_date(val) -> bool:
    if isinstance(val, (pd.Timestamp,)):
        return True
    try:
        pd.to_datetime(val)
        return True
    except (ValueError, TypeError):
        return False


# ── 3. Data Type Drift ─────────────────────────────────────────────

def _detect_type_drift(df: pd.DataFrame) -> list[Issue]:
    issues = []
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        types_found = set()
        for val in non_null.head(500):
            if isinstance(val, (int, np.integer)):
                types_found.add("int")
            elif isinstance(val, (float, np.floating)):
                types_found.add("float")
            elif isinstance(val, bool):
                types_found.add("bool")
            else:
                types_found.add(type(val).__name__)
        # int + float is acceptable, so only flag truly mixed types
        normalized = types_found - {"int", "float"} if {"int", "float"}.issubset(types_found) else types_found
        if len(normalized) > 1:
            for idx in df.index:
                val = df.at[idx, col]
                if pd.notna(val):
                    issues.append(Issue(
                        row_idx=idx, col=col,
                        issue_type="type_drift",
                        detail=f"Mixed types in column: {types_found}",
                        value=val,
                    ))
    return issues


# ── 4. Semantic Inconsistency ──────────────────────────────────────

def _detect_semantic_inconsistency(df: pd.DataFrame, schema: dict[str, str]) -> list[Issue]:
    issues = []
    for col, col_type in schema.items():
        if col_type not in ("categorical", "text"):
            continue
        non_null = df[col].dropna().astype(str)
        if len(non_null) == 0:
            continue
        # Normalize: lowercase, strip, remove dots
        normalized = non_null.str.lower().str.strip().str.replace(".", "", regex=False)
        # Group originals by normalized form
        groups: dict[str, set] = {}
        for orig, norm in zip(non_null, normalized):
            groups.setdefault(norm, set()).add(orig)
        # Flag groups with multiple representations
        flagged_norms = {norm for norm, variants in groups.items() if len(variants) > 1}
        if not flagged_norms:
            continue
        for idx in df.index:
            val = df.at[idx, col]
            if pd.isna(val):
                continue
            norm_val = str(val).lower().strip().replace(".", "")
            if norm_val in flagged_norms:
                variants = groups[norm_val]
                issues.append(Issue(
                    row_idx=idx, col=col,
                    issue_type="semantic_inconsistency",
                    detail=f"Multiple representations: {variants}",
                    value=val,
                ))
    return issues


# ── 5. Duplicate Records ──────────────────────────────────────────

def _detect_duplicates(df: pd.DataFrame) -> list[Issue]:
    issues = []
    dup_mask = df.duplicated(keep=False)
    for idx in df.index[dup_mask]:
        issues.append(Issue(
            row_idx=idx, col="__all__",
            issue_type="duplicate",
            detail="Duplicate row",
        ))
    return issues


# ── 6. Range Violations ───────────────────────────────────────────

def _detect_range_violations(df: pd.DataFrame, schema: dict[str, str]) -> list[Issue]:
    issues = []
    for col, col_type in schema.items():
        if col_type != "numerical":
            continue
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        if numeric_col.isna().all():
            continue
        q1 = numeric_col.quantile(0.25)
        q3 = numeric_col.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        col_lower = col.lower()
        # Business rules for common patterns
        biz_min, biz_max = None, None
        if any(k in col_lower for k in ("age", "yas", "yaş")):
            biz_min, biz_max = 0, 150
        elif any(k in col_lower for k in ("percent", "yüzde", "oran", "rate")):
            biz_min, biz_max = 0, 100
        elif any(k in col_lower for k in ("price", "fiyat", "tutar", "amount")):
            biz_min = 0

        for idx in df.index:
            val = numeric_col.at[idx]
            if pd.isna(val):
                continue
            reason = None
            if val < lower or val > upper:
                reason = f"Statistical outlier (IQR): {val:.2f} outside [{lower:.2f}, {upper:.2f}]"
            if biz_min is not None and val < biz_min:
                reason = f"Business rule violation: {val} < {biz_min}"
            if biz_max is not None and val > biz_max:
                reason = f"Business rule violation: {val} > {biz_max}"
            if reason:
                issues.append(Issue(
                    row_idx=idx, col=col,
                    issue_type="range_violation",
                    detail=reason,
                    value=val,
                ))
    return issues


# ── 7. Sparse Columns ─────────────────────────────────────────────

def _detect_sparse_columns(df: pd.DataFrame) -> list[Issue]:
    issues = []
    threshold = 0.7  # 70% missing -> sparse
    for col in df.columns:
        missing_ratio = df[col].isna().sum() / len(df)
        if missing_ratio >= threshold:
            issues.append(Issue(
                row_idx=None, col=col,
                issue_type="sparse_column",
                detail=f"High missing ratio: {missing_ratio:.1%}",
            ))
    return issues


# ── 8. Meaningless Features ───────────────────────────────────────

def _detect_meaningless_features(df: pd.DataFrame, schema: dict[str, str]) -> list[Issue]:
    issues = []
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        # Constant column
        if non_null.nunique() == 1:
            issues.append(Issue(
                row_idx=None, col=col,
                issue_type="meaningless_feature",
                detail=f"Constant column (single value: {non_null.iloc[0]})",
            ))
            continue
        # ID-like: all unique values
        if non_null.nunique() == len(non_null) and len(non_null) > 10:
            col_lower = col.lower()
            if any(k in col_lower for k in ("id", "index", "key", "uuid", "guid", "no")):
                issues.append(Issue(
                    row_idx=None, col=col,
                    issue_type="meaningless_feature",
                    detail="ID-like column (all unique values)",
                ))
                continue
        # High cardinality categorical
        if schema.get(col) == "categorical":
            ratio = non_null.nunique() / len(non_null)
            if ratio > 0.9 and len(non_null) > 20:
                issues.append(Issue(
                    row_idx=None, col=col,
                    issue_type="meaningless_feature",
                    detail=f"High cardinality categorical ({non_null.nunique()} unique / {len(non_null)} rows)",
                ))
    return issues


# ── User-Defined Rules ────────────────────────────────────────────

def _detect_user_rule_violations(df: pd.DataFrame, rules: dict) -> list[Issue]:
    """
    rules format:
    {
        "column_name": {
            "dtype": "int" | "float" | "date" | "string",
            "regex": "pattern",
            "min_length": 3,
            "max_length": 50,
            "min_value": 0,
            "max_value": 100,
            "allowed_values": ["A", "B", "C"],
        }
    }
    """
    import re
    issues = []
    for col, rule in rules.items():
        if col not in df.columns:
            continue
        for idx in df.index:
            val = df.at[idx, col]
            if pd.isna(val):
                continue
            str_val = str(val)

            if "regex" in rule:
                if not re.match(rule["regex"], str_val):
                    issues.append(Issue(idx, col, "format_issue",
                                        f"Regex mismatch: {rule['regex']}", val))

            if "min_length" in rule and len(str_val) < rule["min_length"]:
                issues.append(Issue(idx, col, "format_issue",
                                    f"Too short: {len(str_val)} < {rule['min_length']}", val))

            if "max_length" in rule and len(str_val) > rule["max_length"]:
                issues.append(Issue(idx, col, "format_issue",
                                    f"Too long: {len(str_val)} > {rule['max_length']}", val))

            if "min_value" in rule:
                try:
                    if float(str_val) < rule["min_value"]:
                        issues.append(Issue(idx, col, "range_violation",
                                            f"Below minimum: {val} < {rule['min_value']}", val))
                except ValueError:
                    pass

            if "max_value" in rule:
                try:
                    if float(str_val) > rule["max_value"]:
                        issues.append(Issue(idx, col, "range_violation",
                                            f"Above maximum: {val} > {rule['max_value']}", val))
                except ValueError:
                    pass

            if "allowed_values" in rule:
                if val not in rule["allowed_values"] and str_val not in rule["allowed_values"]:
                    issues.append(Issue(idx, col, "format_issue",
                                        f"Value not in allowed list: {val}", val))
    return issues
