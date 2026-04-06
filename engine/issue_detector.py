"""Issue Detection Engine - Detects 8 types of data quality issues."""
from __future__ import annotations

import re
import os
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

from .utils import is_missing, missing_mask, is_numeric, is_parseable_date, normalize_semantic


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
    "missing_value": "FFFF00",         # Yellow
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
    risk: str | None = None
    user_risk: str | None = None      # Risk level set by user-defined rule
    rule_label: str | None = None     # Label of the user-defined rule that generated this issue


# ── Public API ─────────────────────────────────────────────────────

def detect_all_issues(
    df: pd.DataFrame,
    schema: dict[str, str],
    user_rules: dict | None = None,
    duplicate_keys: list[str] | None = None,
    fuzzy_threshold: float = 0.0,
    range_rules_path: str | None = None,
) -> list[Issue]:
    """Run all detectors and return a flat list of issues."""
    issues: list[Issue] = []
    issues.extend(_detect_missing_values(df))
    issues.extend(_detect_format_issues(df, schema))
    issues.extend(_detect_type_drift(df))
    issues.extend(_detect_semantic_inconsistency(df, schema))
    issues.extend(_detect_duplicates(df, duplicate_keys, fuzzy_threshold))
    issues.extend(_detect_range_violations(df, schema, range_rules_path))
    issues.extend(_detect_sparse_columns(df))
    issues.extend(_detect_meaningless_features(df, schema))
    if user_rules:
        issues.extend(_detect_user_rule_violations(df, user_rules))
    return issues


# ── 1. Missing Values ─────────────────────────────────────────────

def _detect_missing_values(df: pd.DataFrame) -> list[Issue]:
    issues = []
    sparse_threshold = 0.7
    for col in df.columns:
        mask = missing_mask(df[col])
        missing_ratio = mask.sum() / len(df) if len(df) > 0 else 0
        # Skip columns that are sparse — those are handled by sparse_column detector
        if missing_ratio >= sparse_threshold:
            continue
        for idx in df.index[mask]:
            issues.append(Issue(
                row_idx=idx, col=col,
                issue_type="missing_value",
                detail="Boş, null veya yalnızca boşluk karakteri",
                value=df.at[idx, col],
            ))
    return issues


# ── 2. Format Issues ──────────────────────────────────────────────

def _detect_format_issues(df: pd.DataFrame, schema: dict[str, str]) -> list[Issue]:
    issues = []
    for col, col_type in schema.items():
        if col_type == "numerical":
            for idx in df.index:
                val = df.at[idx, col]
                if is_missing(val):
                    continue
                if not is_numeric(val):
                    issues.append(Issue(
                        row_idx=idx, col=col,
                        issue_type="format_issue",
                        detail="Sayısal kolonda sayısal olmayan değer",
                        value=val,
                    ))
        elif col_type == "date":
            for idx in df.index:
                val = df.at[idx, col]
                if is_missing(val):
                    continue
                if not is_parseable_date(val):
                    issues.append(Issue(
                        row_idx=idx, col=col,
                        issue_type="format_issue",
                        detail="Tarih olarak ayrıştırılamayan değer",
                        value=val,
                    ))
    return issues


# ── 3. Data Type Drift (precision: only minority-type cells) ──────

def _detect_type_drift(df: pd.DataFrame) -> list[Issue]:
    issues = []
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue

        # Classify each value's type
        type_map: dict[int, str] = {}
        type_counts: dict[str, int] = {}
        for idx in non_null.index:
            val = non_null.at[idx]
            t = _classify_value_type(val)
            type_map[idx] = t
            type_counts[t] = type_counts.get(t, 0) + 1

        # int + float together is acceptable
        effective_types = set(type_counts.keys())
        if effective_types == {"int", "float"}:
            continue
        if len(effective_types) <= 1:
            continue

        # Find the majority type
        majority_type = max(type_counts, key=type_counts.get)

        # Flag only minority-type cells
        for idx, t in type_map.items():
            if t != majority_type:
                issues.append(Issue(
                    row_idx=idx, col=col,
                    issue_type="type_drift",
                    detail=f"Beklenen tip '{majority_type}', bulunan '{t}' (kolonda karışık tipler: {set(type_counts.keys())})",
                    value=df.at[idx, col],
                ))
    return issues


def _classify_value_type(val) -> str:
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, (int, np.integer)):
        return "int"
    if isinstance(val, (float, np.floating)):
        return "float"
    return type(val).__name__


# ── 4. Semantic Inconsistency (enhanced normalizer) ───────────────

def _detect_semantic_inconsistency(df: pd.DataFrame, schema: dict[str, str]) -> list[Issue]:
    issues = []
    for col, col_type in schema.items():
        if col_type not in ("categorical", "text"):
            continue
        non_null = df[col].dropna().astype(str)
        if len(non_null) == 0:
            continue

        # Deep normalize and group
        groups: dict[str, set] = {}
        idx_norm: dict[int, str] = {}
        for idx, orig in zip(non_null.index, non_null):
            norm = normalize_semantic(orig)
            groups.setdefault(norm, set()).add(orig)
            idx_norm[idx] = norm

        # Flag groups with multiple representations
        flagged_norms = {norm for norm, variants in groups.items() if len(variants) > 1}
        if not flagged_norms:
            continue

        for idx in df.index:
            val = df.at[idx, col]
            if is_missing(val):
                continue
            norm_val = idx_norm.get(idx)
            if norm_val and norm_val in flagged_norms:
                variants = groups[norm_val]
                issues.append(Issue(
                    row_idx=idx, col=col,
                    issue_type="semantic_inconsistency",
                    detail=f"Farklı yazım biçimleri: {variants}",
                    value=val,
                ))
    return issues


# ── 5. Duplicate Records (full, key-based, fuzzy) ─────────────────

def _detect_duplicates(
    df: pd.DataFrame,
    key_columns: list[str] | None = None,
    fuzzy_threshold: float = 0.0,
) -> list[Issue]:
    issues = []

    # Full row duplicates
    dup_mask = df.duplicated(keep=False)
    for idx in df.index[dup_mask]:
        issues.append(Issue(
            row_idx=idx, col="__all__",
            issue_type="duplicate",
            detail="Tam satır tekrarı",
        ))

    # Key-based duplicates
    if key_columns:
        valid_keys = [k for k in key_columns if k in df.columns]
        if valid_keys:
            key_dup_mask = df.duplicated(subset=valid_keys, keep=False)
            new_dups = key_dup_mask & ~dup_mask
            key_label = ", ".join(valid_keys)
            for idx in df.index[new_dups]:
                issues.append(Issue(
                    row_idx=idx, col="__all__",
                    issue_type="duplicate",
                    detail=f"Anahtar bazlı tekrar ({key_label})",
                ))
            # Update dup_mask to include key-based
            dup_mask = dup_mask | key_dup_mask

    # Fuzzy duplicate detection (string columns, pairwise on sample)
    if fuzzy_threshold > 0:
        issues.extend(_detect_fuzzy_duplicates(df, dup_mask, fuzzy_threshold))

    return issues


def _detect_fuzzy_duplicates(
    df: pd.DataFrame,
    already_flagged: pd.Series,
    threshold: float,
) -> list[Issue]:
    """Compare rows using string similarity. Only checks non-flagged rows."""
    issues = []
    str_cols = df.select_dtypes(include="object").columns.tolist()
    if not str_cols:
        return issues

    unflagged_idx = df.index[~already_flagged].tolist()
    # Limit pairwise comparison to avoid O(n^2) explosion
    sample = unflagged_idx[:2000]
    if len(sample) < 2:
        return issues

    # Pre-build row strings
    row_strs = {}
    for idx in sample:
        row_strs[idx] = " ".join(str(df.at[idx, c]) for c in str_cols)

    flagged: set[int] = set()
    for i in range(len(sample)):
        if sample[i] in flagged:
            continue
        for j in range(i + 1, len(sample)):
            if sample[j] in flagged:
                continue
            ratio = SequenceMatcher(None, row_strs[sample[i]], row_strs[sample[j]]).ratio()
            if ratio >= threshold:
                flagged.add(sample[i])
                flagged.add(sample[j])
                break

    for idx in flagged:
        issues.append(Issue(
            row_idx=idx, col="__all__",
            issue_type="duplicate",
            detail=f"Bulanık eşleşme (benzerlik >= {threshold:.0%})",
        ))
    return issues


# ── 6. Range Violations (config-based) ────────────────────────────

def _load_range_rules(path: str | None = None) -> list[dict]:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "range_rules.yaml")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("rules", [])
    except (FileNotFoundError, yaml.YAMLError):
        return []


def _detect_range_violations(
    df: pd.DataFrame,
    schema: dict[str, str],
    range_rules_path: str | None = None,
) -> list[Issue]:
    issues = []
    biz_rules = _load_range_rules(range_rules_path)

    for col, col_type in schema.items():
        if col_type != "numerical":
            continue
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        if numeric_col.isna().all():
            continue

        # Statistical outlier bounds (IQR)
        q1 = numeric_col.quantile(0.25)
        q3 = numeric_col.quantile(0.75)
        iqr = q3 - q1
        stat_lower = q1 - 3 * iqr
        stat_upper = q3 + 3 * iqr

        # Find matching business rules from config (whole-word match)
        col_lower = col.lower()
        biz_min, biz_max, biz_label = None, None, None
        for rule in biz_rules:
            keywords = rule.get("keywords", [])
            if any(re.search(rf'\b{re.escape(kw)}\b', col_lower) for kw in keywords):
                biz_min = rule.get("min")
                biz_max = rule.get("max")
                biz_label = rule.get("label", "İş kuralı")
                break

        for idx in df.index:
            val = numeric_col.at[idx]
            if pd.isna(val):
                continue
            reason = None
            # Business rule takes priority
            if biz_min is not None and val < biz_min:
                reason = f"{biz_label}: {val} < {biz_min}"
            elif biz_max is not None and val > biz_max:
                reason = f"{biz_label}: {val} > {biz_max}"
            elif val < stat_lower or val > stat_upper:
                reason = f"İstatistiksel aykırı değer (IQR): {val:.2f} [{stat_lower:.2f}, {stat_upper:.2f}] dışında"
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
    threshold = 0.7
    for col in df.columns:
        missing_ratio = missing_mask(df[col]).sum() / len(df)
        if missing_ratio >= threshold:
            issues.append(Issue(
                row_idx=None, col=col,
                issue_type="sparse_column",
                detail=f"Yüksek boşluk oranı: {missing_ratio:.1%}",
            ))
    return issues


# ── 8. Meaningless Features ───────────────────────────────────────

def _detect_meaningless_features(df: pd.DataFrame, schema: dict[str, str]) -> list[Issue]:
    issues = []
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        if non_null.nunique() == 1:
            issues.append(Issue(
                row_idx=None, col=col,
                issue_type="meaningless_feature",
                detail=f"Sabit kolon (tek değer: {non_null.iloc[0]})",
            ))
            continue
        if non_null.nunique() == len(non_null) and len(non_null) > 10:
            col_lower = col.lower()
            if any(k in col_lower for k in ("id", "index", "key", "uuid", "guid", "no")):
                issues.append(Issue(
                    row_idx=None, col=col,
                    issue_type="meaningless_feature",
                    detail="ID benzeri kolon (tüm değerler benzersiz)",
                ))
                continue
        if schema.get(col) == "categorical":
            ratio = non_null.nunique() / len(non_null)
            if ratio > 0.9 and len(non_null) > 20:
                issues.append(Issue(
                    row_idx=None, col=col,
                    issue_type="meaningless_feature",
                    detail=f"Yüksek kardinalite ({non_null.nunique()} benzersiz / {len(non_null)} satır)",
                ))
    return issues


# ── User-Defined Rules ────────────────────────────────────────────

def _detect_user_rule_violations(df: pd.DataFrame, rules: dict) -> list[Issue]:
    """
    rules format:
    {
        "column_name": {
            "rule_key": {"value": ..., "risk": "Düşük"|"Orta"|"Yüksek"|"Kritik", "label": "..."},
            ...
        }
    }
    """
    issues = []
    for col, col_rules in rules.items():
        if col not in df.columns:
            continue

        def _extract(key):
            """Extract value, risk, label from a rule entry."""
            entry = col_rules.get(key)
            if entry is None:
                return None, None, None
            if isinstance(entry, dict) and "value" in entry:
                return entry["value"], entry.get("risk"), entry.get("label")
            return entry, None, None

        def _make_issue(idx, col, issue_type, detail, val, risk, label):
            return Issue(idx, col, issue_type, detail, val, user_risk=risk, rule_label=label)

        # dtype check
        dtype_val, dtype_risk, dtype_label = _extract("dtype")
        if dtype_val is not None:
            expected = dtype_val
            for idx in df.index:
                val = df.at[idx, col]
                if is_missing(val):
                    continue
                mismatch = False
                if expected == "int":
                    if not isinstance(val, (int, np.integer)):
                        try:
                            f = float(str(val))
                            mismatch = f != int(f)
                        except (ValueError, TypeError):
                            mismatch = True
                elif expected == "float":
                    if not isinstance(val, (int, float, np.integer, np.floating)):
                        try:
                            float(str(val).replace(",", "."))
                        except (ValueError, TypeError):
                            mismatch = True
                elif expected == "date":
                    if not is_parseable_date(val):
                        mismatch = True
                elif expected == "string":
                    if not isinstance(val, str):
                        mismatch = True
                if mismatch:
                    issues.append(_make_issue(idx, col, "format_issue",
                                        f"Beklenen tip '{expected}', bulunan: {type(val).__name__}", val,
                                        dtype_risk, dtype_label))

        regex_val, regex_risk, regex_label = _extract("regex")
        min_len_val, min_len_risk, min_len_label = _extract("min_length")
        max_len_val, max_len_risk, max_len_label = _extract("max_length")
        min_v_val, min_v_risk, min_v_label = _extract("min_value")
        max_v_val, max_v_risk, max_v_label = _extract("max_value")
        allowed_val, allowed_risk, allowed_label = _extract("allowed_values")

        for idx in df.index:
            val = df.at[idx, col]
            if is_missing(val):
                continue
            str_val = str(val)

            if regex_val is not None:
                if not re.match(regex_val, str_val):
                    issues.append(_make_issue(idx, col, "format_issue",
                                        f"Regex uyumsuzluğu: {regex_val}", val,
                                        regex_risk, regex_label))

            if min_len_val is not None and len(str_val) < min_len_val:
                issues.append(_make_issue(idx, col, "format_issue",
                                    f"Çok kısa: {len(str_val)} < {min_len_val}", val,
                                    min_len_risk, min_len_label))

            if max_len_val is not None and len(str_val) > max_len_val:
                issues.append(_make_issue(idx, col, "format_issue",
                                    f"Çok uzun: {len(str_val)} > {max_len_val}", val,
                                    max_len_risk, max_len_label))

            if min_v_val is not None:
                try:
                    if float(str_val) < min_v_val:
                        issues.append(_make_issue(idx, col, "range_violation",
                                            f"Minimum altında: {val} < {min_v_val}", val,
                                            min_v_risk, min_v_label))
                except ValueError:
                    pass

            if max_v_val is not None:
                try:
                    if float(str_val) > max_v_val:
                        issues.append(_make_issue(idx, col, "range_violation",
                                            f"Maksimum üstünde: {val} > {max_v_val}", val,
                                            max_v_risk, max_v_label))
                except ValueError:
                    pass

            if allowed_val is not None:
                if val not in allowed_val and str_val not in allowed_val:
                    issues.append(_make_issue(idx, col, "format_issue",
                                        f"İzin verilen listede yok: {val}", val,
                                        allowed_risk, allowed_label))
    return issues
