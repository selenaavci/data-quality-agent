"""Risk Scoring Engine - Assigns Düşük/Orta/Yüksek/Kritik risk levels to issues."""
from __future__ import annotations

import pandas as pd
from .issue_detector import Issue
from .utils import missing_mask

RISK_LEVELS = ("Düşük", "Orta", "Yüksek", "Kritik")

RISK_COLORS = {
    "Düşük": "A8D5BA",     # Soft Green
    "Orta": "F5C242",       # Amber
    "Yüksek": "FF0000",     # Red
    "Kritik": "800020",     # Bordeaux
}

RISK_TEXT_COLORS = {
    "Düşük": "000000",
    "Orta": "000000",
    "Yüksek": "FFFFFF",
    "Kritik": "FFFFFF",
}

# Base risk by issue type
_BASE_RISK = {
    "missing_value": "Orta",
    "format_issue": "Yüksek",
    "type_drift": "Yüksek",
    "semantic_inconsistency": "Düşük",
    "duplicate": "Orta",
    "range_violation": "Yüksek",
    "sparse_column": "Orta",
    "meaningless_feature": "Düşük",
}

_RISK_ORDER = {"Düşük": 0, "Orta": 1, "Yüksek": 2, "Kritik": 3}


def score_issues(issues: list[Issue], df: pd.DataFrame) -> dict[int | None, str]:
    """
    Assign a risk level to each issue and return a mapping of
    row_idx -> highest risk level for that row.
    """
    row_risks: dict[int | None, str] = {}

    for issue in issues:
        # User-defined risk takes priority over computed risk
        if issue.user_risk and issue.user_risk in _RISK_ORDER:
            risk = issue.user_risk
        else:
            risk = _compute_risk(issue, df)
        issue.risk = risk
        key = issue.row_idx
        if key is not None:
            existing = row_risks.get(key, "Düşük")
            if _RISK_ORDER[risk] > _RISK_ORDER[existing]:
                row_risks[key] = risk

    return row_risks


def _compute_risk(issue: Issue, df: pd.DataFrame) -> str:
    base = _BASE_RISK.get(issue.issue_type, "Düşük")
    risk_val = _RISK_ORDER[base]

    # Escalation rules
    if issue.issue_type == "missing_value":
        col = issue.col
        miss_ratio = missing_mask(df[col]).sum() / len(df) if len(df) > 0 else 0
        if miss_ratio > 0.5:
            risk_val = max(risk_val, _RISK_ORDER["Kritik"])
        elif miss_ratio > 0.3:
            risk_val = max(risk_val, _RISK_ORDER["Yüksek"])

    elif issue.issue_type == "format_issue":
        risk_val = max(risk_val, _RISK_ORDER["Yüksek"])

    elif issue.issue_type == "semantic_inconsistency":
        # Pattern-breaking values in numeric/date columns are higher risk
        if issue.detail and ("Sayısal kolonda" in issue.detail or
                             "Tarih kolonda" in issue.detail or
                             "Anlamsız/yer tutucu" in issue.detail):
            risk_val = max(risk_val, _RISK_ORDER["Yüksek"])

    elif issue.issue_type == "type_drift":
        risk_val = max(risk_val, _RISK_ORDER["Kritik"])

    elif issue.issue_type == "range_violation":
        if issue.detail and "İstatistiksel" not in issue.detail:
            risk_val = max(risk_val, _RISK_ORDER["Kritik"])

    elif issue.issue_type == "sparse_column":
        col = issue.col
        miss_ratio = missing_mask(df[col]).sum() / len(df) if len(df) > 0 else 0
        if miss_ratio > 0.9:
            risk_val = max(risk_val, _RISK_ORDER["Yüksek"])

    # Map back to label
    for label, val in _RISK_ORDER.items():
        if val == risk_val:
            return label
    return base
