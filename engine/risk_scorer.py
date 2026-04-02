"""Risk Scoring Engine - Assigns LOW/MEDIUM/HIGH/CRITICAL risk levels to issues."""
from __future__ import annotations

import pandas as pd
from .issue_detector import Issue

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

RISK_COLORS = {
    "LOW": "A8D5BA",       # Soft Green
    "MEDIUM": "F5C242",    # Amber
    "HIGH": "FF0000",      # Red
    "CRITICAL": "800020",  # Bordeaux
}

RISK_TEXT_COLORS = {
    "LOW": "000000",
    "MEDIUM": "000000",
    "HIGH": "FFFFFF",
    "CRITICAL": "FFFFFF",
}

# Base risk by issue type
_BASE_RISK = {
    "missing_value": "MEDIUM",
    "format_issue": "HIGH",
    "type_drift": "HIGH",
    "semantic_inconsistency": "LOW",
    "duplicate": "MEDIUM",
    "range_violation": "HIGH",
    "sparse_column": "MEDIUM",
    "meaningless_feature": "LOW",
}

_RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def score_issues(issues: list[Issue], df: pd.DataFrame) -> dict[int | None, str]:
    """
    Assign a risk level to each issue and return a mapping of
    row_idx -> highest risk level for that row.
    """
    row_risks: dict[int | None, str] = {}

    for issue in issues:
        risk = _compute_risk(issue, df)
        issue.risk = risk  # Attach risk to the issue object
        key = issue.row_idx
        if key is not None:
            existing = row_risks.get(key, "LOW")
            if _RISK_ORDER[risk] > _RISK_ORDER[existing]:
                row_risks[key] = risk

    return row_risks


def _compute_risk(issue: Issue, df: pd.DataFrame) -> str:
    base = _BASE_RISK.get(issue.issue_type, "LOW")
    risk_val = _RISK_ORDER[base]

    # Escalation rules
    if issue.issue_type == "missing_value":
        col = issue.col
        missing_ratio = df[col].isna().sum() / len(df)
        if missing_ratio > 0.5:
            risk_val = max(risk_val, _RISK_ORDER["CRITICAL"])
        elif missing_ratio > 0.3:
            risk_val = max(risk_val, _RISK_ORDER["HIGH"])

    elif issue.issue_type == "format_issue":
        # Format issues in date/numeric columns are more critical
        risk_val = max(risk_val, _RISK_ORDER["HIGH"])

    elif issue.issue_type == "type_drift":
        risk_val = max(risk_val, _RISK_ORDER["CRITICAL"])

    elif issue.issue_type == "range_violation":
        if issue.detail and "Business rule" in issue.detail:
            risk_val = max(risk_val, _RISK_ORDER["CRITICAL"])

    elif issue.issue_type == "sparse_column":
        col = issue.col
        missing_ratio = df[col].isna().sum() / len(df)
        if missing_ratio > 0.9:
            risk_val = max(risk_val, _RISK_ORDER["HIGH"])

    # Map back to label
    for label, val in _RISK_ORDER.items():
        if val == risk_val:
            return label
    return base
