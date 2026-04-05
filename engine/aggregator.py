"""Aggregation layer - Summarizes issues for dashboard consumption."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .issue_detector import Issue, ISSUE_TYPES


@dataclass
class AggregatedSummary:
    total_issues: int
    total_flagged_rows: int
    total_rows: int
    issues_by_type: dict[str, int]
    issues_by_column: dict[str, int]
    risk_distribution: dict[str, int]
    top_risky_columns: list[tuple[str, int, str]]  # (col, issue_count, worst_risk)


def aggregate_issues(
    issues: list[Issue],
    row_risks: dict[int, str],
    total_rows: int,
) -> AggregatedSummary:
    """Build a structured summary from raw issues."""

    # Issues by type
    type_counter: Counter = Counter()
    for issue in issues:
        type_counter[issue.issue_type] += 1

    # Issues by column
    col_counter: Counter = Counter()
    for issue in issues:
        if issue.col != "__all__":
            col_counter[issue.col] += 1
        else:
            col_counter["(tüm satır)"] += 1

    # Risk distribution
    risk_counter: Counter = Counter()
    for risk in row_risks.values():
        risk_counter[risk] += 1

    # Top risky columns: for each column find issue count + worst risk
    col_risks: dict[str, list[str]] = {}
    for issue in issues:
        col = issue.col if issue.col != "__all__" else "(tüm satır)"
        risk = getattr(issue, "risk", None)
        if risk:
            col_risks.setdefault(col, []).append(risk)

    risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    top_risky: list[tuple[str, int, str]] = []
    for col, risks in col_risks.items():
        worst = max(risks, key=lambda r: risk_order.get(r, 0))
        top_risky.append((col, col_counter.get(col, len(risks)), worst))
    top_risky.sort(key=lambda x: (risk_order.get(x[2], 0), x[1]), reverse=True)

    flagged_rows = len(set(
        i.row_idx for i in issues if i.row_idx is not None
    ))

    return AggregatedSummary(
        total_issues=len(issues),
        total_flagged_rows=flagged_rows,
        total_rows=total_rows,
        issues_by_type=dict(type_counter.most_common()),
        issues_by_column=dict(col_counter.most_common()),
        risk_distribution=dict(risk_counter),
        top_risky_columns=top_risky[:10],
    )
