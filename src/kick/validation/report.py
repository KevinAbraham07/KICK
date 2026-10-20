"""
Report generator for Data Quality audits.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Union
from .schema import QualityAuditResult


def generate_markdown_report(
    result: QualityAuditResult,
    match_title: str = "StatsBomb Match Audit",
    match_id: Optional[int] = None
) -> str:
    """
    Produce a clean, professional Markdown report of data quality and schema health.
    """
    d = result.to_dict()
    pass_pct = d["pass_summary"]["completion_rate_pct"]

    report_lines = [
        f"# Data Quality & Integrity Report: {match_title}",
        f"**Match ID**: `{match_id if match_id is not None else 'N/A'}` | **Total Ingested Events**: `{result.total_events:,}`",
        "",
        "## 1. Audit Summary",
        "| Metric | Count | Assessment |",
        "| :--- | :--- | :--- |",
        f"| **Retained Records** | `{result.retained_records_count:,}` | Validated for analysis |",
        f"| **Duplicate Event IDs** | `{result.duplicate_event_ids}` | {'Zero duplicates detected' if result.duplicate_event_ids == 0 else 'Duplicates flagged'} |",
        f"| **Out-of-Bounds Coordinates** | `{result.out_of_bounds_coords}` | {'All coordinates within [0,120]x[0,80]' if result.out_of_bounds_coords == 0 else 'Coordinates exceeded pitch boundaries'} |",
        f"| **Missing In-Play Locations** | `{result.missing_locations}` | Critical active actions missing coordinates |",
        f"| **Flagged Anomalies Total** | `{result.flagged_records_count}` | Total records requiring exclusion or review |",
        "",
        "## 2. Shot & Pass Integrity",
        "### Shots",
        f"- **Total Shots**: `{result.shot_count}`",
        f"- **Missing Outcome Attribute**: `{result.shots_missing_outcome}`",
        f"- **Missing Body Part Attribute**: `{result.shots_missing_body_part}`",
        f"- **Completeness**: `{(1 - (result.shots_missing_outcome + result.shots_missing_body_part) / max(1, result.shot_count*2)) * 100:.1f}%`",
        "",
        "### Passes",
        f"- **Total Passes**: `{result.pass_count}`",
        f"- **Completed Passes**: `{result.completed_passes}` ({pass_pct}%)",
        f"- **Incomplete / Intercepted**: `{result.pass_count - result.completed_passes}`",
        "",
        "## 3. Top Event Types Distribution",
        "| Event Type | Event Count | % of Match Events |",
        "| :--- | :--- | :--- |",
    ]

    for event_type, count in d["top_event_types"].items():
        pct = (count / max(1, result.total_events)) * 100
        report_lines.append(f"| **{event_type}** | `{count:,}` | `{pct:.1f}%` |")

    report_lines.append("")
    report_lines.append("## 4. Mandatory Field Audit")
    if d["missing_mandatory_fields"]:
        report_lines.append("Alert: The following mandatory fields contained missing values:")
        for col, missing_cnt in d["missing_mandatory_fields"].items():
            report_lines.append(f"- `{col}`: `{missing_cnt}` null values")
    else:
        report_lines.append("All mandatory core fields (`id`, `index`, `period`, `timestamp`, `type`) have 100% completeness.")

    return "\n".join(report_lines)


def save_audit_report(
    result: QualityAuditResult,
    output_dir: Union[str, Path],
    filename_prefix: str = "data_quality_report"
) -> Path:
    """
    Saves the audit result as JSON and Markdown files.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = out_path / f"{filename_prefix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    # Save Markdown
    md_path = out_path / f"{filename_prefix}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(generate_markdown_report(result, match_title=filename_prefix))

    return md_path
