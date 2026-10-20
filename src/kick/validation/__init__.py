"""
Validation package for KICK: coordinate validation, schema auditing, and data quality reporting.
"""

from .schema import (
    PitchCoordinates,
    validate_coordinates,
    compute_goal_distance,
    compute_goal_angle,
    QualityAuditResult,
    DataQualityAuditor,
)
from .report import generate_markdown_report, save_audit_report

__all__ = [
    "PitchCoordinates",
    "validate_coordinates",
    "compute_goal_distance",
    "compute_goal_angle",
    "QualityAuditResult",
    "DataQualityAuditor",
    "generate_markdown_report",
    "save_audit_report",
]
