"""
Data schema definitions, coordinate geometry checks, and event auditing rules for KICK.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class PitchCoordinates:
    """
    Standard StatsBomb pitch dimensions (in yards):
    Length (X): 0 to 120 (attacking rightwards towards x=120)
    Width (Y): 0 to 80 (top touchline y=0, bottom touchline y=80)
    Goal Center: (120, 40)
    Goal Posts: y=36 to y=44 (8 yards wide)
    Goal Crossbar: z=2.67 yards (8 feet)
    """
    X_MIN = 0.0
    X_MAX = 120.0
    Y_MIN = 0.0
    Y_MAX = 80.0

    GOAL_X = 120.0
    GOAL_Y = 40.0
    GOAL_Y1 = 36.0  # Left post
    GOAL_Y2 = 44.0  # Right post
    GOAL_HEIGHT = 2.67


def validate_coordinates(
    df: pd.DataFrame,
    x_col: str = "location_x",
    y_col: str = "location_y",
    allow_null: bool = True
) -> pd.Series:
    """
    Returns a boolean Series where True indicates coordinates are valid within pitch boundaries.
    """
    if x_col not in df.columns or y_col not in df.columns:
        return pd.Series(True, index=df.index)

    x = df[x_col]
    y = df[y_col]

    is_null = x.isna() | y.isna()
    in_bounds = (
        (x >= PitchCoordinates.X_MIN) & (x <= PitchCoordinates.X_MAX) &
        (y >= PitchCoordinates.Y_MIN) & (y <= PitchCoordinates.Y_MAX)
    )

    if allow_null:
        return in_bounds | is_null
    return in_bounds


def compute_goal_distance(
    x: pd.Series | np.ndarray | float,
    y: pd.Series | np.ndarray | float
) -> pd.Series | np.ndarray | float:
    """
    Calculates straight-line Euclidean distance in yards to the center of the goal (120, 40).
    """
    dx = PitchCoordinates.GOAL_X - x
    dy = y - PitchCoordinates.GOAL_Y
    return np.sqrt(dx**2 + dy**2)


def compute_goal_angle(
    x: pd.Series | np.ndarray | float,
    y: pd.Series | np.ndarray | float
) -> pd.Series | np.ndarray | float:
    """
    Calculates the visible goal angle (in radians) subtended by the goal posts (y=36 to y=44 at x=120).
    Uses the law of cosines / vector dot product between lines from shooter to both posts.
    """
    # Vector to left post (120, 36)
    v1_x = PitchCoordinates.GOAL_X - x
    v1_y = PitchCoordinates.GOAL_Y1 - y
    d1 = np.sqrt(v1_x**2 + v1_y**2)

    # Vector to right post (120, 44)
    v2_x = PitchCoordinates.GOAL_X - x
    v2_y = PitchCoordinates.GOAL_Y2 - y
    d2 = np.sqrt(v2_x**2 + v2_y**2)

    # Dot product
    dot = (v1_x * v2_x) + (v1_y * v2_y)
    cos_theta = dot / (d1 * d2 + 1e-9)
    # Clip for floating point precision
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


@dataclass
class QualityAuditResult:
    """
    Summary container for data quality checks on an event dataset.
    """
    total_events: int
    event_counts: Dict[str, int]
    missing_mandatory: Dict[str, int]
    out_of_bounds_coords: int
    missing_locations: int
    duplicate_event_ids: int
    shot_count: int
    shots_missing_outcome: int
    shots_missing_body_part: int
    pass_count: int
    completed_passes: int
    flagged_records_count: int
    retained_records_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_events": self.total_events,
            "retained_records": self.retained_records_count,
            "flagged_records": self.flagged_records_count,
            "duplicate_event_ids": self.duplicate_event_ids,
            "out_of_bounds_coords": self.out_of_bounds_coords,
            "missing_locations": self.missing_locations,
            "shot_summary": {
                "total_shots": self.shot_count,
                "missing_outcome": self.shots_missing_outcome,
                "missing_body_part": self.shots_missing_body_part,
            },
            "pass_summary": {
                "total_passes": self.pass_count,
                "completed_passes": self.completed_passes,
                "completion_rate_pct": round((self.completed_passes / max(1, self.pass_count)) * 100, 2),
            },
            "top_event_types": dict(sorted(self.event_counts.items(), key=lambda item: item[1], reverse=True)[:10]),
            "missing_mandatory_fields": {k: v for k, v in self.missing_mandatory.items() if v > 0},
        }


class DataQualityAuditor:
    """
    Audits an event DataFrame and produces a detailed QualityAuditResult.
    """

    MANDATORY_COLS = ["id", "index", "period", "timestamp", "type"]

    def audit(self, df: pd.DataFrame) -> QualityAuditResult:
        total = len(df)
        if total == 0:
            return QualityAuditResult(
                total_events=0,
                event_counts={},
                missing_mandatory={},
                out_of_bounds_coords=0,
                missing_locations=0,
                duplicate_event_ids=0,
                shot_count=0,
                shots_missing_outcome=0,
                shots_missing_body_part=0,
                pass_count=0,
                completed_passes=0,
                flagged_records_count=0,
                retained_records_count=0,
            )

        # 1. Event counts
        event_counts = df["type"].value_counts().to_dict() if "type" in df.columns else {}

        # 2. Mandatory missingness
        missing_mandatory = {
            col: int(df[col].isna().sum()) for col in self.MANDATORY_COLS if col in df.columns
        }

        # 3. Duplicate checks
        dup_ids = int(df["id"].duplicated().sum()) if "id" in df.columns else 0

        # 4. Coordinate checks
        has_x = "location_x" in df.columns
        has_y = "location_y" in df.columns
        if has_x and has_y:
            valid_coords = validate_coordinates(df, "location_x", "location_y", allow_null=True)
            out_of_bounds = int((~valid_coords).sum())
            missing_locs = int((df["location_x"].isna() & df["type"].isin(["Pass", "Shot", "Carry", "Duel", "Pressure"])).sum())
        else:
            out_of_bounds = 0
            missing_locs = 0

        # 5. Shot integrity
        shots = df[df["type"] == "Shot"] if "type" in df.columns else pd.DataFrame()
        shot_count = len(shots)
        shots_missing_outcome = int(shots["shot_outcome"].isna().sum()) if "shot_outcome" in shots.columns else 0
        shots_missing_body_part = int(shots["shot_body_part"].isna().sum()) if "shot_body_part" in shots.columns else 0

        # 6. Pass integrity
        passes = df[df["type"] == "Pass"] if "type" in df.columns else pd.DataFrame()
        pass_count = len(passes)
        # In StatsBomb, if pass_outcome is NaN / None, the pass was completed successfully!
        completed_passes = int(passes["pass_outcome"].isna().sum()) if "pass_outcome" in passes.columns else 0

        # Flagged count = duplicates + out of bounds + shot anomalies
        flagged_count = dup_ids + out_of_bounds + shots_missing_outcome + shots_missing_body_part
        retained_count = total - dup_ids  # Deduplicated clean records retained

        return QualityAuditResult(
            total_events=total,
            event_counts=event_counts,
            missing_mandatory=missing_mandatory,
            out_of_bounds_coords=out_of_bounds,
            missing_locations=missing_locs,
            duplicate_event_ids=dup_ids,
            shot_count=shot_count,
            shots_missing_outcome=shots_missing_outcome,
            shots_missing_body_part=shots_missing_body_part,
            pass_count=pass_count,
            completed_passes=completed_passes,
            flagged_records_count=flagged_count,
            retained_records_count=retained_count,
        )
