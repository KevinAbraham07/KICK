"""
Unit tests for coordinate geometry, boundary validation, and data quality auditing.
"""

import numpy as np
import pandas as pd
import pytest
from kick.ingestion.loader import StatsBombLoader
from kick.validation.schema import (
    PitchCoordinates,
    compute_goal_distance,
    compute_goal_angle,
    validate_coordinates,
    DataQualityAuditor,
)


def test_pitch_geometry_distance():
    # Penalty spot: (108, 40) is exactly 12 yards from goal (120, 40)
    dist = compute_goal_distance(108.0, 40.0)
    assert np.isclose(dist, 12.0)

    # Goal line center: (120, 40)
    dist_zero = compute_goal_distance(120.0, 40.0)
    assert np.isclose(dist_zero, 0.0)

    # Center circle: (60, 40) is 60 yards away
    dist_half = compute_goal_distance(60.0, 40.0)
    assert np.isclose(dist_half, 60.0)


def test_pitch_geometry_angle():
    # Penalty spot: (108, 40)
    # Posts are at (120, 36) and (120, 44). Distance to goal line = 12.
    # Angle is 2 * arctan(4 / 12) = 2 * arctan(1/3) ~ 0.6435 rad (~36.87 degrees)
    angle_pen = compute_goal_angle(108.0, 40.0)
    expected = 2 * np.arctan(4.0 / 12.0)
    assert np.isclose(angle_pen, expected, atol=1e-3)

    # Sharp angle from corner flag (120, 80)
    angle_corner = compute_goal_angle(120.0, 80.0)
    assert angle_corner < 0.1  # Very small visible angle


def test_coordinate_validation():
    df = pd.DataFrame({
        "location_x": [0.0, 60.0, 120.0, -1.0, 125.0, np.nan],
        "location_y": [0.0, 40.0, 80.0, 20.0, 40.0, np.nan],
    })

    # When allowing nulls
    valid_with_null = validate_coordinates(df, allow_null=True)
    assert valid_with_null.tolist() == [True, True, True, False, False, True]

    # When disallowing nulls
    valid_no_null = validate_coordinates(df, allow_null=False)
    assert valid_no_null.tolist() == [True, True, True, False, False, False]


def test_data_quality_auditor(sample_events_df):
    loader = StatsBombLoader()
    df_extracted = loader._extract_coordinates(sample_events_df)

    auditor = DataQualityAuditor()
    result = auditor.audit(df_extracted)

    assert result.total_events == 5
    assert result.shot_count == 2
    assert result.pass_count == 2
    assert result.completed_passes == 1  # event-1 was completed, event-4 was incomplete
    assert result.duplicate_event_ids == 0
    assert result.out_of_bounds_coords == 0
    assert result.shots_missing_outcome == 0
    assert result.shots_missing_body_part == 0
    assert result.retained_records_count == 5


def test_auditor_flags_anomalies():
    bad_df = pd.DataFrame([
        {
            "id": "bad-1",
            "index": 1,
            "period": 1,
            "timestamp": "00:00:10",
            "type": "Shot",
            "location_x": 130.0,  # Out of bounds
            "location_y": 40.0,
            "shot_outcome": None,  # Missing outcome
            "shot_body_part": None,  # Missing body part
        },
        {
            "id": "bad-1",  # Duplicate ID
            "index": 2,
            "period": 1,
            "timestamp": "00:00:15",
            "type": "Pass",
            "location_x": 50.0,
            "location_y": 30.0,
            "pass_outcome": None,
        }
    ])

    auditor = DataQualityAuditor()
    result = auditor.audit(bad_df)

    assert result.total_events == 2
    assert result.duplicate_event_ids == 1
    assert result.out_of_bounds_coords == 1
    assert result.shots_missing_outcome == 1
    assert result.shots_missing_body_part == 1
    assert result.flagged_records_count > 0
