"""
Unit tests for shot feature engineering.
"""

import numpy as np
import pandas as pd
import pytest
from kick.features.shots import ShotFeatureEngineer, BASELINE_FEATURES, FULL_FEATURES


@pytest.fixture
def mock_shots_df():
    return pd.DataFrame([
        {
            "id": "s1",
            "match_id": 101,
            "location_x": 108.0,
            "location_y": 40.0,
            "shot_outcome": "Goal",
            "shot_body_part": "Right Foot",
            "shot_type": "Penalty",
            "shot_first_time": False,
            "shot_one_on_one": True,
            "under_pressure": False,
            "shot_aerial_won": False,
            "shot_technique": "Normal",
            "pass_cross": False,
            "pass_through_ball": False,
            "pass_cut_back": False,
        },
        {
            "id": "s2",
            "match_id": 101,
            "location_x": 102.0,
            "location_y": 48.0,
            "shot_outcome": "Saved",
            "shot_body_part": "Head",
            "shot_type": "Open Play",
            "shot_first_time": True,
            "shot_one_on_one": False,
            "under_pressure": True,
            "shot_aerial_won": True,
            "shot_technique": "Normal",
            "pass_cross": True,
            "pass_through_ball": False,
            "pass_cut_back": False,
        },
        {
            "id": "s3",
            "match_id": 102,
            "location_x": 95.0,
            "location_y": 30.0,
            "shot_outcome": "Blocked",
            "shot_body_part": "Left Foot",
            "shot_type": "Open Play",
            "shot_first_time": False,
            "shot_one_on_one": False,
            "under_pressure": False,
            "shot_aerial_won": False,
            "shot_technique": "Volley",
            "pass_cross": False,
            "pass_through_ball": True,
            "pass_cut_back": False,
        },
    ])


def test_shot_feature_engineer(mock_shots_df):
    engineer = ShotFeatureEngineer(include_penalties=True)
    df_feat = engineer.engineer_features(mock_shots_df)

    assert len(df_feat) == 3
    assert df_feat.loc[0, "is_goal"] == 1
    assert df_feat.loc[1, "is_goal"] == 0
    assert df_feat.loc[2, "is_goal"] == 0

    # Distance and angle
    assert np.isclose(df_feat.loc[0, "distance_to_goal_yards"], 12.0)
    assert df_feat.loc[0, "is_header"] == 0
    assert df_feat.loc[1, "is_header"] == 1

    # Situational features
    assert df_feat.loc[1, "followed_cross"] == 1
    assert df_feat.loc[1, "is_first_time"] == 1
    assert df_feat.loc[1, "is_under_pressure"] == 1
    assert df_feat.loc[2, "followed_through_ball"] == 1
    assert df_feat.loc[2, "technique_volley"] == 1


def test_feature_matrix_shapes(mock_shots_df):
    engineer = ShotFeatureEngineer()
    df_feat = engineer.engineer_features(mock_shots_df)

    X_base, y_base = engineer.get_feature_matrix(df_feat, BASELINE_FEATURES)
    assert X_base.shape == (3, 2)
    assert y_base.tolist() == [1, 0, 0]

    X_full, y_full = engineer.get_feature_matrix(df_feat, FULL_FEATURES)
    assert X_full.shape == (3, len(FULL_FEATURES))
