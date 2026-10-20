"""
Unit tests for xG models, match-aware splitting, and evaluation metrics.
"""

import numpy as np
import pandas as pd
import pytest
from kick.models.xg import MatchAwareSplitter, BaselineLogisticXG, GradientBoostingXG
from kick.models.evaluation import evaluate_xg_model, compute_expected_calibration_error


@pytest.fixture
def synthetic_season_df():
    """
    Generates synthetic match shots across 20 matches.
    """
    np.random.seed(42)
    records = []
    for m in range(20):
        n_shots = np.random.randint(15, 30)
        for s in range(n_shots):
            dist = np.random.uniform(5, 35)
            angle = np.random.uniform(5, 60)
            # Probability inversely related to distance, directly related to angle
            prob = 1 / (1 + np.exp(0.15 * dist - 0.05 * angle - 0.5))
            is_goal = int(np.random.rand() < prob)
            records.append({
                "match_id": 1000 + m,
                "distance_to_goal_yards": dist,
                "goal_angle_deg": angle,
                "is_header": int(np.random.rand() < 0.2),
                "is_first_time": int(np.random.rand() < 0.3),
                "is_one_on_one": int(np.random.rand() < 0.05),
                "is_under_pressure": int(np.random.rand() < 0.4),
                "is_aerial_won": 0,
                "is_open_play": 1,
                "is_free_kick": 0,
                "is_penalty": 0,
                "followed_cross": 0,
                "followed_through_ball": 0,
                "followed_cut_back": 0,
                "technique_volley": 0,
                "technique_half_volley": 0,
                "technique_lob": 0,
                "is_goal": is_goal,
            })
    return pd.DataFrame(records)


def test_match_aware_splitter(synthetic_season_df):
    splitter = MatchAwareSplitter(test_size=0.25, random_state=42)
    train_df, test_df = splitter.split(synthetic_season_df)

    train_matches = set(train_df["match_id"])
    test_matches = set(test_df["match_id"])

    # Must be zero overlap
    assert len(train_matches.intersection(test_matches)) == 0
    # Total matches must equal 20
    assert len(train_matches) + len(test_matches) == 20
    assert len(test_matches) == 5
    assert len(train_matches) == 15


def test_baseline_logistic_xg(synthetic_season_df):
    features = ["distance_to_goal_yards", "goal_angle_deg"]
    X = synthetic_season_df[features].values
    y = synthetic_season_df["is_goal"].values

    model = BaselineLogisticXG(feature_names=features)
    model.fit(X, y)
    probs = model.predict_proba(X)

    # Check bounds
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

    # Check coefficient directions: distance reduces goal probability, angle increases it
    coefs = model.get_coefficients()
    assert coefs["distance_to_goal_yards"] < 0
    assert coefs["goal_angle_deg"] > 0


def test_gradient_boosting_xg(synthetic_season_df):
    features = ["distance_to_goal_yards", "goal_angle_deg", "is_header", "is_first_time"]
    X = synthetic_season_df[features].values
    y = synthetic_season_df["is_goal"].values

    model = GradientBoostingXG(feature_names=features, max_iter=30)
    model.fit(X, y)
    probs = model.predict_proba(X)

    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert len(probs) == len(y)


def test_evaluation_metrics():
    y_true = np.array([0, 1, 0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.1, 0.8, 0.2, 0.7, 0.15, 0.05, 0.9, 0.25])

    res = evaluate_xg_model("TestModel", y_true, y_prob)

    assert res.n_shots == 8
    assert res.n_goals == 3
    assert res.log_loss > 0.0
    assert 0.0 <= res.brier_score <= 1.0
    assert res.roc_auc > 0.9  # Strong ranking
    assert res.expected_calibration_error >= 0.0
