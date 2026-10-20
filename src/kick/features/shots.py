"""
Shot feature engineering for Expected Goals (xG) modeling.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd
from kick.validation.schema import compute_goal_distance, compute_goal_angle

# Core feature definitions
BASELINE_FEATURES = [
    "distance_to_goal_yards",
    "goal_angle_deg",
]

FULL_FEATURES = [
    "distance_to_goal_yards",
    "goal_angle_deg",
    "is_header",
    "is_first_time",
    "is_one_on_one",
    "is_under_pressure",
    "is_aerial_won",
    "is_open_play",
    "is_free_kick",
    "is_penalty",
    "followed_cross",
    "followed_through_ball",
    "followed_cut_back",
    "technique_volley",
    "technique_half_volley",
    "technique_lob",
]


class ShotFeatureEngineer:
    """
    Transforms raw shot event datasets into engineered feature representations for modeling.
    """

    def __init__(self, include_penalties: bool = True):
        self.include_penalties = include_penalties

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives geometric, situational, and assisted-pass features from shot records.
        """
        data = df.copy()

        # Filter out records missing coordinates
        data = data.dropna(subset=["location_x", "location_y"]).copy()

        if not self.include_penalties and "shot_type" in data.columns:
            data = data[data["shot_type"] != "Penalty"].copy()

        # 1. Target variable: is_goal (binary 0/1)
        if "shot_outcome" in data.columns:
            data["is_goal"] = (data["shot_outcome"] == "Goal").astype(int)

        # 2. Geometric features
        data["distance_to_goal_yards"] = compute_goal_distance(
            data["location_x"], data["location_y"]
        )
        data["goal_angle_rad"] = compute_goal_angle(
            data["location_x"], data["location_y"]
        )
        data["goal_angle_deg"] = np.degrees(data["goal_angle_rad"])

        # 3. Body part features
        body_part = data.get("shot_body_part", pd.Series("", index=data.index)).fillna("")
        data["is_header"] = (body_part == "Head").astype(int)
        data["is_right_foot"] = (body_part == "Right Foot").astype(int)
        data["is_left_foot"] = (body_part == "Left Foot").astype(int)

        # 4. Situational and pressure flags
        data["is_first_time"] = data.get("shot_first_time", pd.Series(False, index=data.index)).fillna(False).astype(int)
        data["is_one_on_one"] = data.get("shot_one_on_one", pd.Series(False, index=data.index)).fillna(False).astype(int)
        data["is_under_pressure"] = data.get("under_pressure", pd.Series(False, index=data.index)).fillna(False).astype(int)
        data["is_aerial_won"] = data.get("shot_aerial_won", pd.Series(False, index=data.index)).fillna(False).astype(int)

        # 5. Shot play types
        shot_type = data.get("shot_type", pd.Series("Open Play", index=data.index)).fillna("Open Play")
        data["is_open_play"] = (shot_type == "Open Play").astype(int)
        data["is_free_kick"] = (shot_type == "Free Kick").astype(int)
        data["is_penalty"] = (shot_type == "Penalty").astype(int)

        # 6. Assisted pass context
        data["followed_cross"] = data.get("pass_cross", pd.Series(False, index=data.index)).fillna(False).astype(int)
        data["followed_through_ball"] = data.get("pass_through_ball", pd.Series(False, index=data.index)).fillna(False).astype(int)
        data["followed_cut_back"] = data.get("pass_cut_back", pd.Series(False, index=data.index)).fillna(False).astype(int)

        # 7. Technique indicators
        technique = data.get("shot_technique", pd.Series("Normal", index=data.index)).fillna("Normal")
        data["technique_volley"] = (technique == "Volley").astype(int)
        data["technique_half_volley"] = (technique == "Half Volley").astype(int)
        data["technique_lob"] = (technique == "Lob").astype(int)

        return data

    @staticmethod
    def get_feature_matrix(
        df: pd.DataFrame, feature_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts feature matrix X and target vector y as NumPy arrays.
        """
        for f in feature_names:
            if f not in df.columns:
                df[f] = 0

        X = df[feature_names].values.astype(float)
        y = df["is_goal"].values.astype(int) if "is_goal" in df.columns else np.zeros(len(df))
        return X, y
