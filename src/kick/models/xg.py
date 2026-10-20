"""
Expected Goals (xG) Model implementations: Match-aware splitting, Logistic Regression baseline, and Gradient Boosting.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MatchAwareSplitter:
    """
    Performs leakage-safe train/test splitting grouped by match_id.
    Guarantees that all shots from a given match are assigned strictly to either train or test.
    """

    def __init__(self, test_size: float = 0.20, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits dataframe by match_id.
        """
        if "match_id" not in df.columns:
            raise ValueError("DataFrame must contain 'match_id' column for match-aware splitting.")

        unique_matches = df["match_id"].dropna().unique()
        rng = np.random.default_rng(self.random_state)
        shuffled_matches = rng.permutation(unique_matches)

        n_test = int(len(shuffled_matches) * self.test_size)
        test_match_ids = set(shuffled_matches[:n_test])
        train_match_ids = set(shuffled_matches[n_test:])

        train_df = df[df["match_id"].isin(train_match_ids)].copy()
        test_df = df[df["match_id"].isin(test_match_ids)].copy()

        # Sanity check: zero leakage
        overlap = set(train_df["match_id"]).intersection(set(test_df["match_id"]))
        assert len(overlap) == 0, f"Data leakage detected! Overlapping match IDs: {overlap}"

        return train_df, test_df


class BaselineLogisticXG:
    """
    Interpretable baseline Logistic Regression xG model.
    """

    def __init__(self, feature_names: List[str], C: float = 1.0, random_state: int = 42):
        self.feature_names = feature_names
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=1000, random_state=random_state, solver="lbfgs")),
        ])
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineLogisticXG":
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probability of goal: P(Goal | X).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict_proba.")
        return self.model.predict_proba(X)[:, 1]

    def get_coefficients(self) -> Dict[str, float]:
        """
        Returns dictionary of standardized feature coefficients and intercept.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get coefficients.")
        clf = self.model.named_steps["clf"]
        coefs = dict(zip(self.feature_names, clf.coef_[0].tolist()))
        coefs["intercept"] = float(clf.intercept_[0])
        return coefs

    def save(self, filepath: str | Path):
        joblib.dump({"model": self.model, "features": self.feature_names}, filepath)

    @classmethod
    def load(cls, filepath: str | Path) -> "BaselineLogisticXG":
        data = joblib.load(filepath)
        instance = cls(feature_names=data["features"])
        instance.model = data["model"]
        instance.is_fitted = True
        return instance


class GradientBoostingXG:
    """
    Candidate non-linear Gradient Boosting xG model (HistGradientBoostingClassifier).
    Captures non-linearities and high-order interactions between spatial and situational attributes.
    """

    def __init__(
        self,
        feature_names: List[str],
        learning_rate: float = 0.05,
        max_iter: int = 150,
        min_samples_leaf: int = 20,
        random_state: int = 42,
    ):
        self.feature_names = feature_names
        self.model = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_iter=max_iter,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingXG":
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probability of goal: P(Goal | X).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict_proba.")
        return self.model.predict_proba(X)[:, 1]

    def save(self, filepath: str | Path):
        joblib.dump({"model": self.model, "features": self.feature_names}, filepath)

    @classmethod
    def load(cls, filepath: str | Path) -> "GradientBoostingXG":
        data = joblib.load(filepath)
        instance = cls(feature_names=data["features"])
        instance.model = data["model"]
        instance.is_fitted = True
        return instance
