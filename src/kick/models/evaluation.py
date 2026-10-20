"""
Comprehensive evaluation routines for xG models:
Log loss, Brier score, ROC-AUC, Calibration curves, and Expected Calibration Error (ECE).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve


@dataclass
class ModelEvaluationResult:
    """
    Evaluation metrics on a held-out dataset.
    """
    model_name: str
    n_shots: int
    n_goals: int
    actual_goal_rate: float
    total_predicted_xg: float
    predicted_goal_rate: float
    log_loss: float
    brier_score: float
    roc_auc: float
    expected_calibration_error: float
    calibration_bins: List[Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "n_shots": self.n_shots,
            "n_goals": self.n_goals,
            "actual_goal_rate": round(self.actual_goal_rate, 4),
            "total_predicted_xg": round(self.total_predicted_xg, 2),
            "predicted_goal_rate": round(self.predicted_goal_rate, 4),
            "log_loss": round(self.log_loss, 4),
            "brier_score": round(self.brier_score, 4),
            "roc_auc": round(self.roc_auc, 4),
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "xg_minus_goals": round(self.total_predicted_xg - self.n_goals, 2),
            "calibration_bins": self.calibration_bins,
        }


def compute_expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Tuple[float, List[Dict[str, float]]]:
    """
    Computes Expected Calibration Error (ECE) and returns binned stats.
    ECE = sum(|acc(B_m) - conf(B_m)| * (|B_m| / N))
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_data = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (y_prob >= low) & (y_prob <= high if i == n_bins - 1 else y_prob < high)
        n_in_bin = int(mask.sum())

        if n_in_bin > 0:
            avg_prob = float(y_prob[mask].mean())
            actual_rate = float(y_true[mask].mean())
            bin_error = abs(actual_rate - avg_prob)
            ece += bin_error * (n_in_bin / len(y_true))
            bins_data.append({
                "bin": i + 1,
                "range": f"[{low:.2f}, {high:.2f}]",
                "count": n_in_bin,
                "avg_predicted_xg": round(avg_prob, 4),
                "actual_goal_rate": round(actual_rate, 4),
                "diff": round(actual_rate - avg_prob, 4),
            })
        else:
            bins_data.append({
                "bin": i + 1,
                "range": f"[{low:.2f}, {high:.2f}]",
                "count": 0,
                "avg_predicted_xg": 0.0,
                "actual_goal_rate": 0.0,
                "diff": 0.0,
            })

    return ece, bins_data


def evaluate_xg_model(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> ModelEvaluationResult:
    """
    Computes standard football analytics evaluation suite on predicted probabilities.
    """
    # Clip probabilities to avoid infinite log loss
    y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)

    ll = log_loss(y_true, y_prob_clipped)
    bs = brier_score_loss(y_true, y_prob_clipped)
    try:
        auc = roc_auc_score(y_true, y_prob_clipped)
    except ValueError:
        auc = float("nan")

    ece, bin_stats = compute_expected_calibration_error(y_true, y_prob_clipped, n_bins=n_bins)

    n_shots = len(y_true)
    n_goals = int(np.sum(y_true))
    tot_xg = float(np.sum(y_prob))

    return ModelEvaluationResult(
        model_name=model_name,
        n_shots=n_shots,
        n_goals=n_goals,
        actual_goal_rate=n_goals / max(1, n_shots),
        total_predicted_xg=tot_xg,
        predicted_goal_rate=tot_xg / max(1, n_shots),
        log_loss=float(ll),
        brier_score=float(bs),
        roc_auc=float(auc),
        expected_calibration_error=float(ece),
        calibration_bins=bin_stats,
    )


def compare_models(results: List[ModelEvaluationResult]) -> pd.DataFrame:
    """
    Produces a tidy comparison DataFrame for multiple models.
    """
    rows = []
    for r in results:
        rows.append({
            "Model": r.model_name,
            "Shots": r.n_shots,
            "Actual Goals": r.n_goals,
            "Predicted xG": round(r.total_predicted_xg, 2),
            "Log Loss": round(r.log_loss, 4),
            "Brier Score": round(r.brier_score, 4),
            "ROC-AUC": round(r.roc_auc, 4),
            "ECE": round(r.expected_calibration_error, 4),
        })
    return pd.DataFrame(rows)
