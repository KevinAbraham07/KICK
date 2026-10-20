"""
Machine learning models and evaluation routines for Expected Goals (xG).
"""

from .xg import MatchAwareSplitter, BaselineLogisticXG, GradientBoostingXG
from .evaluation import evaluate_xg_model, ModelEvaluationResult, compare_models

__all__ = [
    "MatchAwareSplitter",
    "BaselineLogisticXG",
    "GradientBoostingXG",
    "evaluate_xg_model",
    "ModelEvaluationResult",
    "compare_models",
]
