"""
Milestone 2: Train and evaluate Expected Goals (xG) models.

Pipeline steps:
1. Aggregate season-wide shot dataset across Premier League 2015/2016.
2. Validate coverage, event schema, and missing data.
3. Engineer spatial, situational, body part, and assist-pass features.
4. Perform leakage-safe match-aware train/test split (80/20).
5. Train Baseline Logistic Regression vs. Candidate Gradient Boosting.
6. Rigorously evaluate using Log Loss, Brier Score, ROC-AUC, ECE, and Calibration.
7. Benchmark against StatsBomb's proprietary xG.
8. Persist trained models and evaluation reports.
"""

import sys
from pathlib import Path

# Configure stdout for utf-8 on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import json
import numpy as np
import pandas as pd

from kick.ingestion.shot_aggregator import SeasonShotAggregator
from kick.features.shots import (
    ShotFeatureEngineer,
    BASELINE_FEATURES,
    FULL_FEATURES,
)
from kick.models.xg import (
    MatchAwareSplitter,
    BaselineLogisticXG,
    GradientBoostingXG,
)
from kick.models.evaluation import (
    evaluate_xg_model,
    compare_models,
)


def main():
    print("=" * 80)
    print("KICK — Milestone 2: Expected Goals (xG) Modelling & Rigorous Evaluation")
    print("=" * 80)

    models_dir = PROJECT_ROOT / "models"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Build Season-Wide Shot Dataset
    # ---------------------------------------------------------
    print("\n[Step 1/5] Ingesting & aggregating season shot dataset...")
    aggregator = SeasonShotAggregator(
        cache_dir=str(PROJECT_ROOT / "data" / "raw"),
        processed_dir=str(processed_dir),
    )

    # Aggregating across 50 Premier League matches (~1,300 shots, ideal for fast local execution & evaluation)
    shots_df = aggregator.aggregate_season(
        competition_id=2,
        season_id=27,
        max_matches=50,
        force_refresh=False,
    )

    print(f"Aggregated {len(shots_df):,} shots across {shots_df['match_id'].nunique()} matches.")

    # ---------------------------------------------------------
    # 2. Coverage & Data Quality Profile
    # ---------------------------------------------------------
    print("\n[Step 2/5] Quantifying dataset coverage & missingness...")
    missing_profile = {
        col: int(shots_df[col].isna().sum())
        for col in ["location_x", "location_y", "shot_outcome", "shot_body_part", "shot_type"]
    }
    outcome_counts = shots_df["shot_outcome"].value_counts().to_dict()
    total_goals = outcome_counts.get("Goal", 0)
    conversion_rate = total_goals / max(1, len(shots_df)) * 100

    print(f"Total Goals: {total_goals} ({conversion_rate:.2f}% conversion rate)")
    print(f"Missing Coordinate Records: {missing_profile['location_x']}")
    print(f"Shot Outcomes: {outcome_counts}")

    # ---------------------------------------------------------
    # 3. Feature Engineering
    # ---------------------------------------------------------
    print("\n[Step 3/5] Engineering spatial, situational, and assist features...")
    engineer = ShotFeatureEngineer(include_penalties=True)
    df_feat = engineer.engineer_features(shots_df)
    print(f"Engineered {len(FULL_FEATURES)} features for {len(df_feat):,} shots.")

    # ---------------------------------------------------------
    # 4. Leakage-Safe Match-Aware Splitting
    # ---------------------------------------------------------
    print("\n[Step 4/5] Splitting dataset by match (80% Train / 20% Test)...")
    splitter = MatchAwareSplitter(test_size=0.20, random_state=42)
    train_df, test_df = splitter.split(df_feat)

    print(f"Train set: {len(train_df):,} shots from {train_df['match_id'].nunique()} matches (Goals: {train_df['is_goal'].sum()})")
    print(f"Test set:  {len(test_df):,} shots from {test_df['match_id'].nunique()} matches (Goals: {test_df['is_goal'].sum()})")

    # Matrices for training
    X_train_base, y_train = engineer.get_feature_matrix(train_df, BASELINE_FEATURES)
    X_test_base, y_test = engineer.get_feature_matrix(test_df, BASELINE_FEATURES)

    X_train_full, _ = engineer.get_feature_matrix(train_df, FULL_FEATURES)
    X_test_full, _ = engineer.get_feature_matrix(test_df, FULL_FEATURES)

    # ---------------------------------------------------------
    # 5. Train Models & Evaluate
    # ---------------------------------------------------------
    print("\n[Step 5/5] Training models and computing evaluation metrics...")

    # A. Baseline Logistic Regression (Distance & Angle only)
    lr_base = BaselineLogisticXG(feature_names=BASELINE_FEATURES)
    lr_base.fit(X_train_base, y_train)
    p_lr_base = lr_base.predict_proba(X_test_base)
    eval_lr_base = evaluate_xg_model("Logistic Regression (Baseline: Dist + Angle)", y_test, p_lr_base)

    # B. Full Logistic Regression (All Features)
    lr_full = BaselineLogisticXG(feature_names=FULL_FEATURES)
    lr_full.fit(X_train_full, y_train)
    p_lr_full = lr_full.predict_proba(X_test_full)
    eval_lr_full = evaluate_xg_model("Logistic Regression (Full Features)", y_test, p_lr_full)

    # C. Candidate Non-linear Model (Gradient Boosting)
    gb_model = GradientBoostingXG(feature_names=FULL_FEATURES, max_iter=120, min_samples_leaf=20, random_state=42)
    gb_model.fit(X_train_full, y_train)
    p_gb = gb_model.predict_proba(X_test_full)
    eval_gb = evaluate_xg_model("Gradient Boosting (HistGBM - Full)", y_test, p_gb)

    # D. StatsBomb Benchmark xG (where available on test set)
    has_sb_xg = test_df["shot_statsbomb_xg"].notna()
    if has_sb_xg.sum() > 0:
        y_test_sb = y_test[has_sb_xg]
        p_sb = test_df.loc[has_sb_xg, "shot_statsbomb_xg"].values.astype(float)
        eval_sb = evaluate_xg_model("StatsBomb Benchmark (Proprietary)", y_test_sb, p_sb)
        all_evals = [eval_lr_base, eval_lr_full, eval_gb, eval_sb]
    else:
        all_evals = [eval_lr_base, eval_lr_full, eval_gb]

    # Save models
    lr_base.save(models_dir / "baseline_logistic_xg.joblib")
    lr_full.save(models_dir / "full_logistic_xg.joblib")
    gb_model.save(models_dir / "gradient_boosting_xg.joblib")

    # Save test predictions for visualization and inspection
    test_df_out = test_df.copy()
    test_df_out["xg_lr_base"] = p_lr_base
    test_df_out["xg_lr_full"] = p_lr_full
    test_df_out["xg_gradient_boosting"] = p_gb
    test_df_out.to_parquet(processed_dir / "test_set_predictions.parquet", index=False)

    # Display comparison table
    comparison_df = compare_models(all_evals)
    print("\n" + "=" * 80)
    print("MODEL COMPARISON (HELD-OUT TEST MATCHES)")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    # Display Logistic Regression Coefficients
    print("\n" + "-" * 80)
    print("LOGISTIC REGRESSION COEFFICIENTS (Interpretable Effects):")
    print("-" * 80)
    coefs = lr_full.get_coefficients()
    for feat, val in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat:<25}: {val:+.4f}")

    # Generate and save Markdown Report
    report_md_path = processed_dir / "xg_model_evaluation_report.md"
    report_json_path = processed_dir / "xg_model_evaluation_report.json"

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in all_evals], f, indent=2)

    report_lines = [
        "# Expected Goals (xG) Model Evaluation Report",
        f"**Dataset**: Premier League 2015/2016 ({shots_df['match_id'].nunique()} matches, {len(shots_df):,} shots)",
        f"**Train Matches**: {train_df['match_id'].nunique()} ({len(train_df):,} shots) | **Test Matches**: {test_df['match_id'].nunique()} ({len(test_df):,} shots)",
        f"**Actual Goals on Test Set**: `{int(y_test.sum())}` (Goal conversion rate: `{y_test.mean()*100:.2f}%`)",
        "",
        "## 1. Held-Out Performance Comparison",
        "| Model | Log Loss | Brier Score | ROC-AUC | Expected Calibration Error (ECE) | Total Pred xG | Actual Goals |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for e in all_evals:
        report_lines.append(
            f"| **{e.model_name}** | `{e.log_loss:.4f}` | `{e.brier_score:.4f}` | `{e.roc_auc:.4f}` | `{e.expected_calibration_error:.4f}` | `{e.total_predicted_xg:.1f}` | `{e.n_goals}` |"
        )

    report_lines.extend([
        "",
        "## 2. Model Calibration Breakdown (HistGBM Candidate)",
        "| Bin Range | Shots Count | Mean Predicted xG | Observed Goal Rate | Calibration Error |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    for b in eval_gb.calibration_bins:
        if b["count"] > 0:
            report_lines.append(
                f"| `{b['range']}` | `{b['count']}` | `{b['avg_predicted_xg']:.3f}` | `{b['actual_goal_rate']:.3f}` | `{b['diff']:+.3f}` |"
            )

    report_lines.extend([
        "",
        "## 3. Interpretable Factor Weights (Standardized Logistic Regression)",
        "| Feature | Coefficient | Qualitative Direction |",
        "| :--- | :--- | :--- |",
    ])

    for feat, val in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True):
        if feat == "intercept":
            continue
        direction = "Higher xG" if val > 0 else "Lower xG"
        report_lines.append(f"| `{feat}` | `{val:+.4f}` | {direction} |")

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nSaved evaluation report to: {report_md_path}")
    print("=" * 80)
    print("Milestone 2 Completed Successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
