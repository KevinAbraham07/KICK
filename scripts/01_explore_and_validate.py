"""
Milestone 1 / Challenge 1: Data Exploration, Schema Inspection, and Validation Pipeline.

This script demonstrates:
1. Discovery of StatsBomb open competitions and seasons.
2. Ingestion of match metadata (Premier League 2015/2016).
3. Deep event inspection on Arsenal vs. Manchester City (Match ID 3754296).
4. Lineup rosters and tactical shape extraction.
5. Schema validation, coordinate checks, and missingness audit.
6. Shot feature engineering exploration (distance, angle, xG distribution).
7. Generation of structured Data Quality Report artifacts.
"""

import sys
from pathlib import Path

# Configure stdout for utf-8 on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src to python path for standalone execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np

from kick.ingestion.loader import StatsBombLoader
from kick.validation.schema import (
    PitchCoordinates,
    DataQualityAuditor,
    compute_goal_distance,
    compute_goal_angle,
    validate_coordinates,
)
from kick.validation.report import generate_markdown_report, save_audit_report


def main():
    print("=" * 80)
    print("KICK — Milestone 1: Data Exploration, Ingestion & Quality Control")
    print("=" * 80)

    loader = StatsBombLoader(cache_dir="data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Competitions Discovery
    print("\n[Step 1/6] Querying available competitions...")
    comps = loader.get_competitions()
    print(f"Total available competition-season packages: {len(comps)}")

    key_competitions = comps[comps["competition_name"].isin([
        "Premier League", "FIFA World Cup", "UEFA Euro", "Champions League", "1. Bundesliga"
    ])][["competition_id", "season_id", "competition_name", "season_name"]].drop_duplicates()
    print("\nSelected Key Competitions:")
    print(key_competitions.head(15).to_string(index=False))

    # 2. Match Ingestion: Premier League 2015/2016
    print("\n[Step 2/6] Loading Premier League 2015/2016 matches...")
    pl_matches = loader.get_matches(competition_id=2, season_id=27)
    print(f"Loaded {len(pl_matches)} matches for Premier League 2015/2016.")

    # Target featured match: Arsenal 2 - 1 Manchester City (Match ID 3754296)
    featured_match_id = 3754296
    match_row = pl_matches[pl_matches["match_id"] == featured_match_id]
    if not match_row.empty:
        row = match_row.iloc[0]
        match_title = f"{row['home_team']} {row['home_score']} - {row['away_score']} {row['away_team']}"
        match_date = str(row['match_date'])
    else:
        match_title = "Arsenal 2 - 1 Manchester City"
        match_date = "2015-12-21"

    print(f"\nFeatured Match: {match_title} ({match_date}) [ID: {featured_match_id}]")

    # 3. Lineup Extraction
    print("\n[Step 3/6] Extracting team lineups...")
    lineups = loader.get_match_lineups(match_id=featured_match_id)
    for team, roster_df in lineups.items():
        starters = roster_df[roster_df["positions"].apply(lambda p: len(p) > 0 and p[0]["start_reason"] == "Starting XI" if isinstance(p, list) and len(p) > 0 and isinstance(p[0], dict) and "start_reason" in p[0] else False)]
        starter_names = starters["player_name"].tolist() if not starters.empty else roster_df["player_name"].head(11).tolist()
        print(f"  {team} (Squad: {len(roster_df)} players, Starters: {len(starter_names)}):")
        print(f"    {', '.join(starter_names[:6])}...")

    # 4. Event Ingestion & Coordinate Extraction
    print("\n[Step 4/6] Ingesting match events and expanding spatial coordinates...")
    events = loader.get_match_events(match_id=featured_match_id, extract_coords=True)
    print(f"Ingested {len(events)} events across {len(events.columns)} columns.")

    # 5. Schema Validation & Data Quality Audit
    print("\n[Step 5/6] Executing Data Quality & Schema Audit...")
    auditor = DataQualityAuditor()
    audit_result = auditor.audit(events)

    report_path = save_audit_report(
        audit_result,
        output_dir=processed_dir,
        filename_prefix="arsenal_mancity_quality_report"
    )
    print(f"Data Quality Report successfully written to: {report_path}")

    audit_summary = audit_result.to_dict()
    print("\n--- Audit Summary ---")
    print(f"Total Events: {audit_summary['total_events']}")
    print(f"Retained Clean Records: {audit_summary['retained_records']}")
    print(f"Duplicate Event IDs: {audit_summary['duplicate_event_ids']}")
    print(f"Out-of-Bounds Coordinates: {audit_summary['out_of_bounds_coords']}")
    print(f"Pass Completion Rate: {audit_summary['pass_summary']['completion_rate_pct']}% ({audit_summary['pass_summary']['completed_passes']}/{audit_summary['pass_summary']['total_passes']})")
    print(f"Total Shots: {audit_summary['shot_summary']['total_shots']} (Missing Outcomes: {audit_summary['shot_summary']['missing_outcome']})")

    # 6. Feature Exploration: Shot Geometry & xG
    print("\n[Step 6/6] Shot Geometry & Feature Engineering Exploration...")
    shots = events[events["type"] == "Shot"].copy()
    shots["distance_to_goal_yards"] = compute_goal_distance(shots["location_x"], shots["location_y"])
    shots["goal_angle_rad"] = compute_goal_angle(shots["location_x"], shots["location_y"])
    shots["goal_angle_deg"] = np.degrees(shots["goal_angle_rad"])

    # Summary by team
    team_shots = shots.groupby("team").agg(
        shots=("id", "count"),
        goals=("shot_outcome", lambda s: (s == "Goal").sum()),
        total_xg=("shot_statsbomb_xg", "sum"),
        avg_distance=("distance_to_goal_yards", "mean"),
        avg_angle_deg=("goal_angle_deg", "mean"),
    ).reset_index()

    print("\nMatch Shot Profile by Team:")
    print(team_shots.to_string(index=False))

    # Inspect the goal events
    goal_events = shots[shots["shot_outcome"] == "Goal"][
        ["player", "team", "minute", "distance_to_goal_yards", "goal_angle_deg", "shot_statsbomb_xg", "shot_body_part"]
    ]
    print("\nGoals Scored in Match:")
    print(goal_events.to_string(index=False))

    # Save cleaned processed datasets to Parquet
    shots_parquet_path = processed_dir / f"shots_{featured_match_id}.parquet"
    shots.to_parquet(shots_parquet_path, index=False)
    print(f"\nSaved processed shots dataset to: {shots_parquet_path}")

    print("\n" + "=" * 80)
    print("Milestone 1 Completed Successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
