"""
StatsBomb Data Loader with disk caching and coordinate extraction.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd
from statsbombpy import sb

logger = logging.getLogger(__name__)


class StatsBombLoader:
    """
    Robust loader for StatsBomb Open Data with local caching.
    """

    def __init__(self, cache_dir: Union[str, Path] = "data/raw"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_competitions(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch available competitions and seasons.
        """
        cache_file = self.cache_dir / "competitions.parquet"
        if not force_refresh and cache_file.exists():
            try:
                return pd.read_parquet(cache_file)
            except Exception as e:
                logger.warning("Failed to read competitions cache: %s. Re-fetching.", e)

        df = sb.competitions()
        try:
            df.to_parquet(cache_file, index=False)
        except Exception as e:
            logger.warning("Could not cache competitions: %s", e)
        return df

    def get_matches(
        self, competition_id: int, season_id: int, force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch matches for a specific competition and season.
        """
        cache_file = self.cache_dir / f"matches_{competition_id}_{season_id}.parquet"
        if not force_refresh and cache_file.exists():
            try:
                return pd.read_parquet(cache_file)
            except Exception as e:
                logger.warning("Failed to read matches cache: %s. Re-fetching.", e)

        df = sb.matches(competition_id=competition_id, season_id=season_id)
        try:
            # Ensure object columns with mixed types (e.g. manager IDs with multiple IDs) are stringified
            save_df = df.copy()
            for col in save_df.select_dtypes(include=["object"]).columns:
                save_df[col] = save_df[col].astype(str)
            save_df.to_parquet(cache_file, index=False)
        except Exception as e:
            logger.warning("Could not cache matches: %s", e)
        return df

    def get_match_events(
        self, match_id: int, force_refresh: bool = False, extract_coords: bool = True
    ) -> pd.DataFrame:
        """
        Fetch event data for a single match, with optional coordinate extraction.
        """
        cache_file = self.cache_dir / f"events_{match_id}.parquet"
        if not force_refresh and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if extract_coords and "location_x" not in df.columns:
                    df = self._extract_coordinates(df)
                return df
            except Exception as e:
                logger.warning("Failed to read cached events for match %s: %s", match_id, e)

        # sb.events returns a DataFrame of all events for this match
        df = sb.events(match_id=match_id)

        # Convert complex objects (e.g. shot_freeze_frame, tactics) to json string if needed for parquet
        for col in df.columns:
            if df[col].dtype == object and len(df) > 0:
                sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(sample, (dict, list)):
                    # Keep location/coordinates as lists or stringify complex nested structures
                    if col not in ("location", "pass_end_location", "carry_end_location", "shot_end_location", "goalkeeper_end_location"):
                        df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)

        if extract_coords:
            df = self._extract_coordinates(df)

        try:
            df.to_parquet(cache_file, index=False)
        except Exception as e:
            logger.warning("Could not cache events for match %s: %s", match_id, e)

        return df

    def get_match_lineups(
        self, match_id: int, force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch lineups for both teams in a match.
        """
        cache_file = self.cache_dir / f"lineups_{match_id}.json"
        if not force_refresh and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {team: pd.DataFrame(roster) for team, roster in data.items()}
            except Exception as e:
                logger.warning("Failed to read cached lineups: %s", e)

        lineups_dict = sb.lineups(match_id=match_id)
        try:
            serializable = {
                team: df.to_dict(orient="records") for team, df in lineups_dict.items()
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
        except Exception as e:
            logger.warning("Could not cache lineups for match %s: %s", match_id, e)

        return lineups_dict

    @staticmethod
    def _extract_coordinates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract numeric x, y (and z if applicable) columns from coordinate list columns.
        """
        df = df.copy()

        # Primary event location [x, y]
        if "location" in df.columns:
            df["location_x"] = df["location"].apply(
                lambda loc: loc[0] if isinstance(loc, (list, tuple)) and len(loc) >= 1 else None
            )
            df["location_y"] = df["location"].apply(
                lambda loc: loc[1] if isinstance(loc, (list, tuple)) and len(loc) >= 2 else None
            )

        # Pass end location [x, y]
        if "pass_end_location" in df.columns:
            df["pass_end_location_x"] = df["pass_end_location"].apply(
                lambda loc: loc[0] if isinstance(loc, (list, tuple)) and len(loc) >= 1 else None
            )
            df["pass_end_location_y"] = df["pass_end_location"].apply(
                lambda loc: loc[1] if isinstance(loc, (list, tuple)) and len(loc) >= 2 else None
            )

        # Carry end location [x, y]
        if "carry_end_location" in df.columns:
            df["carry_end_location_x"] = df["carry_end_location"].apply(
                lambda loc: loc[0] if isinstance(loc, (list, tuple)) and len(loc) >= 1 else None
            )
            df["carry_end_location_y"] = df["carry_end_location"].apply(
                lambda loc: loc[1] if isinstance(loc, (list, tuple)) and len(loc) >= 2 else None
            )

        # Shot end location [x, y, z]
        if "shot_end_location" in df.columns:
            df["shot_end_location_x"] = df["shot_end_location"].apply(
                lambda loc: loc[0] if isinstance(loc, (list, tuple)) and len(loc) >= 1 else None
            )
            df["shot_end_location_y"] = df["shot_end_location"].apply(
                lambda loc: loc[1] if isinstance(loc, (list, tuple)) and len(loc) >= 2 else None
            )
            df["shot_end_location_z"] = df["shot_end_location"].apply(
                lambda loc: loc[2] if isinstance(loc, (list, tuple)) and len(loc) >= 3 else None
            )

        return df
