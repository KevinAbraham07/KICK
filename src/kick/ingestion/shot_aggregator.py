"""
Fast asynchronous aggregator for season-wide match events and shot extraction.
"""

from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OPEN_DATA_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


class SeasonShotAggregator:
    """
    Downloads raw match event JSONs with high concurrency and extracts
    structured shot records and their assisted passes.
    """

    def __init__(self, cache_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.cache_dir = Path(cache_dir)
        self.processed_dir = Path(processed_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def extract_shots_from_events(events: List[Dict[str, Any]], match_id: int) -> List[Dict[str, Any]]:
        """
        Extracts shot records from a raw match event list, joining assisted passes where present.
        """
        # Map passes by event id for key-pass feature lookup
        passes_by_id = {}
        for ev in events:
            if ev.get("type", {}).get("name") == "Pass":
                passes_by_id[ev.get("id")] = ev

        shots_extracted = []
        for ev in events:
            if ev.get("type", {}).get("name") != "Shot":
                continue

            shot_data = ev.get("shot", {})
            loc = ev.get("location", [None, None])
            loc_x = loc[0] if len(loc) >= 1 else None
            loc_y = loc[1] if len(loc) >= 2 else None

            end_loc = shot_data.get("end_location", [None, None, None])
            end_x = end_loc[0] if len(end_loc) >= 1 else None
            end_y = end_loc[1] if len(end_loc) >= 2 else None
            end_z = end_loc[2] if len(end_loc) >= 3 else None

            # Assisted pass linkage
            key_pass_id = shot_data.get("key_pass_id")
            assisted_pass = passes_by_id.get(key_pass_id, {})
            pass_data = assisted_pass.get("pass", {})

            rec = {
                "id": ev.get("id"),
                "match_id": match_id,
                "index": ev.get("index"),
                "period": ev.get("period"),
                "timestamp": ev.get("timestamp"),
                "minute": ev.get("minute"),
                "second": ev.get("second"),
                "team": ev.get("team", {}).get("name"),
                "team_id": ev.get("team", {}).get("id"),
                "player": ev.get("player", {}).get("name"),
                "player_id": ev.get("player", {}).get("id"),
                "position": ev.get("position", {}).get("name"),
                "play_pattern": ev.get("play_pattern", {}).get("name"),
                "under_pressure": bool(ev.get("under_pressure", False)),
                "location_x": loc_x,
                "location_y": loc_y,
                "shot_end_location_x": end_x,
                "shot_end_location_y": end_y,
                "shot_end_location_z": end_z,
                "shot_outcome": shot_data.get("outcome", {}).get("name"),
                "shot_type": shot_data.get("type", {}).get("name"),
                "shot_body_part": shot_data.get("body_part", {}).get("name"),
                "shot_technique": shot_data.get("technique", {}).get("name"),
                "shot_first_time": bool(shot_data.get("first_time", False)),
                "shot_one_on_one": bool(shot_data.get("one_on_one", False)),
                "shot_aerial_won": bool(shot_data.get("aerial_won", False)),
                "shot_statsbomb_xg": shot_data.get("statsbomb_xg"),
                "key_pass_id": key_pass_id,
                # Assisted pass context
                "has_assisted_pass": key_pass_id is not None,
                "pass_type": pass_data.get("type", {}).get("name"),
                "pass_cross": bool(pass_data.get("cross", False)),
                "pass_through_ball": bool(pass_data.get("through_ball", False)),
                "pass_cut_back": bool(pass_data.get("cut_back", False)),
                "pass_switch": bool(pass_data.get("switch", False)),
                "pass_length": pass_data.get("length"),
                "pass_angle": pass_data.get("angle"),
            }
            shots_extracted.append(rec)

        return shots_extracted

    async def _fetch_and_extract_match(
        self, client: httpx.AsyncClient, match_id: int, semaphore: asyncio.Semaphore
    ) -> List[Dict[str, Any]]:
        """
        Fetch single match events and extract shots.
        """
        cache_file = self.cache_dir / f"events_{match_id}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    events = json.load(f)
                return self.extract_shots_from_events(events, match_id)
            except Exception:
                pass

        url = f"{OPEN_DATA_BASE_URL}/events/{match_id}.json"
        async with semaphore:
            for attempt in range(3):
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        events = resp.json()
                        return self.extract_shots_from_events(events, match_id)
                    elif resp.status_code == 404:
                        logger.warning("Match %s not found (404)", match_id)
                        return []
                except Exception as e:
                    if attempt == 2:
                        logger.warning("Error fetching match %s: %s", match_id, e)
                    await asyncio.sleep(0.5 * (attempt + 1))
        return []

    async def aggregate_season_shots_async(
        self, match_ids: List[int], max_connections: int = 25
    ) -> pd.DataFrame:
        """
        Asynchronously fetches matches and aggregates shots across all supplied match IDs.
        """
        limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections)
        semaphore = asyncio.Semaphore(max_connections)
        timeout = httpx.Timeout(30.0, connect=15.0)

        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            tasks = [self._fetch_and_extract_match(client, mid, semaphore) for mid in match_ids]
            results = await asyncio.gather(*tasks)

        all_shots = [shot for match_shots in results for shot in match_shots]
        df = pd.DataFrame(all_shots)
        return df

    def aggregate_season(
        self,
        competition_id: int = 2,
        season_id: int = 27,
        max_matches: Optional[int] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Synchronous entry point that loads season match index and aggregates all shots.
        Saves output to Parquet.
        """
        suffix = f"_{max_matches}" if max_matches is not None else ""
        parquet_path = self.processed_dir / f"season_{competition_id}_{season_id}_shots{suffix}.parquet"
        if not force_refresh and parquet_path.exists():
            logger.info("Loading cached season shots from %s", parquet_path)
            return pd.read_parquet(parquet_path)

        matches_file = self.cache_dir / f"matches_{competition_id}_{season_id}.parquet"
        if not matches_file.exists():
            from kick.ingestion.loader import StatsBombLoader
            loader = StatsBombLoader(cache_dir=self.cache_dir)
            matches_df = loader.get_matches(competition_id, season_id)
        else:
            matches_df = pd.read_parquet(matches_file)

        match_ids = matches_df["match_id"].tolist()
        if max_matches is not None:
            match_ids = match_ids[:max_matches]

        logger.info("Aggregating shots across %d matches...", len(match_ids))
        df_shots = asyncio.run(self.aggregate_season_shots_async(match_ids))

        # Save to processed directory
        df_shots.to_parquet(parquet_path, index=False)
        logger.info("Saved %d aggregated shots to %s", len(df_shots), parquet_path)
        return df_shots
