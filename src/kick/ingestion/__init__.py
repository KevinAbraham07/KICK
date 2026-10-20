"""
Ingestion module for KICK: acquiring, caching, and loading StatsBomb open event data.
"""

from .loader import StatsBombLoader
from .shot_aggregator import SeasonShotAggregator

__all__ = ["StatsBombLoader", "SeasonShotAggregator"]
