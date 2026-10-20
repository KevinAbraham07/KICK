"""
Pytest fixtures for KICK testing.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_events_df():
    """
    Constructs a synthetic yet structurally authentic events DataFrame for unit tests.
    """
    return pd.DataFrame([
        {
            "id": "event-1",
            "index": 1,
            "period": 1,
            "timestamp": "00:01:05.123",
            "type": "Pass",
            "player": "Kevin De Bruyne",
            "team": "Manchester City",
            "location": [60.0, 40.0],
            "pass_end_location": [85.0, 30.0],
            "pass_outcome": None,  # Completed pass
        },
        {
            "id": "event-2",
            "index": 2,
            "period": 1,
            "timestamp": "00:01:10.500",
            "type": "Ball Receipt*",
            "player": "David Silva",
            "team": "Manchester City",
            "location": [85.0, 30.0],
        },
        {
            "id": "event-3",
            "index": 3,
            "period": 1,
            "timestamp": "00:01:15.800",
            "type": "Shot",
            "player": "Sergio Aguero",
            "team": "Manchester City",
            "location": [108.0, 40.0],  # Penalty spot
            "shot_end_location": [120.0, 41.5, 1.2],
            "shot_outcome": "Goal",
            "shot_body_part": "Right Foot",
            "shot_type": "Open Play",
        },
        {
            "id": "event-4",
            "index": 4,
            "period": 1,
            "timestamp": "00:02:00.000",
            "type": "Pass",
            "player": "Mesut Ozil",
            "team": "Arsenal",
            "location": [70.0, 20.0],
            "pass_end_location": [90.0, 25.0],
            "pass_outcome": "Incomplete",  # Incomplete pass
        },
        {
            "id": "event-5",
            "index": 5,
            "period": 1,
            "timestamp": "00:02:10.000",
            "type": "Shot",
            "player": "Theo Walcott",
            "team": "Arsenal",
            "location": [100.0, 55.0],
            "shot_end_location": [120.0, 35.0, 0.5],
            "shot_outcome": "Saved",
            "shot_body_part": "Right Foot",
            "shot_type": "Open Play",
        },
    ])
