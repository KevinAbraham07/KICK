"""
Unit tests for data loader coordinate extraction and cache handling.
"""

import pandas as pd
import numpy as np
import pytest
from kick.ingestion.loader import StatsBombLoader


def test_coordinate_extraction(sample_events_df):
    loader = StatsBombLoader()
    extracted = loader._extract_coordinates(sample_events_df)

    assert "location_x" in extracted.columns
    assert "location_y" in extracted.columns
    assert "pass_end_location_x" in extracted.columns
    assert "pass_end_location_y" in extracted.columns
    assert "shot_end_location_x" in extracted.columns
    assert "shot_end_location_z" in extracted.columns

    # Event 1 (Pass)
    assert extracted.loc[0, "location_x"] == 60.0
    assert extracted.loc[0, "location_y"] == 40.0
    assert extracted.loc[0, "pass_end_location_x"] == 85.0
    assert extracted.loc[0, "pass_end_location_y"] == 30.0

    # Event 3 (Shot)
    assert extracted.loc[2, "location_x"] == 108.0
    assert extracted.loc[2, "location_y"] == 40.0
    assert extracted.loc[2, "shot_end_location_x"] == 120.0
    assert extracted.loc[2, "shot_end_location_z"] == 1.2


def test_coordinate_extraction_null_handling():
    loader = StatsBombLoader()
    df = pd.DataFrame([
        {"id": "e1", "location": None},
        {"id": "e2", "location": []},
        {"id": "e3", "location": [50.0]},
    ])
    extracted = loader._extract_coordinates(df)
    assert pd.isna(extracted.loc[0, "location_x"])
    assert pd.isna(extracted.loc[1, "location_x"])
    assert extracted.loc[2, "location_x"] == 50.0
    assert pd.isna(extracted.loc[2, "location_y"])
