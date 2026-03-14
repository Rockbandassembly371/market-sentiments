# =============================================================================
# AION Open-Source Project: Sentiment Analysis Pipeline
# File: tests/test_prepare_data.py
# Description: Unit tests for data preparation module
# License: Apache License, Version 2.0
# 
# Copyright 2026 AION Contributors
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# AION Attribution: This file is part of the AION open-source ecosystem.
# Please visit https://github.com/aion for more information.
# =============================================================================
"""
Unit tests for the AION data preparation module.

This module contains tests for the prepare_data.py script, including:
- Text cleaning functions
- Sentiment label mapping
- Dataset preparation pipeline
- File I/O operations
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prepare_data import clean_headline, map_sentiment_label, prepare_dataset


class TestCleanHeadline:
    """Tests for the clean_headline function."""

    def test_lowercase_conversion(self) -> None:
        """Test that headlines are converted to lowercase."""
        headline = "AAPL STOCK SOARS"
        result = clean_headline(headline)
        assert result == "aapl stock soars"

    def test_url_removal(self) -> None:
        """Test that URLs are removed from headlines."""
        headline = "Check this out https://example.com/article"
        result = clean_headline(headline)
        assert "http" not in result
        assert "example" not in result

    def test_www_url_removal(self) -> None:
        """Test that www URLs are removed from headlines."""
        headline = "Visit www.example.com for more"
        result = clean_headline(headline)
        assert "www" not in result
        assert "example" not in result

    def test_special_character_removal(self) -> None:
        """Test that special characters are removed."""
        headline = "Stock rises!!! $$$ #trending"
        result = clean_headline(headline)
        assert "$" not in result
        assert "#" not in result
        assert "!!!" not in result

    def test_whitespace_normalization(self) -> None:
        """Test that extra whitespace is normalized."""
        headline = "Stock    rises    with   extra   spaces"
        result = clean_headline(headline)
        assert "  " not in result
        assert result == "stock rises with extra spaces"

    def test_combined_cleaning(
        self,
        sample_headlines: list[str],
        expected_cleaned_headlines: list[str],
    ) -> None:
        """Test combined cleaning operations on sample headlines."""
        for headline, expected in zip(sample_headlines, expected_cleaned_headlines):
            result = clean_headline(headline)
            assert result == expected

    def test_empty_string(self) -> None:
        """Test handling of empty strings."""
        assert clean_headline("") == ""

    def test_none_input(self) -> None:
        """Test handling of None input."""
        assert clean_headline(None) == ""  # type: ignore[arg-type]

    def test_non_string_input(self) -> None:
        """Test handling of non-string input."""
        assert clean_headline(123) == ""  # type: ignore[arg-type]


class TestMapSentimentLabel:
    """Tests for the map_sentiment_label function."""

    def test_positive_label(self) -> None:
        """Test mapping of positive label."""
        assert map_sentiment_label("positive") == 2

    def test_neutral_label(self) -> None:
        """Test mapping of neutral label."""
        assert map_sentiment_label("neutral") == 1

    def test_negative_label(self) -> None:
        """Test mapping of negative label."""
        assert map_sentiment_label("negative") == 0

    def test_case_insensitive(self) -> None:
        """Test that label mapping is case-insensitive."""
        assert map_sentiment_label("POSITIVE") == 2
        assert map_sentiment_label("Neutral") == 1
        assert map_sentiment_label("NEGATIVE") == 0

    def test_whitespace_handling(self) -> None:
        """Test that whitespace is handled correctly."""
        assert map_sentiment_label(" positive ") == 2
        assert map_sentiment_label("\tnegative\n") == 0

    def test_invalid_label(self) -> None:
        """Test handling of invalid labels."""
        assert map_sentiment_label("unknown") is None
        assert map_sentiment_label("") is None

    def test_none_input(self) -> None:
        """Test handling of None input."""
        assert map_sentiment_label(None) is None  # type: ignore[arg-type]


class TestPrepareDataset:
    """Tests for the prepare_dataset function."""

    def test_dataset_split_size(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that dataset is split into correct sizes."""
        train_df, val_df = prepare_dataset(sample_dataframe, test_size=0.2, random_state=42)

        total = len(sample_dataframe)
        train_size = int(total * 0.8)
        val_size = int(total * 0.2)

        assert len(train_df) == train_size
        assert len(val_df) == val_size
        assert len(train_df) + len(val_df) == total

    def test_cleaned_headline_column(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that cleaned headline column is added."""
        train_df, val_df = prepare_dataset(sample_dataframe)

        assert "headline_clean" in train_df.columns
        assert "headline_clean" in val_df.columns

    def test_label_column_added(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that integer label column is added."""
        train_df, val_df = prepare_dataset(sample_dataframe)

        assert "label" in train_df.columns
        assert "label" in val_df.columns
        assert train_df["label"].dtype in [int, "int64", "int32"]

    def test_label_values_correct(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that label values are correctly mapped."""
        train_df, val_df = prepare_dataset(sample_dataframe, random_state=42)

        combined_df = pd.concat([train_df, val_df])
        unique_labels = set(combined_df["label"].unique())

        # Should contain valid sentiment labels
        assert unique_labels.issubset({0, 1, 2})

    def test_missing_required_column(self) -> None:
        """Test handling of missing required columns."""
        df = pd.DataFrame({"other_column": ["test"]})

        with pytest.raises(ValueError, match="Missing required columns"):
            prepare_dataset(df)

    def test_reproducibility(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that results are reproducible with same random state."""
        train1, val1 = prepare_dataset(sample_dataframe, random_state=42)
        train2, val2 = prepare_dataset(sample_dataframe, random_state=42)

        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)
