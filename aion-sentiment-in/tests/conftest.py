# =============================================================================
# AION Open-Source Project: Sentiment Analysis Pipeline
# File: tests/conftest.py
# Description: Pytest fixtures and configuration for AION tests
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
Pytest configuration and fixtures for AION Sentiment Analysis tests.

This module provides shared fixtures, test utilities, and configuration
for the test suite.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_headlines() -> list[str]:
    """
    Provide sample news headlines for testing.

    Returns:
        List of sample headline strings.
    """
    return [
        "AAPL Stock SOARS to New Heights! https://example.com",
        "Tesla misses earnings expectations $$$",
        "Microsoft announces new AI partnership",
        "AMZN shares DROP amid regulatory concerns!!!",
        "Google remains stable in volatile market",
    ]


@pytest.fixture
def sample_sentiment_labels() -> list[str]:
    """
    Provide sample sentiment labels for testing.

    Returns:
        List of sample sentiment label strings.
    """
    return ["positive", "negative", "positive", "negative", "neutral"]


@pytest.fixture
def sample_dataframe(sample_headlines: list[str], sample_sentiment_labels: list[str]) -> pd.DataFrame:
    """
    Provide a sample DataFrame for testing data preparation.

    Args:
        sample_headlines: Fixture providing sample headlines.
        sample_sentiment_labels: Fixture providing sample sentiment labels.

    Returns:
        DataFrame with sample data for testing.
    """
    return pd.DataFrame({
        "headline": sample_headlines,
        "sentiment_label": sample_sentiment_labels,
        "publish_date": ["2024-01-15"] * len(sample_headlines),
        "confidence_score": [0.95, 0.87, 0.92, 0.89, 0.78],
        "ticker": ["AAPL", "TSLA", "MSFT", "AMZN", "GOOGL"],
        "close_price": [185.0, 220.0, 380.0, 155.0, 140.0],
        "returns_1d": [2.5, -1.8, 1.2, -2.1, 0.3],
        "returns_3d": [5.2, -3.5, 2.8, -4.1, 0.8],
        "returns_5d": [7.8, -5.2, 4.1, -6.3, 1.2],
    })


@pytest.fixture
def expected_cleaned_headlines() -> list[str]:
    """
    Provide expected cleaned headlines for testing.

    Returns:
        List of expected cleaned headline strings.
    """
    return [
        "aapl stock soars to new heights",
        "tesla misses earnings expectations",
        "microsoft announces new ai partnership",
        "amzn shares drop amid regulatory concerns",
        "google remains stable in volatile market",
    ]


@pytest.fixture
def expected_labels() -> list[int]:
    """
    Provide expected integer-encoded labels for testing.

    Returns:
        List of expected integer labels.
    """
    return [2, 0, 2, 0, 1]


@pytest.fixture
def temp_data_dir(tmp_path: pytest.TempPathFactory) -> pytest.TempPathFactory:
    """
    Provide a temporary directory for test data output.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Temporary path for test data.
    """
    return tmp_path / "test_data"
