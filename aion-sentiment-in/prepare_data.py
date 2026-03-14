#!/usr/bin/env python3
# =============================================================================
# AION Open-Source Project: Sentiment Analysis Pipeline
# File: prepare_data.py
# Description: Data preparation script for sentiment analysis model training
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
Data Preparation Module for AION Sentiment Analysis.

This module provides functionality to load, clean, and prepare news sentiment
data for training machine learning models. It handles text preprocessing,
label encoding, and train/validation split generation.

Example:
    >>> python prepare_data.py --input data/raw_extracted.csv --output data/
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

# =============================================================================
# CONFIGURATION
# =============================================================================

# Sentiment label mapping (AION standard encoding)
# Supports both full names and ClickHouse Enum8 format (NEG/NEU/POS)
SENTIMENT_LABEL_MAP: dict[str, int] = {
    "positive": 2,
    "neutral": 1,
    "negative": 0,
    "pos": 2,      # ClickHouse Enum8 format
    "neu": 1,
    "neg": 0,
}

# Default configuration values
DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_INPUT_FILE: str = "data/raw_extracted.csv"
DEFAULT_OUTPUT_DIR: str = "data/"

# Text cleaning patterns
URL_PATTERN: str = r"http[s]?://\S+|www\.\S+"
SPECIAL_CHAR_PATTERN: str = r"[^a-zA-Z0-9\s]"
WHITESPACE_PATTERN: str = r"\s+"

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("aion_sentiment.prepare_data")


# =============================================================================
# DATA CLEANING FUNCTIONS
# =============================================================================

def clean_headline(headline: str) -> str:
    """
    Clean a news headline by applying standard text preprocessing.

    This function performs the following cleaning operations:
    1. Converts text to lowercase
    2. Removes URLs (http/https and www links)
    3. Removes special characters (keeping only alphanumeric and spaces)
    4. Removes extra whitespace

    Args:
        headline: The raw headline text to clean.

    Returns:
        The cleaned headline text.

    Example:
        >>> clean_headline("AAPL Stock SOARS! https://example.com $$$")
        'aapl stock soars'
    """
    if not isinstance(headline, str):
        return ""

    # Convert to lowercase
    cleaned = headline.lower()

    # Remove URLs
    cleaned = re.sub(URL_PATTERN, "", cleaned)

    # Remove special characters (keep only alphanumeric and spaces)
    cleaned = re.sub(SPECIAL_CHAR_PATTERN, "", cleaned)

    # Remove extra whitespace
    cleaned = re.sub(WHITESPACE_PATTERN, " ", cleaned).strip()

    return cleaned


def map_sentiment_label(label: str) -> Optional[int]:
    """
    Map a sentiment label string to its integer encoding.

    Args:
        label: The sentiment label string (positive, neutral, negative).

    Returns:
        The integer encoding for the sentiment label, or None if invalid.

    Example:
        >>> map_sentiment_label("positive")
        2
        >>> map_sentiment_label("negative")
        0
    """
    if not isinstance(label, str):
        logger.warning(f"Invalid label type: {type(label)}")
        return None

    label_lower = label.lower().strip()

    if label_lower not in SENTIMENT_LABEL_MAP:
        logger.warning(f"Unknown sentiment label: '{label}'")
        return None

    return SENTIMENT_LABEL_MAP[label_lower]


# =============================================================================
# DATA PREPARATION PIPELINE
# =============================================================================

def load_data(input_path: Path) -> pd.DataFrame:
    """
    Load extracted data from a CSV file.

    Args:
        input_path: Path to the input CSV file.

    Returns:
        DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the input file does not exist.
        pd.errors.EmptyDataError: If the input file is empty.
    """
    logger.info(f"Loading data from {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    return df


def prepare_dataset(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare the dataset for training by cleaning and splitting data.

    This function performs the following steps:
    1. Cleans headlines using standard text preprocessing
    2. Maps sentiment labels to integer encodings
    3. Removes rows with invalid labels
    4. Splits data into train and validation sets (80/20)

    Args:
        df: Input DataFrame with raw data.
        test_size: Proportion of data to use for validation (default: 0.2).
        random_state: Random seed for reproducibility (default: 42).

    Returns:
        Tuple of (train_df, val_df) DataFrames.

    Raises:
        ValueError: If required columns are missing from the input DataFrame.
    """
    required_columns = ["headline", "sentiment_label"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    logger.info("Cleaning headlines...")
    df["headline_clean"] = df["headline"].apply(clean_headline)

    logger.info("Mapping sentiment labels...")
    df["label"] = df["sentiment_label"].apply(map_sentiment_label)

    # Remove rows with invalid labels
    invalid_count = df["label"].isna().sum()
    if invalid_count > 0:
        logger.warning(f"Removing {invalid_count} rows with invalid labels")
        df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)

    # Split into train and validation sets
    logger.info(f"Splitting data (test_size={test_size}, random_state={random_state})...")
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"] if len(df["label"].unique()) > 1 else None,
    )

    logger.info(f"Train set size: {len(train_df)}")
    logger.info(f"Validation set size: {len(val_df)}")

    # Log label distribution
    logger.info("Label distribution in train set:")
    logger.info(train_df["label"].value_counts().sort_index())

    logger.info("Label distribution in validation set:")
    logger.info(val_df["label"].value_counts().sort_index())

    return train_df, val_df


def save_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    Save train and validation datasets to CSV files.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        output_dir: Directory to save the output files.

    Returns:
        Tuple of (train_path, val_path) for the saved files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"

    logger.info(f"Saving train dataset to {train_path}")
    train_df.to_csv(train_path, index=False)

    logger.info(f"Saving validation dataset to {val_path}")
    val_df.to_csv(val_path, index=False)

    return train_path, val_path


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> int:
    """
    Main entry point for the data preparation script.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(
        description="AION Sentiment Analysis - Data Preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_data.py --input data/raw_extracted.csv --output data/
  python prepare_data.py --input extracted.csv --output ./prepared/ --test-size 0.15
        """,
    )

    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path(DEFAULT_INPUT_FILE),
        help=f"Input CSV file path (default: {DEFAULT_INPUT_FILE})",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for train/val CSVs (default: {DEFAULT_OUTPUT_DIR})",
    )

    parser.add_argument(
        "--test-size", "-t",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Validation set proportion (default: {DEFAULT_TEST_SIZE})",
    )

    parser.add_argument(
        "--random-state", "-r",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_STATE})",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        # Load data
        df = load_data(args.input)

        # Prepare dataset
        train_df, val_df = prepare_dataset(
            df,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        # Save datasets
        train_path, val_path = save_datasets(train_df, val_df, args.output)

        logger.info("=" * 60)
        logger.info("AION Data Preparation Complete!")
        logger.info(f"  Train dataset: {train_path} ({len(train_df)} rows)")
        logger.info(f"  Val dataset:   {val_path} ({len(val_df)} rows)")
        logger.info("=" * 60)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data file: {e}")
        return 2
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return 3
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 4


if __name__ == "__main__":
    sys.exit(main())
