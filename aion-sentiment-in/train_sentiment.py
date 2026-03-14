#!/usr/bin/env python3
# =============================================================================
# AION Open-Source Project: Sentiment Analysis Training Pipeline
# File: train_sentiment.py
# Description: Training script for AION-Sentiment-IN model using FinBERT
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
AION Sentiment Analysis Training Script.

This module provides a complete training pipeline for fine-tuning sentiment
analysis models on Indian financial news data. It supports FinBERT as the
primary model with DistilBERT as a fallback option.

Key Features:
    - Automatic model selection (FinBERT preferred, DistilBERT fallback)
    - Data loading from CSV files with headline and label columns
    - Tokenization and preprocessing using HuggingFace transformers
    - Training with evaluation metrics after each epoch
    - Best model checkpointing based on validation F1 score
    - Comprehensive logging and progress reporting

Example:
    >>> python train_sentiment.py --data_dir data --output_dir models
    >>> python train_sentiment.py --model_name distilbert-base-uncased --epochs 5

Attributes:
    DEFAULT_MODEL_NAME (str): Default model to use for fine-tuning.
    FALLBACK_MODEL_NAME (str): Fallback model if primary is unavailable.
    NUM_LABELS (int): Number of sentiment classes (negative, neutral, positive).

"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_callback import TrainerControl, TrainerState, TrainerCallback

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

DEFAULT_MODEL_NAME: str = "ProsusAI/finbert"
FALLBACK_MODEL_NAME: str = "distilbert-base-uncased"
NUM_LABELS: int = 3  # negative, neutral, positive
MAX_LENGTH: int = 128

# Configure logging first (before device detection)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# =============================================================================
# DEVICE CONFIGURATION (Apple Silicon MPS Support)
# =============================================================================


def get_device() -> torch.device:
    """
    Get the optimal available device for training.

    Priority order:
    1. MPS (Apple Silicon GPU) - Best for M1/M2/M3/M4 Macs
    2. CUDA (NVIDIA GPU) - Standard GPU support
    3. CPU - Fallback for systems without GPU

    Returns:
        torch.device: The selected device for training.

    Example:
        >>> device = get_device()
        >>> print(f"Using device: {device}")
        Using device: mps

    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("✓ Apple Silicon MPS (Metal Performance Shaders) detected")
        logger.info("  Using GPU acceleration for Mac M1/M2/M3/M4")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"✓ NVIDIA CUDA detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("⚠ No GPU detected. Training will use CPU (slower).")
        logger.warning("  For faster training, use a Mac with M1/M2/M3/M4 or NVIDIA GPU.")

    return device


# Set global device
DEVICE = get_device()


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================


def load_data_from_csv(
    train_path: str, val_path: str
) -> Tuple[Dataset, Dataset]:
    """
    Load training and validation datasets from CSV files.

    Args:
        train_path: Path to training CSV file with 'headline' and 'label' columns.
        val_path: Path to validation CSV file with 'headline' and 'label' columns.

    Returns:
        Tuple of (train_dataset, val_dataset) as HuggingFace Dataset objects.

    Raises:
        FileNotFoundError: If either CSV file does not exist.
        ValueError: If required columns are missing from the CSV files.

    Example:
        >>> train_ds, val_ds = load_data_from_csv("data/train.csv", "data/val.csv")
        >>> print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    """
    logger.info(f"Loading training data from: {train_path}")
    logger.info(f"Loading validation data from: {val_path}")

    # Validate file existence
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data file not found: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data file not found: {val_path}")

    # Load CSV files
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Validate required columns
    required_columns = {"headline", "label"}
    for name, df in [("train", train_df), ("validation", val_df)]:
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"{name} CSV missing required columns: {missing}. "
                f"Found columns: {list(df.columns)}"
            )

    # Select only required columns and convert to Dataset
    train_dataset = Dataset.from_pandas(train_df[["headline", "label"]])
    val_dataset = Dataset.from_pandas(val_df[["headline", "label"]])

    logger.info(f"Loaded {len(train_dataset)} training samples")
    logger.info(f"Loaded {len(val_dataset)} validation samples")

    # Log label distribution
    train_labels = train_dataset["label"]
    label_counts = pd.Series(train_labels).value_counts().sort_index()
    logger.info(f"Training label distribution: {dict(label_counts)}")

    return train_dataset, val_dataset


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = MAX_LENGTH,
) -> Dataset:
    """
    Tokenize a dataset using the provided tokenizer.

    Args:
        dataset: HuggingFace Dataset with 'headline' and 'label' columns.
        tokenizer: HuggingFace tokenizer for text processing.
        max_length: Maximum sequence length for tokenization.

    Returns:
        Tokenized Dataset with input_ids, attention_mask, and labels.

    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        >>> tokenized = tokenize_dataset(train_dataset, tokenizer)
        >>> print(tokenized[0]["input_ids"][:10])

    """

    def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            examples["headline"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["headline"],
        desc="Tokenizing dataset",
    )

    return tokenized_dataset


# =============================================================================
# MODEL LOADING
# =============================================================================


def load_model_and_tokenizer(
    model_name: str,
    num_labels: int = NUM_LABELS,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Load a pre-trained model and tokenizer with fallback support.

    Attempts to load the specified model. If unavailable, falls back to
    the default DistilBERT model.

    Args:
        model_name: Name or path of the pre-trained model.
        num_labels: Number of classification labels.

    Returns:
        Tuple of (model, tokenizer) ready for fine-tuning.

    Raises:
        RuntimeError: If both primary and fallback models fail to load.

    Example:
        >>> model, tokenizer = load_model_and_tokenizer("ProsusAI/finbert")
        >>> print(f"Model config: {model.config}")

    """
    logger.info(f"Attempting to load model: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        logger.info(f"Successfully loaded model: {model_name}")
        return model, tokenizer

    except Exception as e:
        logger.warning(
            f"Failed to load {model_name}: {e}. "
            f"Attempting fallback: {FALLBACK_MODEL_NAME}"
        )

        try:
            tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(
                FALLBACK_MODEL_NAME,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
            )
            logger.info(f"Successfully loaded fallback model: {FALLBACK_MODEL_NAME}")
            return model, tokenizer

        except Exception as fallback_error:
            raise RuntimeError(
                f"Failed to load both primary ({model_name}) and fallback "
                f"({FALLBACK_MODEL_NAME}) models. "
                f"Fallback error: {fallback_error}"
            ) from fallback_error


# =============================================================================
# EVALUATION METRICS
# =============================================================================


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    """
    Compute evaluation metrics for model predictions.

    Calculates accuracy and F1 score (macro average) for multi-class
    sentiment classification.

    Args:
        eval_pred: EvalPrediction object containing predictions and labels.

    Returns:
        Dictionary with 'accuracy' and 'f1' scores.

    Example:
        >>> metrics = compute_metrics(eval_pred)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

    """
    # Load evaluation metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    # Extract predictions and labels
    predictions, labels = eval_pred

    # Convert logits to predictions
    preds = np.argmax(predictions, axis=-1)

    # Compute metrics
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
    }


# =============================================================================
# CUSTOM CALLBACK FOR EPOCH-END METRICS
# =============================================================================


class EpochMetricsCallback(TrainerCallback):
    """
    Custom callback to print evaluation metrics after each epoch.

    This callback evaluates the model at the end of each epoch and
    prints the accuracy and F1 score to the console.

    Attributes:
        eval_dataset: Validation dataset for evaluation.
        compute_metrics_fn: Function to compute evaluation metrics.

    """

    def __init__(self, eval_dataset: Dataset, compute_metrics_fn: Any):
        """
        Initialize the epoch metrics callback.

        Args:
            eval_dataset: Validation dataset for evaluation.
            compute_metrics_fn: Function to compute evaluation metrics.

        """
        super().__init__()
        self.eval_dataset = eval_dataset
        self.compute_metrics_fn = compute_metrics_fn

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        """
        Evaluate and print metrics at the end of each epoch.

        Args:
            args: Training arguments.
            state: Current trainer state.
            control: Trainer control object.
            **kwargs: Additional keyword arguments.

        Returns:
            Updated TrainerControl object.

        """
        trainer = kwargs.get("trainer")
        if trainer is not None:
            logger.info(f"\n{'='*60}")
            logger.info(f"EPOCH {int(state.epoch)} COMPLETE - EVALUATION METRICS")
            logger.info(f"{'='*60}")

            metrics = trainer.evaluate(self.eval_dataset)

            logger.info(f"Validation Accuracy: {metrics['eval_accuracy']:.4f}")
            logger.info(f"Validation F1 Score: {metrics['eval_f1']:.4f}")
            logger.info(f"Validation Loss: {metrics['eval_loss']:.4f}")
            logger.info(f"{'='*60}\n")

        return control


# =============================================================================
# TRAINING FUNCTION
# =============================================================================


def train_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> Trainer:
    """
    Fine-tune the model on the training dataset.

    Args:
        train_dataset: Tokenized training dataset.
        val_dataset: Tokenized validation dataset.
        model: Pre-trained model for fine-tuning.
        tokenizer: Tokenizer for data collation.
        output_dir: Directory to save model checkpoints.
        num_epochs: Number of training epochs.
        batch_size: Training and evaluation batch size.
        learning_rate: Learning rate for optimizer.

    Returns:
        Trained Trainer object with best model loaded.

    Example:
        >>> trainer = train_model(
        ...     train_ds, val_ds, model, tokenizer,
        ...     output_dir="models/aion-sentiment-in-v1",
        ...     num_epochs=3
        ... )

    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Data collator for padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=100,  # Fixed warmup steps (replaces deprecated warmup_ratio)
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",  # Disable external reporting
        seed=42,
        fp16=False,  # MPS doesn't support fp16, use fp32
        dataloader_pin_memory=False,  # MPS doesn't support pinned memory
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Add custom epoch metrics callback
    epoch_callback = EpochMetricsCallback(val_dataset, compute_metrics)
    trainer.add_callback(epoch_callback)

    # Start training
    logger.info(f"Starting training for {num_epochs} epochs...")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Output directory: {output_dir}")

    trainer.train()

    # Save the best model
    logger.info(f"Saving best model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION - BEST MODEL")
    logger.info("=" * 60)

    final_metrics = trainer.evaluate()
    logger.info(f"Final Accuracy: {final_metrics['eval_accuracy']:.4f}")
    logger.info(f"Final F1 Score: {final_metrics['eval_f1']:.4f}")
    logger.info(f"Final Loss: {final_metrics['eval_loss']:.4f}")
    logger.info("=" * 60)

    return trainer


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the training script.

    Returns:
        Parsed arguments namespace.

    """
    parser = argparse.ArgumentParser(
        description="AION Sentiment Analysis Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing train.csv and val.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/aion-sentiment-in-v1",
        help="Directory to save trained model",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Model name or path (fallback: {FALLBACK_MODEL_NAME})",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training and evaluation batch size",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for optimizer",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=MAX_LENGTH,
        help="Maximum sequence length for tokenization",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the training script.

    Orchestrates the complete training pipeline:
    1. Parse arguments
    2. Load data
    3. Load model and tokenizer
    4. Tokenize data
    5. Train model
    6. Save results

    """
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("AION SENTIMENT ANALYSIS - TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("=" * 60)

    # Step 1: Load data
    train_path = os.path.join(args.data_dir, "train.csv")
    val_path = os.path.join(args.data_dir, "val.csv")

    train_dataset, val_dataset = load_data_from_csv(train_path, val_path)

    # Step 2: Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # Step 3: Tokenize datasets
    logger.info("Tokenizing datasets...")
    train_dataset = tokenize_dataset(train_dataset, tokenizer, args.max_length)
    val_dataset = tokenize_dataset(val_dataset, tokenizer, args.max_length)

    # Step 4: Train model
    trainer = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # Step 5: Save training metadata
    metadata_path = os.path.join(args.output_dir, "training_metadata.json")
    import json

    metadata = {
        "model_name": args.model_name,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Training metadata saved to: {metadata_path}")
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
