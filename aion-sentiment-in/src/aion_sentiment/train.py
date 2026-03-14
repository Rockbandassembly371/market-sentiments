#!/usr/bin/env python3
# =============================================================================
# AION Open-Source Project: Sentiment Analysis Training Pipeline
# File: src/aion_sentiment/train.py
# Description: Training module for AION-Sentiment-IN model
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
AION Sentiment Analysis Training Module.

This module provides programmatic access to the training pipeline for
fine-tuning sentiment analysis models on Indian financial news data.

Key Classes:
    SentimentTrainer: Main class for managing the training process

Example:
    >>> from aion_sentiment.train import SentimentTrainer
    >>> trainer = SentimentTrainer(
    ...     data_dir="data",
    ...     output_dir="models/aion-sentiment-in-v1"
    ... )
    >>> trainer.train(num_epochs=3)

"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_MODEL_NAME: str = "ProsusAI/finbert"
FALLBACK_MODEL_NAME: str = "distilbert-base-uncased"
NUM_LABELS: int = 3
MAX_LENGTH: int = 128
LABEL_MAP: Dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}


# =============================================================================
# TRAINING CALLBACKS
# =============================================================================


class EpochMetricsCallback(TrainerCallback):
    """
    Custom callback to print evaluation metrics after each epoch.

    This callback evaluates the model at the end of each epoch and
    logs the accuracy and F1 score.

    Attributes:
        eval_dataset: Validation dataset for evaluation.

    """

    def __init__(self, eval_dataset: Dataset):
        """
        Initialize the epoch metrics callback.

        Args:
            eval_dataset: Validation dataset for evaluation.

        """
        self.eval_dataset = eval_dataset

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        """
        Evaluate and log metrics at the end of each epoch.

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
# SENTIMENT TRAINER CLASS
# =============================================================================


class SentimentTrainer:
    """
    Main class for training AION sentiment analysis models.

    This class encapsulates the complete training pipeline including:
    - Data loading and preprocessing
    - Model initialization with fallback support
    - Training with evaluation metrics
    - Model checkpointing and saving

    Attributes:
        data_dir: Directory containing train.csv and val.csv.
        output_dir: Directory to save trained models.
        model_name: Name of the base model to fine-tune.
        num_labels: Number of classification labels.
        max_length: Maximum sequence length for tokenization.

    Example:
        >>> trainer = SentimentTrainer(
        ...     data_dir="data",
        ...     output_dir="models/my-model",
        ...     model_name="ProsusAI/finbert"
        ... )
        >>> results = trainer.train(num_epochs=3, batch_size=16)
        >>> print(f"Final F1: {results['f1']:.4f}")

    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        model_name: str = DEFAULT_MODEL_NAME,
        num_labels: int = NUM_LABELS,
        max_length: int = MAX_LENGTH,
    ):
        """
        Initialize the SentimentTrainer.

        Args:
            data_dir: Directory containing train.csv and val.csv.
            output_dir: Directory to save trained models.
            model_name: Name of the base model to fine-tune.
            num_labels: Number of classification labels.
            max_length: Maximum sequence length for tokenization.

        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self._trainer: Optional[Trainer] = None

    def load_data(self) -> Tuple[Dataset, Dataset]:
        """
        Load training and validation datasets from CSV files.

        Returns:
            Tuple of (train_dataset, val_dataset).

        Raises:
            FileNotFoundError: If CSV files don't exist.
            ValueError: If required columns are missing.

        """
        train_path = os.path.join(self.data_dir, "train.csv")
        val_path = os.path.join(self.data_dir, "val.csv")

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

        # Convert to Dataset
        self.train_dataset = Dataset.from_pandas(train_df[["headline", "label"]])
        self.val_dataset = Dataset.from_pandas(val_df[["headline", "label"]])

        logger.info(f"Loaded {len(self.train_dataset)} training samples")
        logger.info(f"Loaded {len(self.val_dataset)} validation samples")

        return self.train_dataset, self.val_dataset

    def load_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load pre-trained model and tokenizer with fallback support.

        Returns:
            Tuple of (model, tokenizer).

        Raises:
            RuntimeError: If both primary and fallback models fail to load.

        """
        logger.info(f"Attempting to load model: {self.model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                ignore_mismatched_sizes=True,
            )
            logger.info(f"Successfully loaded model: {self.model_name}")
            return self.model, self.tokenizer

        except Exception as e:
            logger.warning(
                f"Failed to load {self.model_name}: {e}. "
                f"Attempting fallback: {FALLBACK_MODEL_NAME}"
            )

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_NAME)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    FALLBACK_MODEL_NAME,
                    num_labels=self.num_labels,
                    ignore_mismatched_sizes=True,
                )
                logger.info(f"Successfully loaded fallback model: {FALLBACK_MODEL_NAME}")
                return self.model, self.tokenizer

            except Exception as fallback_error:
                raise RuntimeError(
                    f"Failed to load both primary ({self.model_name}) and fallback "
                    f"({FALLBACK_MODEL_NAME}) models. "
                    f"Fallback error: {fallback_error}"
                ) from fallback_error

    def tokenize_datasets(self) -> Tuple[Dataset, Dataset]:
        """
        Tokenize training and validation datasets.

        Returns:
            Tuple of tokenized (train_dataset, val_dataset).

        """
        if self.train_dataset is None or self.val_dataset is None:
            raise ValueError("Datasets not loaded. Call load_data() first.")

        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")

        logger.info("Tokenizing datasets...")

        def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
            return self.tokenizer(  # type: ignore
                examples["headline"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

        self.train_dataset = self.train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["headline"],
            desc="Tokenizing training dataset",
        )

        self.val_dataset = self.val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["headline"],
            desc="Tokenizing validation dataset",
        )

        return self.train_dataset, self.val_dataset

    @staticmethod
    def compute_metrics(eval_pred: Any) -> Dict[str, float]:
        """
        Compute evaluation metrics for model predictions.

        Args:
            eval_pred: EvalPrediction object containing predictions and labels.

        Returns:
            Dictionary with 'accuracy' and 'f1' scores.

        """
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")

        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=-1)

        accuracy = accuracy_metric.compute(predictions=preds, references=labels)
        f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")

        return {
            "accuracy": accuracy["accuracy"],
            "f1": f1["f1"],
        }

    def train(
        self,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Fine-tune the model on the training dataset.

        Args:
            num_epochs: Number of training epochs.
            batch_size: Training and evaluation batch size.
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for regularization.
            warmup_ratio: Ratio of warmup steps.

        Returns:
            Dictionary containing training results and metrics.

        Raises:
            ValueError: If datasets or model are not loaded.

        """
        # Validate prerequisites
        if self.train_dataset is None or self.val_dataset is None:
            self.load_data()

        if self.model is None or self.tokenizer is None:
            self.load_model()

        if self.train_dataset is None or self.val_dataset is None:
            raise ValueError("Failed to load datasets")
        if self.model is None or self.tokenizer is None:
            raise ValueError("Failed to load model")

        # Tokenize if not already done
        if "input_ids" not in self.train_dataset.column_names:
            self.tokenize_datasets()

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            report_to="none",
            seed=42,
        )

        # Initialize trainer
        self._trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                EpochMetricsCallback(self.val_dataset),
            ],
        )

        # Start training
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
        logger.info(f"Output directory: {self.output_dir}")

        train_result = self._trainer.train()

        # Save the best model
        logger.info(f"Saving best model to: {self.output_dir}")
        self._trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        # Final evaluation
        logger.info("\n" + "=" * 60)
        logger.info("FINAL EVALUATION - BEST MODEL")
        logger.info("=" * 60)

        final_metrics = self._trainer.evaluate()
        logger.info(f"Final Accuracy: {final_metrics['eval_accuracy']:.4f}")
        logger.info(f"Final F1 Score: {final_metrics['eval_f1']:.4f}")
        logger.info(f"Final Loss: {final_metrics['eval_loss']:.4f}")
        logger.info("=" * 60)

        # Save training metadata
        metadata = {
            "model_name": self.model_name,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": self.max_length,
            "train_samples": len(self.train_dataset),
            "val_samples": len(self.val_dataset),
            "final_accuracy": final_metrics["eval_accuracy"],
            "final_f1": final_metrics["eval_f1"],
        }

        metadata_path = os.path.join(self.output_dir, "training_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Training metadata saved to: {metadata_path}")

        return {
            "metrics": final_metrics,
            "metadata": metadata,
            "output_dir": self.output_dir,
        }

    def predict(self, texts: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Make predictions using the trained model.

        Args:
            texts: Single text or list of texts to classify.

        Returns:
            Dictionary with predictions and confidence scores.

        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Train or load model first.")

        if isinstance(texts, str):
            texts = [texts]

        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)

        # Format results
        results = {
            "texts": texts,
            "predictions": [LABEL_MAP[p.item()] for p in predictions],
            "scores": [
                {
                    LABEL_MAP[i]: float(probabilities[j][i].item())
                    for i in range(self.num_labels)
                }
                for j in range(len(texts))
            ],
        }

        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def train_sentiment_model(
    data_dir: str,
    output_dir: str,
    model_name: str = DEFAULT_MODEL_NAME,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> Dict[str, Any]:
    """
    Convenience function to train a sentiment model.

    Args:
        data_dir: Directory containing train.csv and val.csv.
        output_dir: Directory to save trained model.
        model_name: Base model name.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.

    Returns:
        Dictionary with training results.

    Example:
        >>> results = train_sentiment_model(
        ...     data_dir="data",
        ...     output_dir="models/my-model",
        ...     num_epochs=3
        ... )
        >>> print(f"F1 Score: {results['metrics']['eval_f1']:.4f}")

    """
    trainer = SentimentTrainer(
        data_dir=data_dir,
        output_dir=output_dir,
        model_name=model_name,
    )

    return trainer.train(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
