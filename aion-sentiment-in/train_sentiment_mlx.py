#!/usr/bin/env python3
# =============================================================================
# AION Sentiment Analysis - MLX Classification Training
# Uses MLX for native M4 acceleration (NOT PyTorch MPS)
# =============================================================================

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd
from datasets import Dataset
from mlx.utils import tree_map
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_MODEL_NAME: str = "distilbert-base-uncased"
NUM_LABELS: int = 3
MAX_LENGTH: int = 128
BATCH_SIZE: int = 32
LEARNING_RATE: float = 2e-5


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_from_csv(train_path: str, val_path: str) -> Tuple[Dataset, Dataset]:
    """Load training and validation data from CSV files."""
    logger.info(f"Loading training data from: {train_path}")
    logger.info(f"Loading validation data from: {val_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    required_columns = {"headline", "label"}
    for name, df in [("train", train_df), ("validation", val_df)]:
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"{name} CSV missing columns: {missing}")

    train_dataset = Dataset.from_pandas(train_df[["headline", "label"]])
    val_dataset = Dataset.from_pandas(val_df[["headline", "label"]])

    logger.info(f"Loaded {len(train_dataset)} training samples")
    logger.info(f"Loaded {len(val_dataset)} validation samples")

    label_counts = pd.Series(train_dataset["label"]).value_counts().sort_index()
    logger.info(f"Training label distribution: {dict(label_counts)}")

    return train_dataset, val_dataset


# =============================================================================
# BATCH ITERATOR
# =============================================================================

def iterate_batches(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
):
    """Generate batches for training."""
    num_samples = len(input_ids)
    indices = np.random.permutation(num_samples) if shuffle else np.arange(num_samples)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        yield {
            "input_ids": mx.array(input_ids[batch_indices]),
            "attention_mask": mx.array(attention_mask[batch_indices]),
            "labels": mx.array(labels[batch_indices]),
        }


# =============================================================================
# MLX BERT CLASSIFIER
# =============================================================================

class BertClassifier(nn.Module):
    """BERT-based classifier using pure MLX."""
    
    def __init__(self, model_name: str, num_labels: int = 3):
        super().__init__()
        
        # Load weights from HuggingFace model
        from mlx_transformers import AutoModelForSequenceClassification
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.num_labels = num_labels
        
    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


# =============================================================================
# LOSS FUNCTION
# =============================================================================

def loss_fn(model, input_ids, attention_mask, labels):
    """Cross-entropy loss for classification."""
    logits = model(input_ids, attention_mask)
    
    # Cross-entropy loss
    log_probs = mx.log_softmax(logits, axis=-1)
    loss = -log_probs[mx.arange(labels.shape[0]), labels]
    return loss.mean()


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, batches) -> Tuple[float, float]:
    """Evaluate model on validation batches."""
    correct = 0
    total = 0
    total_loss = 0.0
    
    model.eval()
    
    for batch in batches:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        logits = model(input_ids, attention_mask)
        
        # Compute loss
        log_probs = mx.log_softmax(logits, axis=-1)
        loss = -log_probs[mx.arange(labels.shape[0]), labels]
        total_loss += loss.sum().item()
        
        # Compute accuracy
        predictions = mx.argmax(logits, axis=-1)
        correct += (predictions == labels).sum().item()
        total += len(labels)
    
    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_model_mlx(
    model,
    train_dataset: Dataset,
    val_dataset: Dataset,
    tokenizer,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    max_length: int = MAX_LENGTH,
) -> None:
    """Train the model using MLX optimizers."""

    os.makedirs(output_dir, exist_ok=True)

    # Tokenize all data upfront
    logger.info("Tokenizing training data...")
    train_texts = list(train_dataset["headline"])
    train_labels_list = list(train_dataset["label"])
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="np"
    )
    
    logger.info("Tokenizing validation data...")
    val_texts = list(val_dataset["headline"])
    val_labels_list = list(val_dataset["label"])
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="np"
    )

    # Create validation batches
    val_batches = list(iterate_batches(
        val_encodings["input_ids"],
        val_encodings["attention_mask"],
        np.array(val_labels_list),
        batch_size,
        shuffle=False
    ))

    # Initialize optimizer
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Training loop
    best_val_accuracy = 0.0
    num_train = len(train_dataset)

    for epoch in range(num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*60}")
        
        epoch_start = time.time()

        # Training
        model.train()
        total_loss = 0.0
        num_batches = 0

        # Create training batches
        train_batches = list(iterate_batches(
            train_encodings["input_ids"],
            train_encodings["attention_mask"],
            np.array(train_labels_list),
            batch_size,
            shuffle=True
        ))

        for batch in train_batches:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            # Forward and backward pass
            loss, grads = mx.value_and_grad(loss_fn)(
                model, input_ids, attention_mask, labels
            )
            
            # Update parameters
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 50 == 0:
                logger.info(f"  Batch {num_batches}: loss = {loss.item():.4f}")

        epoch_time = time.time() - epoch_start
        avg_train_loss = total_loss / num_batches
        logger.info(f"Training loss: {avg_train_loss:.4f} (time: {epoch_time:.1f}s)")

        # Validation
        logger.info("Running validation...")
        val_loss, val_accuracy = evaluate(model, val_batches)
        logger.info(f"Validation loss: {val_loss:.4f}")
        logger.info(f"Validation accuracy: {val_accuracy:.4f}")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            logger.info(f"New best model! Saving to {output_dir}")

            # Save model weights
            mx.save_safetensors(
                os.path.join(output_dir, "model.safetensors"),
                tree_map(lambda x: x.astype(mx.float32), model.parameters())
            )
            
            # Save tokenizer
            tokenizer.save_pretrained(output_dir)

            # Save config
            config = {
                "num_labels": NUM_LABELS,
                "id2label": {"0": "NEG", "1": "NEU", "2": "POS"},
                "label2id": {"NEG": 0, "NEU": 1, "POS": 2},
                "max_length": MAX_LENGTH,
                "model_source": "aion-analytics/aion-sentiment-in-v2",
                "framework": "mlx",
            }
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

            # Save metadata
            metadata = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "num_labels": NUM_LABELS,
                "max_length": MAX_LENGTH,
                "model_source": "aion-analytics/aion-sentiment-in-v2",
                "framework": "mlx",
            }
            with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"{'='*60}")


# =============================================================================
# MAIN
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AION Sentiment Analysis Training Script (MLX)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing train.csv and val.csv")
    parser.add_argument("--output_dir", type=str, default="models/aion-sentiment-in-v2", help="Output directory")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Model name or path")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH, help="Max sequence length")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("AION SENTIMENT ANALYSIS - TRAINING PIPELINE (MLX)")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Data: {args.data_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("=" * 60)

    # Load data
    train_path = os.path.join(args.data_dir, "train.csv")
    val_path = os.path.join(args.data_dir, "val.csv")

    train_dataset, val_dataset = load_data_from_csv(train_path, val_path)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model = BertClassifier(args.model_name, num_labels=NUM_LABELS)

    # Train
    train_model_mlx(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
