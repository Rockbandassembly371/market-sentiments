#!/usr/bin/env python3
"""
AION Sentiment Analysis - MLX Training Script
Train sentiment classifier using MLX on Apple Silicon M4

Usage:
    python train_mlx.py --data_dir data --output_dir models/aion-sentiment-in-v2 --epochs 3 --batch_size 64
"""

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentClassifier(nn.Module):
    """Simple classifier on top of pretrained transformer using MLX"""
    
    def __init__(self, model_name="distilbert-base-uncased", num_labels=3):
        super().__init__()
        self.base = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        # Convert to MLX-compatible format
        self.base.eval()
        
    def __call__(self, input_ids, attention_mask):
        # Run base model (PyTorch) and convert to MLX
        with torch.no_grad():
            outputs = self.base(
                input_ids=torch.tensor(np.array(input_ids)),
                attention_mask=torch.tensor(np.array(attention_mask))
            )
        return mx.array(outputs.logits.numpy())


def load_data(train_path, val_path):
    """Load train and val CSV files"""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    return train_df, val_df


def create_batches(df, tokenizer, batch_size, max_length=128):
    """Create batches of tokenized inputs"""
    headlines = df['headline'].tolist()
    labels = df['label'].tolist()
    
    batches = []
    for i in range(0, len(headlines), batch_size):
        batch_texts = headlines[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        # Tokenize
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='np'
        )
        
        batches.append({
            'input_ids': mx.array(tokens['input_ids']),
            'attention_mask': mx.array(tokens['attention_mask']),
            'labels': mx.array(batch_labels)
        })
    
    return batches


def loss_fn(predictions, labels):
    """Cross entropy loss"""
    return nn.losses.cross_entropy(predictions, labels).mean()


def train_epoch(model, batches, optimizer, loss_and_grad_fn):
    """Train for one epoch"""
    total_loss = 0.0
    num_batches = 0
    
    for batch in batches:
        # Forward and backward pass
        loss, grads = loss_and_grad_fn(model, batch['input_ids'], batch['attention_mask'], batch['labels'])
        
        # Update parameters
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, batches):
    """Evaluate model on batches"""
    correct = 0
    total = 0
    
    for batch in batches:
        logits = model(batch['input_ids'], batch['attention_mask'])
        predictions = mx.argmax(logits, axis=1)
        
        correct += (predictions == batch['labels']).sum().item()
        total += len(batch['labels'])
    
    return correct / total


def main():
    parser = argparse.ArgumentParser(description='Train sentiment classifier with MLX')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='models/aion-sentiment-in-v2')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased')
    args = parser.parse_args()
    
    print("=" * 60)
    print("AION SENTIMENT - MLX TRAINING (Apple Silicon M4)")
    print("=" * 60)
    print(f"Device: {mx.default_device()}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)
    
    # Load tokenizer (PyTorch for now, convert inputs to MLX)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load data
    print("Loading data...")
    train_df, val_df = load_data(
        f"{args.data_dir}/train.csv",
        f"{args.data_dir}/val.csv"
    )
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    # Create batches
    print("Creating batches...")
    train_batches = create_batches(train_df, tokenizer, args.batch_size, args.max_length)
    val_batches = create_batches(val_df, tokenizer, args.batch_size, args.max_length)
    print(f"Train batches: {len(train_batches)}")
    print(f"Val batches: {len(val_batches)}")
    
    # Initialize model
    print("Initializing model...")
    model = SentimentClassifier(args.model_name)
    
    # Initialize optimizer
    optimizer = optim.Adam(learning_rate=args.lr)
    
    # Create loss and grad function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_batches, optimizer, loss_and_grad_fn)
        
        # Evaluate
        train_acc = evaluate(model, train_batches[:10])  # Sample for speed
        val_acc = evaluate(model, val_batches)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")
    
    # Save model
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save weights
    mx.savez(str(output_path / "weights.npz"), **tree_flatten(model.parameters()))
    print(f"Saved weights to {output_path / 'weights.npz'}")
    
    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    
    # Save config
    config = {
        'model_name': args.model_name,
        'num_labels': 3,
        'id2label': {'0': 'NEG', '1': 'NEU', '2': 'POS'},
        'label2id': {'NEG': 0, 'NEU': 1, 'POS': 2},
        'final_val_accuracy': val_acc
    }
    with open(output_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nFinal Validation Accuracy: {val_acc:.4f}")
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
