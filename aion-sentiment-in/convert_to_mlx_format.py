#!/usr/bin/env python3
"""
Convert VADER-labeled data to MLX-LM fine-tuning format (JSONL).

Sentiment classification as text generation:
- Input: "Headline: {text}\nSentiment:"
- Output: "positive" / "negative" / "neutral"
"""

import pandas as pd
import json
from pathlib import Path

def convert_to_mlx_format(input_csv: str, output_jsonl: str):
    """Convert CSV to MLX-LM JSONL format for LoRA fine-tuning."""
    
    df = pd.read_csv(input_csv)
    
    # Label mapping
    label_map = {
        0: "negative",
        1: "neutral", 
        2: "positive"
    }
    
    with open(output_jsonl, 'w') as f:
        for _, row in df.iterrows():
            headline = row['headline']
            label = label_map.get(row['label'], 'neutral')
            
            # Format as completion task
            example = {
                "prompt": f"Headline: {headline}\n\nSentiment:",
                "completion": f" {label}"
            }
            f.write(json.dumps(example) + '\n')
    
    print(f"Converted {len(df)} examples to {output_jsonl}")

if __name__ == "__main__":
    # Convert train and validation sets
    convert_to_mlx_format("data/train.csv", "data_mlx/train.jsonl")
    convert_to_mlx_format("data/val.csv", "data_mlx/valid.jsonl")
    print("Done!")
