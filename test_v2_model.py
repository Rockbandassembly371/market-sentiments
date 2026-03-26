#!/usr/bin/env python3
"""
Test AION Sentiment Model v2 on 10 Random ClickHouse Headlines

This script:
1. Fetches 10 random news headlines from ClickHouse
2. Runs sentiment analysis using the newly trained v2 model
3. Displays results with confidence scores
"""

import clickhouse_connect
import sys
import os

# Add sentiment package to path
sys.path.insert(0, '/Users/lokeshgupta/aion_open_source/aion-sentiment/src')

from aion_sentiment import AIONSentimentAnalyzer

print("=" * 80)
print("AION SENTIMENT MODEL V2 - RANDOM HEADLINE TEST")
print("=" * 80)
print()

# Step 1: Connect to ClickHouse and fetch 10 random headlines
print("Step 1: Fetching 10 random headlines from ClickHouse...")
print("-" * 80)

client = clickhouse_connect.get_client(
    host='localhost',
    port=8123,  # HTTP port
    database='aion_master'
)

query = """
SELECT 
    title,
    sentiment_score,
    sentiment_label as original_label
FROM aion_master.news_master_v1
WHERE sentiment_label IS NOT NULL
ORDER BY rand()
LIMIT 10
"""

result = client.query(query)
headlines = []
for row in result.result_rows:
    headlines.append({
        'headline': row[0],
        'sentiment_score': row[1],
        'original_label': row[2]
    })

print(f"Fetched {len(headlines)} headlines")
print()

# Step 2: Load model and run predictions
print("Step 2: Running sentiment analysis with v2 model...")
print("-" * 80)

analyzer = AIONSentimentAnalyzer(model_name="/Users/lokeshgupta/aion_open_source/aion-sentiment-in/models/aion-sentiment-in-v2")

import pandas as pd

# Create DataFrame for analysis
df = pd.DataFrame([{'headline': item['headline']} for item in headlines])

# Run analysis
results_df = analyzer.analyze(df, text_column='headline')

# Combine with original data
results = []
for i, item in enumerate(headlines):
    results.append({
        'headline': item['headline'],
        'sentiment_score': item['sentiment_score'],
        'original_label': item['original_label'],
        'predicted_label': results_df['sentiment_label'].iloc[i],
        'confidence': results_df['sentiment_confidence'].iloc[i]
    })

# Step 3: Display results
print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print()

for i, result in enumerate(results, 1):
    print(f"{i}. {result['headline'][:80]}")
    print(f"   Original SHAM Label: {result['original_label']} (score: {result['sentiment_score']:+.2f})")
    print(f"   V2 Model Prediction: {result['predicted_label']} ({result['confidence']:.1%})")
    print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)

# Count predictions
from collections import Counter
pred_counts = Counter([r['predicted_label'] for r in results])
orig_counts = Counter([r['original_label'] for r in results])

print(f"\nOriginal SHAM Labels: {dict(orig_counts)}")
print(f"V2 Model Predictions: {dict(pred_counts)}")
print()
print("=" * 80)
