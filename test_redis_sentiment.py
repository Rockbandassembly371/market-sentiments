#!/usr/bin/env python3
"""
AION Market Sentiment - Redis News Sentiment Test

Extracts 10 news articles from Redis and analyzes sentiment.
This is an isolated test - results are NOT stored back to ClickHouse.
"""

import redis
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, '/Users/lokeshgupta/aion_open_source/aion-sentiment/src')
sys.path.insert(0, '/Users/lokeshgupta/aion_open_source/aion-volweight/src')

from aion_sentiment import AIONSentimentAnalyzer
from aion_volweight import weight_confidence, get_regime

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

print("=" * 80)
print("AION MARKET SENTIMENT - REDIS NEWS TEST")
print("=" * 80)
print()

# Extract 10 news headlines from Redis
print("Step 1: Extracting 10 news articles from Redis...")
print("-" * 80)

news_keys = r.keys("news:published:*")[:10]
news_articles = []

for key in news_keys:
    # Extract headline from key (format: news:published:headline_url)
    headline = key.replace("news:published:", "").split("_https")[0]
    news_articles.append({
        'headline': headline,
        'source': 'Redis'
    })

print(f"Extracted {len(news_articles)} articles from Redis")
print()

# Create DataFrame
df = pd.DataFrame(news_articles)
print("Sample headlines:")
for i, headline in enumerate(df['headline'].head(5).tolist(), 1):
    print(f"  {i}. {headline[:80]}...")
print()

# Step 2: Run sentiment analysis
print("Step 2: Running sentiment analysis...")
print("-" * 80)

analyzer = AIONSentimentAnalyzer()
df = analyzer.analyze(df, text_column='headline')

print("Sentiment predictions:")
for _, row in df.iterrows():
    print(f"  {row['sentiment_label']:8} ({row['sentiment_confidence']:.1%}) - {row['headline'][:60]}...")
print()

# Step 3: VIX adjustment (using current India VIX)
print("Step 3: Applying VIX adjustment...")
print("-" * 80)

# Get current VIX (mock value for test - in production, fetch from market data)
current_vix = 18.5  # Example VIX value
regime = get_regime(current_vix)
print(f"Current India VIX: {current_vix} ({regime} regime)")

df = weight_confidence(df, vix_value=current_vix)

print("VIX-adjusted confidence:")
for _, row in df.iterrows():
    orig_conf = row['sentiment_confidence'] * 100
    adj_conf = row['sentiment_confidence_adjusted'] * 100
    print(f"  {row['sentiment_label']:8} {orig_conf:5.1f}% → {adj_conf:5.1f}% - {row['headline'][:50]}...")
print()

# Step 4: Summary statistics
print("Step 4: Summary Statistics")
print("-" * 80)

sentiment_counts = df['sentiment_label'].value_counts()
print(f"Sentiment Distribution:")
print(f"  Positive:  {sentiment_counts.get('positive', 0):2} articles ({sentiment_counts.get('positive', 0)/len(df)*100:.0f}%)")
print(f"  Neutral:   {sentiment_counts.get('neutral', 0):2} articles ({sentiment_counts.get('neutral', 0)/len(df)*100:.0f}%)")
print(f"  Negative:  {sentiment_counts.get('negative', 0):2} articles ({sentiment_counts.get('negative', 0)/len(df)*100:.0f}%)")
print()

avg_confidence = df['sentiment_confidence'].mean() * 100
avg_adjusted = df['sentiment_confidence_adjusted'].mean() * 100
print(f"Average Confidence: {avg_confidence:.1f}% (raw) → {avg_adjusted:.1f}% (VIX-adjusted)")
print()

# Step 5: Full results table
print("=" * 80)
print("FULL RESULTS TABLE")
print("=" * 80)
print()

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 80)

result_df = df[['headline', 'sentiment_label', 'sentiment_confidence', 'sentiment_confidence_adjusted']].copy()
result_df.columns = ['Headline', 'Sentiment', 'Confidence', 'VIX-Adjusted']
result_df['Confidence'] = result_df['Confidence'].apply(lambda x: f"{x*100:.1f}%")
result_df['VIX-Adjusted'] = result_df['VIX-Adjusted'].apply(lambda x: f"{x*100:.1f}%")

print(result_df.to_string(index=True))
print()

print("=" * 80)
print("TEST COMPLETE - Results NOT stored in ClickHouse (isolated test)")
print("=" * 80)
