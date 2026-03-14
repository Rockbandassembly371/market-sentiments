#!/usr/bin/env python3
# =============================================================================
# AION Sentiment Analysis - Usage Examples
# =============================================================================
# Copyright (c) 2026 AION Open Source Contributors
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
# AION Open Source Project - Financial News Sentiment Analysis
# =============================================================================
"""
Usage Examples for AION Sentiment Analysis.

This script demonstrates various use cases for the AION sentiment
analysis package, including sentiment classification, emotion detection,
and DataFrame analysis.

Run with:
    python example.py
"""

import sys
import os

# Add src to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import json
from typing import List, Dict

from aion_sentiment import (
    AIONSentimentAnalyzer,
    SentimentAnalyzer,
    EmotionAnalyzer,
)


def example_basic_sentiment() -> None:
    """Demonstrate basic sentiment analysis."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Sentiment Analysis")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(device='cpu')  # Use CPU for compatibility
    
    # Sample financial headlines
    headlines = [
        "Stock market reaches all-time high on strong earnings",
        "Company files for bankruptcy, thousands of jobs at risk",
        "Fed maintains current interest rates, markets mixed",
        "Tech stocks rally as AI breakthrough drives optimism",
        "Oil prices crash amid global demand concerns",
    ]
    
    print("\nAnalyzing headlines...\n")
    
    # Analyze each headline
    for headline in headlines:
        result = analyzer.predict(headline)
        print(f"Headline: {headline}")
        print(f"  → Sentiment: {result['label']} ({result['confidence']:.2%})")
        print()


def example_emotion_analysis() -> None:
    """Demonstrate emotion analysis."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Emotion Analysis")
    print("=" * 70)
    
    # Initialize emotion analyzer
    analyzer = EmotionAnalyzer()
    
    # Sample texts with different emotional tones
    texts = [
        "Panic selling grips the market as investors flee risky assets",
        "Investors greedy for returns chase speculative stocks",
        "Hope springs eternal as markets show signs of recovery",
        "Fear and uncertainty dominate trading session",
    ]
    
    print("\nAnalyzing emotions...\n")
    
    for text in texts:
        emotions = analyzer.get_emotions(text)
        dominant = analyzer.get_dominant_emotion(text)
        summary = analyzer.get_emotion_summary(text)
        
        print(f"Text: {text}")
        print(f"  Dominant Emotion: {dominant}")
        print(f"  Scores: fear={emotions['fear']:.2f}, "
              f"greed={emotions['greed']:.2f}, "
              f"panic={emotions['panic']:.2f}, "
              f"optimism={emotions['optimism']:.2f}")
        print(f"  Summary: {summary}")
        print()


def example_dataframe_analysis() -> None:
    """Demonstrate DataFrame analysis."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: DataFrame Analysis")
    print("=" * 70)
    
    # Initialize main analyzer
    analyzer = AIONSentimentAnalyzer()
    
    # Create sample DataFrame with financial news
    df = pd.DataFrame({
        'headline': [
            'Sensex jumps 800 points on strong Q4 earnings',
            'RBI keeps repo rate unchanged at 6.5%',
            'IT stocks decline as US recession fears mount',
            'Reliance Industries reports 10% profit growth',
            'Bank Nifty hits record high on rate cut hopes',
            'Global markets plunge on trade war escalation',
            'Crypto markets tumble on regulatory concerns',
            'Tech rally continues as AI optimism grows',
        ],
        'source': [
            'Economic Times', 'Economic Times', 'Economic Times',
            'Economic Times', 'Reuters', 'Reuters', 'Bloomberg', 'Bloomberg'
        ],
        'timestamp': pd.date_range('2024-01-01', periods=8, freq='H')
    })
    
    print("\nInput DataFrame:")
    print(df[['headline', 'source']].to_string(index=False))
    
    # Analyze
    print("\nAnalyzing headlines...")
    results = analyzer.analyze(df, text_column='headline')
    
    # Display results
    print("\nResults:")
    display_df = results[['headline', 'sentiment_label', 'sentiment_confidence']].copy()
    display_df['sentiment_confidence'] = display_df['sentiment_confidence'].apply(
        lambda x: f"{x:.2%}"
    )
    print(display_df.to_string(index=False))
    
    # Show emotion breakdown
    print("\nEmotion Breakdown:")
    for _, row in results.iterrows():
        emotions = json.loads(row['emotions'])
        print(f"\n{row['headline'][:50]}...")
        print(f"  Sentiment: {row['sentiment_label']}")
        print(f"  Emotions: fear={emotions['fear']:.2f}, "
              f"greed={emotions['greed']:.2f}, "
              f"panic={emotions['panic']:.2f}, "
              f"optimism={emotions['optimism']:.2f}")


def example_batch_processing() -> None:
    """Demonstrate batch processing for large datasets."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Batch Processing")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(device='cpu')
    
    # Simulate large dataset
    headlines = [
        f"Market update {i}: Stocks {'rise' if i % 2 == 0 else 'fall'} "
        f"on {'positive' if i % 3 == 0 else 'mixed'} news"
        for i in range(20)
    ]
    
    print(f"\nProcessing {len(headlines)} headlines in batches...")
    
    # Process in batches for memory efficiency
    results = analyzer.predict_batch(headlines, batch_size=5)
    
    # Summary statistics
    positive = sum(1 for r in results if r['label'] == 'positive')
    negative = sum(1 for r in results if r['label'] == 'negative')
    neutral = sum(1 for r in results if r['label'] == 'neutral')
    
    print(f"\nResults Summary:")
    print(f"  Positive: {positive} ({positive/len(results):.1%})")
    print(f"  Negative: {negative} ({negative/len(results):.1%})")
    print(f"  Neutral:  {neutral} ({neutral/len(results):.1%})")
    
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    print(f"\nAverage Confidence: {avg_confidence:.2%}")


def example_sentiment_scoring() -> None:
    """Demonstrate continuous sentiment scoring."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Continuous Sentiment Scoring")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(device='cpu')
    
    # Sample texts
    texts = [
        ("Excellent earnings beat all expectations", "Very Positive"),
        ("Good quarterly results reported", "Positive"),
        ("Mixed signals from the market", "Neutral"),
        ("Concerns about economic outlook", "Negative"),
        ("Disastrous results trigger selloff", "Very Negative"),
    ]
    
    print("\nContinuous Sentiment Scores (-1 to +1):\n")
    
    for text, expected in texts:
        score = analyzer.get_sentiment_score(text)
        bar_length = int((score + 1) * 20)  # Scale to 0-40
        bar = '█' * bar_length + '░' * (40 - bar_length)
        
        print(f"{text}")
        print(f"  Expected: {expected}")
        print(f"  Score: {score:+.2f} [{bar}]")
        print()


def example_market_dashboard() -> None:
    """Demonstrate a simple market sentiment dashboard."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Market Sentiment Dashboard")
    print("=" * 70)
    
    # Initialize analyzers
    sentiment_analyzer = SentimentAnalyzer(device='cpu')
    emotion_analyzer = EmotionAnalyzer()
    
    # Sample news feed
    news_feed = [
        "Stock market surges on strong earnings reports",
        "Fed signals potential rate cuts ahead",
        "Tech sector leads market rally",
        "Banking stocks show resilience",
        "Oil prices stabilize after volatile week",
        "Inflation data comes in below expectations",
        "Consumer confidence reaches new high",
        "Global markets follow US lead higher",
    ]
    
    print("\nAnalyzing news feed for market sentiment...\n")
    
    # Get sentiment distribution
    sentiments = sentiment_analyzer.predict(news_feed)
    
    positive_count = sum(1 for s in sentiments if s['label'] == 'positive')
    negative_count = sum(1 for s in sentiments if s['label'] == 'negative')
    neutral_count = len(news_feed) - positive_count - negative_count
    
    # Get average emotions
    all_emotions = [emotion_analyzer.get_emotions(news) for news in news_feed]
    avg_emotions = {
        'fear': sum(e['fear'] for e in all_emotions) / len(all_emotions),
        'greed': sum(e['greed'] for e in all_emotions) / len(all_emotions),
        'panic': sum(e['panic'] for e in all_emotions) / len(all_emotions),
        'optimism': sum(e['optimism'] for e in all_emotions) / len(all_emotions),
    }
    
    # Determine market mood
    if avg_emotions['fear'] > 0.4 or avg_emotions['panic'] > 0.3:
        market_mood = "🔴 FEARFUL"
    elif avg_emotions['optimism'] > 0.4:
        market_mood = "🟢 OPTIMISTIC"
    elif avg_emotions['greed'] > 0.3:
        market_mood = "🟡 GREEDY"
    else:
        market_mood = "⚪ NEUTRAL"
    
    # Display dashboard
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "MARKET SENTIMENT DASHBOARD" + " " * 20 + "║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  Market Mood: {market_mood:<52} ║")
    print("╠" + "═" * 68 + "╣")
    print("║  Sentiment Distribution:                              ║")
    print(f"║    Positive: {positive_count}/{len(news_feed)} ({positive_count/len(news_feed):.1%}){' ' * 45}║")
    print(f"║    Negative: {negative_count}/{len(news_feed)} ({negative_count/len(news_feed):.1%}){' ' * 45}║")
    print(f"║    Neutral:  {neutral_count}/{len(news_feed)} ({neutral_count/len(news_feed):.1%}){' ' * 45}║")
    print("╠" + "═" * 68 + "╣")
    print("║  Emotion Scores:                                      ║")
    print(f"║    Fear:     {avg_emotions['fear']:.2f}{' ' * 54}║")
    print(f"║    Greed:    {avg_emotions['greed']:.2f}{' ' * 54}║")
    print(f"║    Panic:    {avg_emotions['panic']:.2f}{' ' * 54}║")
    print(f"║    Optimism: {avg_emotions['optimism']:.2f}{' ' * 54}║")
    print("╚" + "═" * 68 + "╝")


def example_error_handling() -> None:
    """Demonstrate error handling."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Error Handling")
    print("=" * 70)
    
    analyzer = SentimentAnalyzer(device='cpu')
    
    # Test empty input
    print("\n1. Empty input handling:")
    try:
        result = analyzer.predict([])
    except ValueError as e:
        print(f"   Caught expected error: {e}")
    
    # Test empty strings in batch
    print("\n2. Empty strings in batch:")
    texts = ["Valid text", "", "   ", "Another valid text"]
    results = analyzer.predict(texts)
    for text, result in zip(texts, results):
        print(f"   '{text}' → {result['label']}")
    
    # Test invalid DataFrame column
    print("\n3. Invalid DataFrame column:")
    main_analyzer = AIONSentimentAnalyzer()
    df = pd.DataFrame({'title': ['Some text']})
    try:
        main_analyzer.analyze(df, text_column='headline')
    except ValueError as e:
        print(f"   Caught expected error: {e}")


def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 70)
    print(" " * 20 + "AION SENTIMENT ANALYSIS")
    print(" " * 25 + "Usage Examples")
    print("=" * 70)
    
    print("\n⚠️  Note: First run will download models (~400MB for FinBERT)")
    print("   Subsequent runs will use cached models.\n")
    
    try:
        # Run all examples
        example_basic_sentiment()
        example_emotion_analysis()
        example_dataframe_analysis()
        example_batch_processing()
        example_sentiment_scoring()
        example_market_dashboard()
        example_error_handling()
        
        print("\n" + "=" * 70)
        print(" " * 25 + "All Examples Complete!")
        print("=" * 70)
        print("\nFor more information, see README.md")
        print("Report issues at: https://github.com/aion-open-source/aion-sentiment/issues\n")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        raise


if __name__ == '__main__':
    main()
