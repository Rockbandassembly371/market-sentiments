# =============================================================================
# AION Open-Source Project: Sentiment Analysis Demo
# File: demo.py
# Description: Colab notebook as Python script for AION Sentiment Analysis
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
AION Sentiment Analysis Demo.

This script is structured as a Colab notebook with markdown cells in comments.
It demonstrates the AION Sentiment Analysis package functionality including:
- Installation
- Basic sentiment prediction
- Emotion mapping
- Visualization of emotion scores

To run in Colab:
1. Upload this file to Google Colab
2. Run cells sequentially
3. Or convert to notebook using jupytext

Disclaimer: This software is for research and educational purposes only.
Not intended for trading or investment decisions.
"""

# %% [markdown]
# # AION Sentiment Analysis Demo
# 
# Welcome to the AION Sentiment Analysis demonstration notebook!
# 
# This notebook showcases the capabilities of the `aion-sentiment-in` package for:
# - **Sentiment Classification**: Analyzing financial news sentiment
# - **Emotion Detection**: Fine-grained emotion analysis across 8 categories
# - **Visualization**: Creating charts to display emotion scores
# 
# ## Overview
# 
# AION-Sentiment-IN is part of the [AION open-source ecosystem](https://github.com/aion),
# providing pre-trained models for financial NLP applications.
# 
# ### Emotion Categories
# 
# | Emotion | Description |
# |---------|-------------|
# | anger | Hostility, irritation, rage |
# | fear | Anxiety, worry, apprehension |
# | joy | Happiness, delight, pleasure |
# | sadness | Sorrow, grief, unhappiness |
# | trust | Acceptance, confidence, reliance |
# | disgust | Revulsion, contempt, aversion |
# | surprise | Astonishment, unexpectedness |
# | anticipation | Expectation, looking forward |
# 
# ---
# 
# **Disclaimer**: This software is for research and educational purposes only.
# Not intended for trading or investment decisions.

# %% [markdown]
# ## Installation
# 
# Install the AION Sentiment Analysis package from PyPI.

# %%
# Install from PyPI
# Uncomment the line below if running in a fresh environment
# !pip install aion-sentiment-in

# For local development, install from source
# !pip install -e .

# Install visualization dependencies
# !pip install matplotlib

print("Installation complete!")

# %% [markdown]
# ## Import Libraries
# 
# Import the necessary modules for sentiment analysis and visualization.

# %%
from aion_sentiment import AIONSentimentIN, EmotionAnalyzer
import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')

print("Libraries imported successfully!")

# %% [markdown]
# ## Initialize Models
# 
# Create instances of the sentiment model and emotion analyzer.
# 
# **Note**: The first run will download the model from HuggingFace (~500MB).
# Subsequent runs will use cached models.

# %%
# Initialize sentiment model (loads from HuggingFace by default)
print("Loading sentiment model...")
model = AIONSentimentIN()
print(f"Model loaded: {model}")

# Initialize emotion analyzer
print("\nLoading emotion analyzer...")
analyzer = EmotionAnalyzer()
print(f"Emotion analyzer loaded: {analyzer}")

# Check lexicon status
status = analyzer.check_lexicon_status()
print(f"\nLexicon available: {status['available']}")
print(f"Lexicon word count: {status['word_count']}")

# %% [markdown]
# ## Basic Sentiment Prediction
# 
# Analyze sentiment for sample financial news headlines.

# %%
# Sample financial news headlines
headlines = [
    "AAPL stock soars on better-than-expected earnings report",
    "Market crashes amid economic uncertainty and inflation fears",
    "Fed announces interest rate decision, investors remain neutral",
    "Tech sector shows strong growth anticipation for Q4",
    "Oil prices surge following supply disruption concerns",
]

print("=" * 70)
print("SENTIMENT ANALYSIS RESULTS")
print("=" * 70)

# Analyze each headline
results = model.predict_batch(headlines)

for headline, result in zip(headlines, results):
    print(f"\nHeadline: {headline}")
    print(f"  Sentiment: {result['sentiment_label'].upper()}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Scores: neg={result['all_scores']['negative']:.2f}, "
          f"neu={result['all_scores']['neutral']:.2f}, "
          f"pos={result['all_scores']['positive']:.2f}")

# %% [markdown]
# ## Emotion Analysis
# 
# Perform fine-grained emotion analysis on the same headlines.

# %%
print("=" * 70)
print("EMOTION ANALYSIS RESULTS")
print("=" * 70)

emotion_results = analyzer.analyze_batch(headlines)

for headline, result in zip(headlines, emotion_results):
    print(f"\nHeadline: {headline}")
    print(f"  Dominant Emotion: {result.dominant_emotion or 'None'}")
    print(f"  Dominant Score: {result.dominant_score:.2f}")
    
    # Show top 3 emotions
    sorted_emotions = sorted(result.emotions.items(), key=lambda x: x[1], reverse=True)
    top_emotions = [(e, s) for e, s in sorted_emotions if s > 0][:3]
    if top_emotions:
        emotion_str = ", ".join([f"{e} ({s:.2f})" for e, s in top_emotions])
        print(f"  Top Emotions: {emotion_str}")

# %% [markdown]
# ## Visualization: Emotion Scores Bar Chart
# 
# Create a bar chart showing emotion scores for each headline.

# %%
# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 8))

# Emotion categories
emotions = list(emotion_results[0].emotions.keys())
colors = plt.cm.Set3(np.linspace(0, 1, len(emotions)))

# Number of headlines
n_headlines = len(headlines)
bar_width = 0.8 / n_headlines
x = np.arange(len(emotions))

# Plot bars for each headline
for i, (headline, result) in enumerate(zip(headlines, emotion_results)):
    scores = [result.emotions[e] for e in emotions]
    offset = (i - n_headlines / 2 + 0.5) * bar_width
    ax.bar(x + offset, scores, bar_width, label=f'Headline {i+1}', color=colors[i])

# Customize plot
ax.set_xlabel('Emotion Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Emotion Score', fontsize=12, fontweight='bold')
ax.set_title('AION Emotion Analysis: Financial News Headlines', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(emotions, rotation=45, ha='right')
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3)

# Add value labels on bars
for i, result in enumerate(emotion_results):
    for j, emotion in enumerate(emotions):
        score = result.emotions[emotion]
        if score > 0.1:  # Only label significant scores
            offset = (i - n_headlines / 2 + 0.5) * bar_width
            ax.text(j + offset, score + 0.02, f'{score:.2f}', 
                   ha='center', va='bottom', fontsize=7, rotation=90)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualization: Sentiment Distribution
# 
# Show the distribution of sentiment predictions.

# %%
# Aggregate sentiment counts
sentiment_counts = {'negative': 0, 'neutral': 0, 'positive': 0}
for result in results:
    sentiment_counts[result['sentiment_label']] += 1

# Create pie chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
colors_sentiment = ['#ff6b6b', '#c9c9c9', '#51cf66']
axes[0].pie(sentiment_counts.values(), labels=sentiment_counts.keys(), 
            autopct='%1.1f%%', colors=colors_sentiment, startangle=90)
axes[0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')

# Bar chart
axes[1].bar(sentiment_counts.keys(), sentiment_counts.values(), color=colors_sentiment)
axes[1].set_xlabel('Sentiment', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[1].set_title('Sentiment Count', fontsize=14, fontweight='bold')
axes[1].set_ylim(0, max(sentiment_counts.values()) + 1)

for i, (sentiment, count) in enumerate(sentiment_counts.items()):
    axes[1].text(i, count + 0.1, str(count), ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Combined Sentiment + Emotion Analysis
# 
# Demonstrate the combined output from the sentiment model.

# %%
print("=" * 70)
print("COMBINED SENTIMENT + EMOTION ANALYSIS")
print("=" * 70)

# Analyze a single headline with combined output
test_headline = "Stock market crashes on recession fears"
combined_result = model.predict(test_headline)

print(f"\nHeadline: {test_headline}\n")
print(f"Sentiment: {combined_result['sentiment_label'].upper()}")
print(f"Confidence: {combined_result['confidence']:.2%}\n")

print("All Sentiment Scores:")
for label, score in combined_result['all_scores'].items():
    print(f"  {label.capitalize():10s}: {score:.4f}")

print("\nEmotion Scores:")
for emotion, score in sorted(combined_result['emotion_scores'].items(), 
                              key=lambda x: x[1], reverse=True):
    if score > 0:
        print(f"  {emotion.capitalize():15s}: {score:.4f}")

# %% [markdown]
# ## Custom Text Analysis
# 
# Try analyzing your own text!

# %%
# Example: Analyze custom text
custom_texts = [
    "Company reports record profits, stock hits all-time high",
    "Regulatory investigation announced, shares plummet",
    "Analysts maintain hold rating amid mixed signals",
]

print("=" * 70)
print("CUSTOM TEXT ANALYSIS")
print("=" * 70)

for text in custom_texts:
    result = model.predict(text)
    emotion = analyzer.analyze(text)
    
    print(f"\nText: {text}")
    print(f"  Sentiment: {result['sentiment_label'].upper()} ({result['confidence']:.2%})")
    print(f"  Dominant Emotion: {emotion.dominant_emotion or 'None'} ({emotion.dominant_score:.2f})")

# %% [markdown]
# ## Summary
# 
# This demo showcased the key features of AION-Sentiment-IN:
# 
# ✅ **Sentiment Classification**: Negative, Neutral, Positive
# ✅ **Confidence Scores**: Probability for each class
# ✅ **Emotion Analysis**: 8 emotion categories from NRC Lexicon
# ✅ **Batch Processing**: Efficient analysis of multiple texts
# ✅ **Visualization**: Matplotlib charts for emotion scores
# 
# ### Next Steps
# 
# - Explore the [API documentation](https://github.com/aion/aion-sentiment-in#api-reference)
# - Check out the [source code](https://github.com/aion/aion-sentiment-in)
# - Read the [contributing guide](https://github.com/aion/aion-sentiment-in/blob/main/CONTRIBUTING.md)
# - Visit the [AION organization](https://github.com/aion)
# 
# ---
# 
# **Disclaimer**: This software is provided for research and educational purposes only.
# It is not intended for trading or investment decisions. No warranty or guarantee
# of accuracy is provided. Users assume all risks associated with use.
# 
# **Attribution**: This product includes software developed by the AION Project
# (https://github.com/aion).
# 
# **License**: Apache License, Version 2.0

# %%
print("\n" + "=" * 70)
print("DEMO COMPLETE")
print("=" * 70)
print("\nThank you for trying AION Sentiment Analysis!")
print("For more information, visit: https://github.com/aion/aion-sentiment-in")
print("\nRemember: This software is for research and educational purposes only.")
print("Not intended for trading or investment decisions.")
