# AION Sentiment Analysis

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![PyPI](https://img.shields.io/badge/pypi-aion--sentiment-orange)

**Financial News Sentiment Analysis with India-Tuned Models and NRC Emotion Lexicon**

Part of the [AION Open Source Ecosystem](https://github.com/aion-open-source)

</div>

---

## Overview

**AION Sentiment** is a comprehensive Python package for analyzing sentiment and emotions in financial news and text. Built on state-of-the-art transformer models, it provides accurate sentiment classification and emotion detection specifically tuned for Indian financial markets.

### Key Features

- 🎯 **India-Tuned Model**: Default model (`aion-analytics/aion-sentiment-in-v1`) tuned on Indian financial news with 98.55% accuracy
- 📊 **Emotion Detection**: NRC Emotion Lexicon integration for detecting fear, greed, panic, and optimism
- 🔄 **Flexible Model Selection**: Use any HuggingFace model via simple parameter
- 🚀 **Hardware Acceleration**: Automatic detection and utilization of MPS (Apple Silicon), CUDA (NVIDIA), or CPU
- 📈 **DataFrame Integration**: Seamless pandas DataFrame analysis for batch processing
- 🔧 **Production Ready**: Type hints, comprehensive error handling, and extensive test coverage

### Model Information

**Default Model:** `aion-analytics/aion-sentiment-in-v1`
- **Training Data:** 957K Indian financial news headlines from AION analytics database
- **Accuracy:** 98.55% on validation set
- **F1 Score:** 98.65%
- **Labels:** positive, neutral, negative
- **Training Period:** October 2025 - February 2026

The model is automatically downloaded from HuggingFace on first use (~440MB). Subsequent uses load from cache.

**Alternative Models:**
```python
# Use custom HuggingFace model
analyzer = SentimentAnalyzer(model_name="other-model-name")

# Use custom local model
analyzer = SentimentAnalyzer(model_name="/path/to/your/model")
```

### Use Cases

- Financial news sentiment analysis
- Market sentiment monitoring
- Social media sentiment tracking for stocks
- Earnings call transcript analysis
- Investment research automation
- Risk assessment based on news sentiment

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Install from PyPI (Recommended)

```bash
pip install aion-sentiment
```

### Install from Source

```bash
# Clone the repository
git clone https://github.com/aion-open-source/aion-sentiment.git
cd aion-sentiment

# Install in editable mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Verify Installation

```python
from aion_sentiment import AIONSentimentAnalyzer

# Create analyzer instance
analyzer = AIONSentimentAnalyzer()
print("AION Sentiment installed successfully!")
```

---

## Quick Start

### Basic Sentiment Analysis

```python
from aion_sentiment import SentimentAnalyzer

# Initialize the sentiment analyzer
analyzer = SentimentAnalyzer()

# Analyze single text
result = analyzer.predict("Stock market reaches all-time high on strong earnings")
print(f"Sentiment: {result['label']} ({result['confidence']:.2%})")
# Output: Sentiment: positive (0.95)

# Analyze multiple texts
texts = [
    "Company reports record quarterly earnings",
    "Market crashes amid recession fears",
    "Fed maintains current interest rates"
]
results = analyzer.predict(texts)

for text, result in zip(texts, results):
    print(f"{text[:40]}... → {result['label']}")
```

### Emotion Analysis

```python
from aion_sentiment import EmotionAnalyzer

# Initialize the emotion analyzer
analyzer = EmotionAnalyzer()

# Get emotion scores
text = "Panic selling grips the market as investors flee risky assets"
emotions = analyzer.get_emotions(text)

print(f"Fear: {emotions['fear']:.2f}")
print(f"Greed: {emotions['greed']:.2f}")
print(f"Panic: {emotions['panic']:.2f}")
print(f"Optimism: {emotions['optimism']:.2f}")

# Get dominant emotion
dominant = analyzer.get_dominant_emotion(text)
print(f"Dominant emotion: {dominant}")
```

### DataFrame Analysis

```python
import pandas as pd
from aion_sentiment import AIONSentimentAnalyzer

# Initialize the main analyzer
analyzer = AIONSentimentAnalyzer()

# Create sample DataFrame
df = pd.DataFrame({
    'headline': [
        'Tech stocks rally on AI breakthrough',
        'Oil prices crash amid demand concerns',
        'Fed signals potential rate cuts',
        'Banking sector shows resilience'
    ],
    'source': ['Reuters', 'Bloomberg', 'CNBC', 'WSJ']
})

# Analyze all headlines
results = analyzer.analyze(df, text_column='headline')

print(results[['headline', 'sentiment_label', 'sentiment_confidence', 'emotions']])
```

---

## API Reference

### SentimentAnalyzer

Transformer-based sentiment classification for financial text.

```python
from aion_sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer(
    model_name="aion-analytics/aion-sentiment-in-v1",  # HuggingFace model
    device=None  # Auto-detect: 'mps', 'cuda', or 'cpu'
)
```

#### Methods

**`predict(texts: str | list[str]) -> dict | list[dict]`**

Predict sentiment for one or more texts.

```python
# Single text
result = analyzer.predict("Market surges on positive news")
# Returns: {'label': 'positive', 'confidence': 0.95}

# Multiple texts
results = analyzer.predict(["Text 1", "Text 2"])
# Returns: [{'label': 'positive', 'confidence': 0.87}, ...]
```

**`predict_batch(texts: list[str], batch_size: int = 8) -> list[dict]`**

Memory-efficient batch prediction for large datasets.

```python
results = analyzer.predict_batch(large_text_list, batch_size=16)
```

**`get_sentiment_score(text: str) -> float`**

Get continuous sentiment score (-1 to +1).

```python
score = analyzer.get_sentiment_score("Excellent earnings report")
# Returns: 0.85 (positive)
```

---

### EmotionAnalyzer

NRC lexicon-based emotion detection for financial text.

```python
from aion_sentiment import EmotionAnalyzer

analyzer = EmotionAnalyzer(
    lexicon_path=None,  # Auto-download NRC lexicon
    data_dir=None  # Default: ./data
)
```

#### Methods

**`get_emotions(text: str) -> dict`**

Get emotion scores for text.

```python
emotions = analyzer.get_emotions("Fear grips investors as markets tumble")
# Returns: {'fear': 0.75, 'greed': 0.1, 'panic': 0.6, 'optimism': 0.05}
```

**`get_dominant_emotion(text: str) -> str`**

Get the dominant emotion in text.

```python
dominant = analyzer.get_dominant_emotion("Panic selling continues")
# Returns: 'panic'
```

**`get_emotion_summary(text: str) -> str`**

Get human-readable emotion summary.

```python
summary = analyzer.get_emotion_summary("Market crashes on recession fears")
# Returns: "High fear (0.65), Moderate panic (0.45)"
```

**`analyze_texts(texts: list[str]) -> list[dict]`**

Analyze emotions for multiple texts.

```python
results = analyzer.analyze_texts(headline_list)
```

---

### AIONSentimentAnalyzer

Main API for DataFrame analysis combining sentiment and emotion detection.

```python
from aion_sentiment import AIONSentimentAnalyzer

analyzer = AIONSentimentAnalyzer(
    model_name="aion-analytics/aion-sentiment-in-v1",
    device=None,
    lexicon_path=None
)
```

#### Methods

**`analyze(df: pd.DataFrame, text_column: str = 'headline') -> pd.DataFrame`**

Analyze sentiment and emotions for texts in a DataFrame.

Adds columns:
- `sentiment_label`: 'positive', 'neutral', 'negative'
- `sentiment_confidence`: float 0-1
- `emotions`: JSON string with emotion scores

```python
results = analyzer.analyze(df, text_column='headline')
```

---

## Examples

### Real-World Financial News Analysis

```python
import pandas as pd
from aion_sentiment import AIONSentimentAnalyzer
import json

# Initialize analyzer
analyzer = AIONSentimentAnalyzer()

# Sample financial headlines
headlines = [
    "Sensex jumps 800 points on strong Q4 earnings, FII buying",
    "RBI keeps repo rate unchanged at 6.5%, focuses on inflation",
    "IT stocks decline as US recession fears mount",
    "Reliance Industries reports 10% profit growth in Q3",
    "Bank Nifty hits record high on rate cut hopes",
]

# Create DataFrame
df = pd.DataFrame({'headline': headlines})

# Analyze
results = analyzer.analyze(df)

# Display results
for _, row in results.iterrows():
    print(f"\n{row['headline']}")
    print(f"  Sentiment: {row['sentiment_label']} ({row['sentiment_confidence']:.2%})")
    emotions = json.loads(row['emotions'])
    print(f"  Emotions: fear={emotions['fear']:.2f}, "
          f"greed={emotions['greed']:.2f}, "
          f"panic={emotions['panic']:.2f}, "
          f"optimism={emotions['optimism']:.2f}")
```

### Market Sentiment Dashboard

```python
from aion_sentiment import SentimentAnalyzer, EmotionAnalyzer
import pandas as pd

class MarketSentimentDashboard:
    def __init__(self):
        self.sentiment = SentimentAnalyzer()
        self.emotions = EmotionAnalyzer()
    
    def analyze_news_feed(self, headlines: list[str]) -> dict:
        """Analyze a feed of news headlines."""
        sentiments = self.sentiment.predict(headlines)
        
        # Calculate aggregate metrics
        positive = sum(1 for s in sentiments if s['label'] == 'positive')
        negative = sum(1 for s in sentiments if s['label'] == 'negative')
        neutral = len(headlines) - positive - negative
        
        # Calculate average emotion scores
        all_emotions = [self.emotions.get_emotions(h) for h in headlines]
        avg_emotions = {
            'fear': sum(e['fear'] for e in all_emotions) / len(all_emotions),
            'greed': sum(e['greed'] for e in all_emotions) / len(all_emotions),
            'panic': sum(e['panic'] for e in all_emotions) / len(all_emotions),
            'optimism': sum(e['optimism'] for e in all_emotions) / len(all_emotions),
        }
        
        return {
            'total_headlines': len(headlines),
            'sentiment_distribution': {
                'positive': positive / len(headlines),
                'negative': negative / len(headlines),
                'neutral': neutral / len(headlines),
            },
            'average_emotions': avg_emotions,
            'market_mood': self._determine_market_mood(avg_emotions),
        }
    
    def _determine_market_mood(self, emotions: dict) -> str:
        """Determine overall market mood from emotion scores."""
        if emotions['fear'] > 0.5 or emotions['panic'] > 0.4:
            return "Fearful"
        elif emotions['optimism'] > 0.5:
            return "Optimistic"
        elif emotions['greed'] > 0.4:
            return "Greedy"
        else:
            return "Neutral"

# Usage
dashboard = MarketSentimentDashboard()
headlines = [...]  # Your news feed
analysis = dashboard.analyze_news_feed(headlines)
print(f"Market Mood: {analysis['market_mood']}")
```

---

## Configuration

### Device Selection

AION Sentiment automatically detects the best available device:

1. **MPS** (Apple Silicon M1/M2/M3)
2. **CUDA** (NVIDIA GPU)
3. **CPU** (Fallback)

To force a specific device:

```python
# Force CPU usage
analyzer = SentimentAnalyzer(device='cpu')

# Force CUDA
analyzer = SentimentAnalyzer(device='cuda')

# Force MPS (Apple Silicon)
analyzer = SentimentAnalyzer(device='mps')
```

### Custom Model

Use a different HuggingFace model:

```python
# Use Twitter RoBERTa sentiment model
analyzer = SentimentAnalyzer(
    model_name="cardiffnlp/twitter-roberta-base-sentiment"
)

# Use a local model
analyzer = SentimentAnalyzer(
    model_name="/path/to/local/model"
)
```

### Custom Lexicon Path

```python
# Use custom NRC lexicon
analyzer = EmotionAnalyzer(
    lexicon_path="/path/to/nrc_lexicon.txt"
)
```

---

## Testing

### Run All Tests

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=aion_sentiment --cov-report=html
```

### Test Files

- `tests/test_sentiment.py`: Transformer sentiment analysis tests
- `tests/test_emotions.py`: NRC emotion lexicon tests

**Note**: First test run will download:
- transformer model (~400MB)
- NRC lexicon (~1MB)

---

## Performance

### Benchmark Results

| Model | Device | Batch Size | Throughput |
|-------|--------|------------|------------|
| Default | CPU (M2) | 1 | ~50 samples/sec |
| Default | CPU (M2) | 8 | ~200 samples/sec |
| Default | MPS (M2) | 8 | ~350 samples/sec |
| Default | CUDA (A100) | 32 | ~2000 samples/sec |

### Memory Usage

- Model loading: ~800MB
- Per-sample inference: ~50MB
- Recommended: 2GB+ RAM for batch processing

---

## Troubleshooting

### Common Issues

**Model Download Fails**

```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface

# Retry installation
pip install --upgrade transformers torch
```

**MPS Not Detected on Mac**

```python
# Ensure PyTorch supports MPS
import torch
print(torch.backends.mps.is_available())  # Should be True

# Update PyTorch
pip install --upgrade torch
```

**Out of Memory**

```python
# Use smaller batch size
results = analyzer.predict_batch(texts, batch_size=4)

# Clear CUDA cache
import torch
torch.cuda.empty_cache()
```

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/aion-open-source/aion-sentiment.git
cd aion-sentiment

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/aion_sentiment/
```

---

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

```
Copyright (c) 2026 AION Open Source Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## Acknowledgments

- 
- [NRC Emotion Lexicon](https://saifmohammad.com/WebPages/nrc-emotion-intro.html) by Saif Mohammad
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PyTorch](https://pytorch.org/)

---

## AION Open Source

This package is part of the AION Open Source ecosystem. Explore more projects:

- [AION Core](https://github.com/aion-open-source/aion-core)
- [AION Connectors](https://github.com/aion-open-source/aion-connectors)
- [AION Sentiment](https://github.com/aion-open-source/aion-sentiment)

**Website**: [aion-open-source.github.io](https://aion-open-source.github.io)

---

<div align="center">

**Made with ❤️ by AION Open Source Contributors**

[Report Issue](https://github.com/aion-open-source/aion-sentiment/issues) • 
[Request Feature](https://github.com/aion-open-source/aion-sentiment/issues) • 
[Discussions](https://github.com/aion-open-source/aion-sentiment/discussions)

</div>
