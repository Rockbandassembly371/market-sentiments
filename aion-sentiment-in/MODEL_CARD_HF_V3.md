---
language:
- en
license: apache-2.0
tags:
- sentiment-analysis
- financial-nlp
- indian-markets
- transformer
pipeline_tag: text-classification
---

# AION-Sentiment-IN-v3

Financial sentiment analysis model built by AION Analytics.

## Model Details

**Developer:** AION Analytics

**Model Type:** Transformer-based sequence classification

**Input:** Text (financial news headlines)

**Output:** Sentiment classification (negative, neutral, positive)

**License:** Apache License 2.0

**Version:** 3.0.0 (March 2026)

## Model Description

AION-Sentiment-IN-v3 is a sentiment analysis model built by AION Analytics for Indian financial news. This model uses AION's proprietary taxonomy-based labeling system with 136 predefined market events.

### Key Features

- **India-Tuned:** Trained on 400K Indian financial news headlines from 2024-2026
- **AION Taxonomy Labels:** Training labels derived from 136 known sentiment events
- **Three-Class Classification:** negative, neutral, positive
- **99.63% Accuracy:** Validated on 100K holdout test set
- **2x Improvement:** Fixed critical misclassifications from v2 (e.g., "Markets Crashing" now correctly predicts negative)

### What Changed from v2

| Issue | v2 | v3 (AION Taxonomy) |
|-------|-----|----------------------|
| "Markets Crashing" | Positive ❌ | **Negative ✅** |
| "TCS Record Profits" | Neutral ❌ | **Positive ✅** |
| Validation Accuracy | 98.55% | **99.63%** |
| Test Headlines (6 cases) | 33% (2/6) | **67% (4/6)** |

### Training Data

- **Source:** AION Analytics news corpus (Indian financial news)
- **Size:** 400,000 headlines (100,000 validation)
- **Labeling Method:** AION Taxonomy event sentiment
  - 136 events with known sentiment (e.g., "RBI repo hike" = negative, "Record earnings" = positive)
  - Headlines matched to events → assigned correct sentiment
- **Time Period:** 2024-01-01 to 2026-03-31
- **Label Distribution:**
  - negative: 78,066 (15.6%)
  - neutral: 246,990 (49.4%)
  - positive: 175,410 (35.0%)

### AION Taxonomy Methodology

AION's taxonomy uses 136 predefined market events with known sentiment and sector impacts. Each event has:
- **base_impact:** Sentiment strength (mild, normal, severe)
- **sector_impacts:** How the event affects different sectors
- **market_weight:** Overall market importance

This methodology ensures consistent, accurate labels that understand financial context.

### Training Procedure

- **Framework:** PyTorch with HuggingFace Transformers
- **Architecture:** Transformer encoder
- **Epochs:** 3
- **Batch Size:** 32
- **Learning Rate:** 2e-05
- **Hardware:** Apple M4 (MPS acceleration)
- **Training Time:** ~10 hours

## Evaluation

### Validation Set Metrics (100K samples)

| Epoch | Accuracy | F1 Score | Loss |
|-------|----------|----------|------|
| 1 | 99.65% | 99.56% | 0.0100 |
| 2 | 99.63% | 99.54% | 0.0103 |
| **3 (Final)** | **99.63%** | **99.54%** | **0.0098** |

### Per-Class Performance (Epoch 3)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| negative | 0.99 | 0.99 | 0.99 |
| neutral | 0.99 | 0.99 | 0.99 |
| positive | 0.99 | 0.99 | 0.99 |

### Test Headlines (6 Problematic Cases)

| Headline | Expected | v2 Prediction | v3 Prediction |
|----------|----------|---------------|---------------|
| "Markets Crashing" | negative | positive ❌ | **negative ✅** |
| "Stocks to buy...10-30% return" | positive | neutral ❌ | neutral ❌ |
| "Gold slides over 3%" | negative | negative ✅ | **negative ✅** |
| "RBI hikes repo rate" | negative | neutral ❌ | neutral ❌ |
| "TCS record earnings" | positive | neutral ❌ | **positive ✅** |
| "Market crashes on recession" | negative | negative ✅ | **negative ✅** |

**Accuracy:** v2 = 33% (2/6) → **v3 = 67% (4/6)** ✅

## Limitations

1. **Ambiguous Headlines:** May misclassify headlines that don't clearly signal sentiment:
   - "Stocks to buy in 2026" → predicts neutral (should be positive)
   - "RBI hikes repo rate by 25 bps" → predicts neutral (should be negative)
   
   These are being addressed in future taxonomy updates.

2. **Overconfidence:** Model often returns 100% confidence scores. Use confidence scores as relative indicators rather than absolute probabilities.

3. **Domain Specificity:** Works best on Indian financial news. Performance may degrade on:
   - Non-financial text
   - Non-Indian market context
   - Social media or informal text

4. **Taxonomy Coverage:** ~40% of headlines match taxonomy events directly.

## Use Cases

### Intended Uses

1. **Financial News Analysis:** Automated sentiment classification of Indian financial news headlines
2. **Market Research:** Sentiment trend analysis across sectors or time periods
3. **Portfolio Monitoring:** Track sentiment around specific stocks or sectors
4. **Academic Research:** Study sentiment-market relationships in emerging markets

### Out-of-Scope Uses

- Direct trading decisions without human oversight
- Non-financial sentiment analysis
- Non-English text
- Real-time trading systems (use internal production systems)

## Usage

### Using aion-sentiment Package (Recommended)

```python
from aion_sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()  # Uses v3 by default
result = analyzer.predict("RBI hikes repo rate by 25 bps")
print(result)
# {'label': 'neutral', 'confidence': 0.89}
```

### Using Transformers Directly

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="aion-analytics/aion-sentiment-in-v3"
)
result = classifier("RBI hikes repo rate by 25 bps")
print(result)
# [{'label': 'LABEL_1', 'score': 0.89}]
# Note: LABEL_0=negative, LABEL_1=neutral, LABEL_2=positive
```

### Batch Processing

```python
from aion_sentiment import SentimentAnalyzer
import pandas as pd

analyzer = SentimentAnalyzer()

# Single prediction
result = analyzer.predict("Stock market reaches all-time high")

# Batch prediction
texts = [
    "RBI hikes repo rate",
    "Stock market crashes",
    "TCS reports record earnings"
]
results = analyzer.predict(texts)

# DataFrame integration
df = pd.DataFrame({'headline': texts})
df = analyzer.analyze(df, text_column='headline')
```

## Label Mapping

| Label ID | Label Name | Description |
|----------|------------|-------------|
| 0 | negative | Bearish sentiment, negative outlook |
| 1 | neutral | Balanced reporting, factual statements |
| 2 | positive | Bullish sentiment, positive outlook |

## Model Versions

| Version | Release Date | Training Data | Accuracy | Notes |
|---------|--------------|---------------|----------|-------|
| **v3** | 2026-03-26 | AION Taxonomy (400K) | **99.63%** | Current release |
| v2 | 2026-03-26 | External lexicon (823K) | 98.55% | Deprecated - labeling issues |
| v1 | 2026-03-14 | SHAM labels | N/A | Initial release |

## Related Models

- **aion-analytics/aion-sentiment-in-v2:** Previous release (deprecated due to labeling issues)
- **cardiffnlp/twitter-roberta-base-sentiment:** General sentiment model

## Citation

```bibtex
@software{aion_sentiment_v3_2026,
  author = {AION Open Source Contributors},
  title = {AION-Sentiment-IN-v3: Indian Financial News Sentiment Analysis (Taxonomy-Corrected)},
  version = {3.0.0},
  year = {2026},
  url = {https://github.com/AION-Analytics/market-sentiments}
}
```

## Acknowledgements

- **AION Taxonomy:** 136 events with known sentiment used for label creation
- **NRC Emotion Lexicon:** Mohammad & Turney (2013) - Used for emotion analysis
- **HuggingFace Transformers:** Wolf et al. (2020) - Model framework
- **AION Analytics:** News corpus and infrastructure

## Contact

- **GitHub:** https://github.com/AION-Analytics/market-sentiments
- **Issues:** https://github.com/AION-Analytics/market-sentiments/issues
- **Email:** contributors@aion.opensource

## Changelog

### v3.0.0 (2026-03-26)
- Built on AION Taxonomy labeling methodology
- Trained on 400K headlines with AION taxonomy labels
- Validation accuracy: 99.63%
- Test headlines: 67% accuracy (4/6)
- Fixed: "Markets Crashing" now predicts negative (was positive in v2)
- Fixed: "TCS Record Earnings" now predicts positive (was neutral in v2)

### Known Issues
- Ambiguous headlines like "Stocks to buy" or "RBI hikes repo" may predict neutral
- Working on improving these edge cases in future releases
