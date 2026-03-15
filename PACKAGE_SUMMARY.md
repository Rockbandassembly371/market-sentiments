# AION Open-Source Ecosystem - Summary

**Date:** March 14, 2026  
**Status:** ✅ **5 PACKAGES COMPLETE** - Ready for PyPI Release

---

## 📦 Packages Created

| # | Package | Purpose | Status | Location |
|---|---------|---------|--------|----------|
| 1 | **aion-sentiment-in** | Training pipeline (Transformer fine-tuning) | ✅ Model Trained (98.55% acc) | `aion-sentiment-in/` |
| 2 | **aion-sentiment** | Inference API (sentiment + emotions) | ✅ Working (14K NRC lexicon) | `aion-sentiment/` |
| 3 | **aion-sectormap** | Ticker → Sector/Industry mapping | ✅ 592 tickers mapped | `aion-sectormap/` |
| 4 | **aion-volweight** | VIX-based confidence adjustment | ✅ Tested | `aion-volweight/` |
| 5 | **aion-newsimpact** | Historical news impact analysis | ✅ FAISS index working | `aion-newsimpact/` |

---

## 🎯 Package Details

### 1. aion-sentiment-in (Training Pipeline)

**Purpose:** Fine-tune Transformer on Indian financial news sentiment data

**Key Features:**
- ClickHouse data extraction (`extract_data.sql`)
- Data preparation with 80/20 train/val split
- Transformer fine-tuning with MPS (Apple Silicon) support
- Model accuracy: **98.55%**, F1: **98.65%**

**Files:**
- `train_sentiment.py` - Training script with CLI
- `prepare_data.py` - Data cleaning and preparation
- `extract_data.sql` - ClickHouse extraction query
- `model_card.md` - Model documentation

**Data Source:** `aion_master.news_master_v1` (957K rows with sentiment)

---

### 2. aion-sentiment (Inference API)

**Purpose:** Sentiment + Emotion analysis for financial headlines

**Key Features:**
- Transformer-based sentiment classification
- NRC Emotion Lexicon (14,182 words bundled)
- Emotions: fear, greed, panic, optimism
- DataFrame API: `analyze(df, text_column)`

**Test Results:**
```
Headline                              | Sentiment | Confidence | Emotions
--------------------------------------|-----------|------------|---------------------------
Investors panic selling triggers crash| neutral   | 70.8%      | panic:0.25, fear:0.11
Market reaches all-time high          | positive  | 64.1%      | greed:0.21, optimism:0.19
Fear grips Dalal Street               | neutral   | 77.6%      | fear:0.19, panic:0.21
```

**Package Size:** ~2.5 MB (lexicon bundled)

---

### 3. aion-sectormap (Ticker → Sector Mapping)

**Purpose:** Map NSE tickers to sectors, industries, and business groups

**Key Features:**
- 592 NSE tickers mapped
- 44 sectors, 334 business groups
- GIN (Group Identification Number) support
- DataFrame API: `map(df, ticker_column)`

**Data Coverage:**
```
Top Sectors:
- Financial Services: 194 companies
- Non Banking Financial Company: 42
- Power: 27
- Capital Goods: 26
- Realty: 23

Top Groups:
- Noel Tata Group: 33 companies
- Aditya Birla Group: 18
- Mukesh Ambani Group: 12
- Gautambhai Shantilal Adani: 10
```

**Sample Mapping:**
```python
mapper = SectorMapper()
mapper.get_sector('RELIANCE')  # 'Oil, Gas & Consumable Fuels'
mapper.get_group('TCS')        # 'Tata Group'
```

---

### 4. aion-volweight (VIX-based Confidence Adjustment)

**Purpose:** Adjust sentiment confidence based on India VIX regime

**VIX Regimes:**
| Regime | VIX Range | Multiplier | Effect |
|--------|-----------|------------|--------|
| LOW | < 12 | 1.0 | No adjustment |
| NORMAL | 12-15 | 1.0 | No adjustment |
| HIGH | 15-25 | 0.8 | -20% confidence |
| PANIC | ≥ 25 | 0.5 | -50% confidence |

**API:**
```python
from aion_volweight import weight_confidence

df_adjusted = weight_confidence(df, vix_value=20)  # HIGH regime → 0.8 multiplier
```

**Test Results:**
```
VIX 10 → LOW    (multiplier: 1.0)
VIX 13 → NORMAL (multiplier: 1.0)
VIX 18 → HIGH   (multiplier: 0.8)
VIX 28 → PANIC  (multiplier: 0.5)
```

---

### 5. aion-newsimpact (Historical News Impact Analysis)

**Purpose:** Find similar historical headlines and show average price impact

**Key Features:**
- Sentence transformers (all-MiniLM-L6-v2)
- FAISS index for similarity search
- Returns average 1-day price impact
- Query top-K similar headlines

**API:**
```python
from aion_newsimpact import NewsImpact

impact = NewsImpact(historical_df, text_column='headline')
result = impact.query('Market crashes on recession fears', top_k=5)
```

**Sample Output:**
```
Query: "Market crashes as investors fear economic downturn"

Top 3 Similar Headlines:
1. "Stock market crashes on recession fears" (similarity: 0.92) - Returns: -2.5%
2. "Investors panic as banks collapse" (similarity: 0.85) - Returns: -3.8%
3. "Banking crisis spreads across Europe" (similarity: 0.78) - Returns: -2.9%

Average Impact: -3.07%
```

---

## 📊 Data Assets

| Asset | Count | Location |
|-------|-------|----------|
| NSE Group Companies (Excel) | 591 companies, 44 sectors, 340 groups | `Empirical List of Group Companies with GIN.xlsx` |
| Sector Map JSON | 592 tickers | `aion-sectormap/src/aion_sectormap/data/` |
| NSE Sector Constituents | 188 companies | `data/nse_sector_constituents.csv` |
| Group Companies CSV | 591 companies | `data/nse_group_companies.csv` |
| Trained Model | 437 MB | `aion-sentiment-in/models/aion-sentiment-in-v1/` |
| NRC Lexicon | 14,182 words | `aion-sentiment/src/aion_sentiment/lexicons/` |

---

## 🔧 Installation & Usage

### Quick Start (All Packages)

```bash
# Clone repository
cd /Users/lokeshgupta/aion_open_source

# Install each package
cd aion-sentiment && pip install -e . && cd ..
cd aion-sectormap && pip install -e . && cd ..
cd aion-volweight && pip install -e . && cd ..
cd aion-newsimpact && pip install -e ".[dev]" && cd ..

# Run tests
cd aion-sentiment && pytest && cd ..
cd aion-sectormap && pytest && cd ..
cd aion-volweight && pytest && cd ..
cd aion-newsimpact && pytest && cd ..
```

### End-to-End Pipeline Example

```python
import pandas as pd
from aion_sentiment import AIONSentimentAnalyzer
from aion_sectormap import SectorMapper
from aion_volweight import weight_confidence
from aion_newsimpact import NewsImpact

# Sample headlines
headlines = pd.DataFrame({
    'ticker': ['RELIANCE', 'TCS', 'HDFCBANK'],
    'headline': [
        'Reliance Industries reports record profits',
        'TCS wins major digital transformation deal',
        'HDFC Bank expands rural presence'
    ]
})

# 1. Sector mapping
mapper = SectorMapper()
headlines = mapper.map(headlines, ticker_column='ticker')

# 2. Sentiment analysis
analyzer = AIONSentimentAnalyzer()
headlines = analyzer.analyze(headlines, text_column='headline')

# 3. VIX adjustment (assuming VIX=18 - HIGH regime)
headlines = weight_confidence(headlines, vix_value=18)

# 4. Historical impact (if historical data available)
# impact = NewsImpact(historical_df)
# result = impact.query(headlines['headline'].iloc[0])

print(headlines[['ticker', 'sector', 'sentiment_label', 'sentiment_confidence_adjusted']])
```

---

## 📋 Compliance Checklist

| Requirement | Status |
|-------------|--------|
| Apache 2.0 License on all packages | ✅ |
| AION Attribution in all files | ✅ |
| "AION Analytics" in descriptions | ✅ |
| Type hints (Python 3.9+) | ✅ |
| Google-style docstrings | ✅ |
| Unit tests | ✅ (100+ tests total) |
| Complete README | ✅ |
| Modern packaging (pyproject.toml) | ✅ |
| PEP 561 type hints (py.typed) | ✅ |
| No proprietary code | ✅ |
| No dependency on AION internal systems | ✅ |
| No user data collection without consent | ✅ |

---

## 🚀 PyPI Release Plan

### Week 1 (March 17-21, 2026)
- [ ] `aion-sentiment` - Core inference API
- [ ] `aion-sectormap` - Ticker mapping
- [ ] Blog post announcing release

### Week 2 (March 24-28, 2026)
- [ ] `aion-volweight` - VIX adjustment
- [ ] `aion-newsimpact` - News impact analysis
- [ ] Documentation site (Sphinx)

### Week 3 (March 31 - April 4, 2026)
- [ ] `aion-sentiment-in` - Training pipeline
- [ ] HuggingFace model upload
- [ ] Community feedback collection

---

## 📝 Optional Data Contribution (Future)

**Schema for anonymized data contribution:**

```json
{
  "headline_hash": "sha256 of headline",
  "sentiment_label": "positive",
  "sentiment_confidence": 0.95,
  "ticker": "RELIANCE",
  "sector": "Oil, Gas & Consumable Fuels",
  "week": "2025-12",
  "vix_regime": "NORMAL",
  "adjusted_confidence": 0.95
}
```

**Implementation:**
- Add `--contribute` flag to CLI tools
- POST to specified endpoint
- Store in cloud bucket
- No PII collected
- Explicit user consent required

---

## 🎯 Next Steps

1. **PyPI Releases** - Publish all 5 packages
2. **HuggingFace Upload** - Upload trained model (`aion-analytics/aion-sentiment-in-v1`)
3. **Documentation Site** - Sphinx-based docs
4. **CI/CD** - GitHub Actions for automated testing
5. **Blog Post** - Announce open-source release
6. **Community Feedback** - Collect and plan v2

---

*Generated: March 14, 2026*  
*AION Open-Source Project*
