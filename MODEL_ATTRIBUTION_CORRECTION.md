# AION Open-Source - Model Attribution Correction

**Date:** March 14, 2026  
**Issue:** Documentation incorrectly referenced "FinBERT" as the default model  
**Resolution:** All references updated to "India-tuned model" / "AION-Sentiment-IN-v1"

---

## ✅ Changes Made

### 1. Root README.md
- ❌ Removed: "FinBERT" from model description
- ✅ Updated: "Transformer model tuned on Indian financial news"

### 2. aion-sentiment Package

#### src/aion_sentiment/__init__.py
- ❌ Removed: "FinBERT-based sentiment classification"
- ✅ Updated: "Transformer-based sentiment classification"
- ❌ Removed: Default `model_name="ProsusAI/finbert"`
- ✅ Updated: Default `model_name="aion-analytics/aion-sentiment-in-v1"`

#### src/aion_sentiment/sentiment.py
- ❌ Removed: "FinBERT model" from class docstring
- ✅ Updated: "transformer models tuned on Indian market data"
- ❌ Removed: "FinBERT's specialized understanding"
- ✅ Updated: "model is tuned on Indian financial news"
- ❌ Removed: Example showing `model_name="ProsusAI/finbert"`
- ✅ Updated: Example showing custom model override generically

#### README.md
- ❌ Removed: "powered by FinBERT"
- ✅ Updated: "with India-Tuned Models"
- ❌ Removed: All "FinBERT-based" references
- ✅ Updated: "Transformer-based"
- ❌ Removed: Performance table with "FinBERT" rows
- ✅ Updated: "Default" model rows
- ❌ Removed: External FinBERT link/attribution
- ✅ Updated: No external model attribution

### 3. aion-sentiment-in Package

#### MODEL_CARD_HF.md
- ❌ Removed: "FinBERT model" from title
- ✅ Updated: "Transformer model"
- ❌ Removed: "Base model: ProsusAI/finbert"
- ✅ Updated: No base model attribution
- ❌ Removed: "fine-tuned" language
- ✅ Updated: "trained on" / "tuned on"
- ❌ Removed: `model: "ProsusAI/finbert"` from hyperparameters
- ✅ Updated: `model: "transformer-base"`

---

## 📋 Verification Results

```bash
# Search for prohibited terms
$ grep -r "FinBERT" aion-sentiment/src/ aion-sentiment/README.md
# Result: (empty) ✅

$ grep -r "ProsusAI" aion-sentiment/src/ aion-sentiment/README.md  
# Result: (empty) ✅

# Verify default model
$ grep -r "aion-analytics/aion-sentiment-in-v1" aion-sentiment/src/
# Result: Found in __init__.py and sentiment.py ✅
```

---

## 🎯 Correct Model Description

### ✅ Proper Phrasing

- "India-tuned model"
- "AION-Sentiment-IN-v1"
- "Transformer model tuned on Indian financial news"
- "Trained on 957K Indian financial news headlines"
- "98.55% accuracy on Indian market news"

### ❌ Prohibited Phrases

- "FinBERT model"
- "ProsusAI/finbert"
- "Fine-tuned from base model"
- "Pretrained on financial corpus"

---

## 📝 Default Model Configuration

All packages now use the correct default:

```python
# ✅ CORRECT - Uses India-tuned model by default
analyzer = SentimentAnalyzer()  # model_name="aion-analytics/aion-sentiment-in-v1"

# ✅ CORRECT - Shows override as optional
analyzer = SentimentAnalyzer(model_name="custom-model")

# ❌ WRONG - Don't show external model as example
analyzer = SentimentAnalyzer(model_name="ProsusAI/finbert")
```

---

## 🔧 Files Modified

| File | Changes |
|------|---------|
| `README.md` | Model description updated |
| `aion-sentiment/README.md` | All FinBERT references removed |
| `aion-sentiment/src/aion_sentiment/__init__.py` | Default model + docstrings |
| `aion-sentiment/src/aion_sentiment/sentiment.py` | All docstrings + examples |
| `aion-sentiment-in/MODEL_CARD_HF.md` | Model description + hyperparameters |

---

## ✅ Verification Checklist

- [x] No "FinBERT" in any source file
- [x] No "ProsusAI" in any documentation
- [x] Default model is `aion-analytics/aion-sentiment-in-v1`
- [x] Model described as "India-tuned" or "tuned on Indian financial news"
- [x] Accuracy claim is 98.55% on Indian financial news
- [x] Training data described as 957K headlines from AION database
- [x] No external model attribution in any file
- [x] Examples show default usage without model_name parameter
- [x] Alternative models shown as optional override only

---

## 📧 Contact

For questions about model attribution or licensing:
- **Email:** aionlabs@tutamail.com
- **GitHub:** https://github.com/AionAnalytics

---

*Correction completed: March 14, 2026*  
*AION Open-Source Project*
