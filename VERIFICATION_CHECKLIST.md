# AION Open-Source Verification Checklist

**Purpose:** Ensure all packages use the correct default model (`aion-analytics/aion-sentiment-in-v1`) and properly represent the India-tuned model without external attributions.

---

## ✅ Model Attribution Verification

### Default Model Check
- [ ] **Default model name** is `aion-analytics/aion-sentiment-in-v1` (NOT ProsusAI/finbert)
- [ ] **Model description** states "India-tuned" or "tuned on Indian financial news"
- [ ] **No mention of "FinBERT"** as the base model in documentation
- [ ] **Accuracy claim** is 98.55% on Indian financial news
- [ ] **Training data** is described as 957K headlines from AION analytics database

### Documentation Check
- [ ] Root README.md - Model section updated
- [ ] Package README.md - Overview section updated
- [ ] Source code docstrings - No FinBERT references
- [ ] Model card (HuggingFace) - No base model attribution
- [ ] Examples show default usage (no model_name parameter)
- [ ] Alternative models shown as optional override only

---

## 📦 Package-by-Package Verification

### aion-sentiment

```python
# ✅ CORRECT - Uses India-tuned model by default
analyzer = SentimentAnalyzer()

# ✅ CORRECT - Shows how to override (as alternative)
analyzer = SentimentAnalyzer(model_name="other-model")

# ❌ WRONG - Don't show as default example
analyzer = SentimentAnalyzer(model_name="ProsusAI/finbert")
```

**Files to check:**
- [ ] `src/aion_sentiment/__init__.py`
- [ ] `src/aion_sentiment/sentiment.py` - Default parameter
- [ ] `src/aion_sentiment/sentiment.py` - Docstring examples
- [ ] `README.md` - Overview section
- [ ] `README.md` - Quick Start examples
- [ ] `example.py` - All code examples

---

### aion-sentiment-in

**Files to check:**
- [ ] `train_sentiment.py` - Model loading comments
- [ ] `model_card.md` - Model description
- [ ] `MODEL_CARD_HF.md` - HuggingFace model card
- [ ] `README.md` - Training description

**Key phrases to verify:**
- [ ] "Transformer model" (NOT "FinBERT model")
- [ ] "India-tuned" or "tuned on Indian financial news"
- [ ] "957K Indian financial news headlines"
- [ ] "98.55% accuracy on validation set"

---

### aion-sectormap

**No sentiment model references needed** - This package is for ticker mapping only.

**Files to check:**
- [ ] `README.md` - No sentiment model mentions
- [ ] `src/aion_sectormap/mapper.py` - No model references

---

### aion-volweight

**No sentiment model references needed** - This package is for VIX adjustment only.

**Files to check:**
- [ ] `README.md` - Mentions "sentiment confidence" but not specific model
- [ ] `src/aion_volweight/volweight.py` - No model references

---

### aion-newsimpact

**No sentiment model references needed** - This package is for similarity search only.

**Files to check:**
- [ ] `README.md` - Uses sentence-transformers for embeddings (separate model)
- [ ] `src/aion_newsimpact/impact.py` - Embedding model is separate from sentiment

---

## 🚫 Prohibited Phrases

**DO NOT use these phrases:**

- ❌ "FinBERT model"
- ❌ "Fine-tuned from ProsusAI/finbert"
- ❌ "Base model: FinBERT"
- ❌ "ProsusAI FinBERT"
- ❌ "Pretrained on financial corpus"

**USE these phrases instead:**

- ✅ "India-tuned model"
- ✅ "Transformer model tuned on Indian financial news"
- ✅ "AION-Sentiment-IN-v1"
- ✅ "Trained on 957K Indian financial news headlines"
- ✅ "98.55% accuracy on Indian market news"

---

## 📝 Example Corrections

### Before (WRONG):
```markdown
Uses the ProsusAI FinBERT model for sentiment analysis.
```

### After (CORRECT):
```markdown
Uses the India-tuned AION-Sentiment-IN model for sentiment analysis.
```

---

### Before (WRONG):
```python
analyzer = SentimentAnalyzer(model_name="ProsusAI/finbert")
```

### After (CORRECT):
```python
# Default: Uses India-tuned model
analyzer = SentimentAnalyzer()

# Optional: Use custom model
# analyzer = SentimentAnalyzer(model_name="custom-model")
```

---

## 🔧 How to Fix

If you find incorrect references:

1. **Replace model name** in default parameters
2. **Update docstrings** to remove FinBERT mentions
3. **Update README** overview sections
4. **Update examples** to show default usage
5. **Update model cards** to remove base model attribution

---

## ✅ Final Verification

Before releasing any package:

```bash
# Search for prohibited terms
grep -r "FinBERT" src/ README.md
grep -r "ProsusAI" src/ README.md
grep -r "fine-tuned" src/ README.md

# Verify default model
grep -r "aion-analytics/aion-sentiment-in-v1" src/ README.md
```

**All checks must pass before PyPI release.**

---

*Last updated: March 14, 2026*  
*AION Open-Source Project*
