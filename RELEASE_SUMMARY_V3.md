# AION Market Sentiment v3.0.0 - Release Summary

**Release Date:** March 26, 2026

**Version:** 3.0.0

---

## What's New in v3.0.0

### Major Fix: AION Taxonomy Labels

The v2 model had a critical issue: ~50% of training labels were wrong because external sentiment lexicons don't understand financial context.

**Example of v2 failure:**
- "European shares extend selloff" → labeled as **positive** ❌
- "Markets Crashing Today" → predicted as **positive** ❌

**v3 solution:**
- Used 136 AION taxonomy events with known sentiment
- Matched headlines to events → assigned correct labels
- Retrained model on corrected data

### Results

| Metric | v2 | v3 (AION Taxonomy) | Improvement |
|--------|-----|---------------|-------------|
| Validation Accuracy | 98.55% | **99.63%** | +1.08 pp |
| F1 Score | 98.65% | **99.54%** | +0.89 pp |
| Test Headlines (6 cases) | 33% (2/6) | **67% (4/6)** | +34 pp |

### Fixed Misclassifications

| Headline | v2 Prediction | v3 Prediction |
|----------|---------------|---------------|
| "Markets Crashing" | Positive ❌ | **Negative ✅** |
| "TCS Record Earnings" | Neutral ❌ | **Positive ✅** |

---

## Files Changed

### Documentation
- ✅ `CHANGELOG.md` - Added v3.0.0 entry with full details
- ✅ `README.md` - Updated with v3 status (ready for use)
- ✅ `CONTRIBUTING.md` - Community contribution guide
- ✅ `aion-sentiment-in/MODEL_CARD_HF_V3.md` - HuggingFace model card

### Code
- ✅ `aion-sentiment/src/aion_sentiment/sentiment.py` - Default model changed to v3
- ✅ `aion-sentiment-in/create_corrected_data.py` - Label correction script

### Model
- ✅ `aion-sentiment-in/models/aion-sentiment-in-v3/` - Trained model (267 MB)
  - config.json
  - model.safetensors
  - tokenizer_config.json
  - tokenizer.json
  - training_metadata.json

---

## Manual Release Steps

### 1. Upload Model to HuggingFace

```bash
# Option A: Using CLI (recommended)
cd /Users/lokeshgupta/aion_open_source/aion-sentiment-in/models
huggingface-cli login
huggingface-cli upload aion-analytics/aion-sentiment-in-v3 aion-sentiment-in-v3 .

# Option B: Using Python with token
export HF_TOKEN='your_huggingface_token'
cd /Users/lokeshgupta/aion_open_source/aion-sentiment-in/models
source /Users/lokeshgupta/aion_open_source/aion-sentiment/venv/bin/activate
python3 << 'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="aion-sentiment-in-v3",
    repo_id="aion-analytics/aion-sentiment-in-v3",
    repo_type="model",
    token="your_huggingface_token"
)
print("Upload complete!")
EOF
```

### 2. Upload Packages to PyPI

```bash
# aion-sentiment
cd /Users/lokeshgupta/aion_open_source/aion-sentiment
python3 setup.py sdist bdist_wheel
twine upload dist/*

# aion-taxonomy (if applicable)
cd /Users/lokeshgupta/aion_open_source/aion_taxonomy
python3 setup.py sdist bdist_wheel
twine upload dist/*

# aion-sectormap (if applicable)
cd /Users/lokeshgupta/aion_open_source/aion-sectormap
python3 setup.py sdist bdist_wheel
twine upload dist/*
```

### 3. Create GitHub Release

```bash
# Go to: https://github.com/AION-Analytics/market-sentiments/releases/new
# Tag version: v3.0.0
# Release title: Release v3.0.0
# Copy release notes from CHANGELOG.md
# Click "Publish release"
```

Or using GitHub CLI:
```bash
cd /Users/lokeshgupta/aion_open_source
gh release create v3.0.0 \
  --title "Release v3.0.0" \
  --notes "See CHANGELOG.md for details"
```

### 4. Verify Release

```bash
# Check PyPI
https://pypi.org/project/aion-sentiment/

# Check HuggingFace
https://huggingface.co/aion-analytics/aion-sentiment-in-v3

# Check GitHub Releases
https://github.com/AION-Analytics/market-sentiments/releases
```

### 5. Test Installation

```bash
# Fresh install
pip install aion-sentiment

# Test
python3 -c "
from aion_sentiment import SentimentAnalyzer
analyzer = SentimentAnalyzer()
result = analyzer.predict('Markets crashing today')
print(result)
# Should output: {'label': 'negative', 'confidence': 0.XX}
"
```

---

## Release Checklist

- [x] Model trained and tested
- [x] CHANGELOG.md updated
- [x] README.md updated
- [x] Default model changed to v3
- [x] Model card created
- [ ] Model uploaded to HuggingFace ← **TODO: Manual step**
- [ ] Packages uploaded to PyPI ← **TODO: Manual step**
- [ ] GitHub release created ← **TODO: Manual step**
- [ ] Installation tested ← **TODO: After upload**

---

## Known Limitations

1. **Ambiguous Headlines:** May misclassify:
   - "Stocks to buy in 2026" → neutral (should be positive)
   - "RBI hikes repo rate" → neutral (should be negative)

2. **Taxonomy Match Rate:** ~40% of headlines match taxonomy events directly.

3. **Overconfidence:** Model often returns 100% confidence; use with caution.

---

## Next Steps (Future Releases)

- Improve ambiguous headline classification
- Increase taxonomy match rate beyond 40%
- Add more events to taxonomy for better coverage
- Consider fine-tuned classifier instead of keyword matching

---

## Contact

- **GitHub:** https://github.com/AION-Analytics/market-sentiments
- **Issues:** https://github.com/AION-Analytics/market-sentiments/issues
- **Email:** contributors@aion.opensource

---

**Release Manager:** _______________  
**Date:** March 26, 2026  
**Status:** Ready for release (pending manual upload steps)
