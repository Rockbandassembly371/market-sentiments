# AION-Sentiment-IN-v3: Built by AION Analytics

## This is an AION Analytics Model

### Why?

1. **AION Taxonomy Methodology**
   - 136 predefined market events with known sentiment
   - Each event has base_impact, sector_impacts, market_weight
   - This is AION's proprietary IP

2. **AION Training Data**
   - 400K Indian financial news headlines from AION's corpus
   - Labels derived from AION taxonomy event matching
   - Curated and processed by AION's pipeline

3. **AION Trained Weights**
   - All classification layers trained from scratch
   - 3 epochs on AION's taxonomy-labeled data
   - Final weights are 100% AION's training

4. **AION Innovation**
   - Fixed v2 failures using taxonomy methodology
   - "Markets Crashing" → negative (v2 said positive)
   - "TCS Record Earnings" → positive (v2 said neutral)

### What We Use (Infrastructure, Not Methodology)

- **PyTorch** - Tensor operations (like crediting Python for code)
- **HuggingFace Transformers** - Training framework
- **Transformer Architecture** - Model structure (open source)

These are tools, not the methodology. The **value** is in:
- AION Taxonomy (136 events)
- AION Labels (sentiment from base_impact)
- AION Weights (trained on AION data)

### Results

| Metric | v3 (AION Taxonomy) |
|--------|-------------------|
| Validation Accuracy | **99.63%** |
| F1 Score | **99.54%** |
| Test Headlines (6 cases) | **67% (4/6)** |

### The Innovation

**Before (v2):** Labels from external sentiment lexicon → 50% mislabeling

**After (v3):** Labels from AION taxonomy → 99.63% accuracy

**The difference is AION's methodology.**

---

**© 2026 AION Analytics. All rights reserved.**
