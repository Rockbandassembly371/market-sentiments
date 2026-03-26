# Changelog

All notable changes to the AION Market Sentiment project.

---

## [3.0.0] - 2026-03-26

### Fixed

- Sentiment Model v3: Retrained on AION taxonomy labels after discovering labeling issues in v2.
- Critical Misclassifications: "Markets Crashing" now correctly predicts negative (was positive in v2).
- Critical Misclassifications: "TCS Record Earnings" now correctly predicts positive (was neutral in v2).
- Label Source: Switched to AION taxonomy event sentiment (136 events with known sentiment).

### Added

- Model v3: aion-analytics/aion-sentiment-in-v3 with 99.63% validation accuracy.
- Training Data: 400K headlines with AION taxonomy labels (100K validation).
- Label Correction Script: create_corrected_data.py for generating taxonomy-labeled datasets.
- Model Card v3: Comprehensive documentation with v2 vs v3 comparison.

### Changed

- Default Model: aion-sentiment now uses aion-analytics/aion-sentiment-in-v3 (AION taxonomy-trained).
- Validation Accuracy: Improved from 98.55% (v2) to 99.63% (v3).
- Test Headlines Accuracy: Improved from 33% (2/6) to 67% (4/6) on problematic cases.
- Training Time: ~10 hours on Apple M4 (MPS acceleration).

### Known Limitations

- Ambiguous Headlines: "Stocks to buy" and "RBI hikes repo rate" may predict neutral (working on improvement).
- Taxonomy Coverage: ~40% of headlines match taxonomy events directly.
- Overconfidence: Model often returns 100% confidence; use with caution in production.

### Next Steps

- Improve ambiguous headline classification (investment recommendations, policy announcements).
- Increase taxonomy match rate beyond 40%.
- Add more events to taxonomy for better coverage.

### Comparison: v2 vs v3

| Metric | v2 | v3 (AION Taxonomy) | Improvement |
|--------|------------|---------------|-------------|
| Validation Accuracy | 98.55% | 99.63% | +1.08 pp |
| F1 Score | 98.65% | 99.54% | +0.89 pp |
| Test Headlines (6 cases) | 33% (2/6) | 67% (4/6) | +34 pp |
| Training Samples | 823K | 400K | Smaller but corrected |

---

## [2.0.0] - 2026-03-26

- Fixed: Sentiment Model retrained on corrected labels after discovering 40-50% mislabeling in original training data. Accuracy now 98% on test set.
- Fixed: Label Mapping corrected in inference code (0=negative, 1=neutral, 2=positive).
- Fixed: Config.json restored to correct id2label mapping matching training data.
- Added: Taxonomy Package with 136 events across 7 categories.
- Added: Event Classification with keyword-based case-insensitive substring matching.
- Added: Contextual Modifiers for impact level selection based on headline context.
- Added: Sector-Aware Signals with bias propagation (aligned/inverse/neutral).
- Added: Confidence Blending using weighted linear combination.
- Added: Calibration Pipeline for data-driven event_volatility computation.
- Added: Backfill Script supporting 200K classification with --use-body flag.
- Changed: Default Model to aion-analytics/aion-sentiment-in-v2 (India-tuned, 98% accuracy).
- Changed: Training Data switched to corrected labels (823K samples).
- Changed: Taxonomy YAML now includes event_volatility and adjusted base impacts for 14 events.
- Changed: Keyword Coverage enhanced for 22 events with ~100 new keywords.
- Changed: Match Rate improved from 5.8% to 6.6% through keyword enhancement.
- Known: Match Rate at 6.6% with 112 events still uncovered.
- Known: Sentiment model shows overconfidence (often 100%); use with caution.
- Known: Taxonomy is rule-based beta; fine-tuned classifier planned for future.
- Known: Low volatility across calibrated events (homogeneous headlines within categories).
- Next: Community help needed for uncovered events (no_match_events.txt).
- Next: Improve sector mappings for Indian market context.
- Next: Provide more contextual modifier examples.
- Next: Contribute calibration data for rare events.
- Next: Test --use-body flag for improved matching with article body text.
- Resolved: Label Inversion fixed in inference mapping.
- Resolved: Training Data replaced with corrected labels.
- Resolved: Sector Overrides added with flip rule and dynamic threshold.
- Resolved: Documentation cleaned of all external model references.

---

## [1.0.0] - 2026-03-14

- Initial Release: aion-sentiment with transformer-based sentiment analysis.
- Initial Release: aion-sectormap with NSE ticker to sector mapping (592 tickers).
- Initial Release: aion-volweight with VIX-adjusted confidence weighting.
- Initial Release: Base model trained on 957K Indian financial news headlines.

---
