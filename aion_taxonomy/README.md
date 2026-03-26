# AION Taxonomy

**Rule-based event classification and impact scoring for Indian equity markets**

AION Taxonomy provides deterministic event classification and sector-aware impact scoring for Indian financial news. It uses a comprehensive YAML taxonomy of market events, economic indicators, and policy changes to classify headlines and compute sentiment signals.

## Features

- **Event Classification**: Keyword-based matching against 100+ predefined market events
- **Macro Signal Computation**: Compute market-wide sentiment signals from classified events
- **Sector-Specific Impacts**: Sector-level signal computation with bias and multiplier support
- **Confidence Blending**: Combine taxonomy match, model probability, and agreement scores
- **Optional Integration**: Works standalone or integrates with `aion-sentiment-in` and `aion-sectormap`

## Installation

```bash
# Install from source
cd aion_taxonomy
pip install -e .

# Install with optional dependencies
pip install -e ".[sectormap]"  # For ticker-to-sector mapping
pip install -e ".[sentiment]"  # For sentiment model integration
pip install -e ".[dev]"        # For development tools
```

## Quick Start

```python
from aion_taxonomy import TaxonomyPipeline

# Initialize pipeline with taxonomy file
pipeline = TaxonomyPipeline(taxonomy_path="taxonomy_india_v2.yaml")

# Process a headline
result = pipeline.process("RBI hikes repo rate by 25 bps")

print(f"Event: {result['event']['event_id']}")
print(f"Macro Signal: {result['macro_signal']:.3f}")
print(f"Confidence: {result['confidence']:.2%}")
```

## With Sector Mapping

```python
from aion_taxonomy import TaxonomyPipeline
from aion_sectormap import SectorMapper

# Initialize sector mapper
sector_mapper = SectorMapper()

# Initialize pipeline with sector mapper
pipeline = TaxonomyPipeline(
    taxonomy_path="taxonomy_india_v2.yaml",
    sector_mapper=sector_mapper
)

# Process headline with ticker
result = pipeline.process(
    headline="RBI hikes repo rate by 25 bps",
    ticker="HDFCBANK"
)

print(f"Active Sector: {result['active_sector_id']}")
print(f"Sector Signal: {result['active_sector_signal']:.3f}")
```

## With Sentiment Model Agreement

```python
from aion_taxonomy import TaxonomyPipeline
from aion_sentiment import SentimentAnalyzer

# Initialize components
pipeline = TaxonomyPipeline(taxonomy_path="taxonomy_india_v2.yaml")
sentiment_model = SentimentAnalyzer()

# Get model prediction
headline = "RBI hikes repo rate by 25 bps"
model_output = sentiment_model.predict([headline])[0]

# Process with model agreement
result = pipeline.process(
    headline=headline,
    model_output={
        'label': model_output['label'],
        'confidence': model_output['score']
    }
)

print(f"Taxonomy Signal: {result['macro_signal']:.3f}")
print(f"Model Label: {model_output['label']}")
print(f"Agreement Score: {result['confidence_components']['agreement_score']:.2f}")
print(f"Final Confidence: {result['confidence']:.2%}")
```

## Package Structure

```
aion_taxonomy/
├── src/aion_taxonomy/
│   ├── __init__.py       # Package exports
│   ├── loader.py         # YAML loading and validation
│   ├── classifier.py     # Keyword-based event classification
│   ├── impact.py         # Macro and sector signal computation
│   ├── confidence.py     # Confidence blending
│   ├── pipeline.py       # Main entry point
│   └── utils.py          # Helper functions
├── tests/
├── data/
├── README.md
├── setup.py
├── pyproject.toml
└── requirements.txt
```

## API Reference

### TaxonomyPipeline

Main class for processing headlines.

```python
pipeline = TaxonomyPipeline(taxonomy_path, sector_mapper=None)
```

**Methods:**

- `process(headline, ticker=None, date=None, model_output=None)` → Process single headline
- `process_batch(headlines)` → Process multiple headlines
- `get_event_details(event_id)` → Get event by ID
- `list_events()` → List all events in taxonomy

### EventClassifier

Keyword-based event classification.

```python
classifier = EventClassifier(taxonomy)
result = classifier.classify(headline)
```

**Returns:**
```python
{
    'event_id': 'macro_rbi_repo_hike',
    'event_name': 'RBI Repo Rate Hike',
    'category_id': 'macro_economy',
    'subcategory_id': 'monetary_policy',
    'match_score': 0.85,
    'matched_keywords': ['repo rate hike', 'rbi hikes'],
    'base_impact': {'mild': -0.25, 'normal': -0.55, 'severe': -0.85},
    'default_impact': 'normal',
    'market_weight': 0.9,
    'sector_impacts': {...}
}
```

### Signal Computation

```python
from aion_taxonomy import get_macro_signal, get_sector_signal

# Compute macro signal
macro_signal, impact_level = get_macro_signal(event, headline)

# Compute sector signal
sector_result = get_sector_signal(macro_signal, event, 'Banks')
```

### Confidence Computation

```python
from aion_taxonomy import compute_confidence, compute_agreement_score

# Compute agreement between taxonomy and model
agreement = compute_agreement_score(
    taxonomy_signal=0.5,
    model_label='positive',
    model_confidence=0.85
)

# Compute overall confidence
confidence = compute_confidence(
    taxonomy_match=0.8,
    data_quality=0.9,
    model_probability=0.85,
    agreement_score=1.0
)
```

## Taxonomy Format

The taxonomy is defined in YAML format with the following structure:

```yaml
metadata:
  version: 2.0.0
  market: INDIA_EQUITY
  
config:
  impact_scale:
    POSITIVE: 1.0
    NEUTRAL: 0.0
    NEGATIVE: -1.0
  confidence_weights:
    model_probability: 0.4
    taxonomy_match: 0.3
    data_quality: 0.2
    agreement_score: 0.1

sectors:
- id: Financial Services
  beta_default: 1.1
- id: Banks
  beta_default: 1.15

categories:
- id: macro_economy
  subcategories:
  - id: monetary_policy
    events:
    - id: macro_rbi_repo_hike
      name: RBI Repo Rate Hike
      keywords:
      - repo rate hike
      - rbi hikes repo
      base_impact:
        mild: -0.25
        normal: -0.55
        severe: -0.85
      market_weight: 0.9
      sector_impacts:
        Banks:
          multiplier: 1.15
          bias: inverse
          rationale: Funding costs rise
```

## License

Apache License 2.0

## Contributing

See the main AION repository for contribution guidelines.
