# Contributing to AION Market Sentiment

Thank you for your interest in contributing! This guide focuses on high-impact areas where community contributions are most valuable.

---

## Where You Can Help

### 1. Improve Taxonomy Keywords (Highest Priority)

The taxonomy currently matches **~6.6%** of headlines. We need help adding keywords for **112 uncovered events**.

#### Get Started

```bash
# View the list of events without matches
cat aion_taxonomy/no_match_events.txt

# Pick an event category to work on
```

#### Priority Categories

- **RBI Monetary Policy:** repo rate, CRR, SLR, OMO operations
- **Corporate Actions:** CEO/CFO exits, board changes, restructuring
- **Global Events:** Fed decisions, China growth, oil price shocks
- **Sector Events:** Auto sales, bank NPA, IT deal wins, pharma approvals
- **Government Schemes:** PLI, FAME, PM-KISAN, Mudra, Digital India

#### How to Contribute Keywords

1. **Pick an event** from `no_match_events.txt`

2. **Research headline patterns** - Think about how Indian financial news writes about this topic

3. **Add 3-5 keywords** per event in `aion_taxonomy/taxonomy_india_v2_calibrated.yaml`:

```yaml
- id: macro_rbi_repo_hike
  keywords:
  - repo rate hike          # existing
  - rbi hikes repo          # existing
  - rbi raises rates        # your addition
  - monetary policy tightens # your addition
  - central bank hikes      # your addition
```

4. **Test your changes:**

```bash
cd aion-sentiment
python3 backfill_taxonomy.py --limit 10000 --taxonomy-path ../aion_taxonomy/taxonomy_india_v2_calibrated.yaml --dry-run
```

5. **Compare match rates** before and after:

```
Before: 6.6% match rate
After:  7.2% match rate (+0.6 pp improvement)
```

6. **Submit PR** with:
   - Updated YAML file
   - Before/after match rates
   - List of keywords added

---

### 2. Add New Sectors or Events

#### New Sectors

If you identify sectors not covered in the current 32:

```yaml
# Add to sectors list in taxonomy YAML
- id: Renewable Energy
  beta_default: 1.3
```

#### New Events

For new event types:

```yaml
- id: sector_renewable_policy_support
  name: Renewable Energy Policy Support
  seasonal_activation: false
  keywords:
  - renewable energy subsidy
  - solar power incentive
  - wind energy policy
  base_impact:
    mild: 0.15
    normal: 0.35
    severe: 0.55
  default_impact: normal
  market_weight: 0.7
  sector_impacts:
    Power:
      multiplier: 1.2
      bias: aligned
      rationale: Policy support boosts sector
```

---

### 3. Calibrate Event Impacts

Help improve base_impact values using real data:

```bash
# Run backfill on full dataset
python3 backfill_taxonomy.py --limit 200000 --taxonomy-path taxonomy_india_v2_calibrated.yaml

# Run calibration script
cd aion_taxonomy
python3 calibrate_taxonomy.py --min-count 30
```

Review `calibration_summary.txt` for events needing base_impact adjustment.

---

### 4. Build Neural Event Classifier

Current taxonomy is rule-based. Help us build a neural classifier:

- Fine-tune a transformer on event classification
- Integrate with existing pipeline
- Compare accuracy vs. keyword matching

---

## Contribution Process

### Step 1: Fork the Repo

```bash
git clone https://github.com/YOUR_USERNAME/market-sentiments.git
cd market-sentiments
```

### Step 2: Create Branch

```bash
git checkout -b feature/taxonomy-keywords-rbi-policy
```

### Step 3: Make Changes

Edit `aion_taxonomy/taxonomy_india_v2_calibrated.yaml` with your keyword additions.

### Step 4: Test Changes

```bash
cd aion-sentiment
python3 backfill_taxonomy.py --limit 10000 --taxonomy-path ../aion_taxonomy/taxonomy_india_v2_calibrated.yaml --dry-run
```

### Step 5: Document Results

Create a simple summary:

```markdown
## Changes

Event: macro_rbi_repo_hike
Keywords Added: 5
- rbi raises rates
- monetary policy tightens
- central bank hikes
- repo rate raised
- rbi rate decision

## Results

Before: 5.8% match rate (580/10000)
After:  6.4% match rate (640/10000)
Improvement: +0.6 pp
```

### Step 6: Submit PR

```bash
git add aion_taxonomy/taxonomy_india_v2_calibrated.yaml
git commit -m "Add keywords for macro_rbi_repo_hike event"
git push origin feature/taxonomy-keywords-rbi-policy
```

Open pull request with:
- Description of changes
- Before/after match rates
- List of keywords added

---

## Code Style

- **Python:** Follow PEP 8, use type hints
- **YAML:** Consistent indentation (2 spaces), alphabetical where sensible
- **Documentation:** Clear docstrings, examples in docstrings

---

## Questions?

- **Bug Reports:** Use GitHub Issues
- **Discussions:** GitHub Discussions tab
- **Direct Contact:** contributors@aion.opensource

---

## Thank You!

Your contributions help make AION Market Sentiment better for everyone. Every keyword added improves coverage for the entire community.
