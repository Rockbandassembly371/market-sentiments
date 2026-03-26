# Release Checklist - AION Market Sentiment v2.0.0

## Pre-Release

### Code Quality
- [ ] All tests pass (`pytest` in each package)
- [ ] No linting errors (`flake8` or `black --check`)
- [ ] Type checking passes (`mypy`)
- [ ] No FinBERT/BERT references in production code
- [ ] Version numbers updated in all `pyproject.toml` files

### Documentation
- [ ] README.md updated with v2.0.0 status
- [ ] CHANGELOG.md created with all changes
- [ ] CONTRIBUTING.md created with keyword contribution guide
- [ ] Model card updated for HuggingFace
- [ ] Backfill summary report generated

### Model & Data
- [ ] Model trained on VADER-labeled data (823K samples)
- [ ] Validation accuracy ≥ 98%
- [ ] Taxonomy calibrated (24 events with volatility)
- [ ] Backfill complete (200K headlines, 6.6% match rate)
- [ ] No-match events list generated (112 events)

### Testing
- [ ] Sentiment analysis tested on 10 Redis headlines
- [ ] Taxonomy tested on sample headlines
- [ ] Sector mapping verified (592 tickers)
- [ ] VIX adjustment working

---

## Release Execution

### Run Release Script

```bash
# Make script executable
chmod +x release.sh

# Dry run first (recommended)
./release.sh --version 2.0.0 --pypi-upload --hf-upload --gh-release --dry-run

# Actual release
./release.sh --version 2.0.0 --pypi-upload --hf-upload --gh-release
```

### Manual Steps (if needed)

#### PyPI Upload
```bash
# aion-sentiment
cd aion-sentiment
python3 setup.py sdist bdist_wheel
twine upload dist/*

# aion-sectormap
cd ../aion-sectormap
python3 setup.py sdist bdist_wheel
twine upload dist/*

# aion-volweight
cd ../aion-volweight
python3 setup.py sdist bdist_wheel
twine upload dist/*

# aion-taxonomy
cd ../aion_taxonomy
python3 setup.py sdist bdist_wheel
twine upload dist/*
```

#### HuggingFace Upload
```bash
cd aion-sentiment-in/models
huggingface-cli upload aion-analytics/aion-sentiment-in-v2 aion-sentiment-in-v2 .
```

#### GitHub Release
1. Go to: https://github.com/AION-Analytics/market-sentiments/releases/new
2. Tag version: `v2.0.0`
3. Release title: `Release v2.0.0`
4. Copy release notes from CHANGELOG.md
5. Click "Publish release"

---

## Post-Release

### Verification
- [ ] PyPI packages visible: https://pypi.org/user/aion-analytics/
- [ ] HuggingFace model public: https://huggingface.co/aion-analytics/aion-sentiment-in-v2
- [ ] GitHub release published: https://github.com/AION-Analytics/market-sentiments/releases
- [ ] Installation works: `pip install aion-sentiment aion-taxonomy`
- [ ] Quickstart example runs successfully

### Communication
- [ ] Update main README with release announcement
- [ ] Post on relevant forums/communities
- [ ] Notify contributors
- [ ] Update any internal documentation

### Monitoring
- [ ] Watch for installation issues
- [ ] Monitor HuggingFace download counts
- [ ] Track PyPI download statistics
- [ ] Respond to GitHub issues promptly

---

## Rollback Plan (if needed)

### PyPI
```bash
# Delete package from PyPI (requires admin)
twine unregister --repository pypi aion-sentiment
```

### HuggingFace
```bash
# Delete model repo (requires admin)
huggingface-cli delete-repo aion-analytics/aion-sentiment-in-v2
```

### GitHub
- Delete release tag
- Force push to remove commit if critical

---

## Sign-Off

- [ ] Release manager approval
- [ ] Technical lead approval
- [ ] Documentation reviewed
- [ ] All checks passed

**Release Date:** 2026-03-26
**Version:** 2.0.0
**Release Manager:** _______________
