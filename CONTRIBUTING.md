# Contributing to AION Market Sentiment

Thank you for your interest in contributing! This project is designed for the Indian financial community.

## Quick Links

- [Good First Issues](https://github.com/AION-Analytics/market-sentiments/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
- [Help Wanted](https://github.com/AION-Analytics/market-sentiments/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
- [Feature Requests](https://github.com/AION-Analytics/market-sentiments/issues?q=is%3Aissue+is%3Aopen+label%3A%22enhancement%22)

## Why Contribute?

-  **Built for Indian Markets** - NSE/BSE focused sentiment analysis
-  **Low-Latency Design** - Optimized for intraday trading (<100ms inference)
-  **Production-Ready** - 98.55% accuracy on real financial news
- 🤝 **Community-Driven** - Open collaboration for better market intelligence

## How to Contribute

### 1. Report Bugs

Use the [bug report template](https://github.com/AION-Analytics/market-sentiments/issues/new?template=bug_report.md)

Include:
- Python version
- OS details
- Steps to reproduce
- Expected vs actual behavior

### 2. Suggest Features

Use the [feature request template](https://github.com/AION-Analytics/market-sentiments/issues/new?template=feature_request.md)

Include:
- Use case description
- Expected behavior
- Example code (if applicable)

### 3. Submit Code

```bash
# Fork the repo
git fork https://github.com/AION-Analytics/market-sentiments

# Clone your fork
git clone https://github.com/YOUR_USERNAME/market-sentiments

# Create branch
git checkout -b feature/your-feature-name

# Make changes
# Add tests
# Run tests: pytest

# Commit
git commit -m "feat: Add your feature description"

# Push
git push origin feature/your-feature-name

# Open Pull Request
```

### 4. Improve Documentation

- Fix typos
- Add examples
- Improve README sections
- Add tutorials

### 5. Add Data Sources

Help us support more Indian news sources:
- Regional language news (Hindi, Tamil, etc.)
- Social media sentiment (Twitter, StockTwits)
- Options flow data
- FII/DII flow data

## Good First Issues

Look for issues labeled:
- `good first issue` - Perfect for beginners
- `help wanted` - Need community help
- `documentation` - Improve docs
- `tests` - Add test coverage

## Development Setup

```bash
# Clone repo
git clone https://github.com/AION-Analytics/market-sentiments
cd market-sentiments

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
black .
flake8
```

## Code Style

- **Formatting:** Black
- **Linting:** Flake8
- **Type Hints:** Required for all public APIs
- **Docstrings:** Google style

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=aion_sentiment --cov-report=html

# Run specific test
pytest tests/test_sentiment.py -v
```

## Release Process

1. Bump version in `setup.py`
2. Update `CHANGELOG.md`
3. Create PR
4. Maintainer reviews
5. Tag release
6. Publish to PyPI

## Questions?

- **Discord:** [Join our server](https://discord.gg/aion-analytics)
- **Email:** aionlabs@tutamail.com
- **Discussions:** [GitHub Discussions](https://github.com/AION-Analytics/market-sentiments/discussions)

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

---

*Thank you for contributing to open-source financial tools for India!*
