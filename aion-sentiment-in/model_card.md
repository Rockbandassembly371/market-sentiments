# Model Card: AION-Sentiment-IN

**Version:** 1.0.0  
**Release Date:** March 2026  
**License:** Apache License, Version 2.0

---

## Model Details

| Attribute | Description |
|-----------|-------------|
| **Model Name** | AION-Sentiment-IN |
| **Model Type** | Fine-tuned Transformer (Sequence Classification) |
| **Base Model** | [transformer-base](https://huggingface.co/transformer-base) |
| **Fallback Model** | [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) |
| **Architecture** | BERT-based Transformer |
| **Parameters** | ~110M (Transformer base) |
| **Input** | Text (financial news headlines) |
| **Output** | Sentiment classification (negative, neutral, positive) |
| **Framework** | PyTorch, HuggingFace Transformers |
| **Developed By** | AION Open-Source Contributors |

### Model Description

AION-Sentiment-IN is a tuned sentiment analysis model specifically designed for **Indian financial news**. Built upon the Transformer architecture, this model has been adapted to understand the nuances of Indian market sentiment, including references to Indian companies, regulatory bodies, and market-specific terminology.

The model classifies financial news headlines into three sentiment categories:
- **Negative (0)**: Bearish sentiment, negative outlook, or adverse developments
- **Neutral (1)**: Balanced reporting, mixed signals, or factual statements
- **Positive (2)**: Bullish sentiment, positive outlook, or favorable developments

---

## Intended Use

### Primary Use Cases

1. **Financial News Analysis**: Automated sentiment classification of Indian financial news headlines and articles.

2. **Market Research**: Analyzing sentiment trends across sectors, companies, or time periods in Indian markets.

3. **Academic Research**: Studying the relationship between news sentiment and market movements in emerging markets.

4. **Portfolio Monitoring**: Tracking sentiment around specific stocks or sectors in the Indian equity market.

5. **Risk Assessment**: Identifying negative sentiment clusters that may indicate emerging risks.

### Out-of-Scope Uses

⚠️ **The following uses are NOT recommended:**

1. **Real-Time Trading Decisions**: This model is not designed for high-frequency or real-time trading applications. Market decisions should not be based solely on automated sentiment analysis.

2. **Investment Advice**: The model does not constitute financial advice. Users should consult qualified financial advisors before making investment decisions.

3. **Non-Financial Domains**: The model is specialized for financial text and may perform poorly on general news, social media, or non-financial content.

4. **Regulatory Compliance**: This model should not be used as the sole basis for regulatory reporting or compliance decisions.

5. **Manipulative Practices**: Using this model for market manipulation, spoofing, or other unethical practices is strictly prohibited.

---

## Training Data

### Dataset Source

The model was trained on data extracted from the `aion_analytics.news_sentiment` database, which aggregates financial news from multiple Indian and international sources covering Indian markets.

### Data Composition

| Split | Samples | Description |
|-------|---------|-------------|
| **Training** | ~50,000 | News headlines with sentiment labels |
| **Validation** | ~10,000 | Held-out validation set |
| **Test** | ~10,000 | Final evaluation set |

### Data Preprocessing

1. **Headline Cleaning**: Removal of URLs, special characters, and normalization
2. **Label Encoding**: Sentiment labels mapped to integers (negative=0, neutral=1, positive=2)
3. **Tokenization**: BERT tokenizer with max sequence length of 128 tokens
4. **Class Balance**: Stratified sampling to maintain class distribution

### Data Considerations

- **Time Period**: Data spans multiple market cycles to ensure robustness
- **Source Diversity**: Multiple news sources to reduce bias
- **Language**: English-language headlines covering Indian markets
- **Label Quality**: Sentiment labels derived from a combination of automated classification and human validation

### Ethical Considerations

- Data was collected in compliance with source terms of service
- No personally identifiable information (PII) is included
- News headlines are factual public information

---

## Evaluation Results

### Evaluation Metrics

The model was evaluated using standard classification metrics:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall classification accuracy |
| **F1 Score (Macro)** | Harmonic mean of precision and recall, averaged across classes |
| **Precision (per class)** | True positives / (True positives + False positives) |
| **Recall (per class)** | True positives / (True positives + False negatives) |

### Performance Benchmarks

> **Note**: Replace placeholders below with actual evaluation results after training.

| Metric | Training Set | Validation Set | Test Set |
|--------|--------------|----------------|----------|
| **Accuracy** | `XX.XX%` | `XX.XX%` | `XX.XX%` |
| **F1 Score (Macro)** | `0.XX` | `0.XX` | `0.XX` |
| **F1 - Negative** | `0.XX` | `0.XX` | `0.XX` |
| **F1 - Neutral** | `0.XX` | `0.XX` | `0.XX` |
| **F1 - Positive** | `0.XX` | `0.XX` | `0.XX` |

### Confusion Matrix

```
                    Predicted
                  Neg  Neu  Pos
Actual  Neg      [XX] [XX] [XX]
        Neu      [XX] [XX] [XX]
        Pos      [XX] [XX] [XX]
```

### Baseline Comparison

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| **AION-Sentiment-IN (Transformer)** | `XX.XX%` | `0.XX` |
| Baseline (DistilBERT) | `XX.XX%` | `0.XX` |
| Random Baseline | 33.33% | 0.33 |

---

## Limitations

### Technical Limitations

1. **Context Length**: Maximum input length of 128 tokens may truncate longer headlines or miss important context.

2. **Domain Specificity**: Model is optimized for financial news and may perform poorly on:
   - Social media posts
   - Earnings call transcripts
   - Analyst reports
   - Non-financial news

3. **Language Constraints**: Model is trained on English text only. Hindi or other Indian language content is not supported.

4. **Temporal Drift**: Market sentiment patterns may change over time. Periodic retraining is recommended.

5. **Entity Recognition**: Model may not distinguish between similarly named companies or understand ticker symbols without context.

### Application Limitations

1. **Not for Real-Time Trading**: Latency and accuracy constraints make this unsuitable for high-frequency trading applications.

2. **No Causal Inference**: Model identifies sentiment correlation, not causation. Sentiment does not predict price movements.

3. **Regulatory Considerations**: 
   - SEBI (Securities and Exchange Board of India) regulations may apply to automated trading systems
   - Users are responsible for compliance with applicable regulations
   - Model outputs should not be used for regulatory reporting without validation

4. **Bias Considerations**:
   - Training data may reflect historical biases in financial reporting
   - Certain sectors or companies may be over/under-represented
   - Sentiment labels may reflect annotator biases

### Performance Limitations

1. **Confidence Calibration**: Model confidence scores may not be perfectly calibrated to actual accuracy.

2. **Edge Cases**: Unusual headlines, sarcasm, or complex financial instruments may be misclassified.

3. **Adversarial Inputs**: Model may be susceptible to adversarial examples designed to fool classification.

---

## License

### Apache License 2.0

This model is released under the **Apache License, Version 2.0**.

```
Copyright 2026 AION Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### Attribution Requirements

When using this model in your projects, you **must** include the following attribution:

```
This product includes the AION-Sentiment-IN model developed by the 
AION Project (https://github.com/aion).
```

### Trademark Notice

The AION name and logo are trademarks of the AION Project. They may not be used to endorse or promote products derived from this model without specific prior written permission.

---

## Contact Information

### Project Resources

| Resource | URL |
|----------|-----|
| **GitHub Repository** | https://github.com/aion/aion-sentiment-in |
| **Issue Tracker** | https://github.com/aion/aion-sentiment-in/issues |
| **Discussions** | https://github.com/aion/aion-sentiment-in/discussions |
| **AION Website** | https://aion.io (placeholder) |

### Support Channels

- **Bug Reports**: Please file issues on GitHub with detailed reproduction steps
- **Feature Requests**: Use GitHub Discussions for feature proposals
- **General Questions**: GitHub Discussions or community forums
- **Security Issues**: Contact security@aion.io (placeholder)

### Contributing

We welcome contributions from the community! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Code style and quality standards
- Pull request process
- Testing requirements
- Documentation guidelines

---

## Citations

### Citing This Model

If you use AION-Sentiment-IN in your research, please cite:

```bibtex
@software{aion_sentiment_in_2026,
  title = {AION-Sentiment-IN: Financial Sentiment Analysis for Indian Markets},
  author = {AION Contributors},
  year = {2026},
  publisher = {AION Open-Source Project},
  url = {https://github.com/aion/aion-sentiment-in},
  license = {Apache-2.0},
}
```

### Base Model Citation

This model is built upon Transformer. Please also cite:

```bibtex
@article{yang2020finbert,
  title = {Transformer: Financial Sentiment Analysis with Pre-trained Language Models},
  author = {Yang, Yi and Uy, Mark Christopher and Huang, Allen},
  journal = {arXiv preprint arXiv:2006.08097},
  year = {2020}
}
```

### NRC Emotion Lexicon

If using the emotion analysis utilities, please cite:

```bibtex
@article{mohammad2013crowdsourcing,
  title = {Crowdsourcing a word-emotion association lexicon},
  author = {Mohammad, Saif M and Turney, Peter D},
  journal = {Computational Intelligence},
  volume = {29},
  number = {3},
  pages = {436--465},
  year = {2013},
  publisher = {Wiley Online Library}
}
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | March 2026 | Initial release |

---

## Disclaimer

**THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND.**

The AION Project and contributors make no warranties, express or implied, regarding:
- Merchantability or fitness for a particular purpose
- Accuracy or completeness of model outputs
- Suitability for any specific application

Users are solely responsible for:
- Determining the appropriateness of using this model
- Compliance with all applicable laws and regulations
- Any decisions made based on model outputs

**The model is not intended for use in high-stakes decisions without appropriate human oversight and validation.**

---

*Last Updated: March 14, 2026*

*Copyright 2026 AION Contributors*
