# =============================================================================
# AION Open-Source Project: Sentiment Analysis Pipeline
# File: src/aion_sentiment/__init__.py
# Description: AION Sentiment Analysis package initialization
# License: Apache License, Version 2.0
#
# Copyright 2026 AION Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# AION Attribution: This file is part of the AION open-source ecosystem.
# Please visit https://github.com/aion for more information.
# =============================================================================
"""
AION Sentiment Analysis Package.

This package provides tools and utilities for sentiment analysis of financial
news within the AION open-source ecosystem. It includes the AIONSentimentIN
model for sentiment classification and EmotionAnalyzer for fine-grained
emotion detection using the NRC Emotion Lexicon.

Key Features:
    - Sentiment classification (negative/neutral/positive)
    - Eight-category emotion analysis (anger, fear, joy, sadness,
      trust, disgust, surprise, anticipation)
    - HuggingFace model integration with local fallback
    - Batch processing support
    - Type-safe API with comprehensive error handling

Quick Start:
    >>> from aion_sentiment import AIONSentimentIN, EmotionAnalyzer
    >>> 
    >>> # Initialize sentiment model
    >>> model = AIONSentimentIN()
    >>> result = model.predict("AAPL stock soars on earnings beat")
    >>> print(result)
    {'sentiment_label': 'positive', 'confidence': 0.95, 'emotion_scores': {...}}
    >>> 
    >>> # Initialize emotion analyzer
    >>> analyzer = EmotionAnalyzer()
    >>> emotions = analyzer.analyze("Market crashes amid uncertainty")
    >>> print(emotions)
    {'fear': 0.8, 'sadness': 0.6, ...}

Example:
    >>> from aion_sentiment import __version__
    >>> print(__version__)
    '0.1.0'

    >>> from aion_sentiment import AIONSentimentIN
    >>> model = AIONSentimentIN()
    >>> results = model.predict_batch(["Stock rises", "Market falls"])

For more information, visit the AION project documentation:
    https://github.com/aion/aion-sentiment-in

"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "AION Contributors"
__email__ = "contributors@aion.io"
__license__ = "Apache-2.0"
__copyright__ = "Copyright 2026 AION Contributors"

# AION Project Information
AION_PROJECT_NAME = "AION Sentiment Analysis"
AION_PROJECT_URL = "https://github.com/aion/aion-sentiment-in"
AION_ORGANIZATION = "AION Open-Source"
AION_HUGGINGFACE_ORG = "aion-analytics"
AION_DEFAULT_MODEL = "aion-analytics/aion-sentiment-in-v1"

# Import main classes for convenience
from aion_sentiment.model import AIONSentimentIN
from aion_sentiment.emotion import EmotionAnalyzer, EmotionResult

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Project info
    "AION_PROJECT_NAME",
    "AION_PROJECT_URL",
    "AION_ORGANIZATION",
    "AION_HUGGINGFACE_ORG",
    "AION_DEFAULT_MODEL",
    # Main classes
    "AIONSentimentIN",
    "EmotionAnalyzer",
    "EmotionResult",
]
