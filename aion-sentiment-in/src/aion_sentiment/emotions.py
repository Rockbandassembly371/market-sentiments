#!/usr/bin/env python3
# =============================================================================
# AION Open-Source Project: Emotion Analysis Utilities
# File: src/aion_sentiment/emotions.py
# Description: Emotion mapping module for AION Sentiment
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
AION Emotion Analysis Module.

This module provides emotion mapping functionality using the NRC Emotion
Lexicon. It enables fine-grained emotion analysis beyond basic sentiment
classification.

Key Functions:
    map_emotions: Map emotions in text using NRC lexicon
    predict_with_emotions: Combined sentiment and emotion prediction
    initialize_lexicon: Initialize emotion lexicon with download support

Emotion Categories:
    - anger: Feelings of hostility, irritation, or rage
    - fear: Feelings of anxiety, worry, or apprehension
    - joy: Feelings of happiness, delight, or pleasure
    - sadness: Feelings of sorrow, grief, or unhappiness
    - trust: Feelings of acceptance, confidence, or reliance
    - disgust: Feelings of revulsion, contempt, or aversion
    - surprise: Feelings of astonishment or unexpectedness
    - anticipation: Feelings of expectation or looking forward

Example:
    >>> from aion_sentiment.emotions import map_emotions
    >>> emotions = map_emotions("The stock market crashed dramatically")
    >>> print(f"Fear: {emotions['fear']:.2f}, Sadness: {emotions['sadness']:.2f}")

Reference:
    Mohammad, S. M., & Turney, P. D. (2013). Crowdsourcing a word-emotion
    association lexicon. Computational Intelligence, 29(3), 436-465.

"""

from __future__ import annotations

import logging
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional

import torch

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# NRC Emotion Lexicon download URLs
NRC_LEXICON_URLS: List[str] = [
    "https://raw.githubusercontent.com/felipebravom/NRC-Emotion-Lexicon/main/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    "https://saifmohammad.com/WebDocs/NRC-Emotion-Lexicon-v0.92.txt",
]

# Local storage path
LEXICON_DIR: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "lexicons"
)
LEXICON_FILE: str = os.path.join(LEXICON_DIR, "NRC-Emotion-Lexicon.txt")

# Emotion categories
EMOTION_CATEGORIES: List[str] = [
    "anger",
    "fear",
    "joy",
    "sadness",
    "trust",
    "disgust",
    "surprise",
    "anticipation",
]

# Default scores
DEFAULT_EMOTION_SCORES: Dict[str, float] = {emotion: 0.0 for emotion in EMOTION_CATEGORIES}

# Sentiment label mapping
LABEL_MAP: Dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}


# =============================================================================
# LEXICON MANAGEMENT
# =============================================================================


def ensure_lexicon_directory() -> None:
    """Ensure the lexicon directory exists."""
    os.makedirs(LEXICON_DIR, exist_ok=True)


def download_lexicon(url: str, timeout: int = 30) -> bool:
    """
    Download the NRC Emotion Lexicon from a specified URL.

    Args:
        url: URL to download the lexicon from.
        timeout: Request timeout in seconds.

    Returns:
        True if download successful, False otherwise.

    """
    try:
        logger.info(f"Attempting to download lexicon from: {url}")
        ensure_lexicon_directory()

        urllib.request.urlretrieve(url, LEXICON_FILE)

        if os.path.exists(LEXICON_FILE) and os.path.getsize(LEXICON_FILE) > 0:
            logger.info(f"Lexicon successfully downloaded to: {LEXICON_FILE}")
            return True
        else:
            logger.warning("Downloaded file is empty or missing")
            return False

    except Exception as e:
        logger.warning(f"Failed to download lexicon from {url}: {e}")
        return False


def load_lexicon() -> Optional[Dict[str, Dict[str, int]]]:
    """
    Load the NRC Emotion Lexicon from local storage.

    Returns:
        Dictionary mapping words to emotion scores, or None if unavailable.

    """
    if not os.path.exists(LEXICON_FILE):
        logger.warning(f"Lexicon file not found: {LEXICON_FILE}")
        return None

    lexicon: Dict[str, Dict[str, int]] = {}

    try:
        with open(LEXICON_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) != 3:
                    continue

                word, emotion, value = parts
                value = int(value)

                if word not in lexicon:
                    lexicon[word] = {emotion: 0 for emotion in EMOTION_CATEGORIES}

                if emotion.lower() in EMOTION_CATEGORIES:
                    lexicon[word][emotion.lower()] = value

        logger.info(f"Loaded lexicon with {len(lexicon)} words")
        return lexicon

    except Exception as e:
        logger.error(f"Error loading lexicon: {e}")
        return None


def initialize_lexicon() -> Optional[Dict[str, Dict[str, int]]]:
    """
    Initialize the emotion lexicon with automatic download fallback.

    Returns:
        Loaded lexicon dictionary, or None if all attempts fail.

    """
    lexicon = load_lexicon()
    if lexicon is not None:
        return lexicon

    logger.info("Lexicon not found locally. Attempting to download...")

    for url in NRC_LEXICON_URLS:
        if download_lexicon(url):
            lexicon = load_lexicon()
            if lexicon is not None:
                return lexicon

    logger.warning(
        "Failed to download NRC Emotion Lexicon from all sources. "
        "Emotion analysis will use default scores (all zeros). "
        "Manual download instructions:\n"
        "1. Visit: https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm\n"
        "2. Download the word-emotion association lexicon\n"
        "3. Save as: data/lexicons/NRC-Emotion-Lexicon.txt"
    )

    return None


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================


def preprocess_text(text: str) -> str:
    """
    Preprocess text for emotion analysis.

    Args:
        text: Raw input text.

    Returns:
        Preprocessed text suitable for emotion analysis.

    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize preprocessed text into words.

    Args:
        text: Preprocessed text string.

    Returns:
        List of word tokens.

    """
    return text.split()


# =============================================================================
# EMOTION MAPPING
# =============================================================================


def map_emotions(
    text: str,
    lexicon: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, float]:
    """
    Map emotions in text using the NRC Emotion Lexicon.

    Args:
        text: Input text to analyze for emotions.
        lexicon: Pre-loaded lexicon dictionary. If None, attempts to
                 initialize automatically.

    Returns:
        Dictionary mapping emotion categories to normalized scores (0.0-1.0).

    Example:
        >>> emotions = map_emotions("The stock market crashed dramatically")
        >>> print(f"Fear: {emotions['fear']:.2f}")

    """
    if lexicon is None:
        lexicon = initialize_lexicon()

    if lexicon is None:
        logger.debug("Lexicon unavailable - returning default emotion scores")
        return DEFAULT_EMOTION_SCORES.copy()

    processed_text = preprocess_text(text)
    tokens = tokenize_text(processed_text)

    if not tokens:
        return DEFAULT_EMOTION_SCORES.copy()

    emotion_scores: Dict[str, float] = {emotion: 0.0 for emotion in EMOTION_CATEGORIES}
    matched_words = 0

    for token in tokens:
        if token in lexicon:
            matched_words += 1
            for emotion in EMOTION_CATEGORIES:
                emotion_scores[emotion] += lexicon[token].get(emotion, 0)

    if matched_words > 0:
        max_possible = matched_words
        for emotion in EMOTION_CATEGORIES:
            emotion_scores[emotion] = min(1.0, emotion_scores[emotion] / max_possible)

    logger.debug(f"Analyzed {len(tokens)} tokens, matched {matched_words} words")

    return emotion_scores


# =============================================================================
# COMBINED PREDICTION
# =============================================================================


def predict_sentiment(
    text: str,
    model: Any,
    tokenizer: Any,
) -> Dict[str, Any]:
    """
    Predict sentiment for input text using a trained model.

    Args:
        text: Input text to classify.
        model: Trained sentiment classification model.
        tokenizer: Tokenizer for the model.

    Returns:
        Dictionary with sentiment label and scores.

    """
    label_map = {0: "negative", 1: "neutral", 2: "positive"}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()

    scores = {
        label_map[i]: float(probabilities[0][i].item())
        for i in range(len(label_map))
    }

    return {
        "sentiment": label_map[predicted_class],
        "scores": scores,
    }


def predict_with_emotions(
    text: str,
    model: Any,
    tokenizer: Any,
    lexicon: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    """
    Combine sentiment prediction with emotion analysis.

    Args:
        text: Input text to analyze.
        model: Trained sentiment classification model.
        tokenizer: Tokenizer for the sentiment model.
        lexicon: Pre-loaded emotion lexicon.

    Returns:
        Dictionary containing sentiment, scores, and emotions.

    Example:
        >>> result = predict_with_emotions("AAPL stock soars", model, tokenizer)
        >>> print(f"Sentiment: {result['sentiment']}")
        >>> print(f"Joy: {result['emotions']['joy']:.2f}")

    """
    sentiment_result = predict_sentiment(text, model, tokenizer)
    emotion_scores = map_emotions(text, lexicon)

    return {
        "text": text,
        "sentiment": sentiment_result["sentiment"],
        "sentiment_scores": sentiment_result["scores"],
        "emotions": emotion_scores,
    }


# =============================================================================
# BATCH PROCESSING
# =============================================================================


def batch_map_emotions(
    texts: List[str],
    lexicon: Optional[Dict[str, Dict[str, int]]] = None,
) -> List[Dict[str, float]]:
    """
    Map emotions for a batch of texts.

    Args:
        texts: List of input texts to analyze.
        lexicon: Pre-loaded emotion lexicon.

    Returns:
        List of emotion score dictionaries.

    """
    return [map_emotions(text, lexicon) for text in texts]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_emotion_summary(emotion_scores: Dict[str, float]) -> str:
    """
    Generate a human-readable summary of emotion scores.

    Args:
        emotion_scores: Dictionary of emotion scores.

    Returns:
        Formatted string summarizing dominant emotions.

    """
    sorted_emotions = sorted(
        emotion_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    dominant = [(e, s) for e, s in sorted_emotions if s > 0]

    if not dominant:
        return "No dominant emotions detected"

    emotion_strs = [f"{emotion} ({score:.2f})" for emotion, score in dominant[:3]]
    return f"Dominant emotions: {', '.join(emotion_strs)}"


def check_lexicon_status() -> Dict[str, Any]:
    """
    Check the status of the emotion lexicon.

    Returns:
        Dictionary with lexicon status information.

    """
    status: Dict[str, Any] = {
        "exists": os.path.exists(LEXICON_FILE),
        "word_count": 0,
        "file_path": LEXICON_FILE,
        "download_url": NRC_LEXICON_URLS[0],
        "manual_download_page": "https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm",
    }

    if status["exists"]:
        lexicon = load_lexicon()
        if lexicon:
            status["word_count"] = len(lexicon)

    return status


class EmotionAnalyzer:
    """
    High-level interface for emotion analysis.

    This class provides a convenient interface for analyzing emotions
    in text with optional sentiment prediction.

    Example:
        >>> analyzer = EmotionAnalyzer()
        >>> result = analyzer.analyze("Stock market crashes")
        >>> print(result.summary())

    """

    def __init__(self, lexicon: Optional[Dict[str, Dict[str, int]]] = None):
        """
        Initialize the EmotionAnalyzer.

        Args:
            lexicon: Pre-loaded lexicon. If None, will be initialized.

        """
        self.lexicon = lexicon if lexicon is not None else initialize_lexicon()

    def analyze(self, text: str) -> "EmotionResult":
        """
        Analyze emotions in text.

        Args:
            text: Input text to analyze.

        Returns:
            EmotionResult object with analysis results.

        """
        emotions = map_emotions(text, self.lexicon)
        return EmotionResult(text=text, emotions=emotions)

    def analyze_with_sentiment(
        self,
        text: str,
        model: Any,
        tokenizer: Any,
    ) -> "EmotionResult":
        """
        Analyze emotions with sentiment prediction.

        Args:
            text: Input text to analyze.
            model: Trained sentiment model.
            tokenizer: Model tokenizer.

        Returns:
            EmotionResult object with sentiment and emotion results.

        """
        result = predict_with_emotions(text, model, tokenizer, self.lexicon)
        return EmotionResult(
            text=text,
            emotions=result["emotions"],
            sentiment=result["sentiment"],
            sentiment_scores=result["sentiment_scores"],
        )


class EmotionResult:
    """
    Result object for emotion analysis.

    Attributes:
        text: Original input text.
        emotions: Dictionary of emotion scores.
        sentiment: Predicted sentiment (optional).
        sentiment_scores: Sentiment confidence scores (optional).

    """

    def __init__(
        self,
        text: str,
        emotions: Dict[str, float],
        sentiment: Optional[str] = None,
        sentiment_scores: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize EmotionResult.

        Args:
            text: Original input text.
            emotions: Emotion scores dictionary.
            sentiment: Predicted sentiment label.
            sentiment_scores: Sentiment confidence scores.

        """
        self.text = text
        self.emotions = emotions
        self.sentiment = sentiment
        self.sentiment_scores = sentiment_scores or {}

    def summary(self) -> str:
        """
        Get a human-readable summary of the analysis.

        Returns:
            Formatted summary string.

        """
        parts = [f"Text: {self.text}"]

        if self.sentiment:
            parts.append(f"Sentiment: {self.sentiment}")

        emotion_summary = get_emotion_summary(self.emotions)
        parts.append(emotion_summary)

        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary.

        Returns:
            Dictionary representation of the result.

        """
        result: Dict[str, Any] = {
            "text": self.text,
            "emotions": self.emotions,
        }

        if self.sentiment:
            result["sentiment"] = self.sentiment
            result["sentiment_scores"] = self.sentiment_scores

        return result
