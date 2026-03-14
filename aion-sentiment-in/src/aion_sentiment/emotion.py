# =============================================================================
# AION Open-Source Project: Sentiment Analysis Pipeline
# File: src/aion_sentiment/emotion.py
# Description: EmotionAnalyzer class for fine-grained emotion detection
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

This module provides the EmotionAnalyzer class for fine-grained emotion
detection using the NRC Emotion Lexicon. It supports eight primary emotion
categories based on psychological research.

Key Features:
    - NRC Emotion Lexicon integration with automatic download
    - Eight emotion categories: anger, fear, joy, sadness, trust,
      disgust, surprise, anticipation
    - Graceful fallback when lexicon is unavailable
    - Text preprocessing for optimal emotion detection
    - Batch processing support

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
    >>> from aion_sentiment import EmotionAnalyzer
    >>> 
    >>> analyzer = EmotionAnalyzer()
    >>> result = analyzer.analyze("Stock market crashes dramatically")
    >>> print(result.emotions['fear'])
    0.8
    >>> print(result.dominant_emotion)
    'fear'
    >>> 
    >>> # Batch analysis
    >>> texts = ["Market soars", "Economy crashes"]
    >>> results = analyzer.analyze_batch(texts)

Reference:
    Mohammad, S. M., & Turney, P. D. (2013). Crowdsourcing a word-emotion
    association lexicon. Computational Intelligence, 29(3), 436-465.

For more information, visit:
    https://github.com/aion/aion-sentiment-in

"""

from __future__ import annotations

import logging
import os
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# NRC Emotion Lexicon download URLs (multiple sources for redundancy)
NRC_LEXICON_URLS: List[str] = [
    "https://raw.githubusercontent.com/felipebravom/NRC-Emotion-Lexicon/main/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    "https://saifmohammad.com/WebDocs/NRC-Emotion-Lexicon-v0.92.txt",
]

# Local storage path for the lexicon
LEXICON_DIR: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "lexicons"
)
LEXICON_FILE: str = os.path.join(LEXICON_DIR, "NRC-Emotion-Lexicon.txt")

# Supported emotion categories
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

# Default emotion scores (used when lexicon is unavailable)
DEFAULT_EMOTION_SCORES: Dict[str, float] = {emotion: 0.0 for emotion in EMOTION_CATEGORIES}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EmotionResult:
    """
    Result container for emotion analysis.

    Attributes:
        text (str): Original input text.
        emotions (Dict[str, float]): Emotion scores for all categories.
        dominant_emotion (Optional[str]): Emotion with highest score.
        dominant_score (float): Score of the dominant emotion.
        word_count (int): Number of words analyzed.
        matched_words (int): Number of words matched in lexicon.

    Example:
        >>> result = EmotionResult(
        ...     text="Stock market crashes",
        ...     emotions={'fear': 0.8, 'sadness': 0.6, ...},
        ...     dominant_emotion='fear',
        ...     dominant_score=0.8,
        ...     word_count=3,
        ...     matched_words=2
        ... )
        >>> print(result)
        EmotionResult(text='Stock market crashes', dominant_emotion='fear')

    """
    text: str
    emotions: Dict[str, float] = field(default_factory=lambda: DEFAULT_EMOTION_SCORES.copy())
    dominant_emotion: Optional[str] = None
    dominant_score: float = 0.0
    word_count: int = 0
    matched_words: int = 0

    def __post_init__(self) -> None:
        """Compute dominant emotion after initialization."""
        if self.emotions:
            dominant = max(self.emotions.items(), key=lambda x: x[1])
            if dominant[1] > 0:
                self.dominant_emotion = dominant[0]
                self.dominant_score = dominant[1]

    def __repr__(self) -> str:
        """Return string representation."""
        return f"EmotionResult(text='{self.text[:50]}...', dominant_emotion='{self.dominant_emotion}')"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary.

        Returns:
            Dictionary representation of the emotion result.

        """
        return {
            "text": self.text,
            "emotions": self.emotions,
            "dominant_emotion": self.dominant_emotion,
            "dominant_score": self.dominant_score,
            "word_count": self.word_count,
            "matched_words": self.matched_words,
        }


# =============================================================================
# LEXICON MANAGEMENT
# =============================================================================

def ensure_lexicon_directory() -> None:
    """
    Ensure the lexicon directory exists.

    Creates the directory structure if it doesn't exist.

    """
    os.makedirs(LEXICON_DIR, exist_ok=True)


def download_lexicon(url: str, timeout: int = 30) -> bool:
    """
    Download the NRC Emotion Lexicon from a specified URL.

    Args:
        url: URL to download the lexicon from.
        timeout: Request timeout in seconds.

    Returns:
        True if download successful, False otherwise.

    Example:
        >>> success = download_lexicon(NRC_LEXICON_URLS[0])
        >>> if success:
        ...     print("Lexicon downloaded successfully")

    """
    try:
        logger.info(f"Attempting to download lexicon from: {url}")
        ensure_lexicon_directory()

        # Download with timeout
        urllib.request.urlretrieve(url, LEXICON_FILE)

        # Verify file was created and has content
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

    Example:
        >>> lexicon = load_lexicon()
        >>> if lexicon:
        ...     print(f"Loaded {len(lexicon)} words")

    """
    if not os.path.exists(LEXICON_FILE):
        logger.debug(f"Lexicon file not found: {LEXICON_FILE}")
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

    Attempts to load the lexicon from local storage. If not found,
    attempts to download from multiple sources.

    Returns:
        Loaded lexicon dictionary, or None if all attempts fail.

    Example:
        >>> lexicon = initialize_lexicon()
        >>> if lexicon is None:
        ...     print("Lexicon unavailable - using default scores")

    """
    # Try to load existing lexicon
    lexicon = load_lexicon()
    if lexicon is not None:
        return lexicon

    # Attempt to download from multiple sources
    logger.info("Lexicon not found locally. Attempting to download...")

    for url in NRC_LEXICON_URLS:
        if download_lexicon(url):
            lexicon = load_lexicon()
            if lexicon is not None:
                return lexicon

    # All download attempts failed
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

    Performs the following operations:
    - Convert to lowercase
    - Remove URLs
    - Remove special characters (keep alphanumeric and spaces)
    - Remove extra whitespace

    Args:
        text: Raw input text.

    Returns:
        Preprocessed text suitable for emotion analysis.

    Example:
        >>> preprocess_text("AAPL Stock SOARS! https://example.com")
        'aapl stock soars'

    """
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove special characters (keep letters, numbers, spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize preprocessed text into words.

    Args:
        text: Preprocessed text string.

    Returns:
        List of word tokens.

    Example:
        >>> tokenize_text("aapl stock soars")
        ['aapl', 'stock', 'soars']

    """
    return text.split()


# =============================================================================
# EMOTION ANALYZER CLASS
# =============================================================================

class EmotionAnalyzer:
    """
    Emotion analyzer using the NRC Emotion Lexicon.

    This class provides fine-grained emotion detection for text, supporting
    eight primary emotion categories. It automatically handles lexicon
    loading and provides graceful fallback when unavailable.

    Attributes:
        lexicon (Optional[Dict[str, Dict[str, int]]]): Loaded emotion lexicon.
        lexicon_available (bool): Whether lexicon is available for analysis.

    Example:
        >>> analyzer = EmotionAnalyzer()
        >>> result = analyzer.analyze("Market crashes amid uncertainty")
        >>> print(result.dominant_emotion)
        'fear'
        >>> print(result.emotions['fear'])
        0.8
        >>> 
        >>> # Get dictionary representation
        >>> result_dict = result.to_dict()

    Raises:
        Warning: If lexicon is unavailable, analysis uses default scores.

    """

    def __init__(self, lexicon: Optional[Dict[str, Dict[str, int]]] = None) -> None:
        """
        Initialize the EmotionAnalyzer.

        Args:
            lexicon: Optional pre-loaded lexicon dictionary. If None,
                attempts to initialize automatically from local storage
                or download.

        Example:
            >>> # Auto-initialize lexicon
            >>> analyzer = EmotionAnalyzer()
            >>> 
            >>> # Use pre-loaded lexicon
            >>> lexicon = load_lexicon()
            >>> analyzer = EmotionAnalyzer(lexicon=lexicon)

        """
        self.lexicon = lexicon
        self.lexicon_available = False

        # Initialize lexicon if not provided
        if self.lexicon is None:
            self.lexicon = initialize_lexicon()

        self.lexicon_available = self.lexicon is not None

        if not self.lexicon_available:
            logger.warning(
                "EmotionAnalyzer initialized without lexicon. "
                "Emotion scores will be zero. "
                "Install lexicon for full functionality."
            )

    def analyze(self, text: str) -> EmotionResult:
        """
        Analyze emotions in the input text.

        Processes the text and returns emotion scores for all eight
        categories, along with the dominant emotion.

        Args:
            text: Input text string to analyze for emotions.

        Returns:
            EmotionResult object containing:
                - text: Original input text
                - emotions: Dictionary of emotion scores
                - dominant_emotion: Emotion with highest score
                - dominant_score: Score of dominant emotion
                - word_count: Number of words analyzed
                - matched_words: Number of words matched in lexicon

        Raises:
            ValueError: If input text is empty or invalid.

        Example:
            >>> analyzer = EmotionAnalyzer()
            >>> result = analyzer.analyze("Stock market crashes dramatically")
            >>> print(f"Dominant: {result.dominant_emotion}")
            >>> print(f"Fear score: {result.emotions['fear']:.2f}")
            >>> print(f"Joy score: {result.emotions['joy']:.2f}")

        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        # Preprocess and tokenize
        processed_text = preprocess_text(text)
        tokens = tokenize_text(processed_text)

        word_count = len(tokens)
        matched_words = 0

        # Initialize scores
        emotion_scores: Dict[str, float] = {emotion: 0.0 for emotion in EMOTION_CATEGORIES}

        # If lexicon unavailable, return default result
        if not self.lexicon_available or self.lexicon is None:
            return EmotionResult(
                text=text,
                emotions=DEFAULT_EMOTION_SCORES.copy(),
                word_count=word_count,
                matched_words=0,
            )

        # Aggregate emotion scores
        for token in tokens:
            if token in self.lexicon:
                matched_words += 1
                for emotion in EMOTION_CATEGORIES:
                    emotion_scores[emotion] += self.lexicon[token].get(emotion, 0)

        # Normalize scores to [0, 1] range
        if matched_words > 0:
            max_possible = matched_words
            for emotion in EMOTION_CATEGORIES:
                emotion_scores[emotion] = min(
                    1.0, emotion_scores[emotion] / max_possible
                )

        logger.debug(
            f"Analyzed {word_count} tokens, matched {matched_words} words"
        )

        return EmotionResult(
            text=text,
            emotions=emotion_scores,
            word_count=word_count,
            matched_words=matched_words,
        )

    def analyze_batch(self, texts: List[str]) -> List[EmotionResult]:
        """
        Analyze emotions for a batch of texts.

        Args:
            texts: List of input texts to analyze.

        Returns:
            List of EmotionResult objects, one per input text.

        Raises:
            ValueError: If texts list is empty or contains invalid entries.

        Example:
            >>> analyzer = EmotionAnalyzer()
            >>> texts = ["Stock rises", "Market crashes"]
            >>> results = analyzer.analyze_batch(texts)
            >>> for result in results:
            ...     print(f"{result.text}: {result.dominant_emotion}")

        """
        if not texts or not isinstance(texts, list):
            raise ValueError("Input must be a non-empty list of strings")

        results: List[EmotionResult] = []
        for text in texts:
            try:
                result = self.analyze(text)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to analyze text: {text[:50]}... Error: {e}")
                results.append(EmotionResult(
                    text=text,
                    emotions=DEFAULT_EMOTION_SCORES.copy(),
                ))

        return results

    def get_emotion_summary(self, result: EmotionResult) -> str:
        """
        Generate a human-readable summary of emotion analysis.

        Args:
            result: EmotionResult object from analyze() method.

        Returns:
            Formatted string summarizing dominant emotions.

        Example:
            >>> analyzer = EmotionAnalyzer()
            >>> result = analyzer.analyze("Market crashes dramatically")
            >>> summary = analyzer.get_emotion_summary(result)
            >>> print(summary)
            "Dominant emotions: fear (0.80), sadness (0.60)"

        """
        # Sort emotions by score
        sorted_emotions = sorted(
            result.emotions.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Get dominant emotions (score > 0)
        dominant = [(e, s) for e, s in sorted_emotions if s > 0]

        if not dominant:
            return "No dominant emotions detected"

        # Format summary
        emotion_strs = [f"{emotion} ({score:.2f})" for emotion, score in dominant[:3]]
        return f"Dominant emotions: {', '.join(emotion_strs)}"

    def check_lexicon_status(self) -> Dict[str, Any]:
        """
        Check the status of the emotion lexicon.

        Returns:
            Dictionary with lexicon status information:
                - exists: Whether lexicon file exists
                - word_count: Number of words in lexicon
                - available: Whether lexicon is loaded and ready
                - download_url: URL for manual download

        Example:
            >>> analyzer = EmotionAnalyzer()
            >>> status = analyzer.check_lexicon_status()
            >>> if not status['available']:
            ...     print(f"Download from: {status['download_url']}")

        """
        status: Dict[str, Any] = {
            "exists": os.path.exists(LEXICON_FILE),
            "word_count": 0,
            "available": self.lexicon_available,
            "file_path": LEXICON_FILE,
            "download_url": NRC_LEXICON_URLS[0],
            "manual_download_page": "https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm",
        }

        if status["exists"] and self.lexicon is not None:
            status["word_count"] = len(self.lexicon)

        return status

    def __repr__(self) -> str:
        """Return string representation."""
        return f"EmotionAnalyzer(lexicon_available={self.lexicon_available})"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def map_emotions(text: str) -> Dict[str, float]:
    """
    Convenience function to map emotions in text.

    Creates a temporary EmotionAnalyzer and analyzes the text.
    For repeated use, create an EmotionAnalyzer instance instead.

    Args:
        text: Input text to analyze.

    Returns:
        Dictionary mapping emotion categories to scores.

    Example:
        >>> emotions = map_emotions("Stock market crashes")
        >>> print(emotions['fear'])
        0.5

    """
    analyzer = EmotionAnalyzer()
    result = analyzer.analyze(text)
    return result.emotions


def batch_map_emotions(texts: List[str]) -> List[Dict[str, float]]:
    """
    Convenience function to map emotions for multiple texts.

    Args:
        texts: List of input texts.

    Returns:
        List of emotion score dictionaries.

    Example:
        >>> texts = ["Stock rises", "Market falls"]
        >>> emotions_list = batch_map_emotions(texts)

    """
    analyzer = EmotionAnalyzer()
    results = analyzer.analyze_batch(texts)
    return [r.emotions for r in results]


# =============================================================================
# MAIN ENTRY POINT (FOR TESTING)
# =============================================================================

def main() -> None:
    """
    Main entry point for testing emotion utilities.

    Demonstrates the emotion mapping functionality with example texts.

    """
    print("=" * 60)
    print("AION EMOTION ANALYSIS - DEMO")
    print("=" * 60)

    # Initialize analyzer
    analyzer = EmotionAnalyzer()

    # Check lexicon status
    status = analyzer.check_lexicon_status()
    print(f"\nLexicon Status:")
    print(f"  Available: {status['available']}")
    print(f"  Word Count: {status['word_count']}")

    if not status['available']:
        print(f"\n  To download manually, visit:")
        print(f"  {status['manual_download_page']}")

    # Example texts
    example_texts = [
        "AAPL stock soars on better-than-expected earnings",
        "Market crashes amid economic uncertainty",
        "Investors remain neutral on Fed decision",
        "Tech sector shows strong growth anticipation",
    ]

    print("\n" + "=" * 60)
    print("EMOTION ANALYSIS EXAMPLES")
    print("=" * 60)

    for text in example_texts:
        print(f"\nText: {text}")
        result = analyzer.analyze(text)
        summary = analyzer.get_emotion_summary(result)
        print(f"  {summary}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
