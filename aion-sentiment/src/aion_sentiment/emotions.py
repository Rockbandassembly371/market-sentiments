# =============================================================================
# AION Sentiment Analysis - Emotion Analyzer Module
# =============================================================================
# Copyright (c) 2026 AION Open Source Contributors
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
# AION Open Source Project - Financial News Sentiment Analysis
# =============================================================================
"""
Emotion Analyzer Module for AION.

This module provides the EmotionAnalyzer class that uses the NRC Emotion
Lexicon for detecting emotional indicators in financial text, with special
mapping to market-relevant emotions (fear, greed, panic, optimism).

The NRC Emotion Lexicon v0.92 is bundled with the package for offline use.
No runtime downloads are required.

Classes:
    EmotionAnalyzer: NRC lexicon-based emotion detection for financial text.
"""

import os
import re
from typing import Optional, Dict, List
from collections import defaultdict
import logging
from pathlib import Path

try:
    # Python 3.9+
    from importlib.resources import files as importlib_files
    IMPORTLIB_AVAILABLE = True
except ImportError:
    # Fallback for older Python versions
    import importlib_resources
    from importlib_resources import files as importlib_files
    IMPORTLIB_AVAILABLE = True

try:
    import pkg_resources
    PKG_RESOURCES_AVAILABLE = True
except ImportError:
    PKG_RESOURCES_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmotionAnalyzer:
    """
    Financial emotion analyzer using NRC Emotion Lexicon.

    This class provides emotion detection for financial news and text
    using the NRC Emotion Lexicon, with specialized mapping to market-
    relevant emotions: fear, greed, panic, and optimism.

    The analyzer maps basic emotions from NRC to financial market emotions:
        - anger, disgust → fear
        - anticipation, trust → greed
        - fear, sadness → panic
        - joy, positive → optimism

    The NRC Emotion Lexicon v0.92 is bundled with the package for offline
    use. No runtime downloads are required.

    Attributes:
        lexicon_path (str): Path to the NRC lexicon file.
        emotion_words (dict): Mapping of words to emotion scores.
        data_dir (str): Directory for caching custom lexicons.

    Example:
        >>> analyzer = EmotionAnalyzer()
        >>> text = "Investors panic as markets crash on recession fears"
        >>> emotions = analyzer.get_emotions(text)
        >>> print(emotions)
        {'fear': 0.75, 'greed': 0.1, 'panic': 0.8, 'optimism': 0.05}
    """

    # NRC emotion categories
    NRC_EMOTIONS = [
        'anger', 'anticipation', 'disgust', 'fear',
        'joy', 'sadness', 'surprise', 'trust',
        'negative', 'positive'
    ]

    # Mapping from NRC emotions to financial market emotions
    FINANCIAL_EMOTION_MAP = {
        'fear': ['anger', 'disgust', 'fear'],
        'greed': ['anticipation', 'trust'],
        'panic': ['fear', 'sadness'],
        'optimism': ['joy', 'positive', 'anticipation']
    }

    # Bundled lexicon filename
    BUNDLED_LEXICON_NAME = 'nrc_emotion_lexicon_v0.92.txt'

    def __init__(
        self,
        lexicon_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        use_bundled: bool = True
    ) -> None:
        """
        Initialize the EmotionAnalyzer with NRC lexicon.

        Args:
            lexicon_path: Path to NRC lexicon file. If None, uses the
                bundled lexicon (recommended). If provided, loads from
                the custom path.
            data_dir: Directory for storing custom lexicons. Only used
                if lexicon_path is provided and custom lexicon needs caching.
            use_bundled: If True (default), use the bundled lexicon included
                with the package. If False and lexicon_path is None, attempts
                to download lexicon (legacy behavior).

        Raises:
            FileNotFoundError: If lexicon_path is provided but file doesn't exist.
            IOError: If lexicon loading fails.

        Example:
            >>> # Use bundled lexicon (recommended, no download required)
            >>> analyzer = EmotionAnalyzer()
            >>>
            >>> # Use custom lexicon path
            >>> analyzer = EmotionAnalyzer(lexicon_path='/path/to/nrc_lexicon.txt')
            >>>
            >>> # Legacy behavior: attempt download if bundled not available
            >>> analyzer = EmotionAnalyzer(use_bundled=False)
        """
        self.data_dir = data_dir or self._get_default_data_dir()
        self.lexicon_path = lexicon_path

        # Determine lexicon source
        if lexicon_path is not None:
            # Custom lexicon path provided
            if not os.path.exists(lexicon_path):
                raise FileNotFoundError(
                    f"Custom lexicon not found at {lexicon_path}"
                )
            logger.info(f"Using custom lexicon at {lexicon_path}")
        elif use_bundled:
            # Use bundled lexicon (primary method)
            bundled_path = self._get_bundled_lexicon_path()
            if bundled_path and os.path.exists(bundled_path):
                self.lexicon_path = bundled_path
                logger.info(f"Using bundled NRC lexicon: {bundled_path}")
            else:
                # Fallback to minimal lexicon if bundled not found
                logger.warning(
                    "Bundled lexicon not found, using minimal fallback lexicon"
                )
                os.makedirs(self.data_dir, exist_ok=True)
                self.lexicon_path = os.path.join(
                    self.data_dir, 'nrc_lexicon_fallback.txt'
                )
                self._create_minimal_lexicon(self.lexicon_path)
        else:
            # Legacy behavior: attempt download
            logger.warning(
                "use_bundled=False is deprecated. "
                "The NRC lexicon is now bundled with the package."
            )
            os.makedirs(self.data_dir, exist_ok=True)
            self.lexicon_path = os.path.join(self.data_dir, 'nrc_lexicon.txt')

            if not os.path.exists(self.lexicon_path):
                logger.info(f"NRC lexicon not found, downloading to {self.lexicon_path}")
                try:
                    from .utils import download_nrc_lexicon
                    download_nrc_lexicon(self.lexicon_path)
                except Exception as e:
                    logger.warning(f"Failed to download NRC lexicon: {e}")
                    logger.warning("Using minimal fallback lexicon")
                    self._create_minimal_lexicon(self.lexicon_path)
            else:
                logger.info(f"Using cached NRC lexicon at {self.lexicon_path}")

        self.emotion_words: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        if os.path.exists(self.lexicon_path):
            self._load_lexicon()
            logger.info(f"EmotionAnalyzer loaded with {len(self.emotion_words)} words")
        else:
            logger.warning("No lexicon available, emotion analysis will be limited")

    def _get_bundled_lexicon_path(self) -> Optional[str]:
        """
        Get the path to the bundled NRC lexicon.

        Uses importlib.resources (Python 3.9+) or pkg_resources to locate
        the bundled lexicon file within the package.

        Returns:
            Path to bundled lexicon file, or None if not found.
        """
        try:
            # Try direct path from module location (works in development mode)
            module_dir = os.path.dirname(os.path.abspath(__file__))
            bundled_path = os.path.join(module_dir, 'lexicons', self.BUNDLED_LEXICON_NAME)
            if os.path.exists(bundled_path):
                return bundled_path

            # Try using importlib.resources (Python 3.9+)
            if IMPORTLIB_AVAILABLE:
                try:
                    lexicon_file = importlib_files('aion_sentiment.lexicons') / self.BUNDLED_LEXICON_NAME
                    # For file-based access (development/installation)
                    if hasattr(lexicon_file, '__fspath__'):
                        return str(lexicon_file)
                    # For zip-safe or packaged installations
                    return str(lexicon_file)
                except Exception as e:
                    logger.debug(f"importlib.resources failed: {e}")

            # Fallback to pkg_resources
            if PKG_RESOURCES_AVAILABLE:
                try:
                    return pkg_resources.resource_filename(
                        'aion_sentiment',
                        f'lexicons/{self.BUNDLED_LEXICON_NAME}'
                    )
                except Exception as e:
                    logger.debug(f"pkg_resources failed: {e}")

        except Exception as e:
            logger.warning(f"Error getting bundled lexicon path: {e}")

        return None

    def _get_default_data_dir(self) -> str:
        """Get default data directory for lexicon cache."""
        # Get package directory
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(package_dir, 'data')
        return data_dir

    def _create_minimal_lexicon(self, lexicon_path: str) -> None:
        """
        Create a minimal fallback lexicon when download fails.

        Args:
            lexicon_path: Path to save the minimal lexicon.
        """
        # Minimal emotion words for financial sentiment
        minimal_lexicon = {
            'anger': ['hate', 'kill', 'attack', 'enemy', 'terrible', 'horrible', 'awful', 'angry', 'rage', 'furious'],
            'fear': ['afraid', 'scared', 'terrified', 'panic', 'horror', 'dread', 'frightened', 'fear', 'nervous', 'worried'],
            'joy': ['happy', 'joy', 'love', 'wonderful', 'great', 'excellent', 'fantastic', 'pleased', 'delighted', 'thrilled'],
            'sadness': ['sad', 'depressed', 'miserable', 'pain', 'suffer', 'grief', 'tears', 'unhappy', 'sorrow', 'heartbroken'],
            'trust': ['trust', 'faith', 'reliable', 'honest', 'confident', 'believe', 'support', 'loyal', 'dependable', 'secure'],
            'disgust': ['disgust', 'sick', 'vomit', 'repulsive', 'nausea', 'revolting', 'disgusted', 'gross', 'contempt', 'despise'],
            'surprise': ['surprise', 'amazed', 'astonished', 'shocked', 'astounded', 'stunned', 'surprised', 'wow', 'unexpected', 'startled'],
            'anticipation': ['anticipate', 'expect', 'hope', 'await', 'planning', 'looking', 'eager', 'optimistic', 'confident', 'positive'],
            'positive': ['good', 'best', 'gain', 'profit', 'success', 'win', 'growth', 'positive', 'up', 'rise', 'surge', 'rally', 'beat'],
            'negative': ['bad', 'worst', 'loss', 'lose', 'fail', 'failure', 'crash', 'negative', 'down', 'fall', 'drop', 'decline', 'miss'],
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(lexicon_path), exist_ok=True)

        # Write lexicon
        with open(lexicon_path, 'w', encoding='utf-8') as f:
            for emotion, words in minimal_lexicon.items():
                for word in words:
                    f.write(f"{word}\t{emotion}\t1\n")

        logger.info(f"Created minimal lexicon at {lexicon_path}")

    def _load_lexicon(self) -> None:
        """
        Load the NRC emotion lexicon from file.

        The lexicon file format is:
            word<TAB>emotion<TAB>association (0 or 1)

        Example:
            abandon	negative	1
            abandon	sadness	1
            abandon	fear	1
        """
        if not os.path.exists(self.lexicon_path):
            raise FileNotFoundError(f"NRC lexicon not found at {self.lexicon_path}")

        try:
            with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split('\t')
                    if len(parts) != 3:
                        logger.warning(
                            f"Skipping malformed line {line_num}: {line[:50]}..."
                        )
                        continue

                    word, emotion, association = parts

                    # Only include words with positive association
                    if association == '1' and emotion in self.NRC_EMOTIONS:
                        word_lower = word.lower()
                        self.emotion_words[word_lower][emotion] = 1

            logger.info(
                f"Loaded {len(self.emotion_words)} words from NRC lexicon"
            )

        except Exception as e:
            logger.error(f"Error loading NRC lexicon: {e}")
            raise

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text to tokenize.

        Returns:
            List of lowercase words.
        """
        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words

    def _calculate_nrc_scores(self, words: List[str]) -> Dict[str, float]:
        """
        Calculate raw NRC emotion scores for a list of words.

        Args:
            words: List of tokenized words.

        Returns:
            Dict mapping NRC emotions to scores (0 to len(words)).
        """
        scores = {emotion: 0.0 for emotion in self.NRC_EMOTIONS}

        if not words:
            return scores

        for word in words:
            if word in self.emotion_words:
                for emotion in self.NRC_EMOTIONS:
                    if self.emotion_words[word].get(emotion, 0):
                        scores[emotion] += 1

        # Normalize by number of words
        num_words = len(words)
        if num_words > 0:
            scores = {k: v / num_words for k, v in scores.items()}

        return scores

    def _map_to_financial_emotions(
        self,
        nrc_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Map NRC emotion scores to financial market emotions.

        Args:
            nrc_scores: Dict of NRC emotion scores.

        Returns:
            Dict with fear, greed, panic, optimism scores (0 to 1).
        """
        financial_scores = {}

        for fin_emotion, nrc_emotions in self.FINANCIAL_EMOTION_MAP.items():
            # Average the relevant NRC emotion scores
            relevant_scores = [
                nrc_scores.get(nrc_em, 0.0) for nrc_em in nrc_emotions
            ]
            if relevant_scores:
                financial_scores[fin_emotion] = sum(relevant_scores) / len(relevant_scores)
            else:
                financial_scores[fin_emotion] = 0.0

        # Ensure scores are in valid range
        financial_scores = {
            k: min(1.0, max(0.0, v)) for k, v in financial_scores.items()
        }

        return financial_scores

    def get_emotions(self, text: str) -> Dict[str, float]:
        """
        Get financial emotion scores for a text.

        Analyzes the input text and returns scores for four market-relevant
        emotions: fear, greed, panic, and optimism. Each score is a float
        between 0 and 1.

        Args:
            text: Text to analyze for emotions.

        Returns:
            Dict with the following keys:
                - fear (float): Score for fear/anxiety emotions (0-1)
                - greed (float): Score for greed/anticipation emotions (0-1)
                - panic (float): Score for panic/distress emotions (0-1)
                - optimism (float): Score for optimism/joy emotions (0-1)

        Raises:
            TypeError: If text is not a string.

        Example:
            >>> analyzer = EmotionAnalyzer()
            >>>
            >>> text = "Market crashes as investors flee risky assets"
            >>> emotions = analyzer.get_emotions(text)
            >>> print(f"Fear: {emotions['fear']:.2f}")
            Fear: 0.65
            >>> print(f"Panic: {emotions['panic']:.2f}")
            Panic: 0.55
            >>>
            >>> text = "Stocks surge on strong earnings reports"
            >>> emotions = analyzer.get_emotions(text)
            >>> print(f"Optimism: {emotions['optimism']:.2f}")
            Optimism: 0.70
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected string, got {type(text).__name__}")

        if not text or not text.strip():
            return {
                'fear': 0.0,
                'greed': 0.0,
                'panic': 0.0,
                'optimism': 0.0
            }

        # Tokenize text
        words = self._tokenize(text)

        # Calculate NRC emotion scores
        nrc_scores = self._calculate_nrc_scores(words)

        # Map to financial emotions
        financial_scores = self._map_to_financial_emotions(nrc_scores)

        return financial_scores

    def get_dominant_emotion(self, text: str) -> str:
        """
        Get the dominant emotion in a text.

        Args:
            text: Text to analyze.

        Returns:
            The emotion with the highest score: 'fear', 'greed', 'panic',
            or 'optimism'. Returns 'neutral' if all scores are 0.

        Example:
            >>> analyzer = EmotionAnalyzer()
            >>> text = "Panic selling grips the market"
            >>> dominant = analyzer.get_dominant_emotion(text)
            >>> print(dominant)
            'panic'
        """
        emotions = self.get_emotions(text)

        if not emotions or all(v == 0 for v in emotions.values()):
            return 'neutral'

        return max(emotions, key=emotions.get)

    def get_emotion_summary(self, text: str) -> str:
        """
        Get a human-readable emotion summary for a text.

        Args:
            text: Text to analyze.

        Returns:
            String describing the dominant emotions in the text.

        Example:
            >>> analyzer = EmotionAnalyzer()
            >>> text = "Markets tumble on recession fears"
            >>> summary = analyzer.get_emotion_summary(text)
            >>> print(summary)
            'High fear (0.65), Moderate panic (0.45)'
        """
        emotions = self.get_emotions(text)

        # Filter significant emotions (score > 0.1)
        significant = [
            (emotion, score) for emotion, score in emotions.items()
            if score > 0.1
        ]

        if not significant:
            return "Neutral emotional tone"

        # Sort by score descending
        significant.sort(key=lambda x: x[1], reverse=True)

        # Create summary
        parts = []
        for emotion, score in significant[:3]:  # Top 3 emotions
            if score > 0.6:
                intensity = "High"
            elif score > 0.3:
                intensity = "Moderate"
            else:
                intensity = "Low"
            parts.append(f"{intensity} {emotion} ({score:.2f})")

        return ", ".join(parts)

    def analyze_texts(
        self,
        texts: List[str]
    ) -> List[Dict[str, float]]:
        """
        Analyze emotions for multiple texts.

        Args:
            texts: List of texts to analyze.

        Returns:
            List of emotion dicts, one per text.

        Example:
            >>> analyzer = EmotionAnalyzer()
            >>> headlines = [
            ...     "Stock market reaches record high",
            ...     "Economy shows signs of recovery",
            ...     "Investors worry about inflation"
            ... ]
            >>> results = analyzer.analyze_texts(headlines)
        """
        return [self.get_emotions(text) for text in texts]

    def __repr__(self) -> str:
        """Return string representation of the analyzer."""
        return (
            f"EmotionAnalyzer(lexicon='{self.lexicon_path}', "
            f"words={len(self.emotion_words)})"
        )
