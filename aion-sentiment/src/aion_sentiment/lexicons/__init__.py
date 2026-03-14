# =============================================================================
# AION Sentiment Analysis - Lexicons Package
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
Lexicons Package for AION Sentiment Analysis.

This package contains bundled lexicon files for emotion and sentiment
analysis. The NRC Emotion Lexicon v0.92 is included for offline use.

Files:
    nrc_emotion_lexicon_v0.92.txt: NRC Emotion Lexicon Word-level v0.92
        - Contains word-emotion associations for 10 emotion categories
        - Format: word<TAB>emotion<TAB>association (0 or 1)
        - Emotions: anger, anticipation, disgust, fear, joy, sadness,
          surprise, trust, negative, positive

License:
    The NRC Emotion Lexicon is provided under its own license terms.
    Please refer to the NRC Canada website for usage restrictions.
    This bundled version is for research and non-commercial use.

Example:
    >>> from aion_sentiment import EmotionAnalyzer
    >>> analyzer = EmotionAnalyzer()  # Uses bundled lexicon
    >>> emotions = analyzer.get_emotions("Market crashes on recession fears")
    >>> print(emotions)
    {'fear': 0.5, 'greed': 0.0, 'panic': 0.4, 'optimism': 0.0}
"""

import os
from pathlib import Path

# Package directory
LEXICONS_DIR = Path(__file__).parent

# Bundled lexicon paths
NRC_EMOTION_LEXICON = LEXICONS_DIR / "nrc_emotion_lexicon_v0.92.txt"


def get_lexicon_path(lexicon_name: str) -> Path:
    """
    Get the path to a bundled lexicon file.

    Args:
        lexicon_name: Name of the lexicon file.

    Returns:
        Path to the lexicon file.

    Raises:
        FileNotFoundError: If lexicon file doesn't exist.

    Example:
        >>> path = get_lexicon_path("nrc_emotion_lexicon_v0.92.txt")
        >>> print(path.exists())
        True
    """
    lexicon_path = LEXICONS_DIR / lexicon_name
    if not lexicon_path.exists():
        raise FileNotFoundError(f"Lexicon not found: {lexicon_name}")
    return lexicon_path


def list_lexicons() -> list:
    """
    List all available lexicon files in the package.

    Returns:
        List of lexicon filenames.

    Example:
        >>> lexicons = list_lexicons()
        >>> print(lexicons)
        ['nrc_emotion_lexicon_v0.92.txt']
    """
    return [f.name for f in LEXICONS_DIR.glob("*.txt")]
