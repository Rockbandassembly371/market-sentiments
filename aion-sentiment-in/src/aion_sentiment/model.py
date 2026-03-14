# =============================================================================
# AION Open-Source Project: Sentiment Analysis Pipeline
# File: src/aion_sentiment/model.py
# Description: AIONSentimentIN model class for sentiment prediction
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
AION Sentiment Analysis Model.

This module provides the AIONSentimentIN class for sentiment prediction
of financial news text. The model is hosted on HuggingFace and can be
loaded automatically with support for local path override.

Key Features:
    - Automatic model loading from HuggingFace Hub
    - Local path override for development and offline use
    - Batch prediction support
    - Combined sentiment and emotion score output
    - Graceful error handling with informative messages

Model Architecture:
    The AIONSentimentIN model is a fine-tuned transformer-based classifier
    trained on financial news data. It outputs three sentiment classes:
    negative, neutral, and positive, along with confidence scores.

Example:
    >>> from aion_sentiment import AIONSentimentIN
    >>> 
    >>> # Load from HuggingFace (default)
    >>> model = AIONSentimentIN()
    >>> result = model.predict("AAPL stock surges on earnings beat")
    >>> print(result['sentiment_label'])
    'positive'
    >>> print(result['confidence'])
    0.95
    >>> print(result['emotion_scores'])
    {'joy': 0.8, 'anticipation': 0.6, ...}
    >>> 
    >>> # Load from local path
    >>> model_local = AIONSentimentIN(local_path="./models/aion-sentiment-in-v1")
    >>> results = model_local.predict_batch(["Stock rises", "Market falls"])

For more information, visit:
    https://github.com/aion/aion-sentiment-in

"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import optional dependencies
try:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizer,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "transformers library not available. Install with: pip install transformers"
    )

try:
    from aion_sentiment.emotion import EmotionAnalyzer
    EMOTION_AVAILABLE = True
except ImportError:
    EMOTION_AVAILABLE = False
    logger.warning(
        "EmotionAnalyzer not available. Emotion scores will not be computed."
    )


class AIONSentimentIN:
    """
    AION Sentiment Analysis Model for financial news.

    This class provides a high-level interface for sentiment prediction
    using the AION fine-tuned transformer model. Models are loaded from
    HuggingFace Hub by default, with support for local path override.

    The model predicts three sentiment classes (negative, neutral, positive)
    and optionally computes emotion scores using the NRC Emotion Lexicon.

    Attributes:
        model_name_or_path (str): HuggingFace model identifier or local path.
        local_path (Optional[str]): Override path for local model files.
        model (Optional[PreTrainedModel]): Loaded transformer model.
        tokenizer (Optional[PreTrainedTokenizer]): Tokenizer for the model.
        emotion_analyzer (Optional[EmotionAnalyzer]): Emotion analysis module.
        device (str): Device for model inference ('cpu' or 'cuda').

    Example:
        >>> model = AIONSentimentIN()
        >>> result = model.predict("Stock market reaches new highs")
        >>> print(f"Sentiment: {result['sentiment_label']}")
        >>> print(f"Confidence: {result['confidence']:.2%}")

    Raises:
        ImportError: If transformers library is not installed.
        RuntimeError: If model fails to load from both HuggingFace and local path.

    """

    # Label mapping for sentiment classes
    LABEL_MAP: Dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}

    def __init__(
        self,
        model_name_or_path: str = "aion-analytics/aion-sentiment-in-v1",
        local_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the AION Sentiment Analysis model.

        Attempts to load the model from HuggingFace Hub by default.
        If a local_path is provided, attempts to load from local storage first,
        with HuggingFace as fallback.

        Args:
            model_name_or_path: HuggingFace model identifier (e.g.,
                "aion-analytics/aion-sentiment-in-v1") or path to local model.
                Defaults to the official AION model on HuggingFace.
            local_path: Optional local path to model files. If provided,
                attempts to load from this path before trying HuggingFace.
                Useful for development and offline usage.

        Raises:
            ImportError: If transformers library is not installed.
            RuntimeError: If model cannot be loaded from any source.

        Example:
            >>> # Load from HuggingFace
            >>> model = AIONSentimentIN()
            >>> 
            >>> # Load from local path with fallback
            >>> model = AIONSentimentIN(
            ...     model_name_or_path="aion-analytics/aion-sentiment-in-v1",
            ...     local_path="./models/aion-sentiment-in-v1"
            ... )
            >>> 
            >>> # Load from local path only
            >>> model = AIONSentimentIN(local_path="./models/my-custom-model")

        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "The 'transformers' library is required to use AIONSentimentIN. "
                "Install it with: pip install transformers torch"
            )

        self.model_name_or_path = model_name_or_path
        self.local_path = local_path
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.emotion_analyzer: Optional[EmotionAnalyzer] = None
        self.device = self._get_device()
        self._is_loaded = False

        # Initialize emotion analyzer if available
        if EMOTION_AVAILABLE:
            try:
                self.emotion_analyzer = EmotionAnalyzer()
            except Exception as e:
                logger.warning(f"Failed to initialize EmotionAnalyzer: {e}")

        # Load model and tokenizer
        self._load_model()

    def _get_device(self) -> str:
        """
        Determine the best available device for inference.

        Returns:
            Device string: 'cuda' if GPU available, 'cpu' otherwise.

        """
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("Using CPU for inference")
        return device

    def _load_model(self) -> None:
        """
        Load the model and tokenizer from HuggingFace or local path.

        Attempts to load in the following order:
        1. Local path if provided and exists
        2. HuggingFace Hub using model_name_or_path

        Raises:
            RuntimeError: If model cannot be loaded from any source.

        """
        model_loaded = False
        tokenizer_loaded = False

        # Try local path first if provided
        if self.local_path is not None:
            local_path_obj = Path(self.local_path)
            if local_path_obj.exists():
                logger.info(f"Attempting to load model from local path: {self.local_path}")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.local_path)
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        self.local_path
                    )
                    model_loaded = True
                    tokenizer_loaded = True
                    logger.info("Successfully loaded model from local path")
                except Exception as e:
                    logger.warning(f"Failed to load from local path: {e}")

        # Try HuggingFace Hub
        if not model_loaded or not tokenizer_loaded:
            logger.info(f"Attempting to load model from HuggingFace: {self.model_name_or_path}")
            try:
                if not tokenizer_loaded:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name_or_path,
                        trust_remote_code=True,
                    )
                if not model_loaded:
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_name_or_path,
                        trust_remote_code=True,
                    )
                logger.info("Successfully loaded model from HuggingFace")
            except Exception as e:
                logger.error(f"Failed to load from HuggingFace: {e}")

        # Check if loading was successful
        if self.model is None or self.tokenizer is None:
            error_msg = (
                f"Failed to load model from all sources.\n"
                f"  HuggingFace: {self.model_name_or_path}\n"
                f"  Local path: {self.local_path}\n\n"
                f"Possible solutions:\n"
                f"  1. Check your internet connection for HuggingFace access\n"
                f"  2. Verify the model path exists: {self.local_path}\n"
                f"  3. Install required dependencies: pip install transformers torch\n"
                f"  4. Visit https://huggingface.co/{self.model_name_or_path} to verify model exists"
            )
            raise RuntimeError(error_msg)

        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True
        logger.info(f"Model loaded successfully on device: {self.device}")

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single text input.

        Analyzes the input text and returns sentiment classification along
        with confidence scores and emotion analysis.

        Args:
            text: Input text string to analyze for sentiment.

        Returns:
            Dictionary containing:
                - sentiment_label (str): Predicted sentiment class
                  ('negative', 'neutral', or 'positive')
                - confidence (float): Confidence score for the prediction (0.0-1.0)
                - emotion_scores (Dict[str, float]): Emotion scores for eight
                  categories (anger, fear, joy, sadness, trust, disgust,
                  surprise, anticipation). Empty dict if emotion analyzer unavailable.
                - all_scores (Dict[str, float]): Raw scores for all sentiment classes

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If input text is empty or invalid.

        Example:
            >>> model = AIONSentimentIN()
            >>> result = model.predict("AAPL stock soars on earnings beat")
            >>> print(result['sentiment_label'])
            'positive'
            >>> print(f"Confidence: {result['confidence']:.2%}")
            Confidence: 95.23%
            >>> print(result['emotion_scores'].get('joy', 0))
            0.8

        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Move inputs to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        # Extract predictions
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        sentiment_label = self.LABEL_MAP[predicted_class]
        confidence = float(probabilities[0][predicted_class].item())

        # Get all scores
        all_scores = {
            self.LABEL_MAP[i]: float(probabilities[0][i].item())
            for i in range(len(self.LABEL_MAP))
        }

        # Get emotion scores if available
        emotion_scores: Dict[str, float] = {}
        if self.emotion_analyzer is not None:
            try:
                emotion_result = self.emotion_analyzer.analyze(text)
                emotion_scores = emotion_result.emotions if hasattr(emotion_result, 'emotions') else emotion_result
            except Exception as e:
                logger.warning(f"Emotion analysis failed: {e}")

        return {
            "sentiment_label": sentiment_label,
            "confidence": confidence,
            "emotion_scores": emotion_scores,
            "all_scores": all_scores,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for a batch of text inputs.

        Processes multiple texts efficiently and returns a list of
        prediction results.

        Args:
            texts: List of input text strings to analyze.

        Returns:
            List of dictionaries, each containing the same structure as
            the single predict() output:
                - sentiment_label (str)
                - confidence (float)
                - emotion_scores (Dict[str, float])
                - all_scores (Dict[str, float])

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If texts list is empty or contains invalid entries.

        Example:
            >>> model = AIONSentimentIN()
            >>> texts = [
            ...     "Stock market reaches new highs",
            ...     "Economy shows signs of recession",
            ...     "Fed announces interest rate decision"
            ... ]
            >>> results = model.predict_batch(texts)
            >>> for text, result in zip(texts, results):
            ...     print(f"{text}: {result['sentiment_label']}")

        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        if not texts or not isinstance(texts, list):
            raise ValueError("Input must be a non-empty list of strings")

        # Filter out empty strings
        valid_texts = [t for t in texts if t and isinstance(t, str)]
        if not valid_texts:
            raise ValueError("All input texts must be non-empty strings")

        results: List[Dict[str, Any]] = []

        # Process each text individually (can be optimized for batch)
        for text in valid_texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to predict for text: {text[:50]}... Error: {e}")
                # Return default result for failed predictions
                results.append({
                    "sentiment_label": "neutral",
                    "confidence": 0.0,
                    "emotion_scores": {},
                    "all_scores": {"negative": 0.0, "neutral": 1.0, "positive": 0.0},
                })

        return results

    def __repr__(self) -> str:
        """
        Return string representation of the model.

        Returns:
            String containing model configuration information.

        """
        return (
            f"AIONSentimentIN("
            f"model_name_or_path='{self.model_name_or_path}', "
            f"local_path='{self.local_path}', "
            f"device='{self.device}', "
            f"loaded={self._is_loaded})"
        )
