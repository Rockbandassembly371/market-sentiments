# =============================================================================
# AION Sentiment Analysis - Sentiment Analyzer Module
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
Sentiment Analyzer Module for AION.

This module provides the SentimentAnalyzer class that uses transformer models
for financial sentiment classification tuned on Indian market data.

Classes:
    SentimentAnalyzer: Transformer-based sentiment classification for financial text.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Optional, List, Dict, Union
import logging

from .utils import get_device

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Financial sentiment analyzer using transformer models tuned on Indian market data.

    This class provides sentiment classification for financial news and text
    using models from HuggingFace. It supports three sentiment
    labels: positive, neutral, and negative.

    The analyzer automatically detects the best available device (MPS for
    Apple Silicon, CUDA for NVIDIA GPUs, or CPU) and loads the model
    accordingly.

    Attributes:
        model_name (str): Name of the HuggingFace model to use.
        device (str): Device used for inference ('cuda', 'mps', or 'cpu').
        model: The loaded model.
        tokenizer: The tokenizer for the model.

    Example:
        >>> analyzer = SentimentAnalyzer()  # Uses India-tuned model by default
        >>> texts = [
        ...     "Company reports record quarterly earnings",
        ...     "Stock market crashes amid economic uncertainty"
        ... ]
        >>> results = analyzer.predict(texts)
        >>> for result in results:
        ...     print(f"{result['label']}: {result['confidence']:.2%}")
    """
    
    # Mapping from label IDs to human-readable labels
    LABEL_MAP = {
        0: 'positive',
        1: 'neutral',
        2: 'negative'
    }
    
    def __init__(
        self,
        model_name: str = "aion-analytics/aion-sentiment-in-v1",
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the SentimentAnalyzer with India-tuned model.

        Args:
            model_name: HuggingFace model name or path to local model.
                Defaults to "aion-analytics/aion-sentiment-in-v1" which is
                tuned on Indian financial news (98.55% accuracy).
                Can be overridden with any HuggingFace model.
            device: Device to run inference on. Options:
                - 'cuda': NVIDIA GPU with CUDA support
                - 'mps': Apple Silicon (M1/M2/M3)
                - 'cpu': CPU only
                If None, automatically detects best available device.

        Raises:
            ImportError: If transformers or torch is not installed.
            OSError: If model cannot be loaded from HuggingFace or local path.

        Example:
            >>> # Auto-detect device, uses India-tuned model by default
            >>> analyzer = SentimentAnalyzer()
            
            >>> # Use custom model
            >>> analyzer = SentimentAnalyzer(model_name="other-model")
            
            >>> # Use custom local model
            >>> analyzer = SentimentAnalyzer(model_name="/path/to/model")
            >>> 
            >>> # Force CPU usage
            >>> analyzer = SentimentAnalyzer(device='cpu')
            >>> 
            >>> # Use specific model
            >>> analyzer = SentimentAnalyzer(model_name="cardiffnlp/twitter-roberta-base-sentiment")
        """
        self.model_name = model_name
        self.device = get_device() if device is None else device
        
        logger.info(
            f"Initializing SentimentAnalyzer with model '{model_name}' "
            f"on device '{self.device}'"
        )
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=3  # positive, neutral, negative
            )
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise
    
    def predict(
        self,
        texts: Union[str, List[str]]
    ) -> Union[Dict, List[Dict]]:
        """
        Predict sentiment for one or more texts.

        Analyzes the input text(s) and returns sentiment predictions with
        confidence scores. The model is tuned on Indian financial news
        for accurate sentiment classification.

        Args:
            texts: Single text string or list of text strings to analyze.

        Returns:
            For single text: dict with 'label' and 'confidence' keys.
            For multiple texts: list of dicts, each with 'label' and 'confidence'.

            Label values: 'positive', 'neutral', 'negative'
            Confidence: float between 0 and 1

        Raises:
            ValueError: If texts is empty or contains invalid input.
            RuntimeError: If model inference fails.

        Example:
            >>> analyzer = SentimentAnalyzer()
            >>>
            >>> # Single text
            >>> result = analyzer.predict("Stock prices surge on earnings beat")
            >>> print(result)
            {'label': 'positive', 'confidence': 0.95}
            >>> 
            >>> # Multiple texts
            >>> results = analyzer.predict([
            ...     "Market reaches new highs",
            ...     "Economy shows mixed signals",
            ...     "Company faces regulatory challenges"
            ... ])
            >>> for r in results:
            ...     print(f"{r['label']}: {r['confidence']:.2%}")
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        if not texts:
            raise ValueError("Input texts cannot be empty")
        
        # Filter out empty strings and handle them separately
        original_texts = texts
        non_empty_indices = [
            i for i, t in enumerate(texts) if t and str(t).strip()
        ]
        non_empty_texts = [texts[i] for i in non_empty_indices]
        
        # Initialize results with defaults for empty texts
        results = [
            {'label': 'neutral', 'confidence': 0.5}
            for _ in range(len(texts))
        ]
        
        if non_empty_texts:
            try:
                # Tokenize inputs
                encodings = self.tokenizer(
                    non_empty_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                # Move inputs to device
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    # Get probabilities
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    
                    # Get predictions
                    predictions = probabilities.argmax(dim=-1)
                    confidences = probabilities.max(dim=-1).values
                
                # Convert to CPU for result extraction
                predictions = predictions.cpu().numpy()
                confidences = confidences.cpu().numpy()
                
                # Populate results for non-empty texts
                for idx, (pred, conf) in enumerate(zip(predictions, confidences)):
                    original_idx = non_empty_indices[idx]
                    results[original_idx] = {
                        'label': self.LABEL_MAP.get(int(pred), 'neutral'),
                        'confidence': float(conf)
                    }
                
            except Exception as e:
                logger.error(f"Error during sentiment prediction: {e}")
                raise RuntimeError(f"Sentiment prediction failed: {e}")
        
        # Return single result or list
        if single_input:
            return results[0]
        return results
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 8
    ) -> List[Dict]:
        """
        Predict sentiment for texts in batches (memory efficient).
        
        Processes large lists of texts in smaller batches to manage
        memory usage. Useful for processing thousands of headlines.
        
        Args:
            texts: List of text strings to analyze.
            batch_size: Number of texts to process in each batch.
                Defaults to 8.
        
        Returns:
            List of dicts, each with 'label' and 'confidence' keys.
        
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> headlines = [...]  # 1000+ headlines
            >>> results = analyzer.predict_batch(headlines, batch_size=16)
        """
        if not texts:
            return []
        
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.predict(batch)
            all_results.extend(batch_results)
            
            # Clear CUDA cache if using GPU
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        return all_results
    
    def get_sentiment_score(self, text: str) -> float:
        """
        Get a continuous sentiment score for a text.
        
        Converts the categorical sentiment prediction to a continuous
        score between -1 (very negative) and +1 (very positive).
        
        Args:
            text: Text to analyze.
        
        Returns:
            Float between -1 and +1 representing sentiment polarity.
        
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> score = analyzer.get_sentiment_score("Excellent quarterly results")
            >>> print(f"Sentiment score: {score:.2f}")
            Sentiment score: 0.85
        """
        result = self.predict(text)
        label = result['label']
        confidence = result['confidence']
        
        # Convert to continuous score
        if label == 'positive':
            return confidence
        elif label == 'negative':
            return -confidence
        else:  # neutral
            return 0.0
    
    def __repr__(self) -> str:
        """Return string representation of the analyzer."""
        return (
            f"SentimentAnalyzer(model='{self.model_name}', "
            f"device='{self.device}')"
        )
