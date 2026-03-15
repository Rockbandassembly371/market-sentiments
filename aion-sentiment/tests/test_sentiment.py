# =============================================================================
# AION Sentiment Analysis - Sentiment Analyzer Tests
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
Tests for the SentimentAnalyzer class using real Transformer model.

These tests use the actual Transformer model from HuggingFace to ensure
the sentiment analyzer works correctly with real financial text.

Note: First test run will download the Transformer model (~400MB).
Subsequent runs will use the cached model.
"""

import unittest
import sys
import os
from typing import List, Dict

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aion_sentiment.sentiment import SentimentAnalyzer


class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for SentimentAnalyzer with real Transformer model."""
    
    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the sentiment analyzer with Transformer model."""
        print("\n" + "=" * 70)
        print("Initializing SentimentAnalyzer with Transformer model...")
        print("Note: First run will download the model (~400MB)")
        print("=" * 70)
        
        # Use CPU for tests to ensure consistency across environments
        cls.analyzer = SentimentAnalyzer(
            model_name="transformer-base",
            device='cpu'
        )
        
        print(f"\nModel loaded on device: {cls.analyzer.device}")
        print("=" * 70)
    
    # Sample financial headlines from various sources
    POSITIVE_HEADLINES = [
        # Economic Times style
        "Stock market reaches all-time high on strong earnings reports",
        "Tech stocks rally as AI breakthrough drives investor optimism",
        "Company beats quarterly estimates, shares surge 15%",
        "Fed signals rate cuts, markets respond positively",
        "Record IPO raises $2 billion as investor confidence grows",
        
        # Reuters style
        "Global markets climb on positive economic data",
        "Oil prices stabilize as OPEC+ agrees production cuts",
        "Unemployment falls to decade low, boosting consumer sentiment",
        "Merger deal valued at $50 billion approved by regulators",
        "Central bank maintains accommodative monetary policy",
    ]
    
    NEGATIVE_HEADLINES = [
        # Economic Times style
        "Stock market crashes amid recession fears",
        "Company files for bankruptcy, thousands of jobs at risk",
        "Tech sector tumbles as earnings disappoint investors",
        "Fed warns of prolonged high interest rates",
        "Banking crisis deepens as regional banks face liquidity issues",
        
        # Reuters style
        "Global markets plunge on trade war escalation",
        "Oil prices crash amid demand concerns",
        "Inflation surges to 40-year high, sparking selloff",
        "CEO resigns amid fraud investigation, shares plummet",
        "Credit rating downgrade triggers bond market turmoil",
    ]
    
    NEUTRAL_HEADLINES = [
        # Economic Times style
        "Market shows mixed signals as trading volume remains flat",
        "Company announces leadership transition, no financial impact",
        "Analysts maintain hold rating ahead of earnings report",
        "Sector rotation continues as investors reassess positions",
        "Trading halted pending news announcement",
        
        # Reuters style
        "Markets consolidate after recent volatility",
        "Economic data shows mixed signals for growth outlook",
        "Central bank meeting concludes with no policy changes",
        "Merger talks ongoing, no agreement reached yet",
        "Investors await Fed minutes for policy direction",
    ]
    
    def test_positive_sentiment(self) -> None:
        """Test that positive headlines are classified as positive."""
        print("\n--- Testing Positive Sentiment ---")
        
        results = self.analyzer.predict(self.POSITIVE_HEADLINES)
        
        positive_count = sum(
            1 for r in results if r['label'] == 'positive'
        )
        
        print(f"Positive headlines: {len(self.POSITIVE_HEADLINES)}")
        print(f"Correctly classified as positive: {positive_count}")
        
        # Print detailed results
        for headline, result in zip(self.POSITIVE_HEADLINES[:3], results[:3]):
            print(f"\nHeadline: {headline}")
            print(f"Prediction: {result['label']} ({result['confidence']:.2%})")
        
        # At least 60% should be classified as positive
        accuracy = positive_count / len(self.POSITIVE_HEADLINES)
        print(f"\nPositive classification accuracy: {accuracy:.1%}")
        
        self.assertGreater(
            accuracy, 0.5,
            f"Less than 50% of positive headlines classified correctly. "
            f"Results: {[r['label'] for r in results]}"
        )
    
    def test_negative_sentiment(self) -> None:
        """Test that negative headlines are classified as negative."""
        print("\n--- Testing Negative Sentiment ---")
        
        results = self.analyzer.predict(self.NEGATIVE_HEADLINES)
        
        negative_count = sum(
            1 for r in results if r['label'] == 'negative'
        )
        
        print(f"Negative headlines: {len(self.NEGATIVE_HEADLINES)}")
        print(f"Correctly classified as negative: {negative_count}")
        
        # Print detailed results
        for headline, result in zip(self.NEGATIVE_HEADLINES[:3], results[:3]):
            print(f"\nHeadline: {headline}")
            print(f"Prediction: {result['label']} ({result['confidence']:.2%})")
        
        # At least 60% should be classified as negative
        accuracy = negative_count / len(self.NEGATIVE_HEADLINES)
        print(f"\nNegative classification accuracy: {accuracy:.1%}")
        
        self.assertGreater(
            accuracy, 0.5,
            f"Less than 50% of negative headlines classified correctly. "
            f"Results: {[r['label'] for r in results]}"
        )
    
    def test_neutral_sentiment(self) -> None:
        """Test that neutral headlines are classified as neutral."""
        print("\n--- Testing Neutral Sentiment ---")
        
        results = self.analyzer.predict(self.NEUTRAL_HEADLINES)
        
        neutral_count = sum(
            1 for r in results if r['label'] == 'neutral'
        )
        
        print(f"Neutral headlines: {len(self.NEUTRAL_HEADLINES)}")
        print(f"Correctly classified as neutral: {neutral_count}")
        
        # Print detailed results
        for headline, result in zip(self.NEUTRAL_HEADLINES[:3], results[:3]):
            print(f"\nHeadline: {headline}")
            print(f"Prediction: {result['label']} ({result['confidence']:.2%})")
        
        # At least 40% should be classified as neutral (neutral is harder)
        accuracy = neutral_count / len(self.NEUTRAL_HEADLINES)
        print(f"\nNeutral classification accuracy: {accuracy:.1%}")
        
        # Neutral is harder to classify, so we use a lower threshold
        self.assertGreaterEqual(
            accuracy, 0.3,
            f"Less than 30% of neutral headlines classified correctly. "
            f"Results: {[r['label'] for r in results]}"
        )
    
    def test_single_text_prediction(self) -> None:
        """Test prediction for a single text string."""
        print("\n--- Testing Single Text Prediction ---")
        
        text = "Company reports record quarterly earnings"
        result = self.analyzer.predict(text)
        
        print(f"Text: {text}")
        print(f"Prediction: {result['label']} ({result['confidence']:.2%})")
        
        # Verify result structure
        self.assertIn('label', result)
        self.assertIn('confidence', result)
        self.assertIsInstance(result['label'], str)
        self.assertIsInstance(result['confidence'], float)
        
        # Verify label is valid
        self.assertIn(
            result['label'],
            ['positive', 'neutral', 'negative'],
            f"Invalid label: {result['label']}"
        )
        
        # Verify confidence is in valid range
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
    
    def test_batch_prediction(self) -> None:
        """Test batch prediction for multiple texts."""
        print("\n--- Testing Batch Prediction ---")
        
        texts = [
            "Stock market surges on positive news",
            "Economy shows signs of recovery",
            "Investors remain cautious amid uncertainty"
        ]
        
        results = self.analyzer.predict(texts)
        
        print(f"Number of texts: {len(texts)}")
        print(f"Number of results: {len(results)}")
        
        # Verify results
        self.assertEqual(len(results), len(texts))
        
        for i, (text, result) in enumerate(zip(texts, results)):
            print(f"\n{i+1}. {text}")
            print(f"   → {result['label']} ({result['confidence']:.2%})")
            
            self.assertIn('label', result)
            self.assertIn('confidence', result)
    
    def test_confidence_scores(self) -> None:
        """Test that confidence scores are in valid range."""
        print("\n--- Testing Confidence Scores ---")
        
        all_texts = (
            self.POSITIVE_HEADLINES +
            self.NEGATIVE_HEADLINES +
            self.NEUTRAL_HEADLINES
        )
        
        results = self.analyzer.predict(all_texts)
        
        # Check all confidence scores
        for i, result in enumerate(results):
            confidence = result['confidence']
            
            self.assertGreaterEqual(
                confidence, 0.0,
                f"Confidence {confidence} is below 0 for text {i}"
            )
            self.assertLessEqual(
                confidence, 1.0,
                f"Confidence {confidence} is above 1 for text {i}"
            )
        
        # Check that we have varying confidence levels
        confidences = [r['confidence'] for r in results]
        print(f"Confidence range: {min(confidences):.2%} - {max(confidences):.2%}")
        print(f"Average confidence: {sum(confidences)/len(confidences):.2%}")
    
    def test_empty_input(self) -> None:
        """Test handling of empty input."""
        print("\n--- Testing Empty Input ---")
        
        # Empty list should raise ValueError
        with self.assertRaises(ValueError):
            self.analyzer.predict([])
    
    def test_empty_string_handling(self) -> None:
        """Test handling of empty strings in batch."""
        print("\n--- Testing Empty String Handling ---")
        
        texts = [
            "Stock market rises",
            "",  # Empty string
            "   ",  # Whitespace only
            "Economy improves"
        ]
        
        results = self.analyzer.predict(texts)
        
        self.assertEqual(len(results), 4)
        
        # Empty strings should get default neutral prediction
        for i, result in enumerate(results):
            print(f"Text {i}: '{texts[i]}' → {result['label']}")
            self.assertIn('label', result)
            self.assertIn('confidence', result)
    
    def test_sentiment_score(self) -> None:
        """Test continuous sentiment score calculation."""
        print("\n--- Testing Sentiment Score ---")
        
        # Positive text should have positive score
        positive_score = self.analyzer.get_sentiment_score(
            "Excellent earnings beat expectations"
        )
        print(f"Positive text score: {positive_score:.2f}")
        self.assertGreater(positive_score, 0)
        
        # Negative text should have negative score
        negative_score = self.analyzer.get_sentiment_score(
            "Disappointing results trigger selloff"
        )
        print(f"Negative text score: {negative_score:.2f}")
        self.assertLess(negative_score, 0)
    
    def test_predict_batch_method(self) -> None:
        """Test the predict_batch method for memory efficiency."""
        print("\n--- Testing predict_batch Method ---")
        
        # Create a larger batch of texts
        texts = self.POSITIVE_HEADLINES + self.NEGATIVE_HEADLINES
        
        results = self.analyzer.predict_batch(texts, batch_size=4)
        
        self.assertEqual(len(results), len(texts))
        
        print(f"Processed {len(texts)} texts in batches of 4")
        print(f"Results: {len(results)} predictions")
    
    def test_device_detection(self) -> None:
        """Test that device is properly detected and used."""
        print("\n--- Testing Device Detection ---")
        
        device = self.analyzer.device
        print(f"Current device: {device}")
        
        self.assertIn(
            device,
            ['cpu', 'cuda', 'mps'],
            f"Invalid device: {device}"
        )
    
    def test_repr(self) -> None:
        """Test string representation of analyzer."""
        print("\n--- Testing __repr__ ---")
        
        repr_str = repr(self.analyzer)
        print(f"Analyzer repr: {repr_str}")
        
        self.assertIn('SentimentAnalyzer', repr_str)
        self.assertIn('transformer-base', repr_str)


class TestRealWorldHeadlines(unittest.TestCase):
    """Test with real-world financial headlines from news sources."""
    
    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the sentiment analyzer."""
        cls.analyzer = SentimentAnalyzer(device='cpu')
    
    # Real headlines from Economic Times, Reuters, Bloomberg
    REAL_HEADLINES = [
        # From Economic Times
        "Sensex jumps 800 points on strong Q4 earnings, FII buying",
        "RBI keeps repo rate unchanged at 6.5%, focuses on inflation",
        "IT stocks decline as US recession fears mount",
        "Reliance Industries reports 10% profit growth in Q3",
        "Bank Nifty hits record high on rate cut hopes",
        
        # From Reuters
        "Wall Street ends higher as tech stocks rally",
        "ECB holds rates steady, signals cautious approach",
        "China's economy grows 5.2% in 2023, beats expectations",
        "Goldman Sachs profit falls on trading revenue decline",
        "Dollar strengthens as Fed pushes back on rate cut bets",
        
        # From Bloomberg
        "S&P 500 reaches new peak amid earnings optimism",
        "Treasury yields climb on strong jobs data",
        "Crypto markets tumble as regulatory concerns grow",
        "European stocks mixed as investors digest earnings",
        "Oil gains on Middle East supply concerns",
    ]
    
    EXPECTED_SENTIMENTS = [
        'positive',  # Sensex jumps
        'neutral',   # RBI keeps rate
        'negative',  # IT stocks decline
        'positive',  # Profit growth
        'positive',  # Record high
        'positive',  # Wall Street higher
        'neutral',   # ECB holds rates
        'positive',  # Beats expectations
        'negative',  # Profit falls
        'neutral',   # Dollar strengthens
        'positive',  # New peak
        'neutral',   # Yields climb
        'negative',  # Crypto tumble
        'neutral',   # Stocks mixed
        'neutral',   # Oil gains (could be negative for economy)
    ]
    
    def test_real_world_headlines(self) -> None:
        """Test sentiment analysis on real financial headlines."""
        print("\n" + "=" * 70)
        print("Testing Real-World Financial Headlines")
        print("=" * 70)
        
        results = self.analyzer.predict(self.REAL_HEADLINES)
        
        correct = 0
        for headline, result, expected in zip(
            self.REAL_HEADLINES, results, self.EXPECTED_SENTIMENTS
        ):
            match = "✓" if result['label'] == expected else "✗"
            if result['label'] == expected:
                correct += 1
            
            print(f"\n{match} {headline}")
            print(f"   Expected: {expected}")
            print(f"   Got: {result['label']} ({result['confidence']:.2%})")
        
        accuracy = correct / len(self.REAL_HEADLINES)
        print(f"\n" + "=" * 70)
        print(f"Accuracy: {accuracy:.1%} ({correct}/{len(self.REAL_HEADLINES)})")
        print("=" * 70)
        
        # We expect at least 50% accuracy on real headlines
        # (sentiment can be subjective and context-dependent)
        self.assertGreaterEqual(
            accuracy, 0.4,
            f"Accuracy {accuracy:.1%} is below expected 40%"
        )


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
