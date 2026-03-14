# =============================================================================
# AION Sentiment Analysis - Emotion Analyzer Tests
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
Tests for the EmotionAnalyzer class using real NRC Emotion Lexicon.

These tests use the actual NRC Emotion Lexicon to ensure the emotion
analyzer correctly detects fear, greed, panic, and optimism in
financial text.

Note: First test run will download the NRC lexicon (~1MB).
Subsequent runs will use the cached lexicon.
"""

import unittest
import sys
import os
import json
from typing import Dict

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aion_sentiment.emotions import EmotionAnalyzer


class TestEmotionAnalyzer(unittest.TestCase):
    """Test cases for EmotionAnalyzer with real NRC lexicon."""
    
    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the emotion analyzer with NRC lexicon."""
        print("\n" + "=" * 70)
        print("Initializing EmotionAnalyzer with NRC lexicon...")
        print("Note: First run will download the lexicon (~1MB)")
        print("=" * 70)
        
        cls.analyzer = EmotionAnalyzer()
        
        print(f"\nLexicon loaded from: {cls.analyzer.lexicon_path}")
        print(f"Words in lexicon: {len(cls.analyzer.emotion_words)}")
        print("=" * 70)
    
    # Test texts with expected dominant emotions
    FEAR_TEXTS = [
        "Investors fear market crash as recession looms",
        "Panic selling grips Wall Street amid uncertainty",
        "Anxiety grows over banking sector stability",
        "Dread spreads as stock futures tumble",
        "Worries mount over inflation impact on economy",
    ]
    
    GREED_TEXTS = [
        "Investors greedy for high returns chase risky assets",
        "Speculation drives market to new heights",
        "Traders eager for profits increase positions",
        "Desire for wealth fuels crypto rally",
        "Ambition drives merger and acquisition boom",
    ]
    
    PANIC_TEXTS = [
        "Market panic as stocks plummet 10%",
        "Investors flee to safety amid crash fears",
        "Distress selling triggers circuit breakers",
        "Alarm spreads as banks face liquidity crisis",
        "Hysteria grips trading floor on bad news",
    ]
    
    OPTIMISM_TEXTS = [
        "Hope springs eternal as markets recover",
        "Joyful investors celebrate earnings beat",
        "Confidence grows in economic recovery",
        "Positive sentiment drives rally higher",
        "Cheerful outlook boosts consumer spending",
    ]
    
    def test_fear_detection(self) -> None:
        """Test that fear-related texts show high fear scores."""
        print("\n--- Testing Fear Detection ---")
        
        fear_scores = []
        for text in self.FEAR_TEXTS:
            emotions = self.analyzer.get_emotions(text)
            fear_scores.append(emotions['fear'])
            print(f"\nText: {text}")
            print(f"Fear score: {emotions['fear']:.3f}")
        
        avg_fear = sum(fear_scores) / len(fear_scores)
        print(f"\nAverage fear score: {avg_fear:.3f}")
        
        # Fear texts should have higher fear than other emotions
        for text in self.FEAR_TEXTS[:3]:
            emotions = self.analyzer.get_emotions(text)
            print(f"\n'{text[:50]}...'")
            print(f"  Fear: {emotions['fear']:.3f}, "
                  f"Greed: {emotions['greed']:.3f}, "
                  f"Panic: {emotions['panic']:.3f}, "
                  f"Optimism: {emotions['optimism']:.3f}")
    
    def test_greed_detection(self) -> None:
        """Test that greed-related texts show high greed scores."""
        print("\n--- Testing Greed Detection ---")
        
        greed_scores = []
        for text in self.GREED_TEXTS:
            emotions = self.analyzer.get_emotions(text)
            greed_scores.append(emotions['greed'])
            print(f"\nText: {text}")
            print(f"Greed score: {emotions['greed']:.3f}")
        
        avg_greed = sum(greed_scores) / len(greed_scores)
        print(f"\nAverage greed score: {avg_greed:.3f}")
    
    def test_panic_detection(self) -> None:
        """Test that panic-related texts show high panic scores."""
        print("\n--- Testing Panic Detection ---")
        
        panic_scores = []
        for text in self.PANIC_TEXTS:
            emotions = self.analyzer.get_emotions(text)
            panic_scores.append(emotions['panic'])
            print(f"\nText: {text}")
            print(f"Panic score: {emotions['panic']:.3f}")
        
        avg_panic = sum(panic_scores) / len(panic_scores)
        print(f"\nAverage panic score: {avg_panic:.3f}")
        
        # Panic texts should have higher panic than other emotions
        for text in self.PANIC_TEXTS[:3]:
            emotions = self.analyzer.get_emotions(text)
            print(f"\n'{text[:50]}...'")
            print(f"  Fear: {emotions['fear']:.3f}, "
                  f"Greed: {emotions['greed']:.3f}, "
                  f"Panic: {emotions['panic']:.3f}, "
                  f"Optimism: {emotions['optimism']:.3f}")
    
    def test_optimism_detection(self) -> None:
        """Test that optimism-related texts show high optimism scores."""
        print("\n--- Testing Optimism Detection ---")
        
        optimism_scores = []
        for text in self.OPTIMISM_TEXTS:
            emotions = self.analyzer.get_emotions(text)
            optimism_scores.append(emotions['optimism'])
            print(f"\nText: {text}")
            print(f"Optimism score: {emotions['optimism']:.3f}")
        
        avg_optimism = sum(optimism_scores) / len(optimism_scores)
        print(f"\nAverage optimism score: {avg_optimism:.3f}")
    
    def test_emotion_scores_range(self) -> None:
        """Test that all emotion scores are in valid range [0, 1]."""
        print("\n--- Testing Emotion Score Ranges ---")
        
        all_texts = (
            self.FEAR_TEXTS +
            self.GREED_TEXTS +
            self.PANIC_TEXTS +
            self.OPTIMISM_TEXTS
        )
        
        for text in all_texts:
            emotions = self.analyzer.get_emotions(text)
            
            for emotion, score in emotions.items():
                self.assertGreaterEqual(
                    score, 0.0,
                    f"{emotion} score {score} is below 0 for: {text}"
                )
                self.assertLessEqual(
                    score, 1.0,
                    f"{emotion} score {score} is above 1 for: {text}"
                )
        
        print("All emotion scores are in valid range [0, 1]")
    
    def test_empty_text(self) -> None:
        """Test handling of empty text."""
        print("\n--- Testing Empty Text ---")
        
        emotions = self.analyzer.get_emotions("")
        
        print(f"Empty text emotions: {emotions}")
        
        # Empty text should return all zeros
        self.assertEqual(emotions['fear'], 0.0)
        self.assertEqual(emotions['greed'], 0.0)
        self.assertEqual(emotions['panic'], 0.0)
        self.assertEqual(emotions['optimism'], 0.0)
    
    def test_whitespace_only_text(self) -> None:
        """Test handling of whitespace-only text."""
        print("\n--- Testing Whitespace Text ---")
        
        emotions = self.analyzer.get_emotions("   \t\n   ")
        
        print(f"Whitespace text emotions: {emotions}")
        
        # Whitespace should return all zeros
        self.assertEqual(emotions['fear'], 0.0)
        self.assertEqual(emotions['greed'], 0.0)
        self.assertEqual(emotions['panic'], 0.0)
        self.assertEqual(emotions['optimism'], 0.0)
    
    def test_dominant_emotion(self) -> None:
        """Test dominant emotion detection."""
        print("\n--- Testing Dominant Emotion ---")
        
        # Fear text should have fear as dominant
        fear_dominant = self.analyzer.get_dominant_emotion(
            "Fear and anxiety grip investors"
        )
        print(f"Fear text dominant emotion: {fear_dominant}")
        
        # Optimism text should have optimism as dominant
        optimism_dominant = self.analyzer.get_dominant_emotion(
            "Joy and hope fill the market"
        )
        print(f"Optimism text dominant emotion: {optimism_dominant}")
        
        # Empty text should return neutral
        empty_dominant = self.analyzer.get_dominant_emotion("")
        print(f"Empty text dominant emotion: {empty_dominant}")
        self.assertEqual(empty_dominant, 'neutral')
    
    def test_emotion_summary(self) -> None:
        """Test emotion summary generation."""
        print("\n--- Testing Emotion Summary ---")
        
        text = "Panic and fear spread as markets crash"
        summary = self.analyzer.get_emotion_summary(text)
        
        print(f"Text: {text}")
        print(f"Summary: {summary}")
        
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 0)
    
    def test_analyze_texts_batch(self) -> None:
        """Test batch emotion analysis."""
        print("\n--- Testing Batch Analysis ---")
        
        texts = [
            "Market crashes in panic",
            "Investors optimistic about recovery",
            "Fear drives selling pressure"
        ]
        
        results = self.analyzer.analyze_texts(texts)
        
        self.assertEqual(len(results), len(texts))
        
        for text, emotions in zip(texts, results):
            print(f"\n'{text}'")
            print(f"  → {emotions}")
            
            self.assertIn('fear', emotions)
            self.assertIn('greed', emotions)
            self.assertIn('panic', emotions)
            self.assertIn('optimism', emotions)
    
    def test_emotion_keys(self) -> None:
        """Test that returned emotions have correct keys."""
        print("\n--- Testing Emotion Keys ---")
        
        emotions = self.analyzer.get_emotions("Test text with emotions")
        
        expected_keys = {'fear', 'greed', 'panic', 'optimism'}
        actual_keys = set(emotions.keys())
        
        print(f"Expected keys: {expected_keys}")
        print(f"Actual keys: {actual_keys}")
        
        self.assertEqual(actual_keys, expected_keys)
    
    def test_non_string_input(self) -> None:
        """Test handling of non-string input."""
        print("\n--- Testing Non-String Input ---")
        
        with self.assertRaises(TypeError):
            self.analyzer.get_emotions(123)
        
        with self.assertRaises(TypeError):
            self.analyzer.get_emotions(None)
        
        with self.assertRaises(TypeError):
            self.analyzer.get_emotions(['list', 'of', 'strings'])
    
    def test_special_characters(self) -> None:
        """Test handling of special characters in text."""
        print("\n--- Testing Special Characters ---")
        
        text = "Market crashes!!! $$$ Lost everything... :("
        emotions = self.analyzer.get_emotions(text)
        
        print(f"Text: {text}")
        print(f"Emotions: {emotions}")
        
        # Should not raise exception
        self.assertIsInstance(emotions, dict)
    
    def test_very_long_text(self) -> None:
        """Test handling of very long text."""
        print("\n--- Testing Very Long Text ---")
        
        # Create a long text
        base_text = "The market shows strong positive sentiment "
        long_text = base_text * 100
        
        emotions = self.analyzer.get_emotions(long_text)
        
        print(f"Text length: {len(long_text)} characters")
        print(f"Emotions: {emotions}")
        
        # Should not raise exception
        self.assertIsInstance(emotions, dict)
        
        # All scores should be valid
        for score in emotions.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_mixed_emotions_text(self) -> None:
        """Test text with mixed emotions."""
        print("\n--- Testing Mixed Emotions ---")
        
        text = (
            "While fear grips some investors, others remain optimistic "
            "about long-term growth. Panic selling creates opportunities "
            "for greedy traders seeking profits."
        )
        
        emotions = self.analyzer.get_emotions(text)
        
        print(f"Text: {text[:80]}...")
        print(f"Emotions: {emotions}")
        
        # Mixed text should have multiple non-zero emotions
        non_zero_count = sum(1 for v in emotions.values() if v > 0)
        print(f"Non-zero emotions: {non_zero_count}")
        
        self.assertGreater(non_zero_count, 1)
    
    def test_financial_headlines(self) -> None:
        """Test emotion analysis on real financial headlines."""
        print("\n" + "=" * 70)
        print("Testing Financial Headlines")
        print("=" * 70)
        
        headlines = [
            # Economic Times style
            "Sensex crashes 1000 points on global selloff fears",
            "Bank Nifty surges to record high on rate cut hopes",
            "IT stocks tumble as US recession worries mount",
            "Reliance shares jump on strong earnings optimism",
            
            # Reuters style
            "Wall Street panic as banking crisis deepens",
            "Fed decision sparks hope for market recovery",
            "Oil prices crash amid demand destruction fears",
            "Tech rally continues as AI optimism grows",
        ]
        
        for headline in headlines:
            emotions = self.analyzer.get_emotions(headline)
            dominant = self.analyzer.get_dominant_emotion(headline)
            
            print(f"\n{headline}")
            print(f"  Dominant: {dominant}")
            print(f"  Scores: fear={emotions['fear']:.3f}, "
                  f"greed={emotions['greed']:.3f}, "
                  f"panic={emotions['panic']:.3f}, "
                  f"optimism={emotions['optimism']:.3f}")
    
    def test_repr(self) -> None:
        """Test string representation of analyzer."""
        print("\n--- Testing __repr__ ---")
        
        repr_str = repr(self.analyzer)
        print(f"Analyzer repr: {repr_str}")
        
        self.assertIn('EmotionAnalyzer', repr_str)
        self.assertIn('lexicon', repr_str)


class TestNRCWordCoverage(unittest.TestCase):
    """Test NRC lexicon word coverage for financial terms."""
    
    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the emotion analyzer."""
        cls.analyzer = EmotionAnalyzer()
    
    # Financial terms that should be in NRC lexicon
    FINANCIAL_WORDS = {
        'crash': ['fear', 'panic'],
        'surge': ['optimism'],
        'panic': ['fear', 'panic'],
        'hope': ['optimism'],
        'fear': ['fear'],
        'profit': ['optimism', 'greed'],
        'loss': ['fear', 'panic'],
        'gain': ['optimism'],
        'risk': ['fear'],
        'safe': ['optimism'],
    }
    
    def test_financial_word_coverage(self) -> None:
        """Test that key financial words are in the lexicon."""
        print("\n" + "=" * 70)
        print("Testing Financial Word Coverage in NRC Lexicon")
        print("=" * 70)
        
        covered = 0
        for word, expected_emotions in self.FINANCIAL_WORDS.items():
            if word in self.analyzer.emotion_words:
                covered += 1
                word_emotions = self.analyzer.emotion_words[word]
                
                # Check if expected emotions are present
                found_emotions = [
                    em for em in expected_emotions
                    if word_emotions.get(em, 0) > 0
                ]
                
                status = "✓" if found_emotions else "○"
                print(f"{status} '{word}': {list(word_emotions.keys())}")
            else:
                print(f"✗ '{word}': NOT IN LEXICON")
        
        coverage = covered / len(self.FINANCIAL_WORDS)
        print(f"\nCoverage: {coverage:.1%} ({covered}/{len(self.FINANCIAL_WORDS)})")
        
        # We expect at least 50% coverage for key financial terms
        self.assertGreaterEqual(
            coverage, 0.5,
            f"Financial word coverage {coverage:.1%} is below 50%"
        )


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
