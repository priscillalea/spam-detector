"""
test_preprocessing.py — Unit tests for spam_detector.preprocessing.

These tests verify that preprocess_text behaves correctly across all three
normalisation strategies and for edge-case inputs (empty, None, special chars).
"""

import sys
from pathlib import Path

import pytest

# Allow running from project root: pytest tests/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spam_detector.preprocessing import NormStrategy, preprocess_text


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_none_returns_empty(self):
        assert preprocess_text(None) == ""

    def test_empty_string_returns_empty(self):
        assert preprocess_text("") == ""

    def test_whitespace_only_returns_empty(self):
        assert preprocess_text("   ") == ""

    def test_integer_input_returns_empty(self):
        """Non-string inputs should not raise — they return empty string."""
        assert preprocess_text(42) == ""  # type: ignore[arg-type]

    def test_symbols_only_returns_empty(self):
        assert preprocess_text("!!! @@@ ###") == ""

    def test_stopwords_only_returns_empty(self):
        """All English stopwords should be removed, leaving nothing."""
        result = preprocess_text("the a an is was were", strategy="none")
        assert result == ""


# ── Lowercase & symbol removal ────────────────────────────────────────────────

class TestBasicCleaning:
    def test_lowercase(self):
        result = preprocess_text("HELLO WORLD", strategy="none")
        assert result == result.lower()

    def test_punctuation_removed(self):
        result = preprocess_text("hello, world!", strategy="none")
        assert "," not in result
        assert "!" not in result

    def test_digits_kept_by_default(self):
        """Digits like '100' or '1000' are discriminative spam features."""
        result = preprocess_text("win 1000 dollars free", strategy="none")
        assert "1000" in result

    def test_digits_removed_when_flagged(self):
        result = preprocess_text("win 1000 dollars free", strategy="none", keep_digits=False)
        assert "1000" not in result


# ── Stopword removal ──────────────────────────────────────────────────────────

class TestStopwords:
    def test_common_stopwords_removed(self):
        result = preprocess_text("this is a test of the system", strategy="none")
        # 'this', 'is', 'a', 'of', 'the' are stopwords; 'test', 'system' are not
        assert "test" in result
        assert "system" in result
        for sw in ["this", " is ", " a ", " of ", " the "]:
            assert sw not in f" {result} "


# ── Stemming strategy ─────────────────────────────────────────────────────────

class TestStemming:
    def test_running_stemmed(self):
        result = preprocess_text("running", strategy=NormStrategy.STEMMING)
        assert result == "run"

    def test_congratulations_stemmed(self):
        result = preprocess_text("congratulations winner", strategy="stemming")
        # PorterStemmer reduces 'congratulations' to 'congratul'
        assert "congratul" in result

    def test_stemming_is_default(self):
        """Default strategy should be stemming."""
        default = preprocess_text("running")
        stemmed = preprocess_text("running", strategy="stemming")
        assert default == stemmed


# ── Lemmatization strategy ────────────────────────────────────────────────────

class TestLemmatization:
    def test_running_lemmatized(self):
        # WordNetLemmatizer without POS defaults to noun, so 'running' stays
        # as 'running'. This is expected behaviour (not a bug).
        result = preprocess_text("running", strategy=NormStrategy.LEMMATIZATION)
        assert result in ("running", "run")  # both are acceptable

    def test_better_than_stemming_for_english(self):
        """Lemmatization should not produce non-words like 'univers'."""
        stemmed = preprocess_text("university", strategy="stemming")
        lemmatized = preprocess_text("university", strategy="lemmatization")
        # Stemmer produces 'univers', lemmatizer produces 'university'
        assert stemmed != "university"   # stemmer cuts the word
        assert lemmatized == "university"  # lemmatizer keeps valid form


# ── Strategy enum ─────────────────────────────────────────────────────────────

class TestNormStrategy:
    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            preprocess_text("hello", strategy="invalid_strategy")

    def test_string_aliases_work(self):
        """String values like 'stemming' should work as well as the enum."""
        result_str = preprocess_text("running free", strategy="stemming")
        result_enum = preprocess_text("running free", strategy=NormStrategy.STEMMING)
        assert result_str == result_enum


# ── Spam-relevant content ─────────────────────────────────────────────────────

class TestSpamRelevantContent:
    def test_obvious_spam_keywords_survive(self):
        """Critical spam signal words should survive preprocessing."""
        result = preprocess_text(
            "Congratulations! You won a FREE prize worth $1000! Click now!",
            strategy="stemming",
        )
        # After stemming: 'free' stays 'free', 'prize' → 'prize', 'click' stays
        assert "free" in result
        assert "prize" in result
        assert "click" in result

    def test_ham_professional_language_preserved(self):
        result = preprocess_text(
            "Please find the updated report attached. Thank you.",
            strategy="stemming",
        )
        assert "report" in result or "updat" in result  # stem of 'updated'
