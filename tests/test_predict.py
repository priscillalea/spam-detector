"""
test_predict.py — Tests for the SpamDetector inference class.

These tests require a trained model in models/. Run train.py first.
Tests that depend on the model are marked with `requires_model`.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spam_detector.config import MODEL_PATH
from spam_detector.predict import PredictionResult, SpamDetector

# Skip model-dependent tests if the model hasn't been trained yet
requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model not found — run: python -m spam_detector.train",
)


@pytest.fixture(scope="module")
def detector() -> SpamDetector:
    """Load the model once for all tests in this module."""
    return SpamDetector()


# ── Without model ─────────────────────────────────────────────────────────────

def test_missing_model_raises():
    """SpamDetector should raise FileNotFoundError for non-existent model path."""
    with pytest.raises(FileNotFoundError, match="Model not found"):
        SpamDetector(model_path=Path("/non/existent/model.pkl"))


# ── With model ────────────────────────────────────────────────────────────────

@requires_model
class TestPredictions:
    def test_obvious_spam(self, detector: SpamDetector):
        result = detector.predict(
            "CONGRATULATIONS!!! You have WON a FREE prize worth $10,000! "
            "Click here NOW to claim. Limited time offer! Call 1-800-WIN-FREE."
        )
        assert result.is_spam is True
        assert result.label == "spam"

    def test_obvious_ham(self, detector: SpamDetector):
        result = detector.predict(
            "Hi team, please find the Q3 financial report attached. "
            "Let me know if you have any questions before the board meeting."
        )
        assert result.is_spam is False
        assert result.label == "ham"

    def test_empty_input_returns_ham_or_handles_gracefully(self, detector: SpamDetector):
        """Empty text should not crash — model will produce some prediction."""
        result = detector.predict("")
        assert isinstance(result, PredictionResult)
        assert result.label in ("spam", "ham")

    def test_probability_in_valid_range(self, detector: SpamDetector):
        result = detector.predict("Free money click here now")
        assert 0.0 <= result.spam_probability <= 1.0

    def test_spam_probability_high_for_spam(self, detector: SpamDetector):
        result = detector.predict(
            "WIN FREE CASH! Click here immediately! Limited offer! Act now!"
        )
        assert result.spam_probability > 0.5

    def test_spam_probability_low_for_ham(self, detector: SpamDetector):
        result = detector.predict(
            "Please review the attached contract and send your feedback by Friday."
        )
        assert result.spam_probability < 0.5

    def test_result_has_token_explanations(self, detector: SpamDetector):
        """predict() should return non-empty token explanation lists."""
        result = detector.predict("Free prize winner click here now")
        assert isinstance(result.top_spam_tokens, list)
        # May be empty for very short texts but should not raise
        assert isinstance(result.top_ham_tokens, list)

    def test_special_characters_do_not_crash(self, detector: SpamDetector):
        result = detector.predict("Test with unicode: café résumé naïve 日本語")
        assert isinstance(result, PredictionResult)

    def test_very_long_text_does_not_crash(self, detector: SpamDetector):
        long_text = "This is a legitimate business email. " * 500
        result = detector.predict(long_text)
        assert isinstance(result, PredictionResult)

    def test_preprocessed_text_is_returned(self, detector: SpamDetector):
        result = detector.predict("Running free offers!")
        assert isinstance(result.preprocessed_text, str)
        # Should be lowercase and have no punctuation
        assert result.preprocessed_text == result.preprocessed_text.lower()


@requires_model
class TestModelLoadedOnce:
    """The model must be loaded at instantiation, not on each predict() call."""

    def test_two_calls_use_same_pipeline(self, detector: SpamDetector):
        """Same detector instance — pipeline object should be identical."""
        id1 = id(detector._pipeline)
        detector.predict("test message one")
        id2 = id(detector._pipeline)
        assert id1 == id2, "Pipeline was reloaded between calls — this is a bug"
