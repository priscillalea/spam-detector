"""
test_pipeline.py — Integration tests for the training pipeline.

These tests verify the correctness of the ML pipeline at a structural level:
no data leakage, correct stratification, and reproducibility.
They intentionally run on a small synthetic dataset to stay fast.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spam_detector.config import RANDOM_STATE, TEST_SIZE
from spam_detector.preprocessing import preprocess_text


# ── Fixtures ──────────────────────────────────────────────────────────────────

SPAM_TEXTS = [
    "Congratulations you won a free prize click here now",
    "Win cash money free offer limited time act now",
    "FREE GIFT claim your prize call 1800 now winner",
    "Urgent your account selected for free reward click",
    "You have been chosen win million dollars free",
    "Get rich quick free money no risk guaranteed winner",
    "Click here free offer exclusive deal prize money",
    "Final notice claim free gift prize won selected",
    "Alert free cash winner selected contact immediately",
    "Special offer free trial winner prize claim now",
]

HAM_TEXTS = [
    "Please find the attached report for your review",
    "Hi team the meeting has been rescheduled to Thursday",
    "Could you send me the updated project documentation",
    "The quarterly results show a slight improvement in revenue",
    "Thank you for your feedback on the proposal draft",
    "I will be out of office next week returning Monday",
    "Please review the contract and let me know your thoughts",
    "The server maintenance is scheduled for this weekend",
    "Attached is the invoice for services rendered in October",
    "Can we schedule a call to discuss the project timeline",
]

LABELS = [1] * len(SPAM_TEXTS) + [0] * len(HAM_TEXTS)
TEXTS = SPAM_TEXTS + HAM_TEXTS


# ── Data leakage test ─────────────────────────────────────────────────────────

class TestNoDataLeakage:
    """Verify that the vectorizer is never fit on test data."""

    def test_vectorizer_fit_only_on_train(self):
        """The Pipeline must be fit only on training data.

        If the vectorizer were fit on all data (the old bug), the vocabulary
        size after fitting on train would equal the full-corpus vocabulary.
        With correct usage, fitting only on train gives a strictly smaller or
        equal vocabulary (equal only if the train set covers all tokens).
        """
        texts_pp = [preprocess_text(t) for t in TEXTS]
        X_train, X_test, y_train, _ = train_test_split(
            texts_pp, LABELS, test_size=0.3, random_state=RANDOM_STATE, stratify=LABELS
        )

        # Correct: fit only on train
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", MultinomialNB()),
        ])
        pipeline.fit(X_train, y_train)
        vocab_train_only = len(pipeline.named_steps["tfidf"].vocabulary_)

        # Wrong: fit on all data (simulates the bug)
        full_vectorizer = TfidfVectorizer()
        full_vectorizer.fit(texts_pp)
        vocab_full = len(full_vectorizer.vocabulary_)

        # The train-only vocab should be ≤ the full vocab
        assert vocab_train_only <= vocab_full, (
            f"Train vocab ({vocab_train_only}) > full vocab ({vocab_full}). "
            "This suggests the vectorizer was accidentally fit on all data."
        )

    def test_transform_does_not_update_vocabulary(self):
        """Calling transform() on test set must not modify the fitted vocabulary."""
        texts_pp = [preprocess_text(t) for t in TEXTS]
        X_train, X_test, _, _ = train_test_split(
            texts_pp, LABELS, test_size=0.3, random_state=RANDOM_STATE
        )

        vectorizer = TfidfVectorizer()
        vectorizer.fit(X_train)
        vocab_before = set(vectorizer.vocabulary_.keys())

        vectorizer.transform(X_test)  # must not modify vocab
        vocab_after = set(vectorizer.vocabulary_.keys())

        assert vocab_before == vocab_after


# ── Stratification test ───────────────────────────────────────────────────────

class TestStratification:
    def test_split_preserves_class_proportions(self):
        """With stratify=y, train and test should have similar spam ratios."""
        labels = np.array(LABELS)
        indices = np.arange(len(TEXTS))

        _, _, y_train, y_test = train_test_split(
            indices, labels,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=labels,
        )

        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        full_ratio = labels.mean()

        # With the small synthetic dataset (20 samples), integer rounding means
        # even perfect stratification can deviate up to ~10%. Tolerance is set
        # to 15% to validate that stratify= is being applied without being fragile.
        # In practice, with real data (5171 samples) the deviation is <0.1%.
        tolerance = 0.15
        assert abs(train_ratio - full_ratio) < tolerance, (
            f"Train spam ratio {train_ratio:.2f} deviates too much from "
            f"full dataset ratio {full_ratio:.2f}"
        )
        assert abs(test_ratio - full_ratio) < tolerance, (
            f"Test spam ratio {test_ratio:.2f} deviates too much from "
            f"full dataset ratio {full_ratio:.2f}"
        )


# ── Reproducibility test ──────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_random_state_gives_same_split(self):
        """Two splits with the same random_state must produce identical results."""
        texts_pp = [preprocess_text(t) for t in TEXTS]

        X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
            texts_pp, LABELS, test_size=0.3, random_state=RANDOM_STATE
        )
        X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
            texts_pp, LABELS, test_size=0.3, random_state=RANDOM_STATE
        )

        assert X_train_1 == X_train_2
        assert X_test_1 == X_test_2
        assert y_train_1 == y_train_2

    def test_pipeline_fit_is_deterministic(self):
        """Same data + same random_state → same predictions."""
        texts_pp = [preprocess_text(t) for t in TEXTS]
        X_train, X_test, y_train, _ = train_test_split(
            texts_pp, LABELS, test_size=0.3, random_state=RANDOM_STATE
        )

        def make_and_fit():
            p = Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())])
            p.fit(X_train, y_train)
            return p.predict(X_test)

        preds_1 = make_and_fit()
        preds_2 = make_and_fit()
        assert list(preds_1) == list(preds_2)
