"""
predict.py — Inference class for the spam detector.

The :class:`SpamDetector` class loads the trained Pipeline once at
instantiation. This is the correct pattern for use in web apps (Streamlit,
Flask, FastAPI) or anywhere the model is called repeatedly.

Note on probabilities
---------------------
The trained Pipeline includes ``predict_proba``, but these probabilities
should not be presented as "confidence" without a caveat. Naive Bayes tends
to output extreme probabilities (near 0 or 1) that are poorly calibrated.
Logistic Regression is better calibrated. LinearSVC probabilities come from
CalibratedClassifierCV and are approximate.

The app is expected to label them as "model score" or "estimated probability"
and include a brief explanation, not as ground-truth confidence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spam_detector.config import LABEL_ENCODER_PATH, MODEL_PATH
from spam_detector.preprocessing import NormStrategy, preprocess_text

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of a single spam prediction.

    Attributes
    ----------
    label:
        ``'spam'`` or ``'ham'``.
    is_spam:
        Boolean convenience flag.
    spam_probability:
        Estimated probability that the message is spam, in [0.0, 1.0].
        Based on the model's ``predict_proba`` output. May be poorly
        calibrated depending on the underlying classifier.
    top_spam_tokens:
        Top tokens associated with spam classification for this input.
    top_ham_tokens:
        Top tokens associated with ham classification for this input.
    preprocessed_text:
        The cleaned text that was fed to the model (useful for debugging).
    """

    label: str
    is_spam: bool
    spam_probability: float
    top_spam_tokens: list[tuple[str, float]] = field(default_factory=list)
    top_ham_tokens: list[tuple[str, float]] = field(default_factory=list)
    preprocessed_text: str = ""


class SpamDetector:
    """Loads the trained pipeline once and provides a simple prediction API.

    Parameters
    ----------
    model_path:
        Path to the serialised sklearn Pipeline (``model.pkl``).
    metadata_path:
        Path to the JSON metadata file saved during training.

    Examples
    --------
    >>> detector = SpamDetector()
    >>> result = detector.predict("Congratulations! You won a free prize!")
    >>> result.label
    'spam'
    >>> result.spam_probability
    0.97
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        metadata_path: Path = LABEL_ENCODER_PATH,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}.\n"
                "Run: python -m spam_detector.train"
            )

        logger.info("Loading model from %s …", model_path)
        self._pipeline: Pipeline = joblib.load(model_path)

        # Load metadata to retrieve normalisation strategy used during training
        self._strategy: str = NormStrategy.STEMMING  # safe default
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                meta = json.load(f)
            raw_strategy = meta.get("normalisation_strategy", "stemming")
            # The JSON may store 'NormStrategy.LEMMATIZATION' (enum str repr).
            # Strip the class prefix to get the plain value ('lemmatization').
            if "." in raw_strategy:
                raw_strategy = raw_strategy.split(".", 1)[1].lower()
            self._strategy = NormStrategy(raw_strategy)
        else:
            logger.warning(
                "Metadata file not found at %s — using default strategy '%s'.",
                metadata_path,
                self._strategy,
            )

        # Pre-extract feature names for token-level explanations
        try:
            tfidf = self._pipeline.named_steps["tfidf"]
            self._feature_names: Optional[np.ndarray] = np.array(
                tfidf.get_feature_names_out()
            )
        except (AttributeError, KeyError):
            self._feature_names = None

        logger.info(
            "Model loaded. Strategy=%s  Vocabulary size=%s",
            self._strategy,
            len(self._feature_names) if self._feature_names is not None else "unknown",
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, text: str, top_n: int = 10) -> PredictionResult:
        """Predict whether an email is spam.

        Parameters
        ----------
        text:
            Raw email text (will be preprocessed internally).
        top_n:
            Number of top tokens to return for each class in the explanation.

        Returns
        -------
        PredictionResult
            Prediction with label, probability, and token explanations.
        """
        preprocessed = preprocess_text(text, strategy=self._strategy)

        # Predict
        label_num: int = int(self._pipeline.predict([preprocessed])[0])
        is_spam: bool = label_num == 1
        label: str = "spam" if is_spam else "ham"

        # Probability
        spam_probability: float = 0.5  # fallback
        try:
            proba = self._pipeline.predict_proba([preprocessed])[0]
            spam_probability = float(proba[1])  # index 1 = spam class
        except AttributeError:
            logger.debug("predict_proba not available for this model.")

        # Token explanations
        top_spam, top_ham = self._explain(preprocessed, top_n=top_n)

        return PredictionResult(
            label=label,
            is_spam=is_spam,
            spam_probability=spam_probability,
            top_spam_tokens=top_spam,
            top_ham_tokens=top_ham,
            preprocessed_text=preprocessed,
        )

    def _explain(
        self,
        preprocessed_text: str,
        top_n: int = 10,
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """Extract top tokens influencing spam / ham decision.

        For linear models (LR, LinearSVC): uses model coefficients.
        For Naive Bayes: uses log-probability ratio (log P(token|spam) / P(token|ham)).
        Returns empty lists if explanation is not available.

        Parameters
        ----------
        preprocessed_text:
            Already-preprocessed input text.
        top_n:
            Number of tokens to return per class.

        Returns
        -------
        tuple[list[tuple[str, float]], list[tuple[str, float]]]
            (top_spam_tokens, top_ham_tokens), each a list of (token, score) pairs.
        """
        if self._feature_names is None:
            return [], []

        clf = self._pipeline.named_steps["clf"]
        tfidf = self._pipeline.named_steps["tfidf"]

        # Transform this specific input to get its feature vector
        x_vec = tfidf.transform([preprocessed_text])
        # Only consider features present in this input
        active_feature_indices = x_vec.nonzero()[1]

        if len(active_feature_indices) == 0:
            return [], []

        active_tokens = self._feature_names[active_feature_indices]

        try:
            # Linear models: coef_[0] gives the log-odds for the positive (spam) class
            if hasattr(clf, "coef_"):
                coef = clf.coef_[0] if clf.coef_.ndim > 1 else clf.coef_
                scores = coef[active_feature_indices]
            # Naive Bayes: log P(feature|spam) - log P(feature|ham)
            elif hasattr(clf, "feature_log_prob_"):
                # feature_log_prob_ shape: (n_classes, n_features)
                log_prob_spam = clf.feature_log_prob_[1][active_feature_indices]
                log_prob_ham = clf.feature_log_prob_[0][active_feature_indices]
                scores = log_prob_spam - log_prob_ham
            # CalibratedClassifierCV wrapping LinearSVC
            elif hasattr(clf, "estimators_"):
                inner = clf.estimators_[0]
                if hasattr(inner, "coef_"):
                    coef = inner.coef_[0] if inner.coef_.ndim > 1 else inner.coef_
                    scores = coef[active_feature_indices]
                else:
                    return [], []
            else:
                return [], []
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not extract token scores: %s", exc)
            return [], []

        token_scores = list(zip(active_tokens, scores.tolist()))

        top_spam = sorted(token_scores, key=lambda x: x[1], reverse=True)[:top_n]
        top_ham = sorted(token_scores, key=lambda x: x[1])[:top_n]

        return top_spam, top_ham
