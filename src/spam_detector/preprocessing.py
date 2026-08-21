"""
preprocessing.py — Text cleaning and normalisation for the spam detector.

Three normalisation strategies are supported and can be compared experimentally:

    'none'        — lowercase + symbol removal only (baseline)
    'stemming'    — PorterStemmer (fast, produces non-words like "univers")
    'lemmatization' — WordNetLemmatizer (slower, linguistically correct)

Design notes
------------
- The NLTK stopword set and the normaliser object are instantiated once at
  module level, not inside the per-document function. This avoids recreating
  them thousands of times during dataset processing.
- Digits are intentionally *kept* by default. Numbers like "100%", "$1000",
  "1-800" are discriminative features for spam detection.
- `preprocess_text` is a pure function (no side effects) so it can be used
  both inside sklearn Pipelines and standalone (e.g. in the Streamlit app).
"""

from __future__ import annotations

import logging
import re
from enum import Enum

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

logger = logging.getLogger(__name__)


class NormStrategy(str, Enum):
    """Normalisation strategy for :func:`preprocess_text`."""

    NONE = "none"
    STEMMING = "stemming"
    LEMMATIZATION = "lemmatization"


# ── Download required NLTK resources (idempotent, quiet after first run) ──────
def _ensure_nltk_resources() -> None:
    """Download NLTK data if not already present."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", name)
            nltk.download(name, quiet=True)


_ensure_nltk_resources()

# ── Module-level singletons — instantiated once, reused across all calls ──────
_STOP_WORDS: frozenset[str] = frozenset(stopwords.words("english"))
_STEMMER: PorterStemmer = PorterStemmer()
_LEMMATIZER: WordNetLemmatizer = WordNetLemmatizer()

# Compiled regex — no numbers removed (digits are informative for spam)
_SYMBOL_RE = re.compile(r"[^a-z0-9\s]")


# ──────────────────────────────────────────────────────────────────────────────

def preprocess_text(
    text: str | None,
    strategy: NormStrategy | str = NormStrategy.STEMMING,
    keep_digits: bool = True,
) -> str:
    """Clean and normalise a single email text.

    Parameters
    ----------
    text:
        Raw email string. ``None`` or non-string values return an empty string.
    strategy:
        One of ``'none'``, ``'stemming'`` or ``'lemmatization'``.
        See :class:`NormStrategy` for details.
    keep_digits:
        If ``True`` (default), digits are kept in the output. Digits like
        "1000" or "800" are discriminative features in spam email text.

    Returns
    -------
    str
        Preprocessed text, or ``""`` for empty / non-string input.

    Examples
    --------
    >>> preprocess_text("Running FREE offers at 100%!", strategy="stemming")
    'run free offer 100'
    >>> preprocess_text("Running FREE offers!", strategy="lemmatization")
    'running free offer'
    >>> preprocess_text("Running FREE offers!", strategy="none")
    'running free offer'
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    strategy = NormStrategy(strategy)

    # 1. Lowercase
    text = text.lower()

    # 2. Remove symbols (optionally keep digits)
    if keep_digits:
        text = _SYMBOL_RE.sub(" ", text)
    else:
        text = re.sub(r"[^a-z\s]", " ", text)

    # 3. Tokenise by whitespace
    tokens = text.split()

    # 4. Remove stopwords
    tokens = [t for t in tokens if t not in _STOP_WORDS]

    # 5. Apply normalisation strategy
    if strategy is NormStrategy.STEMMING:
        tokens = [_STEMMER.stem(t) for t in tokens]
    elif strategy is NormStrategy.LEMMATIZATION:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    # NormStrategy.NONE — no further transformation

    return " ".join(tokens)


def preprocess_series(
    texts,  # pd.Series
    strategy: NormStrategy | str = NormStrategy.STEMMING,
    keep_digits: bool = True,
):
    """Apply :func:`preprocess_text` to a pandas Series.

    Parameters
    ----------
    texts:
        A ``pd.Series`` of raw email strings.
    strategy:
        Normalisation strategy passed to :func:`preprocess_text`.
    keep_digits:
        Passed to :func:`preprocess_text`.

    Returns
    -------
    pd.Series
        Series of preprocessed strings.
    """
    return texts.apply(
        preprocess_text, strategy=strategy, keep_digits=keep_digits
    )
