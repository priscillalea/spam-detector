"""
config.py — Centralised paths and constants for the spam_detector package.

All other modules import from here instead of hard-coding strings.
"""

from pathlib import Path

# ── Repository root (two levels up from this file: src/spam_detector/config.py)
ROOT_DIR: Path = Path(__file__).resolve().parents[2]

# ── Data
DATA_DIR: Path = ROOT_DIR
DATASET_PATH: Path = DATA_DIR / "spam_ham_dataset.csv"

# ── Artefacts produced by train.py
MODELS_DIR: Path = ROOT_DIR / "models"
MODEL_PATH: Path = MODELS_DIR / "spam_detector_model.pkl"
VECTORIZER_PATH: Path = MODELS_DIR / "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH: Path = MODELS_DIR / "label_encoder.json"

# ── Reports produced by evaluate.py
REPORTS_DIR: Path = ROOT_DIR / "reports"

# ── Column names in the raw CSV
COL_TEXT: str = "text"
COL_LABEL: str = "label"
LABEL_SPAM: str = "spam"
LABEL_HAM: str = "ham"

# ── Train / test split
TEST_SIZE: float = 0.25
RANDOM_STATE: int = 42

# ── TF-IDF defaults (overridable in train.py experiments)
TFIDF_DEFAULTS: dict = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
    "max_features": 50_000,
}
