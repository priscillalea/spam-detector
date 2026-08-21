"""
train.py — Full training pipeline for the spam detector.

Running this script:
    python -m spam_detector.train

What it does
------------
1. Loads and validates the raw CSV (no assumptions about pre-encoded columns).
2. Investigates duplicate texts — before and after the split — and logs the
   impact. This is intentionally documented, not silently fixed.
3. Runs a grid of experiments comparing:
   - Normalisation strategies : none | stemming | lemmatization
   - Classifiers              : MultinomialNB | ComplementNB |
                                LogisticRegression | LinearSVC
4. Evaluates each combination with StratifiedKFold(5) cross-validation on
   the training set, then reports final test-set metrics for the winner.
5. Saves the final Pipeline (vectorizer + model) and a JSON metadata file
   to the models/ directory.

Design decisions
----------------
- sklearn.Pipeline is used so that the vectorizer is fit ONLY on training data
  within each CV fold. This is the correct way to avoid data leakage.
- stratify=y is used in train_test_split to preserve class proportions.
- The test set is held out entirely until the final evaluation step.
- LinearSVC does not natively support predict_proba. If it wins, it is wrapped
  in CalibratedClassifierCV so the app can show calibrated probabilities.
- Digits are kept in preprocessing (see preprocessing.py for rationale).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Allow running as a script: python -m spam_detector.train
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spam_detector.config import (
    COL_LABEL,
    COL_TEXT,
    DATASET_PATH,
    LABEL_ENCODER_PATH,
    LABEL_HAM,
    LABEL_SPAM,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    TFIDF_DEFAULTS,
    VECTORIZER_PATH,
)
from spam_detector.preprocessing import NormStrategy, preprocess_series

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── 1. Data loading ───────────────────────────────────────────────────────────

def load_data(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load and validate the raw CSV.

    The raw dataset may contain artefact columns (``Unnamed: 0``, a pre-encoded
    ``label_num``) from previous processing runs. These are dropped here so the
    pipeline always derives labels from the authoritative ``label`` column.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with exactly two columns: ``text`` and ``label``.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at the given path.
    ValueError
        If required columns are missing or labels are unexpected.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Please download it from:\n"
            "  https://www.kaggle.com/datasets/venky73/spam-mails-dataset\n"
            "and place spam_ham_dataset.csv in the project root."
        )

    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path.name)

    # Drop artefact columns — keep only what we need
    drop_cols = [c for c in df.columns if c not in (COL_TEXT, COL_LABEL)]
    if drop_cols:
        logger.info("Dropping artefact columns: %s", drop_cols)
        df = df.drop(columns=drop_cols)

    # Validate
    for col in (COL_TEXT, COL_LABEL):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataset.")

    unexpected = set(df[COL_LABEL].unique()) - {LABEL_SPAM, LABEL_HAM}
    if unexpected:
        raise ValueError(f"Unexpected label values: {unexpected}")

    logger.info(
        "Label distribution — ham: %d (%.1f%%)  spam: %d (%.1f%%)",
        (df[COL_LABEL] == LABEL_HAM).sum(),
        (df[COL_LABEL] == LABEL_HAM).mean() * 100,
        (df[COL_LABEL] == LABEL_SPAM).sum(),
        (df[COL_LABEL] == LABEL_SPAM).mean() * 100,
    )
    return df


# ── 2. Duplicate analysis ─────────────────────────────────────────────────────

def analyse_duplicates(df: pd.DataFrame, y_train_idx, y_test_idx) -> None:
    """Log duplicate statistics before and after the train/test split.

    This is intentionally surfaced rather than silently removed because:
    - Duplicates in the full dataset inflate per-class token frequencies.
    - If the same text appears in both train and test, it inflates test metrics.
    - Documenting this is more honest than hiding it.

    Parameters
    ----------
    df:
        Full DataFrame (before split).
    y_train_idx, y_test_idx:
        Index objects from the train/test split.
    """
    total_dups = df[COL_TEXT].duplicated().sum()
    logger.info("=== Duplicate Analysis ===")
    logger.info("Duplicate texts in full dataset: %d / %d", total_dups, len(df))

    train_texts = set(df.loc[y_train_idx, COL_TEXT])
    test_texts = set(df.loc[y_test_idx, COL_TEXT])
    cross_split = train_texts & test_texts

    logger.info(
        "Texts appearing in BOTH train and test: %d "
        "(potential metric inflation)",
        len(cross_split),
    )
    if cross_split:
        logger.warning(
            "Cross-split duplicates found. Results may be slightly optimistic. "
            "Consider deduplication if you want conservative estimates."
        )


# ── 3. Experiment grid ────────────────────────────────────────────────────────

def _build_pipeline(
    classifier: Any,
    tfidf_params: dict | None = None,
) -> Pipeline:
    """Create a sklearn Pipeline: TfidfVectorizer → classifier."""
    params = {**TFIDF_DEFAULTS, **(tfidf_params or {})}
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**params)),
            ("clf", classifier),
        ]
    )


CLASSIFIERS: dict[str, Any] = {
    "MultinomialNB": MultinomialNB(),
    "ComplementNB": ComplementNB(),
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),
    "LinearSVC": LinearSVC(
        max_iter=2000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),
}

NORM_STRATEGIES: list[str] = [
    NormStrategy.NONE,
    NormStrategy.STEMMING,
    NormStrategy.LEMMATIZATION,
]


def run_experiments(
    X_train: pd.Series,
    y_train: pd.Series,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """Cross-validate all (normalisation × classifier) combinations.

    The TF-IDF vectorizer is fitted inside each CV fold via the Pipeline,
    so there is no data leakage between folds.

    Parameters
    ----------
    X_train:
        Raw (non-preprocessed) training texts.
    y_train:
        Binary labels (1 = spam, 0 = ham).
    cv_folds:
        Number of folds for StratifiedKFold.

    Returns
    -------
    pd.DataFrame
        Results table sorted by mean CV F1-spam (descending).
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    results = []

    total = len(NORM_STRATEGIES) * len(CLASSIFIERS)
    logger.info(
        "Running %d experiment(s) × %d-fold CV (%d total fits)…",
        total,
        cv_folds,
        total * cv_folds,
    )

    for strategy in NORM_STRATEGIES:
        logger.info("── Normalisation: %s", strategy)
        X_preprocessed = preprocess_series(X_train, strategy=strategy)

        for clf_name, clf in CLASSIFIERS.items():
            t0 = time.perf_counter()
            pipeline = _build_pipeline(clf)

            scores = cross_validate(
                pipeline,
                X_preprocessed,
                y_train,
                cv=cv,
                scoring={
                    "f1_spam": "f1",          # F1 for class 1 (spam)
                    "precision_spam": "precision",
                    "recall_spam": "recall",
                    "roc_auc": "roc_auc",
                    "accuracy": "accuracy",
                },
                n_jobs=-1,
            )

            elapsed = time.perf_counter() - t0
            row = {
                "normalisation": strategy,
                "classifier": clf_name,
                "cv_f1_spam_mean": scores["test_f1_spam"].mean(),
                "cv_f1_spam_std": scores["test_f1_spam"].std(),
                "cv_recall_spam_mean": scores["test_recall_spam"].mean(),
                "cv_precision_spam_mean": scores["test_precision_spam"].mean(),
                "cv_roc_auc_mean": scores["test_roc_auc"].mean(),
                "cv_accuracy_mean": scores["test_accuracy"].mean(),
                "elapsed_s": round(elapsed, 1),
            }
            results.append(row)
            logger.info(
                "  %-20s  F1-spam=%.4f ± %.4f  ROC-AUC=%.4f  (%.1fs)",
                clf_name,
                row["cv_f1_spam_mean"],
                row["cv_f1_spam_std"],
                row["cv_roc_auc_mean"],
                elapsed,
            )

    results_df = pd.DataFrame(results).sort_values(
        "cv_f1_spam_mean", ascending=False
    ).reset_index(drop=True)
    return results_df


# ── 4. Final evaluation ───────────────────────────────────────────────────────

def final_evaluation(
    best_strategy: str,
    best_clf_name: str,
    X_train: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Pipeline:
    """Train the winning combination on the full training set and evaluate on test.

    Parameters
    ----------
    best_strategy:
        Winning normalisation strategy name.
    best_clf_name:
        Winning classifier name (key in :data:`CLASSIFIERS`).
    X_train, X_test:
        Raw (non-preprocessed) text splits.
    y_train, y_test:
        Label splits.

    Returns
    -------
    Pipeline
        Fitted sklearn Pipeline ready for inference.
    """
    logger.info("=== Final Evaluation ===")
    logger.info("Best combination: %s + %s", best_strategy, best_clf_name)

    clf = CLASSIFIERS[best_clf_name]

    # LinearSVC does not natively support predict_proba.
    # Wrapping in CalibratedClassifierCV gives calibrated probabilities.
    if isinstance(clf, LinearSVC):
        logger.info(
            "LinearSVC detected — wrapping in CalibratedClassifierCV "
            "to enable predict_proba for the UI."
        )
        clf = CalibratedClassifierCV(clf, cv=3)

    pipeline = _build_pipeline(clf)

    X_train_pp = preprocess_series(X_train, strategy=best_strategy)
    X_test_pp = preprocess_series(X_test, strategy=best_strategy)

    pipeline.fit(X_train_pp, y_train)
    y_pred = pipeline.predict(X_test_pp)

    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    # ROC-AUC (requires predict_proba or decision_function)
    try:
        y_proba = pipeline.predict_proba(X_test_pp)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        logger.info("ROC-AUC on test set: %.4f", auc)
    except AttributeError:
        logger.warning("Model does not support predict_proba — skipping ROC-AUC.")

    f1 = f1_score(y_test, y_pred)
    logger.info("F1-spam on test set: %.4f", f1)

    return pipeline


# ── 5. Artefact saving ────────────────────────────────────────────────────────

def save_artefacts(
    pipeline: Pipeline,
    best_strategy: str,
    best_clf_name: str,
    results_df: pd.DataFrame,
) -> None:
    """Persist the trained Pipeline and experiment metadata.

    Parameters
    ----------
    pipeline:
        Fitted sklearn Pipeline (vectorizer + classifier).
    best_strategy:
        Winning normalisation strategy.
    best_clf_name:
        Winning classifier name.
    results_df:
        Full experiment results from :func:`run_experiments`.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save the full Pipeline (vectorizer is embedded inside)
    joblib.dump(pipeline, MODEL_PATH)
    logger.info("Pipeline saved → %s", MODEL_PATH)

    # Save metadata for the app and README
    metadata = {
        "normalisation_strategy": str(best_strategy),
        "classifier": best_clf_name,
        "tfidf_params": {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in TFIDF_DEFAULTS.items()
        },
        "cv_results_top5": results_df.head(5).to_dict(orient="records"),
    }
    with open(LABEL_ENCODER_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved → %s", LABEL_ENCODER_PATH)

    # Also save the results CSV for the notebook / README
    results_csv = MODELS_DIR / "cv_results.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info("CV results saved → %s", results_csv)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """End-to-end training pipeline."""
    logger.info("========== Spam Detector — Training Pipeline ==========")

    # 1. Load data
    df = load_data()

    # 2. Encode labels (from authoritative 'label' column, not pre-encoded)
    y = (df[COL_LABEL] == LABEL_SPAM).astype(int)

    # 3. Train/test split — stratified to preserve class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        df[COL_TEXT],
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        "Split — train: %d  test: %d  (stratified, random_state=%d)",
        len(X_train),
        len(X_test),
        RANDOM_STATE,
    )

    # 4. Duplicate analysis
    analyse_duplicates(df, X_train.index, X_test.index)

    # 5. Cross-validated experiment grid
    results_df = run_experiments(X_train, y_train)

    logger.info("\n=== Experiment Results (sorted by CV F1-spam) ===")
    logger.info("\n%s", results_df.to_string(index=False))

    # 6. Select best combination by CV F1-spam
    best = results_df.iloc[0]
    best_strategy = best["normalisation"]
    best_clf_name = best["classifier"]

    # 7. Final evaluation on held-out test set
    pipeline = final_evaluation(
        best_strategy, best_clf_name,
        X_train, X_test, y_train, y_test,
    )

    # 8. Save artefacts
    save_artefacts(pipeline, best_strategy, best_clf_name, results_df)

    logger.info("========== Training complete ==========")


if __name__ == "__main__":
    main()
