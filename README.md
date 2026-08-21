# Spam Detector

> Email classification with classical NLP — TF-IDF vectorization and comparative model evaluation on the Enron Spam Dataset.

![CI](https://github.com/priscillalea/spam-detector/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

This project builds a spam email classifier as a rigorous ML exercise — not just as a demo, but as an experimentally correct pipeline. The emphasis is on doing the basics right: no data leakage, proper cross-validation, honest evaluation metrics, and transparent model selection.

**Why classical ML?** For this dataset and task size, TF-IDF + linear models are fast, interpretable, and competitive. Deep learning and LLMs would add complexity without a clear performance justification here.

---

## Pipeline

```
Raw CSV (Enron Spam Dataset)
  ↓
Data loading & validation        — drops artefact columns, validates labels
  ↓
Train / Test split (75/25)       — stratified by class, random_state=42
  ↓
Duplicate analysis               — documented, not silently hidden
  ↓
Preprocessing comparison         — none | stemming | lemmatization
  ↓
sklearn.Pipeline                 — TF-IDF fit only on train (no leakage)
  ↓
5-fold StratifiedKFold CV        — 12 combinations evaluated
  ↓
Model selection by F1-spam       — not accuracy
  ↓
Final evaluation on held-out test set
  ↓
Artefact persistence + Streamlit UI
```

---

## Dataset

**[Enron Spam Dataset](http://www2.aueb.gr/users/ion/data/enron-spam/)** — `enron-1` subset, processed and published by [venky73 on Kaggle](https://www.kaggle.com/datasets/venky73/spam-mails-dataset).

| | Count | % |
|---|---|---|
| Ham (legitimate) | 3,672 | 71.0% |
| Spam | 1,499 | 29.0% |
| **Total** | **5,171** | |

**License:** CC0 — Public Domain. Dataset is included in the repository.

> **Duplicate note:** 178 duplicate texts exist in the dataset (54 appear in both train and test splits). This is documented in the training logs rather than silently removed — the cross-split overlap is small and does not materially affect metrics, but it means results are slightly optimistic.

---

## Preprocessing

Three strategies were compared experimentally:

| Strategy | Description |
|---|---|
| `none` | Lowercase + symbol removal only |
| `stemming` | PorterStemmer — fast, produces non-words (e.g., "univers") |
| `lemmatization` | WordNetLemmatizer — slower, linguistically correct |

**Digits are preserved** by default. Numbers like "100%", "$1000", "1-800" are informative spam signals and removing them loses relevant features.

---

## Model Comparison

All experiments use the same TF-IDF configuration: `ngram_range=(1,2)`, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`, `max_features=50000`.

Evaluated with **5-fold StratifiedKFold CV**. Primary metric: **F1-spam** (balances precision and recall for the minority class — accuracy alone is misleading with 71/29% class imbalance).

| Normalisation | Classifier | F1-spam ↓ | Recall-spam | Precision-spam | ROC-AUC |
|---|---|---|---|---|---|
| **lemmatization** | **LinearSVC** | **0.9816** | **0.9947** | 0.9689 | **0.9989** |
| none | LinearSVC | 0.9803 | 0.9947 | 0.9664 | 0.9987 |
| stemming | LinearSVC | 0.9785 | 0.9911 | 0.9663 | 0.9987 |
| lemmatization | ComplementNB | 0.9634 | 0.9600 | 0.9670 | 0.9973 |
| none | ComplementNB | 0.9616 | 0.9564 | 0.9668 | 0.9974 |
| stemming | MultinomialNB | 0.9186 | 0.8585 | 0.9879 | 0.9973 |

**MultinomialNB note:** High precision (0.988) but low recall (0.859) — it misses ~14% of spam. For a spam filter, false negatives (spam that passes) are costly.

**Normalisation finding:** Lemmatization gives a small but consistent edge over stemming and baseline. The difference is small (~0.3pp F1), which is expected — the Enron dataset is relatively clean text.

---

## Final Model

**LinearSVC + Lemmatization**, wrapped in `CalibratedClassifierCV` to enable probability outputs.

**Test-set results (held-out, never seen during training or model selection):**

| Metric | Ham | Spam |
|---|---|---|
| Precision | 0.99 | 0.98 |
| Recall | 0.99 | 0.98 |
| F1-score | 0.99 | 0.98 |
| **ROC-AUC** | | **0.9990** |

**Why LinearSVC?** It consistently ranked first across all normalisation strategies. For TF-IDF sparse features, linear models with a large margin (SVM) generally outperform probabilistic models. LinearSVC with `class_weight='balanced'` handles the class imbalance correctly without requiring oversampling.

**Why not Logistic Regression?** LR showed high recall-spam (0.998) but lower precision (0.914), meaning it over-predicts spam. LinearSVC achieves a better precision/recall balance for this dataset.

> **Note on probabilities:** The UI displays calibrated probability scores from `CalibratedClassifierCV`. These are approximate — they reflect relative model confidence, not ground-truth certainty.

---

## Project Structure

```
spam-detector/
├── src/
│   └── spam_detector/
│       ├── config.py           # paths and constants
│       ├── preprocessing.py    # text cleaning, 3 normalisation strategies
│       ├── train.py            # full pipeline: load → split → CV → evaluate → save
│       ├── evaluate.py         # metrics report generation
│       └── predict.py          # SpamDetector class (loads model once)
├── app/
│   └── app.py                  # Streamlit UI
├── notebooks/
│   └── 01_eda.ipynb            # EDA and visualisations
├── tests/
│   ├── test_preprocessing.py   # 20 unit tests
│   ├── test_predict.py         # 11 integration tests
│   └── test_pipeline.py        # 5 structural correctness tests
├── models/                     # generated artefacts (gitignored)
├── reports/
│   └── model_comparison.md     # full experiment results
├── spam_ham_dataset.csv        # Enron dataset (CC0)
├── requirements.txt
├── pyproject.toml              # ruff + pytest config
└── .github/workflows/ci.yml    # CI: lint + test on push
```

---

## Getting Started

**Requirements:** Python 3.10+

```bash
git clone https://github.com/priscillalea/spam-detector.git
cd spam-detector

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

**Train the model:**

```bash
python -m spam_detector.train
# Working directory: src/
cd src
python -m spam_detector.train
```

This will:
- Run 12 model/preprocessing combinations with 5-fold CV
- Print a ranked comparison table
- Save `models/spam_detector_model.pkl` and `models/label_encoder.json`
- Write `reports/model_comparison.md`

**Run the app:**

```bash
streamlit run app/app.py
```

**Deploy to Streamlit Community Cloud (free):**

1. Push the repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo
3. Set **Main file path** to `app/app.py`
4. Add a `packages.txt` file at the project root if any system dependencies are needed (not required here)
5. The app will be live at `https://<your-username>-spam-detector.streamlit.app`

> **Note:** Streamlit Cloud runs `train.py` is not executed automatically on deploy. Include the trained `models/` artefacts in the repo, or add a startup script. The simplest approach for a portfolio project is to commit `models/spam_detector_model.pkl` and `models/label_encoder.json` (even if the raw `.pkl` files are gitignored locally, you can push them once for the live demo).

**Run tests:**

```bash
python -m pytest tests/ -v
```

---

## Interpretability

The app shows which tokens in each message pushed the model toward spam or ham. This uses the LinearSVC decision boundary (coefficient vector) — only tokens present in the specific message are shown, not global feature importance.

For Naive Bayes, the equivalent is the log-probability ratio: `log P(token|spam) - log P(token|ham)`.

This is a meaningful and technically honest form of local explanation for linear models — no LIME or SHAP needed.

---

## Limitations

- **Domain:** Trained on Enron employee emails (~2000s). Performance on modern spam (social media phishing, HTML-heavy newsletters) may differ significantly.
- **Language:** English only. Stopwords, stemming, and lemmatization are English-specific.
- **No drift monitoring:** The model has no mechanism to detect when the spam distribution shifts over time.
- **Probabilities:** The displayed confidence scores are calibrated approximations, not ground-truth probabilities.
- **Static model:** No online learning or retraining pipeline.

---

## Dataset Attribution

Dataset derived from the [Enron Spam Dataset](http://www2.aueb.gr/users/ion/data/enron-spam/) (Metsis et al., 2006), processed and published on Kaggle by [venky73](https://www.kaggle.com/venky73) under **CC0 — Public Domain**.

---

## License

Code: [MIT License](LICENSE)  
Dataset: CC0 — Public Domain (see attribution above)

---

**Priscilla Leandro** — [LinkedIn](https://www.linkedin.com/in/priscillaleandro/) · [GitHub](https://github.com/priscillalea)
