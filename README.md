# Spam Detector

Binary email classification — `spam` vs `ham` — using TF-IDF + classical ML.

## Structure

```
spam-detector/
├── notebooks/
│   └── spam_detector.ipynb   # ← everything lives here
├── models/                   # artefacts saved by the notebook (gitignored)
├── spam_ham_dataset.csv      # dataset (gitignored)
├── requirements.txt
└── .gitignore
```

## Quickstart

```bash
pip install -r requirements.txt
jupyter notebook notebooks/spam_detector.ipynb
```

Run all cells top-to-bottom. The notebook:

1. Loads and explores the dataset (EDA)
2. Preprocesses text (lowercase → symbol removal → stopwords → stemming)
3. Compares 4 classifiers via 5-fold cross-validation
4. Evaluates the winner on the held-out test set
5. Saves the trained pipeline to `models/`
6. Demonstrates inference on new emails

## Dataset

[Spam Mails Dataset](https://www.kaggle.com/datasets/venky73/spam-mails-dataset) — place `spam_ham_dataset.csv` in the project root.
