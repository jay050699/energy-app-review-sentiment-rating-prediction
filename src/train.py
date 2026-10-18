"""Train and evaluate text models for energy-app reviews.

The same group-aware train/validation/test partition is used for sentiment
and rating prediction. Model selection uses only validation data; test data
is evaluated once after selection and final training on train + validation.

Sentiment classes are proxies derived from star ratings (1-2 Negative,
3 Neutral, 4-5 Positive), not independently human-annotated sentiment.
"""

from pathlib import Path
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATA_PATH = Path("data/processed/energy_app_reviews_cleaned.csv")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = Path("models")
SEED = 42


def load_dataset():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing {DATA_PATH}; run python src/preprocessing.py first")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    required = {"clean_review_text", "sentiment", "rating"}
    absent = required.difference(df.columns)
    if absent:
        raise ValueError(f"Missing processed dataset columns: {sorted(absent)}")

    df = df.dropna(subset=list(required)).copy()
    df["clean_review_text"] = (
        df["clean_review_text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df = df.loc[df["clean_review_text"].ne("")].reset_index(drop=True)
    df["rating"] = df["rating"].astype(int)

    if not df["rating"].isin([1, 2, 3, 4, 5]).all():
        raise ValueError("Ratings must be integers between 1 and 5")
    if not df["sentiment"].isin(["Negative", "Neutral", "Positive"]).all():
        raise ValueError("Unexpected sentiment labels")
    expected_sentiment = df["rating"].map({
        1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"
    })
    if not (df["sentiment"] == expected_sentiment).all():
        raise ValueError("The sentiment labels are inconsistent with star ratings")

    # Prevent identical processed text from being shared between partitions,
    # even when the original reviews belong to different apps or platforms.
    groups = df["clean_review_text"].str.casefold().to_numpy()
    return df, groups


def make_partitions(df, groups):
    ratings = df["rating"].to_numpy()
    positions = np.arange(len(df))

    # Hold out approximately 20% as an untouched test set.
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    development_idx, test_idx = next(outer.split(positions, ratings, groups))

    # Split the remainder approximately 75/25 = 60% train, 20% validation.
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + 1)
    train_local, validation_local = next(
        inner.split(development_idx, ratings[development_idx], groups[development_idx])
    )
    train_idx = development_idx[train_local]
    validation_idx = development_idx[validation_local]

    group_sets = [set(groups[ix]) for ix in (train_idx, validation_idx, test_idx)]
    if (group_sets[0] & group_sets[1]) or (group_sets[0] & group_sets[2]) or (group_sets[1] & group_sets[2]):
        raise RuntimeError("Review-text leakage detected between partitions")

    for name, idx in (("Train", train_idx), ("Validation", validation_idx), ("Test", test_idx)):
        counts = df.iloc[idx]["rating"].value_counts().sort_index()
        print(f"{name:11}: {len(idx):>6,} reviews | ratings {counts.to_dict()}")
        if len(counts) != 5:
            raise ValueError(f"{name} partition does not contain all five rating classes")

    return train_idx, validation_idx, test_idx


def sentiment_candidates():
    return {
        "CountVectorizer + Naive Bayes": Pipeline([
            ("vectorizer", CountVectorizer(max_features=5000, min_df=3, max_df=0.9, ngram_range=(1, 2))),
            ("model", MultinomialNB()),
        ]),
        "CountVectorizer + Tuned Naive Bayes": Pipeline([
            ("vectorizer", CountVectorizer(max_features=5000, min_df=3, max_df=0.9, ngram_range=(1, 2))),
            ("model", MultinomialNB(alpha=0.5)),
        ]),
        "Character TF-IDF + Logistic Regression": Pipeline([
            ("vectorizer", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=20000)),
            ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=SEED)),
        ]),
        "TF-IDF + LinearSVC": Pipeline([
            ("vectorizer", TfidfVectorizer(max_features=10000, min_df=2, max_df=0.95, ngram_range=(1, 2))),
            ("model", LinearSVC(C=1.0, class_weight="balanced", random_state=SEED)),
        ]),
    }


def rating_candidates():
    return {
        "CountVectorizer + Naive Bayes": Pipeline([
            ("vectorizer", CountVectorizer(max_features=5000, min_df=3, max_df=0.9, ngram_range=(1, 2))),
            ("model", MultinomialNB()),
        ]),
        "CountVectorizer + Logistic Regression": Pipeline([
            ("vectorizer", CountVectorizer(max_features=5000, min_df=3, max_df=0.9, ngram_range=(1, 2))),
            ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=SEED)),
        ]),
        "TF-IDF + LinearSVC": Pipeline([
            ("vectorizer", TfidfVectorizer(max_features=10000, min_df=2, max_df=0.95, ngram_range=(1, 2))),
            ("model", LinearSVC(C=1.0, class_weight="balanced", random_state=SEED)),
        ]),
        "Character TF-IDF + Logistic Regression": Pipeline([
            ("vectorizer", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=20000)),
            ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=SEED)),
        ]),
    }


def metrics(y_true, y_pred, rating=False):
    output = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if rating:
        output["mean_absolute_error"] = mean_absolute_error(y_true, y_pred)
        output["within_one_star_accuracy"] = float(np.mean(np.abs(np.asarray(y_true, dtype=int) - np.asarray(y_pred, dtype=int)) <= 1))
    return output


def choose_on_validation(task, models, X_train, y_train, X_val, y_val, rating=False):
    rows = []
    print(f"\n=== {task.upper()} MODEL SELECTION (VALIDATION SET ONLY) ===", flush=True)

    for i, (name, candidate) in enumerate(models.items()):
        started = time.perf_counter()
        model = clone(candidate)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        result = metrics(y_val, pred, rating=rating)
        record = {
            "stage": "Baseline" if i == 0 else "Candidate",
            "selection_criterion": "Validation Macro F1",
            "model_name": name,
            "accuracy": round(result["accuracy"], 4),
            "macro_precision": round(result["macro_precision"], 4),
            "macro_recall": round(result["macro_recall"], 4),
            "macro_f1": round(result["macro_f1"], 4),
            "weighted_f1": round(result["weighted_f1"], 4),
            "mean_absolute_error": round(result["mean_absolute_error"], 4) if rating else "",
            "best_params": str(candidate.get_params()["model"]),
            "comment": "Validation-only metrics, not held-out test metrics",
        }
        rows.append(record)
        extra = f", MAE={result['mean_absolute_error']:.4f}" if rating else ""
        print(f"{name}: accuracy={result['accuracy']:.4f}, macro_f1={result['macro_f1']:.4f}{extra} ({time.perf_counter() - started:.1f}s)", flush=True)

    # Explicit, predeclared criterion. MAE is only a rating-task tiebreaker.
    selected = max(
        rows,
        key=lambda r: (r["macro_f1"], -float(r["mean_absolute_error"]) if rating else r["weighted_f1"], r["accuracy"]),
    )["model_name"]
    for r in rows:
        if r["model_name"] == selected:
            r["stage"] = "Selected"
            r["comment"] = "Selected by validation Macro F1; final metrics in final_model_summary.csv"
    print(f"Selected {task.lower()} model: {selected}", flush=True)
    return selected, pd.DataFrame(rows)


def save_classification_report(y_true, y_pred, path):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(path)


def save_confusion_matrix(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(title=title, xlabel="Predicted", ylabel="Actual")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > threshold else "black", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df, groups = load_dataset()
    print(f"Loaded {len(df):,} processed reviews from {df['app_name'].nunique() if 'app_name' in df else 'unknown'} apps")
    print("Sentiment labels are derived from ratings, not independently annotated.")
    train_idx, val_idx, test_idx = make_partitions(df, groups)

    X = df["clean_review_text"].to_numpy()
    y_sentiment = df["sentiment"].to_numpy()
    y_rating = df["rating"].to_numpy()
    train_val_idx = np.concatenate([train_idx, val_idx])

    tasks = [
        {
            "name": "Sentiment",
            "target": y_sentiment,
            "candidates": sentiment_candidates(),
            "is_rating": False,
            "report": REPORTS_DIR / "final_baseline_vs_tuned_model_comparison.csv",
            "classification": REPORTS_DIR / "classification_report_final_sentiment_model.csv",
            "confusion": FIGURES_DIR / "confusion_matrix_final_sentiment_model.png",
            "model_path": MODELS_DIR / "final_sentiment_model.pkl",
            "classes": ["Negative", "Neutral", "Positive"],
        },
        {
            "name": "Rating",
            "target": y_rating,
            "candidates": rating_candidates(),
            "is_rating": True,
            "report": REPORTS_DIR / "final_rating_baseline_vs_tuned_comparison.csv",
            "classification": REPORTS_DIR / "classification_report_final_rating_model.csv",
            "confusion": FIGURES_DIR / "confusion_matrix_final_rating_model.png",
            "model_path": MODELS_DIR / "final_rating_model.pkl",
            "classes": [1, 2, 3, 4, 5],
        },
    ]
    final_rows = []

    for task in tasks:
        selected, comparison = choose_on_validation(
            task["name"], task["candidates"],
            X[train_idx], task["target"][train_idx],
            X[val_idx], task["target"][val_idx],
            rating=task["is_rating"],
        )
        comparison.to_csv(task["report"], index=False)
        print(f"\nRefitting {selected} on train + validation...", flush=True)
        started = time.perf_counter()
        final_model = clone(task["candidates"][selected])
        final_model.fit(X[train_val_idx], task["target"][train_val_idx])
        prediction = final_model.predict(X[test_idx])
        final_metrics = metrics(task["target"][test_idx], prediction, rating=task["is_rating"])
        print(f"{task['name']} held-out TEST: {final_metrics} ({time.perf_counter() - started:.1f}s)", flush=True)

        save_classification_report(task["target"][test_idx], prediction, task["classification"])
        save_confusion_matrix(task["target"][test_idx], prediction, task["classes"],
                              f"Final {task['name']} Model — Held-out Test Set", task["confusion"])
        joblib.dump(final_model, task["model_path"])
        final_rows.append({
            "task": task["name"].lower(),
            "final_model": selected,
            "accuracy": round(final_metrics["accuracy"], 4),
            "macro_precision": round(final_metrics["macro_precision"], 4),
            "macro_recall": round(final_metrics["macro_recall"], 4),
            "macro_f1": round(final_metrics["macro_f1"], 4),
            "weighted_f1": round(final_metrics["weighted_f1"], 4),
            "mae": round(final_metrics["mean_absolute_error"], 4) if task["is_rating"] else "",
            "within_one_star_accuracy": round(final_metrics["within_one_star_accuracy"], 4) if task["is_rating"] else "",
            "train_count": len(train_idx),
            "validation_count": len(val_idx),
            "test_count": len(test_idx),
            "split_method": "StratifiedGroupKFold (groups = normalized cleaned review text)",
            "selection_method": "Validation Macro F1; held-out test evaluated after selection",
        })

    pd.DataFrame(final_rows).to_csv(REPORTS_DIR / "final_model_summary.csv", index=False)
    print("\n=== FINAL HELD-OUT TEST SUMMARY ===", flush=True)
    print(pd.DataFrame(final_rows)[["task", "final_model", "accuracy", "macro_f1", "weighted_f1", "mae"]].to_string(index=False))
    print("\nSaved existing model and report filenames. Training complete.", flush=True)


if __name__ == "__main__":
    main()
