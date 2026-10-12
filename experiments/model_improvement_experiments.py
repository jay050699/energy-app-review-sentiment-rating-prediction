from pathlib import Path
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


warnings.filterwarnings("ignore", category=ConvergenceWarning)


DATA_PATH = Path("data/processed/energy_app_reviews_cleaned.csv")
OUTPUT_PATH = Path("reports/model_improvement_experiments.csv")


def evaluate_classification(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "weighted_f1": f1_score(y_test, preds, average="weighted"),
    }


def evaluate_rating(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "weighted_f1": f1_score(y_test, preds, average="weighted"),
        "mae": mean_absolute_error(y_test, preds),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_columns = ["clean_review_text", "sentiment", "rating"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.dropna(subset=required_columns).copy()
    df["clean_review_text"] = df["clean_review_text"].astype(str)
    df["rating"] = df["rating"].astype(int)

    print("Dataset shape:", df.shape)
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())
    print("\nRating distribution:")
    print(df["rating"].value_counts().sort_index())

    x = df["clean_review_text"]
    y_sentiment = df["sentiment"]
    y_rating = df["rating"]

    x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(
        x,
        y_sentiment,
        test_size=0.2,
        random_state=42,
        stratify=y_sentiment,
    )

    x_train_r, x_test_r, y_train_r, y_test_r = train_test_split(
        x,
        y_rating,
        test_size=0.2,
        random_state=42,
        stratify=y_rating,
    )

    sentiment_models = {
        "Current config - CountVectorizer + MultinomialNB": Pipeline(
            [
                (
                    "vectorizer",
                    CountVectorizer(
                        max_features=5000,
                        min_df=3,
                        max_df=0.9,
                        ngram_range=(1, 2),
                    ),
                ),
                ("model", MultinomialNB(alpha=0.5)),
            ]
        ),
        "TF-IDF + LinearSVC": Pipeline(
            [
                (
                    "vectorizer",
                    TfidfVectorizer(
                        max_features=10000,
                        min_df=2,
                        max_df=0.95,
                        ngram_range=(1, 2),
                    ),
                ),
                ("model", LinearSVC(C=1.0, class_weight="balanced")),
            ]
        ),
        "CountVectorizer + LinearSVC": Pipeline(
            [
                (
                    "vectorizer",
                    CountVectorizer(
                        max_features=10000,
                        min_df=2,
                        max_df=0.95,
                        ngram_range=(1, 2),
                    ),
                ),
                ("model", LinearSVC(C=1.0, class_weight="balanced")),
            ]
        ),
        "TF-IDF + ComplementNB": Pipeline(
            [
                (
                    "vectorizer",
                    TfidfVectorizer(
                        max_features=10000,
                        min_df=2,
                        max_df=0.95,
                        ngram_range=(1, 2),
                    ),
                ),
                ("model", ComplementNB(alpha=0.5)),
            ]
        ),
        "Character TF-IDF + LogisticRegression": Pipeline(
            [
                (
                    "vectorizer",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=3,
                        max_features=20000,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    rating_models = {
        "Current config - CountVectorizer + LogisticRegression": Pipeline(
            [
                (
                    "vectorizer",
                    CountVectorizer(
                        max_features=5000,
                        min_df=3,
                        max_df=0.9,
                        ngram_range=(1, 2),
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "TF-IDF + LinearSVC": Pipeline(
            [
                (
                    "vectorizer",
                    TfidfVectorizer(
                        max_features=10000,
                        min_df=2,
                        max_df=0.95,
                        ngram_range=(1, 2),
                    ),
                ),
                ("model", LinearSVC(C=1.0, class_weight="balanced")),
            ]
        ),
        "CountVectorizer + LinearSVC": Pipeline(
            [
                (
                    "vectorizer",
                    CountVectorizer(
                        max_features=10000,
                        min_df=2,
                        max_df=0.95,
                        ngram_range=(1, 2),
                    ),
                ),
                ("model", LinearSVC(C=1.0, class_weight="balanced")),
            ]
        ),
        "TF-IDF + LogisticRegression": Pipeline(
            [
                (
                    "vectorizer",
                    TfidfVectorizer(
                        max_features=10000,
                        min_df=2,
                        max_df=0.95,
                        ngram_range=(1, 2),
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Character TF-IDF + LogisticRegression": Pipeline(
            [
                (
                    "vectorizer",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=3,
                        max_features=20000,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "CountVectorizer + ComplementNB": Pipeline(
            [
                (
                    "vectorizer",
                    CountVectorizer(
                        max_features=10000,
                        min_df=2,
                        max_df=0.95,
                        ngram_range=(1, 2),
                    ),
                ),
                ("model", ComplementNB(alpha=0.5)),
            ]
        ),
    }

    results = []

    print("\n=== SENTIMENT EXPERIMENTS ===")
    for name, model in sentiment_models.items():
        metrics = evaluate_classification(model, x_train_s, x_test_s, y_train_s, y_test_s)
        row = {
            "task": "sentiment",
            "model": name,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "mae": None,
        }
        results.append(row)
        print(
            f"{name:50} "
            f"accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"weighted_f1={metrics['weighted_f1']:.4f}"
        )

    print("\n=== RATING EXPERIMENTS ===")
    for name, model in rating_models.items():
        metrics = evaluate_rating(model, x_train_r, x_test_r, y_train_r, y_test_r)
        row = {
            "task": "rating",
            "model": name,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "mae": metrics["mae"],
        }
        results.append(row)
        print(
            f"{name:55} "
            f"accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"weighted_f1={metrics['weighted_f1']:.4f} "
            f"mae={metrics['mae']:.4f}"
        )

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved experiment results to: {OUTPUT_PATH}")

    print("\nBest sentiment by Macro F1:")
    print(
        results_df[results_df["task"] == "sentiment"]
        .sort_values("macro_f1", ascending=False)
        .head(3)
        .to_string(index=False)
    )

    print("\nBest rating by Macro F1:")
    print(
        results_df[results_df["task"] == "rating"]
        .sort_values("macro_f1", ascending=False)
        .head(3)
        .to_string(index=False)
    )

    print("\nBest rating by MAE:")
    print(
        results_df[results_df["task"] == "rating"]
        .sort_values("mae", ascending=True)
        .head(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()