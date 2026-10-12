from pathlib import Path

import joblib
import pandas as pd
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt


DATA_PATH = Path("data/processed/energy_app_reviews_cleaned.csv")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = Path("models")


def ensure_directories():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {DATA_PATH}. "
            "Run python src/preprocessing.py first."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = ["clean_review_text", "sentiment", "rating"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.dropna(subset=required_columns).copy()
    df["clean_review_text"] = df["clean_review_text"].astype(str)
    df["rating"] = df["rating"].astype(int)

    return df


def evaluate_classification_model(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "predictions": predictions,
        "accuracy": accuracy_score(y_test, predictions),
        "macro_precision": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "weighted_f1": f1_score(y_test, predictions, average="weighted"),
    }


def evaluate_rating_model(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "predictions": predictions,
        "accuracy": accuracy_score(y_test, predictions),
        "macro_precision": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "weighted_f1": f1_score(y_test, predictions, average="weighted"),
        "mean_absolute_error": mean_absolute_error(y_test, predictions),
    }


def save_classification_report(y_true, y_pred, output_path):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_path)


def save_confusion_matrix(y_true, y_pred, labels, title, output_path):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(matrix)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, matrix[i, j], ha="center", va="center")

    fig.colorbar(image)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def results_row(stage, selection_criterion, model_name, metrics, best_params, comment):
    return {
        "stage": stage,
        "selection_criterion": selection_criterion,
        "model_name": model_name,
        "accuracy": round(metrics["accuracy"], 4),
        "macro_precision": round(metrics.get("macro_precision", 0), 4),
        "macro_recall": round(metrics.get("macro_recall", 0), 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "weighted_f1": round(metrics["weighted_f1"], 4),
        "mean_absolute_error": round(metrics.get("mean_absolute_error", 0), 4)
        if "mean_absolute_error" in metrics
        else "",
        "best_params": best_params,
        "comment": comment,
    }


def main():
    ensure_directories()

    df = load_dataset()

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
        "Baseline - CountVectorizer + Naive Bayes": Pipeline(
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
                ("model", MultinomialNB()),
            ]
        ),
        "Tuned - CountVectorizer + Naive Bayes": Pipeline(
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
        "Final - Character TF-IDF + Logistic Regression": Pipeline(
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
        "Baseline - CountVectorizer + Naive Bayes": Pipeline(
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
                ("model", MultinomialNB()),
            ]
        ),
        "Tuned - CountVectorizer + Logistic Regression": Pipeline(
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
        "Final - TF-IDF + LinearSVC": Pipeline(
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
    }

    print("\n=== SENTIMENT MODELS ===")
    sentiment_results = {}

    for name, model in sentiment_models.items():
        metrics = evaluate_classification_model(
            model, x_train_s, x_test_s, y_train_s, y_test_s
        )
        sentiment_results[name] = {"model": model, "metrics": metrics}

        print(
            f"{name:50} "
            f"accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"weighted_f1={metrics['weighted_f1']:.4f}"
        )

    print("\n=== RATING MODELS ===")
    rating_results = {}

    for name, model in rating_models.items():
        metrics = evaluate_rating_model(model, x_train_r, x_test_r, y_train_r, y_test_r)
        rating_results[name] = {"model": model, "metrics": metrics}

        print(
            f"{name:55} "
            f"accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"weighted_f1={metrics['weighted_f1']:.4f} "
            f"mae={metrics['mean_absolute_error']:.4f}"
        )

    final_sentiment_name = "Final - Character TF-IDF + Logistic Regression"
    final_rating_name = "Final - TF-IDF + LinearSVC"

    final_sentiment_model = sentiment_results[final_sentiment_name]["model"]
    final_sentiment_metrics = sentiment_results[final_sentiment_name]["metrics"]

    final_rating_model = rating_results[final_rating_name]["model"]
    final_rating_metrics = rating_results[final_rating_name]["metrics"]

    save_classification_report(
        y_test_s,
        final_sentiment_metrics["predictions"],
        REPORTS_DIR / "classification_report_final_sentiment_model.csv",
    )

    save_classification_report(
        y_test_r,
        final_rating_metrics["predictions"],
        REPORTS_DIR / "classification_report_final_rating_model.csv",
    )

    save_confusion_matrix(
        y_test_s,
        final_sentiment_metrics["predictions"],
        labels=["Negative", "Neutral", "Positive"],
        title="Final Sentiment Model Confusion Matrix",
        output_path=FIGURES_DIR / "confusion_matrix_final_sentiment_model.png",
    )

    save_confusion_matrix(
        y_test_r,
        final_rating_metrics["predictions"],
        labels=[1, 2, 3, 4, 5],
        title="Final Rating Model Confusion Matrix",
        output_path=FIGURES_DIR / "confusion_matrix_final_rating_model.png",
    )

    sentiment_comparison = pd.DataFrame(
        [
            results_row(
                "Baseline",
                "Reference baseline",
                "CountVectorizer + Naive Bayes",
                sentiment_results["Baseline - CountVectorizer + Naive Bayes"]["metrics"],
                "Default settings",
                "Simple baseline model for sentiment classification.",
            ),
            results_row(
                "Tuned",
                "Previous selected model",
                "CountVectorizer + Naive Bayes",
                sentiment_results["Tuned - CountVectorizer + Naive Bayes"]["metrics"],
                "alpha = 0.5",
                "Previous final model before additional experiments.",
            ),
            results_row(
                "Final",
                "Best Macro F1 and Weighted F1",
                "Character TF-IDF + Logistic Regression",
                final_sentiment_metrics,
                "char_wb ngrams 3-5, max_features = 20000, class_weight = balanced",
                "Selected because it improved class-balanced sentiment performance.",
            ),
        ]
    )

    rating_comparison = pd.DataFrame(
        [
            results_row(
                "Baseline",
                "Reference baseline",
                "CountVectorizer + Naive Bayes",
                rating_results["Baseline - CountVectorizer + Naive Bayes"]["metrics"],
                "Default settings",
                "Simple baseline model for exact rating prediction.",
            ),
            results_row(
                "Tuned",
                "Previous selected model",
                "CountVectorizer + Logistic Regression",
                rating_results[
                    "Tuned - CountVectorizer + Logistic Regression"
                ]["metrics"],
                "C = 1.0, class_weight = balanced",
                "Previous final model before additional experiments.",
            ),
            results_row(
                "Final",
                "Best accuracy, weighted F1, and MAE",
                "TF-IDF + LinearSVC",
                final_rating_metrics,
                "C = 1.0, class_weight = balanced",
                "Selected because it reduced average star-rating error.",
            ),
        ]
    )

    sentiment_comparison.to_csv(
        REPORTS_DIR / "final_baseline_vs_tuned_model_comparison.csv", index=False
    )
    rating_comparison.to_csv(
        REPORTS_DIR / "final_rating_baseline_vs_tuned_comparison.csv", index=False
    )

    pd.DataFrame(
        [
            {
                "task": "sentiment",
                "final_model": "Character TF-IDF + Logistic Regression",
                "accuracy": round(final_sentiment_metrics["accuracy"], 4),
                "macro_f1": round(final_sentiment_metrics["macro_f1"], 4),
                "weighted_f1": round(final_sentiment_metrics["weighted_f1"], 4),
                "mae": "",
            },
            {
                "task": "rating",
                "final_model": "TF-IDF + LinearSVC",
                "accuracy": round(final_rating_metrics["accuracy"], 4),
                "macro_f1": round(final_rating_metrics["macro_f1"], 4),
                "weighted_f1": round(final_rating_metrics["weighted_f1"], 4),
                "mae": round(final_rating_metrics["mean_absolute_error"], 4),
            },
        ]
    ).to_csv(REPORTS_DIR / "final_model_summary.csv", index=False)

    joblib.dump(final_sentiment_model, MODELS_DIR / "final_sentiment_model.pkl")
    joblib.dump(final_rating_model, MODELS_DIR / "final_rating_model.pkl")

    print("\nFinal sentiment model:")
    print(final_sentiment_name)
    print(
        {
            "accuracy": round(final_sentiment_metrics["accuracy"], 4),
            "macro_f1": round(final_sentiment_metrics["macro_f1"], 4),
            "weighted_f1": round(final_sentiment_metrics["weighted_f1"], 4),
        }
    )

    print("\nFinal rating model:")
    print(final_rating_name)
    print(
        {
            "accuracy": round(final_rating_metrics["accuracy"], 4),
            "macro_f1": round(final_rating_metrics["macro_f1"], 4),
            "weighted_f1": round(final_rating_metrics["weighted_f1"], 4),
            "mae": round(final_rating_metrics["mean_absolute_error"], 4),
        }
    )

    print("\nReports and final models saved successfully.")


if __name__ == "__main__":
    main()