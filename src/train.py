import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from data_loader import load_processed_reviews
from evaluate import (
    evaluate_classification_model,
    evaluate_rating_model,
    save_classification_report,
    save_confusion_matrix,
)


def train_final_models():
    """
    Train final sentiment classification and rating prediction models.

    Final sentiment model:
        CountVectorizer + Multinomial Naive Bayes, alpha=0.5

    Final rating model:
        CountVectorizer + Logistic Regression, C=1.0, class_weight='balanced'
    """

    reports_dir = Path("reports")
    figures_dir = Path("reports/figures")
    models_dir = Path("models")

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Loading processed dataset")
    print("=" * 80)

    df = load_processed_reviews()

    print(f"Dataset shape: {df.shape}")

    X = df["clean_review_text"]
    y_sentiment = df["sentiment"]
    y_rating = df["rating"]

    custom_stopwords = [
        "app",
        "apps",
        "nu",
        "sa",
        "la",
        "si",
        "de",
    ]

    # ------------------------------------------------------------------
    # Sentiment Classification
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("Training final sentiment model")
    print("=" * 80)

    X_train_sentiment, X_test_sentiment, y_train_sentiment, y_test_sentiment = train_test_split(
        X,
        y_sentiment,
        test_size=0.2,
        random_state=42,
        stratify=y_sentiment,
    )

    sentiment_model = Pipeline(
        steps=[
            (
                "vectorizer",
                CountVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.90,
                    stop_words=custom_stopwords,
                    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
                ),
            ),
            (
                "model",
                MultinomialNB(alpha=0.5),
            ),
        ]
    )

    sentiment_model.fit(X_train_sentiment, y_train_sentiment)

    y_pred_sentiment = sentiment_model.predict(X_test_sentiment)

    sentiment_results = evaluate_classification_model(
        "Final Sentiment Model - CountVectorizer + Naive Bayes",
        y_test_sentiment,
        y_pred_sentiment,
    )

    print("\nFinal sentiment model results:")
    print(sentiment_results)

    save_classification_report(
        y_test_sentiment,
        y_pred_sentiment,
        reports_dir / "script_classification_report_final_sentiment_model.csv",
    )

    save_confusion_matrix(
        y_test_sentiment,
        y_pred_sentiment,
        labels=["Negative", "Neutral", "Positive"],
        title="Confusion Matrix - Final Sentiment Model",
        output_path=figures_dir / "script_confusion_matrix_final_sentiment_model.png",
    )

    joblib.dump(
        sentiment_model,
        models_dir / "final_sentiment_model.pkl"
    )

    # ------------------------------------------------------------------
    # Rating Prediction
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("Training final rating prediction model")
    print("=" * 80)

    X_train_rating, X_test_rating, y_train_rating, y_test_rating = train_test_split(
        X,
        y_rating,
        test_size=0.2,
        random_state=42,
        stratify=y_rating,
    )

    rating_model = Pipeline(
        steps=[
            (
                "vectorizer",
                CountVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.90,
                    stop_words=custom_stopwords,
                    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
                ),
            ),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    rating_model.fit(X_train_rating, y_train_rating)

    y_pred_rating = rating_model.predict(X_test_rating)

    rating_results = evaluate_rating_model(
        "Final Rating Model - CountVectorizer + Logistic Regression",
        y_test_rating,
        y_pred_rating,
    )

    print("\nFinal rating model results:")
    print(rating_results)

    save_classification_report(
        y_test_rating,
        y_pred_rating,
        reports_dir / "script_classification_report_final_rating_model.csv",
    )

    save_confusion_matrix(
        y_test_rating,
        y_pred_rating,
        labels=[1, 2, 3, 4, 5],
        title="Confusion Matrix - Final Rating Model",
        output_path=figures_dir / "script_confusion_matrix_final_rating_model.png",
    )

    joblib.dump(
        rating_model,
        models_dir / "final_rating_model.pkl"
    )

    # ------------------------------------------------------------------
    # Save combined script result summary
    # ------------------------------------------------------------------

    results_df = pd.DataFrame(
        [
            sentiment_results,
            rating_results,
        ]
    )

    results_df.to_csv(
        reports_dir / "script_final_model_results.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("Training completed")
    print("=" * 80)
    print("Saved final sentiment model to: models/final_sentiment_model.pkl")
    print("Saved final rating model to: models/final_rating_model.pkl")
    print("Saved script result summary to: reports/script_final_model_results.csv")


if __name__ == "__main__":
    train_final_models()