import argparse
import joblib
import pandas as pd

from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from preprocessing import clean_text, download_nltk_resources


def load_models():
    """
    Load final sentiment and rating prediction models.
    """

    sentiment_model_path = Path("models/final_sentiment_model.pkl")
    rating_model_path = Path("models/final_rating_model.pkl")

    if not sentiment_model_path.exists() or not rating_model_path.exists():
        raise FileNotFoundError(
            "Model files not found. Please run 'python src/train.py' first."
        )

    sentiment_model = joblib.load(sentiment_model_path)
    rating_model = joblib.load(rating_model_path)

    return sentiment_model, rating_model


def prepare_review_text(review_text):
    """
    Clean new review text using the same preprocessing logic.
    """

    download_nltk_resources()

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    return clean_text(review_text, stop_words, lemmatizer)


def predict_review(review_text):
    """
    Predict sentiment and rating for a single review.
    """

    sentiment_model, rating_model = load_models()

    cleaned_text = prepare_review_text(review_text)

    predicted_sentiment = sentiment_model.predict([cleaned_text])[0]
    predicted_rating = rating_model.predict([cleaned_text])[0]

    result = {
        "review_text": review_text,
        "cleaned_text": cleaned_text,
        "predicted_sentiment": predicted_sentiment,
        "predicted_rating": int(predicted_rating),
    }

    return result


def run_examples():
    """
    Run prediction on example energy app reviews.
    """

    example_reviews = [
        "The charging station worked perfectly and the app was very easy to use.",
        "I cannot log in and the payment keeps failing every time.",
        "The service is okay but the app needs some improvements.",
        "Terrible app. Nothing works and customer support never responds.",
        "Excellent experience, very useful and reliable charging service.",
    ]

    predictions = []

    for review in example_reviews:
        predictions.append(predict_review(review))

    predictions_df = pd.DataFrame(predictions)

    print("\nExample predictions:")
    print(predictions_df[["review_text", "predicted_sentiment", "predicted_rating"]])


def main():
    """
    Command-line interface for prediction.
    """

    parser = argparse.ArgumentParser(
        description="Predict sentiment and rating for an energy app review."
    )

    parser.add_argument(
        "--text",
        type=str,
        help="Review text to classify."
    )

    args = parser.parse_args()

    if args.text:
        result = predict_review(args.text)

        print("\nPrediction result")
        print("=" * 80)
        print(f"Review              : {result['review_text']}")
        print(f"Cleaned text        : {result['cleaned_text']}")
        print(f"Predicted sentiment : {result['predicted_sentiment']}")
        print(f"Predicted rating    : {result['predicted_rating']}")
    else:
        run_examples()


if __name__ == "__main__":
    main()