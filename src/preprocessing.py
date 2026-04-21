import re
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def download_nltk_resources():
    """
    Download required NLTK resources if they are not already available.
    """

    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")


def map_rating_to_sentiment(rating):
    """
    Convert numeric star rating into sentiment label.

    1, 2 -> Negative
    3    -> Neutral
    4, 5 -> Positive
    """

    if rating in [1, 2]:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    elif rating in [4, 5]:
        return "Positive"
    else:
        return "Unknown"


def clean_text(text, stop_words, lemmatizer):
    """
    Clean raw review text for NLP modeling.

    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove non-letter characters
    4. Remove extra spaces
    5. Remove stopwords
    6. Apply lemmatization
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(cleaned_words)


def preprocess_energy_reviews():
    """
    Load raw energy app reviews, clean the text, create sentiment labels,
    and save the processed dataset.
    """

    input_path = "data/raw/energy_app_reviews_raw.csv"
    output_path = "data/processed/energy_app_reviews_cleaned.csv"
    report_path = "reports/preprocessing_summary.csv"

    print("=" * 80)
    print("Loading raw dataset")
    print("=" * 80)

    df = pd.read_csv(input_path)

    print(f"Raw dataset shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # Keep only rows with review text and rating
    df = df.dropna(subset=["review_text", "rating"])

    # Remove duplicate reviews for the same app
    duplicate_count = df.duplicated(subset=["app_name", "review_text"]).sum()
    print(f"\nDuplicate reviews found: {duplicate_count}")

    df = df.drop_duplicates(subset=["app_name", "review_text"])

    # Ensure rating is integer
    df["rating"] = df["rating"].astype(int)

    # Create sentiment labels
    df["sentiment"] = df["rating"].apply(map_rating_to_sentiment)

    # Original text length features
    df["word_count"] = df["review_text"].apply(lambda text: len(str(text).split()))
    df["char_count"] = df["review_text"].apply(lambda text: len(str(text)))

    print("\nDownloading / checking NLTK resources...")
    download_nltk_resources()

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    print("\nCleaning review text...")

    df["clean_review_text"] = df["review_text"].apply(
        lambda text: clean_text(text, stop_words, lemmatizer)
    )

    df["clean_word_count"] = df["clean_review_text"].apply(
        lambda text: len(str(text).split())
    )

    # Remove rows where cleaned text became empty
    before_empty_filter = len(df)
    df = df[df["clean_word_count"] > 0]
    after_empty_filter = len(df)

    removed_empty_rows = before_empty_filter - after_empty_filter

    print(f"\nRows removed because cleaned text was empty: {removed_empty_rows}")

    print("\nProcessed dataset shape:", df.shape)

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    print("\nRating distribution:")
    print(df["rating"].value_counts().sort_index())

    print("\nAverage cleaned word count by sentiment:")
    print(df.groupby("sentiment")["clean_word_count"].mean())

    print("\nFirst 5 cleaned examples:")
    print(df[["rating", "sentiment", "review_text", "clean_review_text"]].head())

    # Save processed dataset
    df.to_csv(output_path, index=False, encoding="utf-8")

    # Save preprocessing summary
    summary = pd.DataFrame(
        {
            "metric": [
                "raw_rows",
                "processed_rows",
                "duplicates_removed",
                "empty_clean_text_rows_removed",
                "negative_reviews",
                "neutral_reviews",
                "positive_reviews",
            ],
            "value": [
                len(pd.read_csv(input_path)),
                len(df),
                duplicate_count,
                removed_empty_rows,
                (df["sentiment"] == "Negative").sum(),
                (df["sentiment"] == "Neutral").sum(),
                (df["sentiment"] == "Positive").sum(),
            ],
        }
    )

    summary.to_csv(report_path, index=False, encoding="utf-8")

    print("\n" + "=" * 80)
    print("Files saved")
    print("=" * 80)
    print(f"Processed dataset saved to: {output_path}")
    print(f"Preprocessing summary saved to: {report_path}")


if __name__ == "__main__":
    preprocess_energy_reviews()