"""Preprocess English-language energy app reviews for text classification.

Reads the existing raw dataset and writes the existing processed dataset and
preprocessing summary. Source-platform and review metadata are preserved.
"""

import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


RAW_PATH = Path("data/raw/energy_app_reviews_raw.csv")
PROCESSED_PATH = Path("data/processed/energy_app_reviews_cleaned.csv")
SUMMARY_PATH = Path("reports/preprocessing_summary.csv")
ENGLISH_THRESHOLD = 0.92


def download_nltk_resources():
    """Download the resources used by the existing project when necessary."""
    for resource in ("stopwords", "wordnet", "omw-1.4"):
        nltk.download(resource, quiet=True)


def map_rating_to_sentiment(rating):
    """Map 1–2 stars to Negative, 3 to Neutral, and 4–5 to Positive."""
    if rating in (1, 2):
        return "Negative"
    if rating == 3:
        return "Neutral"
    if rating in (4, 5):
        return "Positive"
    return "Unknown"


def clean_text(text, stop_words, lemmatizer):
    """Normalize review text while preserving explicit negation."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\bcan['’]t\b", "can not", text)
    text = re.sub(r"\bwon['’]t\b", "will not", text)
    text = re.sub(r"\b(\w+)n['’]t\b", r"\1 not", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return " ".join(
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    )


def preprocess_energy_reviews():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset missing: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH, low_memory=False)
    raw_rows = len(df)
    required = {
        "app_name", "review_text", "rating", "platform", "review_id",
        "language", "language_probability",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Expected the English-only "
            "Google Play / Apple review collection."
        )

    print(f"Raw dataset: {raw_rows:,} rows, {len(df.columns)} columns")
    df = df.dropna(subset=["app_name", "review_text", "rating", "platform", "review_id"]).copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    before_rating = len(df)
    df = df[df["rating"].isin((1, 2, 3, 4, 5))].copy()
    invalid_rating_rows = before_rating - len(df)
    df["rating"] = df["rating"].astype(int)

    # The collector has already filtered English reviews; verify this again.
    probability = pd.to_numeric(df["language_probability"], errors="coerce")
    english_mask = (
        df["language"].astype(str).str.casefold().eq("en")
        & probability.ge(ENGLISH_THRESHOLD)
    )
    non_english_rows = int((~english_mask).sum())
    df = df.loc[english_mask].copy()

    # Repeated wording with different star ratings is ambiguous for text models.
    # Drop every member of conflicting groups, instead of retaining a random label.
    df["_text_key"] = (
        df["review_text"].astype(str).str.casefold()
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )
    duplicate_keys = ["app_name", "_text_key"]
    conflicting_mask = df.groupby(duplicate_keys)["rating"].transform("nunique").gt(1)
    conflicting_rows = int(conflicting_mask.sum())
    df = df.loc[~conflicting_mask].copy()

    same_label_duplicate_rows = int(
        df.duplicated(subset=duplicate_keys + ["rating"]).sum()
    )
    df = df.drop_duplicates(subset=duplicate_keys + ["rating"]).copy()
    df.drop(columns=["_text_key"], inplace=True)

    df["sentiment"] = df["rating"].map(map_rating_to_sentiment)
    df["word_count"] = df["review_text"].astype(str).str.split().str.len()
    df["char_count"] = df["review_text"].astype(str).str.len()

    download_nltk_resources()
    stop_words = set(stopwords.words("english")) - {"no", "nor", "not", "never"}
    lemmatizer = WordNetLemmatizer()
    print("Cleaning text (this may take a few minutes)...")
    df["clean_review_text"] = df["review_text"].apply(
        lambda t: clean_text(t, stop_words, lemmatizer)
    )
    df["clean_word_count"] = df["clean_review_text"].str.split().str.len()
    before_empty = len(df)
    df = df[df["clean_word_count"].gt(0)].copy()
    empty_clean_rows = before_empty - len(df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False, encoding="utf-8")

    summary = {
        "raw_rows": raw_rows,
        "invalid_rating_rows_removed": invalid_rating_rows,
        "non_english_rows_removed": non_english_rows,
        "conflicting_rating_rows_removed": conflicting_rows,
        "duplicates_removed": same_label_duplicate_rows,
        "empty_clean_text_rows_removed": empty_clean_rows,
        "processed_rows": len(df),
        "negative_reviews": int(df["sentiment"].eq("Negative").sum()),
        "neutral_reviews": int(df["sentiment"].eq("Neutral").sum()),
        "positive_reviews": int(df["sentiment"].eq("Positive").sum()),
        "google_play_reviews": int(df["platform"].eq("google_play").sum()),
        "apple_app_store_reviews": int(df["platform"].eq("apple_app_store").sum()),
        "unique_apps": int(df["app_name"].nunique()),
    }
    pd.DataFrame(summary.items(), columns=["metric", "value"]).to_csv(
        SUMMARY_PATH, index=False, encoding="utf-8"
    )

    print("\n=== PREPROCESSING RESULTS ===")
    print(f"Raw reviews:                  {raw_rows:,}")
    print(f"Non-English rows removed:     {non_english_rows:,}")
    print(f"Conflicting-rating rows:      {conflicting_rows:,}")
    print(f"Same-label duplicates:        {same_label_duplicate_rows:,}")
    print(f"Empty cleaned text removed:   {empty_clean_rows:,}")
    print(f"Final processed reviews:      {len(df):,}")
    print("\nRating distribution:")
    print(df["rating"].value_counts().sort_index().to_string())
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts().to_string())
    print(f"\nSaved: {PROCESSED_PATH}")
    print(f"Saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    preprocess_energy_reviews()
