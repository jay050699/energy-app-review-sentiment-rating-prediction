import pandas as pd


def check_sample_quality():
    """
    Check the quality of the collected sample review dataset.

    This script helps us decide whether the selected energy-sector apps
    are suitable for the final NLP sentiment analysis and rating prediction project.
    """

    input_path = "data/raw/energy_app_reviews_sample.csv"

    print("=" * 80)
    print("Loading sample dataset")
    print("=" * 80)

    df = pd.read_csv(input_path)

    print(f"Dataset shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df[["app_name", "category", "rating", "review_text"]].head())

    print("\n" + "=" * 80)
    print("Missing values")
    print("=" * 80)
    print(df.isnull().sum())

    print("\n" + "=" * 80)
    print("Review count by app")
    print("=" * 80)
    print(df["app_name"].value_counts())

    print("\n" + "=" * 80)
    print("Rating distribution")
    print("=" * 80)
    print(df["rating"].value_counts().sort_index())

    print("\n" + "=" * 80)
    print("Rating distribution by app")
    print("=" * 80)
    rating_by_app = pd.crosstab(df["app_name"], df["rating"])
    print(rating_by_app)

    print("\n" + "=" * 80)
    print("Duplicate review check")
    print("=" * 80)

    duplicate_count = df.duplicated(subset=["app_name", "review_text"]).sum()
    print(f"Duplicate reviews based on app_name + review_text: {duplicate_count}")

    print("\n" + "=" * 80)
    print("Review length analysis")
    print("=" * 80)

    df["review_text"] = df["review_text"].fillna("")
    df["char_count"] = df["review_text"].apply(len)
    df["word_count"] = df["review_text"].apply(lambda text: len(str(text).split()))

    print("\nCharacter count summary:")
    print(df["char_count"].describe())

    print("\nWord count summary:")
    print(df["word_count"].describe())

    short_reviews = df[df["word_count"] < 3]
    print(f"\nReviews with fewer than 3 words: {len(short_reviews)}")

    print("\n" + "=" * 80)
    print("Average review length by app")
    print("=" * 80)

    length_by_app = df.groupby("app_name")[["char_count", "word_count"]].mean()
    print(length_by_app.sort_values("word_count", ascending=False))

    print("\n" + "=" * 80)
    print("Sample reviews from each app")
    print("=" * 80)

    for app_name in df["app_name"].unique():
        print("\n" + "-" * 80)
        print(f"App: {app_name}")
        print("-" * 80)

        sample_reviews = df[df["app_name"] == app_name][["rating", "review_text"]].head(3)

        for _, row in sample_reviews.iterrows():
            print(f"Rating: {row['rating']}")
            print(f"Review: {row['review_text']}")
            print()

    print("\n" + "=" * 80)
    print("Sample reviews by rating")
    print("=" * 80)

    for rating in sorted(df["rating"].dropna().unique()):
        print("\n" + "-" * 80)
        print(f"Rating: {rating}")
        print("-" * 80)

        sample_reviews = df[df["rating"] == rating][["app_name", "review_text"]].head(3)

        for _, row in sample_reviews.iterrows():
            print(f"App: {row['app_name']}")
            print(f"Review: {row['review_text']}")
            print()

    output_path = "reports/sample_data_quality_summary.csv"

    summary = (
        df.groupby("app_name")
        .agg(
            review_count=("review_text", "count"),
            average_rating=("rating", "mean"),
            average_word_count=("word_count", "mean"),
            min_word_count=("word_count", "min"),
            max_word_count=("word_count", "max"),
        )
        .reset_index()
    )

    summary.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("Data quality summary saved")
    print("=" * 80)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    check_sample_quality()