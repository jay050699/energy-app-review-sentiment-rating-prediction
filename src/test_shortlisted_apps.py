import pandas as pd
from google_play_scraper import reviews, Sort


def collect_reviews_for_shortlisted_apps():
    """
    Collect a medium-size test sample from shortlisted energy-sector apps.

    This script checks whether the selected apps are suitable for the final
    Energy App Review Sentiment Analysis and Rating Prediction project.
    """

    app_config_path = "config/energy_apps_shortlist.csv"
    output_data_path = "data/raw/energy_app_reviews_shortlist_test.csv"
    output_report_path = "reports/shortlisted_apps_quality.csv"

    apps_df = pd.read_csv(app_config_path)

    all_reviews = []

    reviews_per_app = 500

    print("=" * 80)
    print("Loading shortlisted energy apps")
    print("=" * 80)
    print(apps_df[["app_name", "app_id", "category", "country"]])

    for _, app in apps_df.iterrows():
        app_name = app["app_name"]
        app_id = app["app_id"]
        category = app["category"]
        country = app["country"]

        print("\n" + "=" * 80)
        print(f"Collecting reviews for: {app_name}")
        print(f"App ID: {app_id}")
        print(f"Country: {country}")
        print("=" * 80)

        try:
            result, continuation_token = reviews(
                app_id,
                lang="en",
                country=country,
                sort=Sort.NEWEST,
                count=reviews_per_app,
            )

            print(f"Reviews collected: {len(result)}")

            for review in result:
                review_text = review.get("content")

                all_reviews.append(
                    {
                        "app_name": app_name,
                        "app_id": app_id,
                        "category": category,
                        "country": country,
                        "review_text": review_text,
                        "rating": review.get("score"),
                        "review_date": review.get("at"),
                        "thumbs_up_count": review.get("thumbsUpCount"),
                        "app_version": review.get("reviewCreatedVersion"),
                    }
                )

        except Exception as error:
            print(f"Error collecting reviews for {app_name}: {error}")

    df = pd.DataFrame(all_reviews)

    print("\n" + "=" * 80)
    print("Shortlist test dataset summary")
    print("=" * 80)

    print("Shape:", df.shape)

    if df.empty:
        print("No reviews collected. Check app IDs or country settings.")
        return

    print("\nReview count by app:")
    print(df["app_name"].value_counts())

    print("\nRating distribution:")
    print(df["rating"].value_counts().sort_index())

    print("\nMissing values:")
    print(df.isnull().sum())

    df["review_text"] = df["review_text"].fillna("")
    df["word_count"] = df["review_text"].apply(lambda text: len(str(text).split()))
    df["char_count"] = df["review_text"].apply(lambda text: len(str(text)))

    print("\nWord count summary:")
    print(df["word_count"].describe())

    print("\nRating distribution by app:")
    rating_by_app = pd.crosstab(df["app_name"], df["rating"])
    print(rating_by_app)

    print("\nAverage review length by app:")
    app_quality = (
        df.groupby("app_name")
        .agg(
            review_count=("review_text", "count"),
            average_rating=("rating", "mean"),
            average_word_count=("word_count", "mean"),
            min_word_count=("word_count", "min"),
            max_word_count=("word_count", "max"),
        )
        .reset_index()
        .sort_values("average_word_count", ascending=False)
    )

    print(app_quality)

    print("\nExample reviews:")
    print(df[["app_name", "rating", "review_text"]].head(10))

    df.to_csv(output_data_path, index=False, encoding="utf-8")
    app_quality.to_csv(output_report_path, index=False, encoding="utf-8")

    print("\n" + "=" * 80)
    print("Files saved")
    print("=" * 80)
    print(f"Shortlist test data saved to: {output_data_path}")
    print(f"Shortlist quality report saved to: {output_report_path}")


if __name__ == "__main__":
    collect_reviews_for_shortlisted_apps()