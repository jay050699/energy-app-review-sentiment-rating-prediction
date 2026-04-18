import pandas as pd
from google_play_scraper import reviews, Sort


def collect_energy_app_reviews():
    """
    Collect Google Play reviews for shortlisted energy-sector apps.

    The app list is stored in:
        config/energy_apps_shortlist.csv

    The collected raw dataset is saved to:
        data/raw/energy_app_reviews_raw.csv

    A summary report is saved to:
        reports/final_dataset_summary.csv
    """

    app_config_path = "config/energy_apps_shortlist.csv"
    output_data_path = "data/raw/energy_app_reviews_raw.csv"
    output_report_path = "reports/final_dataset_summary.csv"

    reviews_per_app = 500

    apps_df = pd.read_csv(app_config_path)

    all_reviews = []

    print("=" * 80)
    print("Starting final energy app review collection")
    print("=" * 80)
    print(f"Apps selected: {len(apps_df)}")
    print(f"Target reviews per app: {reviews_per_app}")

    for _, app in apps_df.iterrows():
        app_name = app["app_name"]
        app_id = app["app_id"]
        category = app["category"]
        country = app["country"]

        print("\n" + "=" * 80)
        print(f"Collecting reviews for: {app_name}")
        print(f"App ID: {app_id}")
        print(f"Category: {category}")
        print(f"Country: {country}")
        print("=" * 80)

        try:
            result, _ = reviews(
                app_id,
                lang="en",
                country=country,
                sort=Sort.NEWEST,
                count=reviews_per_app,
            )

            print(f"Reviews collected: {len(result)}")

            for review in result:
                all_reviews.append(
                    {
                        "app_name": app_name,
                        "app_id": app_id,
                        "category": category,
                        "country": country,
                        "review_text": review.get("content"),
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
    print("Final raw dataset summary")
    print("=" * 80)

    print("Shape:", df.shape)

    if df.empty:
        print("No reviews collected. Please check app IDs and country settings.")
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
        .sort_values("review_count", ascending=False)
    )

    print("\nFinal app-level summary:")
    print(summary)

    df.to_csv(output_data_path, index=False, encoding="utf-8")
    summary.to_csv(output_report_path, index=False, encoding="utf-8")

    print("\n" + "=" * 80)
    print("Files saved")
    print("=" * 80)
    print(f"Raw dataset saved to: {output_data_path}")
    print(f"Dataset summary saved to: {output_report_path}")


if __name__ == "__main__":
    collect_energy_app_reviews()