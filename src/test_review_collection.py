import pandas as pd
from google_play_scraper import reviews, Sort


def collect_sample_reviews():
    """
    Collect a small sample of Google Play reviews from selected energy-sector apps.

    This is only a feasibility test.
    The goal is to check whether the app reviews are useful for NLP sentiment analysis
    and rating prediction before collecting the full dataset.
    """

    apps = [
        {
            "app_name": "Octopus Electroverse",
            "app_id": "energy.octopus.electricjuice.android",
            "category": "EV Charging / Energy Mobility",
            "country": "gb",
        },
        {
            "app_name": "E.ON Next Home",
            "app_id": "com.eonnext.home.eu",
            "category": "Energy Supplier",
            "country": "gb",
        },
        {
            "app_name": "E.ON Myline",
            "app_id": "ro.eon.myline",
            "category": "Energy Supplier",
            "country": "ro",
        },
        {
            "app_name": "Shell",
            "app_id": "com.shell.sitibv.retail",
            "category": "Energy / Mobility",
            "country": "gb",
        },
        {
            "app_name": "PlugShare",
            "app_id": "com.xatori.Plugshare",
            "category": "EV Charging / Mobility",
            "country": "gb",
        },
    ]

    all_reviews = []

    for app in apps:
        print("\n" + "=" * 80)
        print(f"Collecting reviews for: {app['app_name']}")
        print(f"App ID: {app['app_id']}")
        print("=" * 80)

        try:
            result, continuation_token = reviews(
                app["app_id"],
                lang="en",
                country=app["country"],
                sort=Sort.NEWEST,
                count=20,
            )

            print(f"Reviews collected: {len(result)}")

            for review in result:
                all_reviews.append(
                    {
                        "app_name": app["app_name"],
                        "app_id": app["app_id"],
                        "category": app["category"],
                        "review_text": review.get("content"),
                        "rating": review.get("score"),
                        "review_date": review.get("at"),
                        "thumbs_up_count": review.get("thumbsUpCount"),
                        "app_version": review.get("reviewCreatedVersion"),
                    }
                )

        except Exception as error:
            print(f"Error collecting reviews for {app['app_name']}: {error}")

    df = pd.DataFrame(all_reviews)

    print("\n" + "=" * 80)
    print("Final sample dataset summary")
    print("=" * 80)

    print("Shape:", df.shape)

    if not df.empty:
        print("\nColumns:")
        print(df.columns.tolist())

        print("\nReview count by app:")
        print(df["app_name"].value_counts())

        print("\nRating distribution:")
        print(df["rating"].value_counts().sort_index())

        print("\nFirst 5 rows:")
        print(df[["app_name", "rating", "review_text"]].head())

        output_path = "data/raw/energy_app_reviews_sample.csv"
        df.to_csv(output_path, index=False, encoding="utf-8")

        print(f"\nSample dataset saved to: {output_path}")
    else:
        print("No reviews were collected.")


if __name__ == "__main__":
    collect_sample_reviews()