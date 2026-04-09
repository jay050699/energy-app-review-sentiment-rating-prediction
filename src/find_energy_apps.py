from google_play_scraper import search


def search_energy_apps():
    """
    Search Google Play Store for energy-sector apps.

    This script helps us find app IDs before collecting reviews.
    We search by app/company name and print the top matching results.
    """

    search_queries = [
        "Octopus Energy",
        "E.ON Next",
        "British Gas",
        "EDF Energy",
        "OVO Energy",
        "Shell Recharge",
        "ChargePoint",
        "Electrify America",
        "Vattenfall InCharge",
        "EnBW mobility+",
    ]

    country = "gb"
    language = "en"

    for query in search_queries:
        print("\n" + "=" * 80)
        print(f"Search query: {query}")
        print("=" * 80)

        try:
            results = search(
                query,
                lang=language,
                country=country,
                n_hits=5
            )

            if not results:
                print("No results found.")
                continue

            for index, app in enumerate(results, start=1):
                title = app.get("title")
                app_id = app.get("appId")
                developer = app.get("developer")
                score = app.get("score")
                installs = app.get("installs")

                print(f"\nResult {index}")
                print(f"Title     : {title}")
                print(f"App ID    : {app_id}")
                print(f"Developer : {developer}")
                print(f"Score     : {score}")
                print(f"Installs  : {installs}")

        except Exception as error:
            print(f"Error while searching for {query}: {error}")


if __name__ == "__main__":
    search_energy_apps()