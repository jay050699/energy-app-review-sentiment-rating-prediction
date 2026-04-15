import pandas as pd
from google_play_scraper import search


def search_energy_apps():
    """
    Search Google Play Store for energy-sector apps and save candidate results.

    This helps us identify valid app IDs before collecting the final review dataset.
    """

    search_queries = [
        # Energy suppliers
        "Octopus Energy",
        "E.ON Next",
        "British Gas",
        "EDF Energy",
        "OVO Energy",
        "Scottish Power",
        "Vattenfall",
        "EnBW",
        "E.ON Myline",

        # EV charging / mobility
        "Octopus Electroverse",
        "Shell Recharge",
        "PlugShare",
        "ChargePoint",
        "Electrify America",
        "EVgo",
        "Ionity",
        "BP Pulse",
        "Tesla",
    ]

    countries = ["gb", "us", "de"]
    language = "en"

    all_results = []

    for country in countries:
        print("\n" + "#" * 100)
        print(f"Searching country: {country}")
        print("#" * 100)

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
                    row = {
                        "country": country,
                        "search_query": query,
                        "result_rank": index,
                        "title": app.get("title"),
                        "app_id": app.get("appId"),
                        "developer": app.get("developer"),
                        "score": app.get("score"),
                        "installs": app.get("installs"),
                    }

                    all_results.append(row)

                    print(f"\nResult {index}")
                    print(f"Title     : {row['title']}")
                    print(f"App ID    : {row['app_id']}")
                    print(f"Developer : {row['developer']}")
                    print(f"Score     : {row['score']}")
                    print(f"Installs  : {row['installs']}")

            except Exception as error:
                print(f"Error while searching for {query}: {error}")

    df = pd.DataFrame(all_results)

    output_path = "reports/app_search_results.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    print("\n" + "=" * 80)
    print("App search results saved")
    print("=" * 80)
    print(f"Saved to: {output_path}")
    print(f"Total candidate rows: {len(df)}")


if __name__ == "__main__":
    search_energy_apps()