# Dataset notes

## Source and scope

The dataset contains publicly available Google Play and Apple App Store reviews for 19 energy-related applications, covering utilities, EV charging, mobility, and smart-home energy services. Reviews are retrieved across multiple storefronts, with language acceptance based on review text rather than country alone.

## Quality controls

- Raw English-filtered dataset: **71,067** reviews, 17 columns.
- English confidence threshold: **0.92** (language detection is approximate; short/ambiguous texts can still be misclassified).
- Collector minimum text length: **5 words** for language evaluation.
- Stable review IDs used for deduplication within each app/platform across storefronts.
- During preprocessing, **72 rows** belonging to same-app identical-text, conflicting-rating groups were excluded; **one** cleaned review was empty and excluded.
- Final modeling dataset: **70,994** reviews from 19 applications.
- Raw platform counts: Google Play **60,839**; Apple App Store **10,228**.

## Labels

1–2 stars → Negative; 3 stars → Neutral; 4–5 stars → Positive. This is a **rating-derived proxy**, not independently annotated sentiment ground truth.

## Final rating counts (processed)

| Rating | Count |
| --- | --- |
| 1 | 29951 |
| 2 | 6383 |
| 3 | 5539 |
| 4 | 6354 |
| 5 | 22767 |

## Data access and reproducibility

- `data/raw/energy_app_reviews_raw.csv` and `data/processed/energy_app_reviews_cleaned.csv` are generated locally and not tracked in Git.
- `src/collect_reviews.py` collects and filters review candidates; `src/preprocessing.py` creates the modeling dataset.
- Collection is a snapshot of public app-store availability. Subsequent collection may produce different content and counts.
- The complete source review text should not be redistributed as a public repo dataset without checking applicable terms and rights.
