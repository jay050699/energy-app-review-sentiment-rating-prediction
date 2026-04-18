# Dataset Notes

## Final Dataset Choice

The final dataset used in this project is a custom-collected public Google Play review dataset from energy-sector mobile apps.

The dataset focuses on customer reviews from apps related to:

- Energy supplier digital services
- EV charging and mobility
- Smart energy and home energy services

## Project Dataset Name

Energy App Reviews Dataset

## Data Source

The review data is collected from public Google Play Store app reviews using the `google-play-scraper` Python package.

## Selected Apps

| App Name | Category |
|---|---|
| Octopus Electroverse | EV Charging / Energy Mobility |
| E.ON Next Home | Energy Supplier Digital Service |
| E.ON Myline | Energy Supplier Digital Service |
| PlugShare | EV Charging / Mobility |
| Shell | Energy / Mobility |
| ScottishPower | Energy Supplier Digital Service |
| Hive | Smart Energy / Home Energy |
| Vattenfall InCharge | EV Charging / Energy Mobility |

## Dataset Columns

| Column | Description |
|---|---|
| app_name | Name of the mobile application |
| app_id | Google Play app package ID |
| category | Energy-sector category |
| country | Country used for review collection |
| review_text | Written user review |
| rating | Star rating from 1 to 5 |
| review_date | Review timestamp |
| thumbs_up_count | Number of helpful votes |
| app_version | App version associated with the review, if available |

## Target Variables

This project uses the rating column in two ways:

### Sentiment Classification

Ratings are mapped into sentiment classes:

| Rating | Sentiment |
|---|---|
| 1, 2 | Negative |
| 3 | Neutral |
| 4, 5 | Positive |

### Rating Prediction

The project also predicts the original 1 to 5 star rating.

## Dataset Limitations

- The dataset represents app users only, not all energy customers.
- Reviews are public Google Play reviews and may be biased toward very positive or very negative experiences.
- Some apps have fewer available reviews than others.
- Some app reviews may discuss app usability rather than the full energy service experience.
- Star ratings are used as proxy labels for sentiment.

## Reproducibility

The raw dataset is not committed to GitHub because raw review data can be large and source-dependent.

The dataset can be recreated by running:

```bash
python src/collect_reviews.py