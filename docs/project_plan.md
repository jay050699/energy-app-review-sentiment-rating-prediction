# Project plan

## Objective

Explore what text analytics can learn from energy-app customer feedback by classifying review sentiment and predicting the reviewer's 1–5 star rating.

## Scope

- Data: English-language public app-store reviews from 19 energy-related applications.
- Outputs: reusable Python scripts, exploration/feature notebooks, model evaluation, report CSVs, figures, and Excel summary.
- Tasks: three-class sentiment proxy from star ratings and five-class rating prediction.

## Method

1. Retrieve Google Play and Apple App Store reviews and associated metadata.
2. Filter unusable and uncertain-language content and deduplicate records.
3. Prepare cleaned review text and rating-derived sentiment labels.
4. Explore rating balance, app coverage, review lengths, and common terms.
5. Build a shared text-group-aware training/validation/test split.
6. Compare candidate pipelines on validation Macro F1; refit the selected models and evaluate once on held-out test data.
7. Save reusable models locally, publish aggregate evaluation reports and reproducible code.

## Current verified run

- Raw reviews: **71,067**
- Cleaned reviews: **70,994**
- Train / validation / test: **42,596 / 14,199 / 14,199**
- Sentiment model: **TF-IDF + LinearSVC**, accuracy **0.8083**, Macro F1 **0.6583**.
- Rating model: **CountVectorizer + Naive Bayes**, accuracy **0.6766**, Macro F1 **0.4800**, MAE **0.5650**.

## Limitations

Sentiment comes from ratings rather than independent annotations. App-store reviewers are not a representative sample. Identical-text grouping prevents only exact normalized review overlap, not all semantic near-duplicates. The work is an independent ML demonstration, not a deployed production system.
