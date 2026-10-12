# Energy App Review Sentiment Analysis and Rating Prediction

This project analyzes public Google Play reviews from energy-related mobile apps and uses classical NLP models to classify review sentiment and predict star ratings.

I chose energy supplier apps, EV charging apps, mobility apps, and smart home energy services because the reviews are usually very direct. Users often mention problems with login, payments, meter readings, charging sessions, updates, app crashes, and customer support. That makes the dataset useful for practicing text analytics on realistic customer feedback.

The project follows a complete machine learning workflow: collecting reviews, cleaning text, exploring review patterns, extracting NLP features, training baseline, tuned, and experiment models, comparing metrics, and saving the results in reports that can be reviewed later.

---

## What the project covers

The project focuses on two prediction tasks:

1. **Sentiment classification**  
   Predict whether a review is negative, neutral, or positive.

2. **Rating prediction**  
   Predict the original 1-to-5 star rating from the review text.

The sentiment task gives a quick view of customer mood. The rating task is harder because the model has to separate five rating classes, especially the middle ratings where the text is often less clear.

---

## Why I built this

Energy companies depend more and more on digital apps for customer-facing services such as account management, payments, meter readings, EV charging, and support. Public reviews can show where users are satisfied and where they repeatedly face issues.

Instead of reading reviews one by one, this project turns review text into structured outputs:

- sentiment labels
- predicted star ratings
- rating and sentiment distributions
- common positive and negative terms
- model comparison reports
- reusable prediction scripts

The goal was not only to train a model, but to organize the whole project in a way that is easy to review, reproduce, and explain.

---

## Technical scope

The project includes:

- collecting public Google Play review data with Python
- preparing an app shortlist for energy-related services
- cleaning and preprocessing real-world review text
- creating sentiment labels from star ratings
- exploring rating and sentiment distributions
- extracting features with CountVectorizer and TF-IDF
- training Naive Bayes, Logistic Regression, and LinearSVC models
- comparing baseline, tuned, and experiment models
- evaluating models with Accuracy, Macro F1, Weighted F1, and MAE
- saving reports, figures, and a summary Excel workbook
- creating reusable scripts for collection, preprocessing, training, and prediction

---

## Dataset

The reviews were collected from public Google Play Store reviews using the `google-play-scraper` Python package.

After preprocessing, the final dataset used for modeling contained:

| Metric                    | Value |
|---------------------------|-------|
| Total reviews             | 3,084 |
| Number of apps            | 8     |
| Average rating            | 2.99  |
| Average review word count | 22.56 |
| Positive reviews          | 1,436 |
| Negative reviews          | 1,440 |
| Neutral reviews           | 208   |

Raw and processed datasets are not committed to the repository. They are created locally by running the data collection and preprocessing scripts.

---

## Apps included

The dataset includes reviews from apps related to energy supplier services, EV charging, energy mobility, and smart home energy.

| App                  | Type                            |
|----------------------|---------------------------------|
| Octopus Electroverse | EV charging / energy mobility   |
| E.ON Myline          | Energy supplier digital service |
| E.ON Next Home       | Energy supplier digital service |
| PlugShare            | EV charging / mobility          |
| Shell                | Energy / mobility               |
| ScottishPower        | Energy supplier digital service |
| Hive                 | Smart home / energy service     |
| Vattenfall InCharge  | EV charging / energy mobility   |

---

## Labeling approach

The original 1-to-5 star ratings were mapped into sentiment labels for the sentiment classification task.

| Rating | Sentiment |
|--------|-----------|
| 1, 2   | Negative  |
| 3      | Neutral   |
| 4, 5   | Positive  |

For the rating prediction task, the model predicts the original star rating directly:

- 1 star
- 2 stars
- 3 stars
- 4 stars
- 5 stars

Rating prediction is more difficult than sentiment classification because it requires the model to separate five rating levels instead of three broader sentiment groups.

---

## Workflow

The project follows this workflow:

1. Select energy-related mobile apps.
2. Collect public Google Play reviews.
3. Check the collected dataset.
4. Clean and preprocess review text.
5. Create sentiment labels from star ratings.
6. Explore rating and sentiment distributions.
7. Analyze frequent positive and negative words.
8. Extract features using CountVectorizer and TF-IDF.
9. Train baseline models.
10. Tune selected models.
11. Compare model results.
12. Save reports, figures, and classification summaries.
13. Build reusable scripts for training and prediction.
14. Prepare the repository for GitHub review.

---

## Text preprocessing

The preprocessing pipeline includes:

- lowercasing
- URL removal
- special character removal
- extra whitespace cleanup
- stopword removal
- lemmatization
- duplicate review removal
- empty cleaned-text filtering

Both the original review text and cleaned review text are preserved during the workflow, so the cleaning process can be checked later.

---

## Feature extraction

Two feature extraction methods were used.

| Method          | Why it was used                                                             |
|-----------------|-----------------------------------------------------------------------------|
| CountVectorizer | Simple, interpretable, and strong for baseline text classification          |
| TF-IDF          | Useful for weighting terms based on their importance across the dataset     |

Both methods used unigrams and bigrams so the models could learn from single words as well as short phrases.

Examples of useful phrases:

- charging station
- not working
- customer service
- easy use
- payment issue

---

## Models used

The project uses classical machine learning models from scikit-learn.

| Model                   | Why it was used                                                         |
|-------------------------|-------------------------------------------------------------------------|
| Multinomial Naive Bayes | Fast and reliable baseline for text classification                      |
| Logistic Regression     | Strong linear model for sparse text features                            |
| LinearSVC               | Strong margin-based classifier often effective for TF-IDF text features |

The models are built using scikit-learn pipelines, where vectorization and classification are handled together.

---

## Exploratory analysis

The review dataset is strongly polarized. Many reviews are clearly positive or clearly negative, while neutral and middle-rating reviews are less common.

Common positive terms included:

- easy
- good
- great
- charging
- charger
- work
- service
- excellent

Common negative terms included:

- work
- update
- time
- charging
- useless
- card
- issue
- account

A clear pattern is that negative reviews often contain more detailed problem descriptions, especially around login issues, app updates, failed payments, charging sessions, and app reliability.

---

## Evaluation metrics

Different metrics were used because accuracy alone can be misleading on imbalanced data.

| Metric      | Why it matters                                           |
|-------------|----------------------------------------------------------|
| Accuracy    | Shows the overall percentage of correct predictions      |
| Macro F1    | Gives equal importance to each class                     |
| Weighted F1 | Accounts for class imbalance                             |
| MAE         | Measures average rating error for the 1-to-5 rating task |

Macro F1 was especially important because neutral reviews and middle ratings had fewer examples.

---

## Sentiment classification results

The final sentiment model was selected after comparing baseline, tuned, and additional experiment models.

| Stage          | Model                                  | Accuracy | Macro F1 | Weighted F1 |
|----------------|----------------------------------------|----------|----------|-------------|
| Baseline       | CountVectorizer + Naive Bayes          | 0.8039   | 0.5946   | 0.7853      |
| Previous tuned | CountVectorizer + Naive Bayes          | 0.7974   | 0.5983   | 0.7842      |
| Final          | Character TF-IDF + Logistic Regression | 0.7925   | 0.6322   | 0.7963      |

Final selected sentiment model:

| Component      | Value               | 
|----------------|---------------------|
| Feature method | Character TF-IDF    |
| Analyzer       | `char_wb`           |
| N-gram range   | 3 to 5              |
| Model          | Logistic Regression |
| Class weight   | balanced            |

The CountVectorizer + Naive Bayes baseline produced the highest accuracy, but the Character TF-IDF + Logistic Regression model gave better Macro F1 and Weighted F1. I selected the Character TF-IDF model because the dataset is imbalanced, and a small drop in accuracy was acceptable for better class-balanced performance.

Character-level TF-IDF also helps with short app reviews, spelling variations, and repeated issue patterns that may not always appear as clean full words.

The final sentiment model performs well on clearly positive and clearly negative reviews. Neutral reviews remain more difficult because there are fewer neutral examples and many neutral reviews contain mixed feedback.

---

## Rating prediction results

The final rating model was selected after comparing baseline, tuned, and additional experiment models.

| Stage          | Model                                 | Accuracy | Macro F1 | Weighted F1 | MAE    |
|----------------|---------------------------------------|----------|----------|-------------|--------|
| Baseline       | CountVectorizer + Naive Bayes         | 0.6596   | 0.3504   | 0.6122      | 0.8104 |
| Previous tuned | CountVectorizer + Logistic Regression | 0.6321   | 0.3978   | 0.6344      | 0.7115 |
| Final          | TF-IDF + LinearSVC                    | 0.6710   | 0.4008   | 0.6534      | 0.6645 |

Final selected rating model:

| Component      | Value               |
|----------------|---------------------|
| Feature method | TF-IDF              |
| Model          | LinearSVC           |
| C              | 1.0                 |
| Class weight   | balanced            |

The TF-IDF + LinearSVC model was selected as the final rating model because it achieved the best accuracy, Weighted F1, and MAE among the tested models. The MAE improvement is important because star ratings are ordered: predicting 4 instead of 5 is a smaller error than predicting 1 instead of 5.

Rating prediction is still harder than sentiment classification because the model has to separate five rating levels. The model performs better on clear 1-star and 5-star reviews, while 2-star, 3-star, and 4-star reviews are more difficult because they are less frequent and often less clearly worded.

---

## Main findings

- Classical NLP models can separate positive and negative reviews reasonably well.
- Neutral reviews are harder to classify because they are less frequent and often contain mixed language.
- Exact star-rating prediction is harder than sentiment classification.
- Middle ratings are more difficult to predict than extreme ratings.
- Character-level TF-IDF improved the sentiment model by increasing Macro F1 and Weighted F1.
- TF-IDF + LinearSVC improved rating prediction by reducing average rating error.
- Negative reviews often contain more specific issue-related terms than positive reviews.
---

## Limitations

This project has some limitations:

- The dataset represents public mobile app reviewers only, not all customers.
- Reviews are highly polarized toward 1-star and 5-star ratings.
- Neutral and middle-rating classes are underrepresented.
- Some reviews discuss app usability rather than the full energy service experience.
- Star ratings are used as proxy labels for sentiment.
- Google Play reviews can change over time, so future data collection may produce slightly different results.
- The project uses classical machine learning models, not transformer-based language models.

These limitations are expected for a lightweight, reproducible NLP project built with public review data.

---

## Possible improvements

I would try the following improvements in a next version:

- Tune LinearSVC and Logistic Regression hyperparameters more deeply.
- Compare word-level TF-IDF, character-level TF-IDF, and combined feature pipelines.
- Tune vectorizer settings such as `max_features`, `min_df`, `max_df`, and n-gram range.
- Collect more reviews to improve neutral and middle-rating classes.
- Treat rating prediction as an ordinal classification problem instead of a standard multi-class task.
- Compare the classical ML pipeline with a small transformer-based model.

I kept the current version focused on classical machine learning because it is easier to reproduce, faster to run, and more interpretable for a first complete NLP workflow.

---

## Repository contents

| Folder/File        | Purpose                                                                    |
|--------------------|----------------------------------------------------------------------------|
| `config/`          | App shortlist used for review collection                                   |
| `data/`            | Placeholder folders for raw and processed local data                       |
| `docs/`            | Project plan and dataset notes                                             |
| `notebooks/`       | Step-by-step exploration, preprocessing, and modeling notebooks            |
| `reports/`         | CSV reports, figures, and Excel summary workbook                           |
| `src/`             | Reusable Python scripts                                                    |
| `models/`          | Placeholder folder for locally generated trained models                    |
| `README.md`        | Main project documentation                                                 |
| `requirements.txt` | Python dependencies                                                        |
| `config.yaml`      | Project configuration                                                      |
| `.gitignore`       | Excludes generated datasets, models, cache files, and virtual environments |
| `experiments/`     | Additional model comparison experiments                                    |

---

## Project structure

```
energy-app-review-sentiment-rating-prediction/
│
├── config/
│   └── energy_apps_shortlist.csv
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
│
├── docs/
│   ├── dataset_notes.md
│   └── project_plan.md
│
├── experiments/
│   └── model_improvement_experiments.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_text_preprocessing.ipynb
│   └── 03_model_training_evaluation.ipynb
│
├── reports/
│   ├── excel/
│   │   └── energy_app_review_project_summary.xlsx
│   ├── figures/
│   ├── average_rating_by_app.csv
│   ├── classification_report_final_rating_model.csv
│   ├── classification_report_final_sentiment_model.csv
│   ├── eda_summary.csv
│   ├── feature_extraction_summary.csv
│   ├── final_baseline_vs_tuned_model_comparison.csv
│   ├── final_dataset_summary.csv
│   ├── final_rating_baseline_vs_tuned_comparison.csv
│   ├── preprocessing_summary.csv
│   ├── rating_distribution.csv
│   ├── rating_model_results.csv
│   ├── rating_model_tuning_results.csv
│   ├── sentiment_distribution.csv
│   ├── sentiment_model_results.csv
│   ├── sentiment_model_tuning_results.csv
│   ├── top_countvectorizer_features.csv
│   ├── top_negative_words.csv
│   ├── top_positive_words.csv
│   ├── top_tfidf_features.csv
│   ├── final_model_summary.csv
│   └── model_improvement_experiments.csv
│
├── src/
│   ├── __init__.py
│   ├── collect_reviews.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── .gitkeep
│
├── config.yaml
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How to run the project

### 1. Clone the repository

```bash
git clone https://github.com/jay050699/energy-app-review-sentiment-rating-prediction.git
cd energy-app-review-sentiment-rating-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Collect public Google Play reviews

```bash
python src/collect_reviews.py
```

This creates the raw dataset locally:

```
data/raw/energy_app_reviews_raw.csv
```

The raw dataset is not committed to GitHub because it can be recreated using the collection script.

### 5. Preprocess the review dataset

```bash
python src/preprocessing.py
```

This creates the cleaned dataset locally:

```
data/processed/energy_app_reviews_cleaned.csv
```

### 6. Train the final models

```bash
python src/train.py
```

This trains and saves the final models locally:

```
models/final_sentiment_model.pkl
models/final_rating_model.pkl
```

It also creates final reports in the `reports/` folder.

### 7. Run prediction examples

Run the default examples:

```bash
python src/predict.py
```

Run prediction on custom review text:

```bash
python src/predict.py --text "The app keeps crashing and I cannot submit my meter reading."
```

Example output:

```
Predicted sentiment : Negative
Predicted rating    : 1
```

### 8. Run notebooks

The notebooks can be run in this order:

```
01_data_exploration.ipynb
02_text_preprocessing.ipynb
03_model_training_evaluation.ipynb
```

---

## Reports

The `reports/` folder contains:

- dataset summaries
- sentiment and rating distributions
- average rating by app
- feature extraction summaries
- model comparison results
- final classification reports
- figures
- Excel summary workbook

The Excel workbook provides a quick business-friendly summary of the project.

---

## Data and model files

Raw data, processed data, and trained model files are not committed to GitHub.

They are created locally when the scripts are run:

```bash
python src/collect_reviews.py
python src/preprocessing.py
python src/train.py
```

This keeps the repository lightweight and avoids committing generated files that can be reproduced.

---

## Tools and libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Google Play Scraper
- Joblib
- Jupyter Notebook
- Git
- GitHub

---

## Final note

This project is a practical NLP project built around real public review text. It is intentionally kept lightweight and reproducible, using classical machine learning models instead of heavier deep learning models.

The main purpose was to build a complete, reviewable workflow: from collecting raw text to producing model results, reports, figures, and reusable prediction scripts.