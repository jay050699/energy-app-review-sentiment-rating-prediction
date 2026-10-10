# Energy App Review Sentiment Analysis and Rating Prediction

## Project Overview

This project analyzes public Google Play reviews from energy-related mobile apps using a classical NLP and machine learning pipeline.

The dataset contains Google Play Store reviews from energy supplier apps, EV charging apps, mobility apps, and smart home energy services. The goal is to classify customer sentiment and predict star ratings based on review text.

The main idea is to show how app reviews can be turned into useful signals about customer satisfaction, complaints, and product issues.

---

## Business Motivation

Energy companies increasingly provide digital services through mobile applications. Customers use these apps for tasks such as checking energy usage, submitting meter readings, managing accounts, making payments, finding EV charging stations, and reporting technical issues.

Instead of reading reviews manually, this project uses NLP models to group reviews by sentiment and estimate the likely star rating from the text.

---

## Key Skills Demonstrated

* Collecting public review data using Python
* Cleaning and preprocessing text data
* Creating sentiment labels from star ratings
* Building scikit-learn NLP pipelines
* Comparing baseline and tuned machine learning models
* Evaluating classification models using Accuracy, Macro F1, Weighted F1, and MAE
* Saving trained models with Joblib
* Creating reusable scripts for training and prediction
* Organizing a machine learning project for GitHub

---

## Dataset

The dataset was collected from public Google Play Store reviews using the `google-play-scraper` Python package.

After preprocessing, the dataset used for training and evaluation contained:
 
| Metric                    | Value |
|---------------------------|-------|
| Total reviews             | 3,070 |
| Number of apps            |     8 |
| Average rating            |  2.97 |
| Average review word count | 22.57 |
| Positive reviews          | 1,411 |
| Negative reviews          | 1,439 |
| Neutral reviews           |   220 |

---

## Selected Apps

The dataset includes reviews from apps related to energy supplier services, EV charging, energy mobility, and smart home energy.

Example apps include:

* Octopus Electroverse
* E.ON Myline
* E.ON Next Home
* PlugShare
* Shell
* ScottishPower
* Hive
* Vattenfall InCharge

---

## Project Tasks

The project focuses on two prediction tasks.

### 1. Sentiment Classification

The original 1-to-5 star ratings were mapped into three sentiment classes:

| Rating | Sentiment |  
|--------|-----------|
| 1, 2   | Negative  |
| 3      | Neutral   |
| 4, 5   | Positive  |

### 2. Rating Prediction

The second task predicts the original star rating:

```
1, 2, 3, 4, or 5 stars
```

Rating prediction is more difficult than sentiment classification because the model must distinguish between five rating levels instead of three sentiment classes.

---

## Project Workflow

The project follows this end-to-end workflow:

1. Define energy-sector app list
2. Collect public app reviews
3. Perform data quality checks
4. Clean and preprocess review text
5. Create sentiment labels from ratings
6. Perform exploratory data analysis
7. Extract text features using CountVectorizer and TF-IDF
8. Train baseline sentiment classification models
9. Tune sentiment classification models
10. Train baseline rating prediction models
11. Tune rating prediction models
12. Compare model results
13. Save evaluation reports and visualizations
14. Create reusable training and prediction scripts

---

## Text Preprocessing

The review text was cleaned using the following steps:

* Lowercasing
* URL removal
* Special character removal
* Extra whitespace removal
* Stopword removal
* Lemmatization
* Duplicate review removal
* Empty cleaned-text filtering

Both the original review text and cleaned review text were preserved.

---

## Feature Extraction

Two classical NLP feature extraction methods were used.

### CountVectorizer

CountVectorizer converts text into word and phrase count features. It is simple, interpretable, and works well as a baseline for text classification tasks.

### TF-IDF Vectorizer

TF-IDF gives higher importance to terms that are frequent in a review but not too common across all reviews. It is commonly used with linear models such as Logistic Regression.

Both methods used unigrams and bigrams to capture words and short phrases such as:

```
charging station
not working
customer service
easy use
payment issue
```

---

## Models Used

The project compares the following classical machine learning models:

* Multinomial Naive Bayes
* Logistic Regression

Model pipelines combine text vectorization and classification in a single reusable workflow.

---

## Exploratory Data Analysis Insights

The review dataset showed strong polarization between positive and negative feedback.

Common positive terms included:

```
easy
good
great
charging
charger
work
service
excellent
```

Common negative terms included:

```
work
update
time
charging
useless
card
issue
account
```

Negative reviews often contained more detailed explanations of technical problems, login issues, payment failures, or charging difficulties.

---

## Sentiment Classification Results

The final sentiment model was selected after comparing the baseline and tuned models.

|  Stage   |            Model              | Accuracy | Macro F1 | Weighted F1 |
|----------|-------------------------------|----------|----------|-------------|    
| Baseline | CountVectorizer + Naive Bayes | 0.8062   | 0.5960   | 0.7855      |
| Tuned    | CountVectorizer + Naive Bayes | 0.8013   | 0.6203   | 0.7878      |

The baseline model had slightly higher accuracy, but the tuned model was chosen because it improved Macro F1 and Weighted F1. For this dataset, that trade-off is useful because the neutral class is much smaller than the positive and negative classes.

---

## Sentiment Model Interpretation

The final sentiment model performs well on clearly positive and clearly negative reviews.

The neutral class remains harder to classify because it has fewer examples and often contains mixed language. For example, a neutral review may include both positive and negative comments in the same text.

This is a realistic result for public app review data, where users often leave reviews after very positive or very negative experiences.

---

## Rating Prediction Results

The final rating prediction model was selected after comparing baseline and tuned models.

|  Stage   |                Model                  | Accuracy | Macro F1  | Weighted F1 |  MAE   |
|----------|---------------------------------------|----------|-----------|-------------|--------|    
| Baseline | CountVectorizer + Naive Bayes         | 0.6840   |  0.3958   |   0.6399    | 0.7394 |
| Tuned    | CountVectorizer + Logistic Regression | 0.6466   |  0.4302   |   0.6426    | 0.7296 |

Tuned model parameters:

- Model: Logistic Regression
- C: 1.0
- Class weight: balanced

The Naive Bayes baseline reached higher accuracy, but the tuned Logistic Regression model performed better on Macro F1, Weighted F1, and MAE. I selected it because rating prediction is imbalanced, so accuracy alone was not the best selection criterion.

---

## Rating Model Interpretation

The rating prediction task confirms that exact star-rating prediction is more challenging than sentiment classification.

The dataset is highly polarized, with many 1-star and 5-star reviews and fewer 2-star, 3-star, and 4-star reviews.

Because of this, the models perform best on extreme ratings and struggle with middle ratings.

Mean Absolute Error was included because ratings are ordered. Predicting 4 instead of 5 is a smaller error than predicting 1 instead of 5.

---

## Final Conclusion

This project shows how classical NLP and machine learning models can be used to analyze customer reviews from energy-related mobile apps.

The sentiment classification model performed well on clearly positive and clearly negative reviews. Neutral reviews were more difficult because they appeared less often and sometimes contained mixed feedback.

The rating prediction task was harder because the model had to predict five ordered rating classes instead of three sentiment groups. The tuned Logistic Regression model was selected because it improved Macro F1, Weighted F1, and MAE, even though the baseline model had slightly higher accuracy.

From a project perspective, this repository demonstrates the full workflow: review collection, preprocessing, exploratory analysis, feature extraction, model training, evaluation, model saving, and prediction using reusable scripts.


---

## Limitations

* The dataset represents public mobile app users only, not all energy customers.
* Reviews are highly polarized toward 1-star and 5-star ratings.
* Neutral and middle-rating classes are underrepresented.
* Some reviews discuss app usability rather than the complete energy service experience.
* Star ratings are used as proxy labels for sentiment.
* The dataset is source-dependent because Google Play reviews may change over time.

---

## Tools and Libraries

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Matplotlib
* Google Play Scraper
* Joblib
* Jupyter Notebook
* Git and GitHub

---

## Project Structure

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
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_text_preprocessing.ipynb
│   └── 03_model_training_evaluation.ipynb
│
├── reports/
│   ├── figures/
│   ├── eda_summary.csv
│   ├── sentiment_model_results.csv
│   ├── sentiment_model_tuning_results.csv
│   ├── rating_model_results.csv
│   └── rating_model_tuning_results.csv
│
├── src/
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
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone <repository-url>
cd energy-app-review-sentiment-rating-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
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

The raw dataset is not committed to GitHub because it can be recreated using the script.

### 5. Preprocess the review dataset

```bash
python src/preprocessing.py
```

This creates the cleaned dataset locally:

```
data/processed/energy_app_reviews_cleaned.csv
```

### 6. Train final models using reusable scripts

```bash
python src/train.py
```

This trains and saves:

```
models/final_sentiment_model.pkl
models/final_rating_model.pkl
```

It also creates final script-based reports in the `reports/` folder.

### 7. Run prediction demo

Run default example predictions:

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

## Note on Raw Data and Models

The raw dataset, processed dataset, and trained model files are not committed to GitHub and can be recreated by running:

```bash
python src/collect_reviews.py
python src/preprocessing.py
python src/train.py
```

Generated reports and figures are included to make the project results easy to review.

