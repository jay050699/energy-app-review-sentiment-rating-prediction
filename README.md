# Energy App Review Sentiment Analysis and Rating Prediction

## Project Overview

This project builds an end-to-end Natural Language Processing and Machine Learning pipeline to analyze public customer reviews from energy-sector mobile applications.

The dataset contains Google Play Store reviews from energy supplier apps, EV charging apps, energy mobility apps, and smart home energy services. The goal is to classify customer sentiment and predict star ratings based on review text.

The project is designed as a practical text analytics workflow for understanding customer feedback patterns in the energy sector.

## Business Motivation

Energy companies increasingly provide digital services through mobile applications. Customers use these apps for tasks such as checking energy usage, submitting meter readings, managing accounts, making payments, finding EV charging stations, and reporting technical issues.

Public app reviews contain useful feedback about customer satisfaction, usability problems, charging issues, login problems, payment failures, and service quality.

This project shows how NLP can be used to extract insights from unstructured review text and support customer experience monitoring.

## Dataset

The dataset was collected from public Google Play Store reviews using the `google-play-scraper` Python package.

The final cleaned dataset contains:

 Metric                     Value 
 -------------------------  ----- 
 Total reviews              3,070 
 Number of apps                 8 
 Average rating              2.97 
 Average review word count  22.57 
 Positive reviews           1,411 
 Negative reviews           1,439 
 Neutral reviews              220 

## Selected App Categories

The dataset includes reviews from apps related to:

* Energy supplier digital services
* EV charging and mobility
* Smart energy and home energy services

Example apps include:

* Octopus Electroverse
* E.ON Myline
* E.ON Next Home
* PlugShare
* Shell
* ScottishPower
* Hive
* Vattenfall InCharge

## Project Tasks

This project contains two machine learning tasks.

### 1. Sentiment Classification

The original 1-to-5 star ratings were mapped into three sentiment classes:

 Rating  Sentiment 
 ------  --------- 
 1, 2    Negative  
 3       Neutral   
 4, 5    Positive  

### 2. Rating Prediction

The second task predicts the original star rating:

1, 2, 3, 4, or 5 stars

Rating prediction is more difficult than sentiment classification because the model must distinguish between five rating levels instead of three sentiment classes.

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

## Text Preprocessing

The review text was cleaned using the following steps:

* Lowercasing
* URL removal
* Special character removal
* Extra whitespace removal
* Stopword removal
* Lemmatization
* Short empty-text filtering
* Duplicate review removal

Both the original review text and cleaned review text were preserved.

## Feature Extraction

Two classical NLP feature extraction methods were used:

### CountVectorizer

CountVectorizer converts text into word and phrase count features.

It is simple, interpretable, and works well as a baseline for text classification tasks.

### TF-IDF Vectorizer

TF-IDF gives higher importance to terms that are frequent in a review but not too common across all reviews.

It is commonly used with linear models such as Logistic Regression.

Both methods used unigrams and bigrams to capture words and short phrases such as:

charging station
not working
customer service
easy use
payment issue

## Models Used

The project compares the following classical machine learning models:

* Multinomial Naive Bayes
* Logistic Regression

Model pipelines combine text vectorization and classification in a single reusable workflow.

## Sentiment Classification Results

The final sentiment model was selected after comparing baseline and tuned models.

  Stage     Model                          Accuracy  Macro F1  Weighted F1  Parameters                
  --------  -----------------------------  --------  --------  -----------  ------------------------- 
  Baseline  CountVectorizer + Naive Bayes    0.8062    0.5960       0.7855  Default baseline settings 
  Tuned     CountVectorizer + Naive Bayes    0.8013    0.6203       0.7878  alpha = 0.5               

The tuned CountVectorizer + Naive Bayes model was selected as the final sentiment model because it achieved the best macro F1-score.

Although the baseline model had slightly higher accuracy, the tuned model performed better when considering all sentiment classes more equally.

## Sentiment Model Interpretation

The final sentiment model performs well on clearly positive and clearly negative reviews.

The neutral class remains harder to classify because it has fewer examples and often contains mixed language. For example, a neutral review may include both positive and negative comments in the same text.

This is a realistic result for public app review data, where users often leave reviews after very positive or very negative experiences.

## Rating Prediction Results

Rating prediction was treated as a secondary task because exact 1-to-5 star prediction is more difficult than sentiment classification.

  Stage     Model                                  Accuracy  Macro F1  Weighted F1     MAE  Parameters                       
  --------  -------------------------------------  --------  --------  -----------  ------  -------------------------------- 
  Baseline  CountVectorizer + Naive Bayes            0.6840    0.3958       0.6399  0.7394  Default baseline settings        
  Tuned     CountVectorizer + Logistic Regression    0.6466    0.4302       0.6426  0.7296  C = 1.0, class_weight = balanced 

The tuned CountVectorizer + Logistic Regression model was selected as the final rating model because it achieved a better macro F1-score and slightly lower Mean Absolute Error.

The baseline Naive Bayes model achieved higher exact accuracy, but it mainly performed well on the dominant 1-star and 5-star classes.

## Rating Model Interpretation

The rating prediction task confirms that exact star-rating prediction is more challenging than sentiment classification.

The dataset is highly polarized, with many 1-star and 5-star reviews and fewer 2-star, 3-star, and 4-star reviews.

Because of this, the models perform best on extreme ratings and struggle with middle ratings.

Mean Absolute Error was included because ratings are ordered. Predicting 4 instead of 5 is a smaller error than predicting 1 instead of 5.

## Key Insights from EDA

The review dataset showed strong polarization between positive and negative feedback.

Common positive terms included:

easy
good
great
charging
charger
work
service
excellent

Common negative terms included:

work
update
time
charging
useless
card
issue
account

Negative reviews often contained more detailed explanations of technical problems, login issues, payment failures, or charging difficulties.

## Limitations

* The dataset represents public mobile app users only, not all energy customers.
* Reviews are highly polarized toward 1-star and 5-star ratings.
* Neutral and middle-rating classes are underrepresented.
* Some reviews discuss app usability rather than the complete energy service experience.
* Star ratings are used as proxy labels for sentiment.
* The dataset is source-dependent because Google Play reviews may change over time.

## Conclusion

This project demonstrates a complete NLP pipeline for energy-sector review analytics.

The sentiment classification task produced reliable results for positive and negative customer feedback. The final tuned sentiment model achieved strong overall performance while improving macro F1 compared with the baseline.

The rating prediction task was more difficult because exact star ratings require more fine-grained classification. The tuned rating model improved macro F1 and Mean Absolute Error, but middle ratings remained challenging due to class imbalance.

Overall, the project shows how machine learning can be applied to public energy app reviews to identify customer satisfaction patterns, common complaints, and digital service issues.

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

## Project Structure

nlp-yelp-sentiment-rating-prediction/
│
├── config/
│   └── energy_apps_shortlist.csv
│
├── data/
│   ├── raw/
│   └── processed/
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
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── README.md
├── requirements.txt
└── .gitignore

## How to Run the Project

### 1. Clone the repository

git clone <repository-url>
cd nlp-yelp-sentiment-rating-prediction


### 2. Create a virtual environment

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Activate it on macOS/Linux:

source .venv/bin/activate


### 3. Install dependencies

pip install -r requirements.txt

This creates the raw dataset locally:

data/raw/energy_app_reviews_raw.csv


### 4. Collect public Google Play reviews

python src/collect_reviews.py


### 5. Preprocess the review dataset

python src/preprocessing.py

This creates the cleaned dataset locally:

data/processed/energy_app_reviews_cleaned.csv


### 6. Train final models using reusable scripts

python src/train.py

This trains and saves:

models/final_sentiment_model.pkl
models/final_rating_model.pkl


### 7. Run prediction demo

Run default example predictions:

python src/predict.py

Run prediction on custom review text:

python src/predict.py --text "Write a review here"


### 8. Run Notebooks

The notebooks can be run in this order:

01_data_exploration.ipynb
02_text_preprocessing.ipynb
03_model_training_evaluation.ipynb


## Note on Raw Data

The raw and processed datasets are not committed to GitHub because they can be recreated using the collection and preprocessing scripts.

The generated reports and figures are included to make the project results easy to review.
