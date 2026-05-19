# Project Plan

## Project Title

NLP Sentiment Analysis and Rating Prediction

## Objective

The objective of this project is to build an end-to-end NLP machine learning pipeline that uses customer review text to predict sentiment and rating.

The project will include two related tasks:

1. Sentiment classification
2. Rating prediction

## Main Workflow

The project follows this workflow:

1. Select and download a review dataset
2. Load the raw dataset
3. Inspect dataset columns and missing values
4. Clean the dataset
5. Create sentiment labels from ratings
6. Explore review and rating patterns
7. Preprocess review text
8. Convert text into numerical features
9. Train machine learning models
10. Evaluate model performance
11. Compare models
12. Save trained models
13. Create a prediction script
14. Document results in README

## Dataset Requirement

The dataset should contain at least:

- Review text
- Rating score or sentiment label

Preferred columns:

- review_text
- rating
- date
- product/business/category name

## Planned Feature Extraction Methods

### CountVectorizer

CountVectorizer converts text into word-count features.

It is simple, interpretable, and works well with Naive Bayes.

### TF-IDF Vectorizer

TF-IDF gives higher importance to meaningful words and lower importance to very common words.

It often works well with Logistic Regression.

## Planned Models

### Multinomial Naive Bayes

This model is commonly used for text classification and is fast to train.

### Logistic Regression

This model is a strong baseline for text classification and works well with sparse TF-IDF features.

## Planned Evaluation Metrics

The project will use:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Expected Final Outputs

The final project will include:

- Cleaned dataset
- EDA notebook
- Text preprocessing notebook
- Model training notebook
- Reusable Python scripts
- Saved model files
- Evaluation results
- Confusion matrix plots
- README documentation



# Final Project Conclusion

This project developed an end-to-end NLP pipeline for analyzing customer reviews from energy-sector mobile applications.

The sentiment classification task showed strong performance for positive and negative reviews. The final tuned sentiment model, CountVectorizer + Naive Bayes with alpha = 0.5, achieved an accuracy of 0.8013 and a macro F1-score of 0.6203.

The rating prediction task was more challenging because the model had to predict exact 1-to-5 star ratings. The final tuned rating model, CountVectorizer + Logistic Regression with balanced class weights, achieved an accuracy of 0.6466, macro F1-score of 0.4302, and Mean Absolute Error of 0.7296.

The results show that sentiment classification is more reliable than exact rating prediction for this dataset. This is mainly because public app reviews are highly polarized, with many 1-star and 5-star reviews and fewer middle-rating examples.

Overall, the project demonstrates how classical NLP and machine learning methods can be used to analyze energy-sector customer feedback, identify common satisfaction and complaint patterns, and support customer experience monitoring.
