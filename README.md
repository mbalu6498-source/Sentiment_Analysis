# SENTIMENT ANALYSIS

Sentiment Analysis is a Natural Language Processing (NLP) technique used to determine the emotional tone behind a piece of text.
This project analyzes textual reviews and classifies them into positive or negative sentiments using machine learning algorithms.
The project demonstrates a complete NLP pipeline including text preprocessing, feature extraction, model training, and evaluation.

# Technologies Used

Python
Pandas
NumPy
NLTK
Scikit-learn

# Workflow

Load the dataset using Pandas
>Perform text preprocessing:

Convert text to lowercase
Remove punctuation
Remove stopwords

>Convert text into numerical features using:
TF-IDF Vectorizer

Split the dataset into training and testing sets

>Apply a classification algorithm:
Logistic Regression  and Naïve Bayes

Evaluate the model using accuracy score

>Machine Learning Algorithms

Logistic Regression
Multinomial Naïve Bayes

#Performance Evaluation

>The model is evaluated using:
Accuracy Score
Confusion Matrix
Classification Repor

# sentiment_analysis/
│
├── app.py 
├── requirements.txt 
│
├── src/
│ ├── data_preprocessing.py 
│ ├── model_trainer.py 
│ ├── model_predictor.py 
│
├── models/
│ ├── sentiment_model.pkl
│ ├── tfidf_vectorizer.pkl
│ └── label_encoder.pkl
│
├── templates/
│ ├── index.html
│ └── result.html
│
├── static/
│ └── style.css
│
└── dataset/
└── test (1).csv
