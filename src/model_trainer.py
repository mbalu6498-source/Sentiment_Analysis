import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from src.data_preprocessing import clean_text


def train_model(csv_path="dataset/test (1).csv"):
    # ✅ Robust CSV loading
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin1")

    df = df[['text', 'sentiment']].dropna()
    df['clean_text'] = df['text'].astype(str).apply(clean_text)

    le = LabelEncoder()
    df['sentiment_label'] = le.fit_transform(df['sentiment'])

    X = df['clean_text']
    y = df['sentiment_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )
    X_train_tfidf = tfidf.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/sentiment_model.pkl")
    joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
    joblib.dump(le, "models/label_encoder.pkl")

    print("✅ Model trained and saved successfully!")


if __name__ == "__main__":
    train_model()
