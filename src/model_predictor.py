import joblib 
from src.data_preprocessing import clean_text

model = joblib.load("models/sentiment_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


def predict_sentiment(text):
    cleaned_text = clean_text(text)
    vector = tfidf.transform([cleaned_text])
    prediction = model.predict(vector)
    return label_encoder.inverse_transform(prediction)[0]