import joblib

# loads pretrained models
def load_models():
    sentiment_model = joblib.load("models/model_sentiment.pkl")
    rating_model = joblib.load("models/model_rating.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return sentiment_model, rating_model, vectorizer

# converts textual review -> numerical insights(sentiment and rating values)
def predict_review(text, sentiment_model, rating_model, vectorizer):
    X = vectorizer.transform([text]).toarray()
    sentiment = sentiment_model.predict(X)[0]
    rating = rating_model.predict(X)[0]
    return sentiment, rating
