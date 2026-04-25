import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import joblib
import os
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]
    
    return " ".join(tokens)

def main():
    print("Loading dataset...")
    try:
        df = pd.read_csv('dataset.csv')
    except FileNotFoundError:
        print("Error: dataset.csv not found!")
        return

    print("Preprocessing text...")
    df['processed_review'] = df['Feedback'].apply(preprocess_text)

    # Features and labels
    X = df['processed_review']
    y_sentiment = df['Sentiment Label']
    y_rating = df['Ratings']

    # Single train/test split
    X_train, X_test, y_train_s, y_test_s, y_train_r, y_test_r = train_test_split(
        X,
        y_sentiment,
        y_rating,
        test_size=0.2,
        random_state=42
    )
    print("Extracting features (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=9000,ngram_range=(1,2),min_df=2,max_df=0.9)
    
    # Fit and transform on sentiment training data, transform test data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Sentiment Classification Model (Logistic Regression)...")
    sentiment_model = LogisticRegression(max_iter=2000,C=2,solver='lbfgs')
    sentiment_model.fit(X_train_tfidf, y_train_s)
    y_pred_s = sentiment_model.predict(X_test_tfidf)
    acc = accuracy_score(y_test_s, y_pred_s)
    print(f"Sentiment Model Accuracy: {acc:.4f}")

    print("Training Rating Prediction Model (Ridge Regression)...")
    rating_model = Ridge(alpha=1.0, random_state=42)

    rating_model.fit(X_train_tfidf, y_train_r)

    y_pred_r = rating_model.predict(X_test_tfidf)

    mae = mean_absolute_error(y_test_r, y_pred_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))

    print(f"Rating Model MAE: {mae:.4f}")
    print(f"Rating Model RMSE: {rmse:.4f}")

  #  print("Saving models and vectorizer...")
    joblib.dump(sentiment_model, 'model_sentiment.pkl')
    joblib.dump(rating_model, 'model_rating.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    #print("Training complete! Models saved as model_sentiment.pkl, model_rating.pkl, and vectorizer.pkl.")

if __name__ == "__main__":
    main()