import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def aggregate_hospitals(df):
    grouped = df.groupby("Hospital Name").agg({
        "Hospital Name": "first",
        "City": "first",
        "pred_sentiment": "mean",
        "pred_rating": "mean",
        "Feedback": "count"
    }).reset_index(drop=True)

    grouped.columns = [
        "name",
        "city",
        "avg_sentiment",
        "avg_predicted_rating",
        "review_count"
    ]
    grouped["hospital_id"] = range(1, len(grouped) + 1)
    return grouped