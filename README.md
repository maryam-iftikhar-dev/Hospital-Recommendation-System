# AI-Powered Hospital Recommendation System

## Overview
This project presents an AI-driven system for analyzing hospital reviews and generating intelligent hospital recommendations. The system integrates Natural Language Processing (NLP), machine learning models, and heuristic-based A* search to provide both review-level insights and hospital-level rankings.

Users can:
- Analyze custom hospital reviews (sentiment + predicted rating)
- Filter hospitals based on multiple criteria
- Obtain ranked recommendations using an informed search algorithm

---

## Features

### 1. NLP-Based Review Analysis
- Text preprocessing (tokenization, stopword removal, normalization)
- Sentiment classification (Positive / Negative)
- Rating prediction (1–5 scale)
- Star-based visualization of predicted rating

### 2. Hospital Recommendation System
- Filtering by:
  - City
  - Minimum rating
  - Minimum sentiment
  - Minimum number of reviews
- Top-K hospital selection
- Dynamic ranking based on heuristic scoring

### 3. A* Search-Based Ranking
- Each hospital represented as a state
- Heuristic function combines:
  - Average sentiment
  - Predicted rating
  - Review count
  - Consistency
- Cost function based on deviation from target rating
- Efficient selection of top-ranked hospitals

---

## Project Structure
PROJECT/
│
├── app.py
├── dataset.csv
│
├── data/
│ └── dataset.csv
│
├── model_training/
│ ├── train_model.py
│ └── pyproject.toml
│
├── models/
│ ├── model_sentiment.pkl
│ ├── model_rating.pkl
│ └── vectorizer.pkl
│
├── utils/
│ ├── data_preprocessing.py
│ └── model_utils.py


---

## Installation

Install required dependencies:

```bash
pip install streamlit pandas scikit-learn nltk joblib

---

## Execution Instructions

Run the application using Streamlit:

```bash
streamlit run app.py

The application will open in your default web browser.
