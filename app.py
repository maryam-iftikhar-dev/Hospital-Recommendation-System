import streamlit as st
import pandas as pd
import heapq
import math
from dataclasses import dataclass, field
from typing import Optional
import joblib
import os


from utils.model_utils import load_models, predict_review
from utils.data_preprocessing import aggregate_hospitals , preprocess_text

MODEL_DIR = "models"

vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
sentiment_model = joblib.load(os.path.join(MODEL_DIR, "model_sentiment.pkl"))
rating_model = joblib.load(os.path.join(MODEL_DIR, "model_rating.pkl"))

# =========================
# DATA STRUCTURE
# =========================
@dataclass
class HospitalNode:
    hospital_id: int
    name: str
    city: str
    avg_sentiment: float
    avg_predicted_rating: float
    review_count: int
    rating_std: float

    g_cost: float = field(default=0.0, compare=False)
    h_cost: float = field(default=0.0, compare=False)

    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost

    def __lt__(self, other: "HospitalNode") -> bool:
        return self.f_cost < other.f_cost


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_processed_data():
    df = pd.read_csv("data/dataset.csv")

    sentiment_model, rating_model, vectorizer = load_models()

    sentiments = []
    ratings = []

    for text in df["Feedback"]:
        s, r = predict_review(str(text), sentiment_model, rating_model, vectorizer)
        sentiments.append(s)
        ratings.append(r)

    # Fix sentiment (handle model inconsistencies)
    # Standard binary conversion
    sentiments = [1 if float(s) >= 0.5 else 0 for s in sentiments]

    df["pred_sentiment"] = sentiments
    df["pred_rating"] = ratings

    # Ensure required columns exist
    df = df.dropna(subset=["Hospital Name", "City"])

    grouped = aggregate_hospitals(df)
    grouped["rating_std"] = 0.5

    return grouped.to_dict("records")
# =========================
# HEURISTIC + COST
# =========================
def heuristic(node: HospitalNode,
              w_sentiment=0.35,
              w_rating=0.40,
              w_volume=0.15,
              w_consistency=0.10,
              max_reviews=500):

    rating_norm = (node.avg_predicted_rating - 1.0) / 4.0
    review_norm = min(math.log(node.review_count + 1) /
                      math.log(max_reviews + 1), 1.0)
    std_norm = 1.0 / (1.0 + node.rating_std)

    score = (w_sentiment * node.avg_sentiment +
             w_rating * rating_norm +
             w_volume * review_norm +
             w_consistency * std_norm)

    return round(score, 6)


def cost_function(node: HospitalNode, target_rating: float):
    return abs(node.avg_predicted_rating - target_rating) / 4.0


def apply_filters(hospitals, city, min_rating, min_sentiment, min_reviews):
    filtered = []

    for h in hospitals:
        if city and h.city != city:
            continue
        if h.avg_predicted_rating < min_rating:
            continue
        if h.avg_sentiment < min_sentiment:
            continue
        if h.review_count < min_reviews:
            continue

        filtered.append(h)

    return filtered


# =========================
# A* SEARCH
# =========================
def astar_search(hospitals, target_rating, k, **heuristic_kwargs):
    if not hospitals:
        return [], []

    trace = []
    visited = set()
    results = []

    pq = []

    for idx, node in enumerate(hospitals):
        h_val = heuristic(node, **heuristic_kwargs)
        g_val = cost_function(node, target_rating)

        h_astar = 1.0 - h_val
        f_val = g_val + h_astar

        node.g_cost = g_val
        node.h_cost = h_val

        heapq.heappush(pq, (f_val, idx, node))

    step = 0

    while pq and len(results) < k:
        f_val, _, current = heapq.heappop(pq)
        step += 1

        if current.hospital_id in visited:
            continue

        visited.add(current.hospital_id)
        results.append(current)

        trace.append({
            "Step": step,
            "Hospital": current.name,
            "City": current.city,
            "g(n)": round(current.g_cost, 4),
            "h(n)": round(current.h_cost, 4),
            "f(n)": round(f_val, 4),
        })

    return results, trace


def results_dataframe(hospitals):
    rows = []

    for rank, h in enumerate(hospitals, 1):
        rows.append({
            "Rank": rank,
            "Hospital": h.name,
            "City": h.city,
            "Sentiment": f"{h.avg_sentiment:.3f}",
            "Rating": f"{h.avg_predicted_rating:.2f}",
            "Reviews": h.review_count,
        })

    return pd.DataFrame(rows)


# =========================
# MAIN APP
# =========================
def main():
    st.set_page_config(page_title="AI Hospital Recommender", layout="wide")

    st.title("AI-Powered Hospital Recommender")

    raw_data = load_processed_data()
    hospitals = [HospitalNode(**row) for row in raw_data]

    # session state for button control
    if "run_search" not in st.session_state:
        st.session_state.run_search = False

    # =========================
    # SIDEBAR
    # =========================
    with st.sidebar:
        st.header("Preferences")

        city = st.selectbox("City", ["All"] + sorted(set(h.city for h in hospitals)))
        min_rating = st.slider("Min Rating", 1.0, 5.0, 3.0)
        min_sentiment = st.slider("Min Sentiment", 0.0, 1.0, 0.5)
        min_reviews = st.number_input("Min Reviews", 1, 200, 10)
        top_k = st.slider("Top K", 3, 20, 10)
        target_rating = st.slider("Target Rating", 1.0, 5.0, 4.0)

        st.subheader("Heuristic Weights")
        w_sent = st.slider("Sentiment", 0.0, 1.0, 0.35)
        w_rat = st.slider("Rating", 0.0, 1.0, 0.40)
        w_vol = st.slider("Reviews", 0.0, 1.0, 0.15)
        w_con = st.slider("Consistency", 0.0, 1.0, 0.10)

        if st.button("Get Recommendations", type="primary"):
            st.session_state.run_search = True

    # =========================
    # METRICS
    # =========================
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Hospitals", len(hospitals))

    # =========================
    # RUN SEARCH ONLY ON CLICK
    # =========================
    if st.session_state.run_search:

        filtered = apply_filters(
            hospitals,
            city if city != "All" else None,
            min_rating,
            min_sentiment,
            min_reviews
        )

        col2.metric("Filtered", len(filtered))
        col3.metric("Predictions Used", len(raw_data))

        if filtered:
            heuristic_kwargs = dict(
                w_sentiment=w_sent,
                w_rating=w_rat,
                w_volume=w_vol,
                w_consistency=w_con
            )

            ranked, trace = astar_search(filtered, target_rating, top_k, **heuristic_kwargs)

            st.success(f"Top {len(ranked)} Hospitals")

            st.dataframe(results_dataframe(ranked), hide_index=True)

            # Top 3 cards
            st.subheader("Top Hospitals")
            for i, h in enumerate(ranked[:3]):
                st.markdown(f"""
                ###  #{i+1} {h.name}
                 {h.city}  
                 Rating: {h.avg_predicted_rating:.2f}  
                 Sentiment: {h.avg_sentiment:.3f}
                """)

        else:
            st.warning("No hospitals match filters.")

    else:
        st.info("Set preferences and click 'Get Recommendations'")

    # =========================
    # REVIEW SECTION
    # =========================
    st.divider()
    st.subheader(" Add Your Review")

    hospital_names = sorted(set(h.name for h in hospitals))
    selected_hospital = st.selectbox("Select Hospital", hospital_names)

    user_review = st.text_area("Write your review")

    # Persist review analysis across reruns so it doesn't vanish
    if "review_analysis" not in st.session_state:
        st.session_state.review_analysis = None

    if st.button("Analyze & Submit Review"):
        if not user_review.strip():
            st.warning("Please enter a review")
        else:
            processed_text = preprocess_text(user_review)
                
                # Vectorize
            features = vectorizer.transform([processed_text])
                
                # Predict Sentiment
            sentiment_pred = sentiment_model.predict(features)[0]
            sentiment_probs = sentiment_model.predict_proba(features)[0]
            pred_rating = rating_model.predict(features)[0]
                
                # Sentiment text and confidence
            if sentiment_pred == 1:
                    sentiment_label = "Positive"
                    confidence = sentiment_probs[1]
                    color = "green"
            else:
                    sentiment_label = "Negative"
                    confidence = sentiment_probs[0]
                    color = "red"

            # Store in session state so result survives st.rerun()
            st.session_state.review_analysis = {
                "pred_sent": sentiment_pred,
                "pred_rating": pred_rating,
                "just_submitted": True
            }

            # Load existing dataset
            df_existing = pd.read_csv("data/dataset.csv")

            new_row = pd.DataFrame([{
                "Feedback": user_review,
                "Sentiment Label": int(sentiment_pred),
                "Ratings": float(round(pred_rating, 2)),
                "Hospital Name": selected_hospital,
                "City": next(h.city for h in hospitals if h.name == selected_hospital)
            }])

            # Append safely
            df_updated = pd.concat([df_existing, new_row], ignore_index=True)

            # Save clean
            df_updated.to_csv("data/dataset.csv", index=False)

            st.cache_data.clear()
            st.rerun()

    # Display analysis result OUTSIDE the button block so it persists
    if st.session_state.review_analysis:
        analysis = st.session_state.review_analysis
        pred_sent = analysis["pred_sent"]
        pred_rating = analysis["pred_rating"]

        st.subheader("Analysis Result")

        if pred_sent == 1:
            label = "Positive "
            color = "green"
        else:
            label = "Negative "
            color = "red"

        stars = int(round(pred_rating))

        st.markdown(
            f"""
            **Sentiment:** <span style='color:{color}; font-weight:bold;'>{label}</span>  

            **Rating:** {pred_rating:.2f} / 5.0  

            **Stars:** {'⭐' * stars}{'☆' * (5 - stars)}
            """,
            unsafe_allow_html=True
        )

        st.info("Analysis completed using trained ML models.")

        if analysis.get("just_submitted"):
            st.success("Review saved!")
            st.session_state.review_analysis["just_submitted"] = False



    # =========================
    # RECENT REVIEWS
    # =========================
    st.subheader("Recent Reviews")

    df_reviews = pd.read_csv("data/dataset.csv")

    if "Hospital Name" in df_reviews.columns:
        recent = df_reviews[df_reviews["Hospital Name"] == selected_hospital]
        st.dataframe(recent[["Feedback", "Ratings"]], hide_index=True)


if __name__ == "__main__":
    main()
