import streamlit as st
import numpy as np
import re
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# -------------------------------
# Load model
# -------------------------------
model = load_model("sentiment_model.h5")

# -------------------------------
# Load tokenizer
# -------------------------------
with open("tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer = tokenizer_from_json(f.read())

# -------------------------------
# Preprocessing
# -------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------------
# Star Generator ⭐
# -------------------------------
def generate_stars(rating):
    full = int(rating)
    half = 1 if rating - full >= 0.5 else 0
    empty = 5 - full - half

    return "⭐" * full + "✨" * half + "☆" * empty

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="MovieCom", layout="centered")

st.title("🎬 MovieCom")
st.subheader("Smart Movie Review Analyzer ⭐")

st.write("Enter a movie review and get sentiment + star rating")

movie_name = st.text_input("🎥 Enter Movie Name")

review = st.text_area("✍️ Write your review here:")

if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a review!")
    else:
        cleaned = preprocess_text(review)

        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=100)

        prediction = model.predict(padded)[0][0]

        # Sentiment
        if prediction >= 0.5:
            st.success("😊 Sentiment: Positive")
        else:
            st.error("😞 Sentiment: Negative")

        # Rating
        rating = round(prediction * 5, 1)
        stars = generate_stars(rating)

        st.markdown(f"### ⭐ Rating: {rating}/5")
        st.markdown(f"### {stars}")

        # Confidence
        st.progress(float(prediction))
        st.write(f"Confidence Score: {round(prediction, 3)}")

        if movie_name:
            st.write(f"🎬 Movie: {movie_name}")