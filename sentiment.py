# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 11:42:27 2025

@author: ADMIN
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------- Data Load ----------------
st.title("📌 Sentiment Analysis Web App using NLP")
st.write("Analyze movie reviews and classify as Positive or Negative")

# Load your dataset
df = pd.read_csv(r"E:\jd\sentiment Dataset.csv")   # Change to your filename

# ---------------- Text Cleaning ----------------
def clean_text(text):
    text = re.sub('<.*?>', '', text)  
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = text.lower()
    return text

df['clean_review'] = df['review'].apply(clean_text)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_review'], df['sentiment'], test_size=0.2, random_state=42
)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# -------- Model Training --------
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

st.subheader(f"📊 Model Accuracy: **{accuracy:.2f}**")

# ---------------- WordCloud ----------------
st.subheader("☁ WordCloud of Reviews")

all_words = ' '.join(df['clean_review'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis("off")
st.pyplot(plt)

# ---------------- Prediction Section ----------------
st.subheader("🔍 Try Your Own Review")

user_input = st.text_area("Enter a movie review here...")

if st.button("Predict Sentiment"):
    cleaned = clean_text(user_input)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    if prediction == "positive":
        st.success("😊 Sentiment: **Positive**")
    else:
        st.error("😞 Sentiment: **Negative**")
