import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\HP\Desktop\Text Sentiment Analyzer for Movie\dataset\IMDB Dataset.csv")
    df = df.sample(frac=0.1, random_state=42)  
    df['cleaned'] = df['review'].apply(clean_text)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df

df = load_data()

@st.cache_resource
def train_model(data):
    X_train, X_test, y_train, y_test = train_test_split(data['cleaned'], data['sentiment'], test_size=0.2, random_state=42)
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])
    model.fit(X_train, y_train)
    return model

model = train_model(df)

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Analyze the **sentiment** of movie or news reviews using machine learning!")

user_input = st.text_area("✍️ Enter your review here:")

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        pred = model.predict([cleaned])[0]
        proba = model.predict_proba([cleaned])[0]

        # Show result
        sentiment = "😊 Positive" if pred == 1 else "😠 Negative"
        confidence = round(max(proba) * 100, 2)
        st.subheader(f"Sentiment: {sentiment}")
        st.info(f"Confidence: {confidence}%")

        # WordCloud
        st.subheader("☁️ Word Cloud of Your Review")
        wc = WordCloud(width=600, height=300, background_color="white").generate(cleaned)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        # Bar chart
        st.subheader("📊 Sentiment Probabilities")
        st.bar_chart({"Negative 😠": [proba[0]], "Positive 😊": [proba[1]]})