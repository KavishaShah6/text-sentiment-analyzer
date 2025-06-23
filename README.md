# 🎬 Text Sentiment Analyzer for Movie Reviews

A lightweight and interactive **Streamlit** web app that classifies movie or news reviews into **Positive** or **Negative** sentiments using Natural Language Processing (NLP) and a trained Machine Learning model.

---

## 🚀 Features

- 🔤 Input any review and get instant sentiment analysis
- 🤖 Pre-trained Logistic Regression model
- 🧼 Cleaned text using NLTK (stopwords, punctuation, lowercase)
- 📊 Confidence score for prediction
- 🎈 Word Cloud generation for visual insights

---

## 📷 Preview

> *"The storyline was gripping and the acting was phenomenal!"*

✅ **Sentiment:** Positive  
📈 **Confidence:** 94%

---

## 🧠 How It Works

1. The review is preprocessed (lowercase, stopwords removed, etc.)
2. It is vectorized using **TF-IDF**
3. Passed into a trained **Logistic Regression** model
4. Outputs predicted label with probability

---

## 🛠 Tech Stack

- **Frontend**: Streamlit
- **ML Model**: Scikit-learn (Logistic Regression)
- **NLP**: NLTK
- **Visualization**: WordCloud, Matplotlib

---

