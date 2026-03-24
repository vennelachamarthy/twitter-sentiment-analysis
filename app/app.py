import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="Twitter",
    layout="centered"
)

@st.cache_resource
def load_model():
    import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, 'models', 'logistic_regression_model.pkl'))
vectorizer = joblib.load(os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl'))
    return model, vectorizer

model, vectorizer = load_model()

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

st.title("Twitter Sentiment Analyzer")
st.markdown("Classify any tweet as **Positive** or **Negative** using Machine Learning")
st.markdown("---")

tweet = st.text_area(
    "Enter a tweet below:",
    placeholder="Type or paste a tweet here...",
    height=100
)

if st.button("Analyze Sentiment", type="primary"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet first.")
    else:
        cleaned = clean_text(tweet)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized)[0]
        conf_score = round(max(confidence) * 100, 2)

        if prediction == 1:
            st.success(f"Positive  —  {conf_score}% confidence")
        else:
            st.error(f"Negative  —  {conf_score}% confidence")

        st.markdown("#### Confidence breakdown")
        col1, col2 = st.columns(2)
        col1.metric("Negative", f"{round(confidence[0]*100, 2)}%")
        col2.metric("Positive", f"{round(confidence[1]*100, 2)}%")

st.markdown("---")

st.markdown("#### Try these examples")
examples = [
    "I love this beautiful sunny day!",
    "This is the worst experience I've ever had.",
    "Just had an amazing lunch with my friends!",
    "My flight got cancelled and I missed the meeting.",
]

for example in examples:
    if st.button(example):
        st.rerun()

st.markdown("---")
st.caption("Built with Python | scikit-learn | Streamlit | Trained on Sentiment140 1.6M tweets")