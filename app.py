# app.py

import streamlit as st
import joblib
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

st.title("📧 Spam or Ham Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = vectorizer.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.error("🚨 This is a Spam message!")
    else:
        st.success("✅ This is a Ham message.")
