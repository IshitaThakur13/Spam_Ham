# spam_classifier.py

import pandas as pd
import numpy as np
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score
import joblib

# Load data
df = pd.read_csv(r'C:\Users\hp\OneDrive\Dokumen\spam.csv', encoding='ISO-8859-1')[['v1', 'v2']]
df.columns = ['target', 'text']
df.drop_duplicates(inplace=True)

# Encode labels
df['target'] = df['target'].map({'ham': 0, 'spam': 1})

# Preprocessing
ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

df['transformed_text'] = df['text'].apply(transform_text)

# Vectorization
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'spam_model.pkl')
joblib.dump(tfidf, 'vectorizer.pkl')
