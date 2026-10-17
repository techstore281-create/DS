import warnings
warnings.filterwarnings("ignore")

import re
import io
import zipfile
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load the drug reviews dataset directly from the UCI repository
url = (
    "https://archive.ics.uci.edu/static/public/462/"
    "drug+review+dataset+drugs+com.zip"
)

response = requests.get(url, timeout=120)
response.raise_for_status()

with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    print("Files in zip:", z.namelist())

    # Pick the training file by name (it is tab-separated)
    train_file = next(
        n for n in z.namelist()
        if n.lower().endswith("train_raw.tsv")
    )

    with z.open(train_file) as f:
        df = pd.read_csv(f, sep="\t")

df = df[['review', 'rating']].dropna()


# Assign sentiment labels based on ratings
def label_from_rating(r):
    if r >= 7:
        return 'positive'
    if r >= 4:
        return 'neutral'
    return 'negative'


df['sentiment'] = df['rating'].apply(label_from_rating)
df = df[['review', 'sentiment']]

# Sample 5000 reviews
df = df.sample(
    n=5000,
    random_state=42
).reset_index(drop=True)

# Display dataset overview
print("=" * 62)
print("DRUG REVIEWS DATASET — OVERVIEW")
print("=" * 62)

print(f"Sampled reviews : {len(df)}")
print(f"Sentiment counts : {dict(df['sentiment'].value_counts())}")

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Clean and preprocess review text
def clean_text(text):
    text = text.lower()

    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)

    tokens = text.split()

    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) > 2
    ]

    return ' '.join(tokens)


df['clean_review'] = df['review'].apply(clean_text)

# Display a sample cleaned review
print("\nSample cleaned review:")
print(" ", df['clean_review'].iloc[0][:150], "...")

# Convert text into TF-IDF features
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = tfidf.fit_transform(df['clean_review'])
y = df['sentiment']

# Split the dataset into training and testing sets
X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Define classification models
models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# Train models and compare performance
print("\n" + "=" * 62)
print("CLASSIFIER PERFORMANCE (Weighted Average)")
print("=" * 62)

print(
    f"{'Classifier':<24}"
    f"{'Acc':>7}"
    f"{'Prec':>7}"
    f"{'Rec':>7}"
    f"{'F1':>7}"
)

print("-" * 62)

records = []

for tag, clf in models.items():
    # Train the classifier
    clf.fit(X_tr, y_tr)

    # Predict sentiment labels
    y_hat = clf.predict(X_te)

    # Calculate evaluation metrics
    acc = accuracy_score(y_te, y_hat)

    prec = precision_score(
        y_te,
        y_hat,
        average='weighted',
        zero_division=0
    )

    rec = recall_score(
        y_te,
        y_hat,
        average='weighted',
        zero_division=0
    )

    f1 = f1_score(
        y_te,
        y_hat,
        average='weighted',
        zero_division=0
    )

    # Store results
    records.append((tag, acc, prec, rec, f1, clf))

    # Display model performance
    print(
        f"{tag:<24}"
        f"{acc:>7.4f}"
        f"{prec:>7.4f}"
        f"{rec:>7.4f}"
        f"{f1:>7.4f}"
    )

# Select the best model based on weighted F1-score
best = max(records, key=lambda r: r[4])

print("-" * 62)
print(f"Best Model : {best[0]}")
print(f"F1 (weighted) : {best[4]:.4f}")

# Generate the confusion matrix
cm = confusion_matrix(
    y_te,
    best[5].predict(X_te),
    labels=['negative', 'neutral', 'positive']
)

print("\nConfusion Matrix (Best Model)")
print(" neg neu pos")

for i, lbl in enumerate(['negative', 'neutral', 'positive']):
    print(
        f"Actual {lbl:<8} "
        f"{cm[i, 0]:>4} "
        f"{cm[i, 1]:>4} "
        f"{cm[i, 2]:>4}"
    )

# Test the model on a new medical review
new_review = (
    "I have been taking this medicine for three weeks and "
    "the headaches have completely disappeared. "
    "No side effects at all, highly recommended!"
)

# Preprocess the new review
clean = clean_text(new_review)

# Transform the review using the trained TF-IDF vectorizer
vec = tfidf.transform([clean])

# Predict sentiment
pred = best[5].predict(vec)[0]

# Display prediction results
print("\n" + "=" * 62)
print("NEW MEDICAL REVIEW — SENTIMENT PREDICTION")
print("=" * 62)

print(f"Review : {new_review}")
print(f"Predicted : {pred.upper()}")

print("=" * 62)