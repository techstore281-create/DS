
import warnings
warnings.filterwarnings("ignore")  # Suppress FutureWarnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load the diabetes dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

# Display dataset overview
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"Shape : {df.shape}")
print(f"Features : {list(df.columns[:-1])}")
print("Target : Outcome (0 = No Diabetes, 1 = Diabetes)")
print(f"Class Balance : {dict(df['Outcome'].value_counts())}")

# Handle zero values in selected columns
zero_cols = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]

df[zero_cols] = df[zero_cols].replace(0, np.nan)

# Replace missing values with column medians
df[zero_cols] = df[zero_cols].fillna(
    df[zero_cols].median()
)

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Standardize the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define classification models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

# Train models and compare performance
print("\n" + "=" * 60)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 60)

print(
    f"{'Model':<22}"
    f"{'Acc':>8}"
    f"{'Prec':>8}"
    f"{'Rec':>8}"
    f"{'F1':>8}"
)

print("-" * 60)

summary = []

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Store model performance
    summary.append((name, acc, prec, rec, f1))

    # Print performance metrics
    print(
        f"{name:<22}"
        f"{acc:>8.4f}"
        f"{prec:>8.4f}"
        f"{rec:>8.4f}"
        f"{f1:>8.4f}"
    )

# Identify the best model based on accuracy
best = max(summary, key=lambda x: x[1])

print("-" * 60)
print(f"Best Model : {best[0]} (Accuracy = {best[1]:.4f})")

best_model = models[best[0]]

# Create a new patient record
new_patient = pd.DataFrame([{
    'Pregnancies': 2,
    'Glucose': 150,
    'BloodPressure': 80,
    'SkinThickness': 30,
    'Insulin': 120,
    'BMI': 32.5,
    'DiabetesPedigreeFunction': 0.6,
    'Age': 45
}])

# Scale the new patient's features
new_scaled = scaler.transform(new_patient)

# Predict diabetes outcome and probability
pred = best_model.predict(new_scaled)[0]
prob = best_model.predict_proba(new_scaled)[0][1]

# Display prediction results
print("\n" + "=" * 60)
print("NEW PATIENT PREDICTION")
print("=" * 60)

print(
    f"Input Features : "
    f"{new_patient.to_dict(orient='records')[0]}"
)

print(
    f"Prediction : "
    f"{'Diabetes Risk Present' if pred == 1 else 'No Diabetes Risk'}"
)

print(f"Risk Probability: {prob:.4f} ({prob * 100:.2f}%)")

print("=" * 60)
