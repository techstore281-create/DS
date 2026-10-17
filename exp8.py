
# Experiment 8: Explainable AI (XAI) in Healthcare
# Dataset: Breast Cancer Wisconsin Diagnostic Dataset

# STEP 1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

# STEP 2: Load the healthcare dataset
data = load_breast_cancer()

# Convert the dataset into a pandas DataFrame
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Display dataset information
print("Dataset loaded successfully!")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("\nFirst five rows:")
print(X.head())

# STEP 3: Display the target class distribution
print("\nTarget Class Distribution:")
print(y.map({0: "Malignant", 1: "Benign"}).value_counts())

plt.figure(figsize=(6, 4))
y.map({0: "Malignant (0)", 1: "Benign (1)"}).value_counts().reindex(
    ["Malignant (0)", "Benign (1)"], fill_value=0
).plot(kind="bar", color=["tomato", "mediumseagreen"])

plt.title("Target Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# STEP 4: Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.27,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# STEP 5: Train the Random Forest classification model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")

# STEP 6: Predict the test data
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation")
print("----------------")
print("Accuracy:", round(accuracy, 4))
print("Accuracy Percentage:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=data.target_names
))

# STEP 7: Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
)

display.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# STEP 8: Global XAI - Random Forest Feature Importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Random Forest Features:")
print(importance.head(10))

plt.figure(figsize=(9, 5))
importance.head(10).sort_values().plot(kind="barh")
plt.title("Random Forest Feature Importance (Top 10)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# STEP 9: Global XAI - Permutation Importance
perm = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="accuracy"
)

perm_importance = pd.Series(
    perm.importances_mean,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Permutation Features:")
print(perm_importance.head(10))

plt.figure(figsize=(9, 5))
perm_importance.head(10).sort_values().plot(kind="barh")
plt.title("Permutation Feature Importance (Top 10)")
plt.xlabel("Mean decrease in accuracy")
plt.tight_layout()
plt.show()

# STEP 10: Local XAI - Explain one individual prediction using SHAP
explainer = shap.TreeExplainer(model)

# Select the first patient from the test dataset
patient = X_test.iloc[[0]]
patient_prediction = model.predict(patient)[0]
patient_probabilities = model.predict_proba(patient)[0]

print("\nIndividual Patient Explanation")
print("------------------------------")
print("Predicted class:", data.target_names[patient_prediction])
print("Actual class:", data.target_names[y_test.iloc[0]])
print("Probability of malignant:", round(patient_probabilities[0], 4))
print("Probability of benign:", round(patient_probabilities[1], 4))

# Calculate SHAP values for the selected patient
patient_shap = explainer.shap_values(patient)

# Handle SHAP output formats across library versions
if isinstance(patient_shap, list):
    shap_values_class = patient_shap[1][0]
elif isinstance(patient_shap, np.ndarray) and patient_shap.ndim == 3:
    # Some versions return (samples, features, classes)
    shap_values_class = patient_shap[0, :, 1]
else:
    # Binary-class explanations may return a 2D array
    shap_values_class = patient_shap[0]

# Display the five largest absolute feature contributions
local_importance = pd.Series(
    shap_values_class,
    index=X.columns
)

print("\nTop 5 Features Influencing This Patient's Prediction:")
print(local_importance.reindex(
    local_importance.abs().sort_values(ascending=False).head(5).index
))

# STEP 11: SHAP global summary / beeswarm plot
# SHAP explains the model's output for the test dataset
shap_test = explainer.shap_values(X_test)

if isinstance(shap_test, list):
    shap_values_benign = shap_test[1]
elif isinstance(shap_test, np.ndarray) and shap_test.ndim == 3:
    shap_values_benign = shap_test[:, :, 1]
else:
    shap_values_benign = shap_test

# Display the SHAP beeswarm plot
shap.summary_plot(
    shap_values_benign,
    X_test,
    max_display=10,
    show=False
)

plt.title("SHAP Beeswarm Plot (Class: Benign)")
plt.tight_layout()
plt.show()

# STEP 12: Final conclusion
print("\nCONCLUSION")
print("Healthcare data was used to train a Random Forest classifier.")
print("Model performance was evaluated using accuracy and a confusion matrix.")
print("Feature importance and permutation importance explained global behavior.")
print("SHAP explained feature contributions for individual and overall predictions.")
print("These explanations support model interpretation, but do not replace")
print("professional medical diagnosis or clinical judgment.")
