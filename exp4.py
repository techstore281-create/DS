
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

df = pd.read_csv("support2.csv")

print("Dataset Shape:", df.shape)

print("\nColumn Info:")
df.info()

print("\nMissing Values:")
print(df.isna().sum())

features = [
    'age', 'sex', 'race', 'num.co', 'diabetes', 'dementia',
    'ca', 'meanbp', 'wblc', 'hrt', 'resp', 'temp', 'pafi',
    'alb', 'bili', 'crea', 'sod', 'ph', 'glucose', 'bun',
    'urine', 'adlp'
]

target = 'death'

df_model = df[features + [target]].copy()

# Handle missing values
for col in df_model.columns:
    if col != target:
        if df_model[col].dtype == 'object':
            df_model[col] = df_model[col].fillna(
                df_model[col].mode()[0]
            )
        else:
            df_model[col] = df_model[col].fillna(
                df_model[col].median()
            )

# Convert categorical variables into dummy variables
df_model = pd.get_dummies(
    df_model,
    columns=['sex', 'race', 'ca'],
    drop_first=True,
    dtype=int
)

# Define features and target
X = df_model.drop(columns=['death'])
y = df_model['death']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train the Random Forest classifier
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

# Calculate ROC-AUC score
auc = roc_auc_score(y_test, y_prob)

print("\nROC-AUC:", round(auc, 3))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=['Survived', 'Died']
    )
)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Feature importance visualization
importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=True)

plt.figure(figsize=(6, 4))
importance.tail(15).plot(
    kind='barh',
    color='#4C72B0'
)

plt.title("Biomarker Hazard Importance")
plt.xlabel("Relative Importance")
plt.ylabel("")
plt.tight_layout()
plt.show()

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(5, 4))
plt.plot(
    fpr,
    tpr,
    color='#F0A04B',
    linewidth=2,
    label=f'Prognostic Model (AUC = {auc:.2f})'
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--',
    color='blue',
    linewidth=1
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Risk Discrimination Curve (ROC)")
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Survived', 'Mortality']
)

fig, ax = plt.subplots(figsize=(4, 3))

disp.plot(
    cmap='Blues',
    values_format='d',
    ax=ax,
    colorbar=True
)

plt.title("Prognostic Outcome Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.show()
