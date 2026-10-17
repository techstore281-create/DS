import pandas as pd
import numpy as np
import zipfile
import io
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# LOAD RAW DATA FROM UCI
# ---------------------------------------------------------
url = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"

response = requests.get(url, timeout=60)
response.raise_for_status()

with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    with z.open("diabetic_data.csv") as f:
        raw = pd.read_csv(f)

# ---------------------------------------------------------
# REBUILD THE CLEANED + TRANSFORMED DATASET
# (same steps as the preprocessing script)
# ---------------------------------------------------------
df_demographics = raw[
    ['patient_nbr', 'race', 'gender', 'age', 'weight']
].copy()

df_clinical = raw[
    [
        'patient_nbr', 'encounter_id', 'time_in_hospital',
        'num_lab_procedures', 'num_medications',
        'number_outpatient', 'number_inpatient',
        'number_emergency', 'readmitted'
    ]
].copy()

df_integrated = pd.merge(
    df_demographics, df_clinical, on='patient_nbr', how='inner'
)

df_cleaned = df_integrated.replace(['?', 'Unknown/Invalid'], np.nan)
df_cleaned = df_cleaned.dropna(subset=['gender', 'age', 'readmitted'])

df_cleaned['age'] = (
    df_cleaned['age'].str.extract(r'(\d+)').astype(float)
)

df_cleaned = df_cleaned[df_cleaned['time_in_hospital'].between(1, 14)]
df_cleaned = df_cleaned[df_cleaned['num_lab_procedures'].between(1, 132)]

df_transformed = df_cleaned.copy()
df_transformed['readmission'] = df_transformed['readmitted'].map(
    {'NO': 0, '>30': 0, '<30': 1}
)
df_transformed = df_transformed.dropna(subset=['readmission'])

num_features = [
    'age', 'time_in_hospital', 'num_lab_procedures',
    'num_medications', 'number_outpatient',
    'number_inpatient', 'number_emergency'
]

scaler = StandardScaler()
df_transformed[num_features] = scaler.fit_transform(
    df_transformed[num_features]
)

df = pd.get_dummies(
    df_transformed,
    columns=['race', 'gender'],
    drop_first=True,
    dtype=int
)

print("Dataset ready:", df.shape)


# ---------------------------------------------------------
# STEP 1: Descriptive statistics
# ---------------------------------------------------------
print("Descriptive Statistics:")

display(
    df[
        [
            'age',
            'time_in_hospital',
            'num_lab_procedures',
            'num_medications',
            'number_outpatient',
            'number_inpatient',
            'number_emergency'
        ]
    ].describe().round(2)
)


# ---------------------------------------------------------
# STEP 2: Readmission distribution
# ---------------------------------------------------------
plt.figure(figsize=(4, 3))

sns.countplot(
    x='readmission',
    hue='readmission',
    data=df,
    palette='Set2',
    legend=False
)

plt.title("Readmission Distribution (0: No, 1: Yes)")
plt.xlabel("Readmission")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# STEP 3: Distribution of numerical features
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

sns.histplot(
    df['age'], kde=True, ax=axes[0], color='#4C72B0'
).set(title="Age")

sns.histplot(
    df['time_in_hospital'], kde=True, ax=axes[1], color='#55A868'
).set(title="Time in Hospital")

sns.histplot(
    df['num_lab_procedures'], kde=True, ax=axes[2], color='#8172B2'
).set(title="Laboratory Procedures")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# STEP 4: Compare features by readmission status
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

sns.boxplot(
    x='readmission', y='age', hue='readmission',
    data=df, ax=axes[0], palette='Set2', legend=False
).set(title="Age vs Readmission")

sns.boxplot(
    x='readmission', y='time_in_hospital', hue='readmission',
    data=df, ax=axes[1], palette='Set2', legend=False
).set(title="Hospital Stay vs Readmission")

sns.boxplot(
    x='readmission', y='num_lab_procedures', hue='readmission',
    data=df, ax=axes[2], palette='Set2', legend=False
).set(title="Lab Procedures vs Readmission")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# STEP 5: Correlation matrix
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))

core_cols = [
    'age',
    'time_in_hospital',
    'num_lab_procedures',
    'num_medications',
    'number_outpatient',
    'number_inpatient',
    'number_emergency',
    'readmission'
]

sns.heatmap(
    df[core_cols].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()