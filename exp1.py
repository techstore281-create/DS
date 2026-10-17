import pandas as pd
import numpy as np
import zipfile
import io
import requests

# Load the dataset directly from the UCI repository
url = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"

response = requests.get(url, timeout=60)
response.raise_for_status()

with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    with z.open("diabetic_data.csv") as f:
        df = pd.read_csv(f)

print("Dataset Shape:", df.shape)

print("\nColumn Info:")
df.info()

print("\nMissing Values Count:")
print(df.isna().sum())

print("\nFirst Five Rows:")
print(df.head())


# Integrate demographic and clinical data
df_demographics = df[
    ['patient_nbr', 'race', 'gender', 'age', 'weight']
].copy()

df_clinical = df[
    [
        'patient_nbr',
        'encounter_id',
        'time_in_hospital',
        'num_lab_procedures',
        'num_medications',
        'number_outpatient',
        'number_inpatient',
        'number_emergency',
        'readmitted'
    ]
].copy()

df_integrated = pd.merge(
    df_demographics,
    df_clinical,
    on='patient_nbr',
    how='inner'
)

print("\nIntegrated Dataset Shape:", df_integrated.shape)
print(df_integrated.head())


# Clean the dataset
df_cleaned = df_integrated.copy()

df_cleaned = df_cleaned.replace(
    ['?', 'Unknown/Invalid'],
    np.nan
)

df_cleaned = df_cleaned.dropna(
    subset=['gender', 'age', 'readmitted']
)

df_cleaned['age'] = (
    df_cleaned['age']
    .str.extract(r'(\d+)')
    .astype(float)
)

df_cleaned = df_cleaned[
    df_cleaned['time_in_hospital'].between(1, 14)
]

df_cleaned = df_cleaned[
    df_cleaned['num_lab_procedures'].between(1, 132)
]

print(f"\nRows before cleaning: {len(df_integrated)}")
print(f"Rows after cleaning: {len(df_cleaned)}")
print(
    f"Records removed: "
    f"{len(df_integrated) - len(df_cleaned)}"
)


# Transform the dataset
df_transformed = df_cleaned.copy()

df_transformed['readmission'] = (
    df_transformed['readmitted'].map({
        'NO': 0,
        '>30': 0,
        '<30': 1
    })
)

df_transformed = df_transformed.dropna(
    subset=['readmission']
)


# Standardize numerical features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

num_features = [
    'age',
    'time_in_hospital',
    'num_lab_procedures',
    'num_medications',
    'number_outpatient',
    'number_inpatient',
    'number_emergency'
]

df_transformed[num_features] = scaler.fit_transform(
    df_transformed[num_features]
)


# Convert categorical variables into numerical columns
df_final = pd.get_dummies(
    df_transformed,
    columns=['race', 'gender'],
    drop_first=True,
    dtype=int
)

print("\nFinal Dataset Shape:", df_final.shape)
print(df_final.head())