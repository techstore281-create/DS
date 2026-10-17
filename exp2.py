
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned and transformed dataset
df = pd.read_csv(
    "Raj_Parihar_BEA170_diabetes_cleaned_transformed.csv"
)

# STEP 1: Display descriptive statistics
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


# STEP 2: Readmission distribution
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


# STEP 3: Distribution of numerical features
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

sns.histplot(
    df['age'],
    kde=True,
    ax=axes[0],
    color='#4C72B0'
).set(title="Age")

sns.histplot(
    df['time_in_hospital'],
    kde=True,
    ax=axes[1],
    color='#55A868'
).set(title="Time in Hospital")

sns.histplot(
    df['num_lab_procedures'],
    kde=True,
    ax=axes[2],
    color='#8172B2'
).set(title="Laboratory Procedures")

plt.tight_layout()
plt.show()


# STEP 4: Compare features by readmission status
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

sns.boxplot(
    x='readmission',
    y='age',
    hue='readmission',
    data=df,
    ax=axes[0],
    palette='Set2',
    legend=False
).set(title="Age vs Readmission")

sns.boxplot(
    x='readmission',
    y='time_in_hospital',
    hue='readmission',
    data=df,
    ax=axes[1],
    palette='Set2',
    legend=False
).set(title="Hospital Stay vs Readmission")

sns.boxplot(
    x='readmission',
    y='num_lab_procedures',
    hue='readmission',
    data=df,
    ax=axes[2],
    palette='Set2',
    legend=False
).set(title="Lab Procedures vs Readmission")

plt.tight_layout()
plt.show()


# STEP 5: Correlation matrix
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
