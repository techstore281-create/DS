"""
Healthcare Data Science Pipeline - Heart Disease Prediction
Steps: Collect -> Clean -> Integrate -> Transform -> Model -> Evaluate
"""

import os
import io
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

SEP = "=" * 70

def header(title):
    print(f"\n{SEP}\n>>> {title}\n{SEP}")


# =====================================================================
# STEP 1: DATA COLLECTION
# =====================================================================
header("STEP 1: DATA COLLECTION")

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "heart-disease/processed.cleveland.data")

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

def collect_data():
    """Download raw data from UCI repository."""
    local_path = "data/raw_heart.csv"
    if os.path.exists(local_path):
        print(f"[+] Loading cached data: {local_path}")
        df = pd.read_csv(local_path, names=COLUMNS, na_values="?")
    else:
        print(f"[+] Downloading from: {URL}")
        try:
            r = requests.get(URL, timeout=30)
            r.raise_for_status()
            with open(local_path, "w") as f:
                f.write(r.text)
            print(f"[+] Saved to {local_path}")
            df = pd.read_csv(io.StringIO(r.text), names=COLUMNS, na_values="?")
        except Exception as e:
            print(f"[!] Download failed ({e}); using sklearn-style fallback")
            # Fallback: generate synthetic-but-valid sample
            np.random.seed(42)
            n = 303
            df = pd.DataFrame({
                "age":      np.random.randint(29, 77, n),
                "sex":      np.random.randint(0, 2, n),
                "cp":       np.random.randint(0, 4, n),
                "trestbps": np.random.randint(94, 200, n),
                "chol":     np.random.randint(126, 564, n),
                "fbs":      np.random.randint(0, 2, n),
                "restecg":  np.random.randint(0, 3, n),
                "thalach":  np.random.randint(71, 202, n),
                "exang":    np.random.randint(0, 2, n),
                "oldpeak":  np.round(np.random.uniform(0, 6.2, n), 1),
                "slope":    np.random.randint(0, 3, n),
                "ca":       np.random.randint(0, 4, n),
                "thal":     np.random.randint(1, 4, n),
                "target":   np.random.randint(0, 5, n),
            })
    print(f"[+] Collected: shape={df.shape}")
    return df


raw_df = collect_data()
print("\n--- Raw Data Preview ---")
print(raw_df.head(10).to_string())
print(f"\nRaw dtypes:\n{raw_df.dtypes}")


# =====================================================================
# STEP 2: DATA CLEANING
# =====================================================================
header("STEP 2: DATA CLEANING")

def clean_data(df):
    df = df.copy()

    # 2.1 Handle non-numeric values (e.g., '?' already turned into NaN)
    print(f"[+] Missing values before cleaning:\n{df.isnull().sum().to_string()}")

    # 2.2 Impute numeric columns with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # 2.3 Remove duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[+] Removed {before - len(df)} duplicate rows.")

    # 2.4 Outlier clipping via IQR (on continuous features only)
    continuous = ["trestbps", "chol", "thalach", "oldpeak"]
    for col in continuous:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < low) | (df[col] > high)).sum()
        df[col] = df[col].clip(low, high)
        print(f"[+] Clipped {n_out:3d} outliers in '{col}' -> [{low:.2f}, {high:.2f}]")

    # 2.5 Binarize target (UCI: 0 = no disease, 1-4 = disease)
    df["target"] = (df["target"] > 0).astype(int)

    # 2.6 Fix types
    int_cols = ["age","sex","cp","fbs","restecg","exang","slope","ca","thal","target"]
    for c in int_cols:
        df[c] = df[c].astype(int)

    print(f"\n[+] Cleaned shape: {df.shape}")
    print(f"[+] Missing values after cleaning: {df.isnull().sum().sum()}")
    return df


clean_df = clean_data(raw_df)
print("\n--- Cleaned Data Preview ---")
print(clean_df.head(10).to_string())


# =====================================================================
# STEP 3: DATA INTEGRATION
# =====================================================================
header("STEP 3: DATA INTEGRATION")

def integrate_data(df):
    """
    Simulate integration with a secondary source:
    - Clinical notes / lifestyle metadata (BMI, smoker, diabetes)
    - Merge on a synthetic patient_id
    """
    df = df.copy()
    df.insert(0, "patient_id", [f"P{str(i).zfill(4)}" for i in range(len(df))])

    np.random.seed(42)
    lifestyle = pd.DataFrame({
        "patient_id": df["patient_id"],
        "bmi":        np.round(np.random.normal(27, 4, len(df)).clip(15, 45), 1),
        "smoker":     np.random.choice([0, 1], len(df), p=[0.6, 0.4]),
        "diabetes":   np.random.choice([0, 1], len(df), p=[0.85, 0.15]),
    })

    print(f"[+] Secondary source (lifestyle): shape={lifestyle.shape}")
    merged = df.merge(lifestyle, on="patient_id", how="left")
    print(f"[+] Integrated dataset shape: {merged.shape}")
    print(f"[+] Columns: {list(merged.columns)}")
    return merged


integrated_df = integrate_data(clean_df)
print("\n--- Integrated Data Preview ---")
print(integrated_df.head(10).to_string())


# =====================================================================
# STEP 4: DATA TRANSFORMATION
# =====================================================================
header("STEP 4: DATA TRANSFORMATION")

def transform_data(df):
    df = df.copy()

    # 4.1 Feature engineering
    df["age_group"]     = pd.cut(df["age"], bins=[0,40,55,70,120],
                                 labels=["young","middle","senior","elderly"]).astype(str)
    df["chol_ratio"]    = df["chol"] / df["age"]
    df["hr_reserve"]    = 220 - df["age"] - df["thalach"]
    df["bp_chol_risk"]  = df["trestbps"] * df["chol"] / 1000
    df["metabolic_risk"]= df["bmi"] * df["fbs"] + df["diabetes"]
    print("[+] Engineered 5 new features.")

    # 4.2 One-hot encode categorical
    cat_cols = ["cp", "restecg", "slope", "thal", "age_group"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f"[+] One-hot encoded {len(cat_cols)} categorical features.")

    # 4.3 Split features/target
    drop_cols = ["patient_id", "target"]
    X = df.drop(columns=drop_cols)
    y = df["target"]

    # 4.4 Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4.5 Handle class imbalance via SMOTE
    print(f"[+] Class distribution BEFORE SMOTE: {dict(y_train.value_counts())}")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"[+] Class distribution AFTER  SMOTE: {dict(pd.Series(y_train_bal).value_counts())}")

    # 4.6 Scaling
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train_bal), columns=X.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)

    print(f"[+] Final shapes -> Train: {X_train_s.shape}, Test: {X_test_s.shape}")
    return X_train_s, X_test_s, y_train_bal, y_test, X.columns


X_train, X_test, y_train, y_test, feature_names = transform_data(integrated_df)


# =====================================================================
# STEP 5: MODELING
# =====================================================================
header("STEP 5: MODEL TRAINING")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
}

results = {}
trained = {}

for name, model in models.items():
    print(f"\n[>] Training: {name} ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()

    results[name] = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1":        f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_prob),
        "CV-AUC":    cv,
    }
    trained[name] = model


# =====================================================================
# STEP 6: EVALUATION & REPORTING
# =====================================================================
header("STEP 6: EVALUATION")

results_df = pd.DataFrame(results).T.round(4)
print("\n--- Model Comparison ---")
print(results_df.to_string())

best_name = results_df["ROC-AUC"].idxmax()
best_model = trained[best_name]
print(f"\n[★] Best Model (by ROC-AUC): {best_name}")

y_pred_best = best_model.predict(X_test)
print("\n--- Classification Report (Best Model) ---")
print(classification_report(y_test, y_pred_best,
                            target_names=["No Disease", "Disease"]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
print("--- Confusion Matrix ---")
print(pd.DataFrame(cm,
                   index=["Actual No","Actual Yes"],
                   columns=["Pred No","Pred Yes"]).to_string())

# Feature importance (if tree-based)
if hasattr(best_model, "feature_importances_"):
    header("FEATURE IMPORTANCE (Top 15)")
    imp = pd.Series(best_model.feature_importances_,
                    index=feature_names).sort_values(ascending=False)
    print(imp.head(15).to_string())


# =====================================================================
# STEP 7: VISUALIZATION (saved to outputs/)
# =====================================================================
header("STEP 7: VISUALIZATIONS")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Target distribution
integrated_df["target"].value_counts().plot(
    kind="bar", ax=axes[0, 0], color=["#4C72B0","#C44E52"])
axes[0, 0].set_title("Target Distribution")
axes[0, 0].set_xticklabels(["No Disease","Disease"], rotation=0)

# (b) Correlation heatmap
num_df = integrated_df.select_dtypes(include=[np.number])
sns.heatmap(num_df.corr(), cmap="coolwarm", ax=axes[0, 1],
            cbar_kws={"shrink": 0.7})
axes[0, 1].set_title("Correlation Heatmap")

# (c) ROC curves
for name in models:
    y_prob = trained[name].predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1, 0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test,y_prob):.3f})")
axes[1, 0].plot([0,1],[0,1],"k--",alpha=0.5)
axes[1, 0].set_xlabel("FPR"); axes[1, 0].set_ylabel("TPR")
axes[1, 0].set_title("ROC Curves"); axes[1, 0].legend(fontsize=8)

# (d) Model comparison bar chart
results_df[["Accuracy","F1","ROC-AUC"]].plot(kind="bar", ax=axes[1, 1])
axes[1, 1].set_title("Model Metrics Comparison")
axes[1, 1].set_ylim(0, 1.05)
axes[1, 1].tick_params(axis='x', rotation=20)

plt.tight_layout()
out_path = "outputs/healthcare_report.png"
plt.savefig(out_path, dpi=120)
print(f"[+] Saved plots to: {out_path}")

# Save processed dataset
integrated_df.to_csv("outputs/processed_heart.csv", index=False)
print(f"[+] Saved processed dataset to: outputs/processed_heart.csv")

header("PIPELINE COMPLETE ✅")
print(f"Records processed : {len(integrated_df)}")
print(f"Features used     : {len(feature_names)}")
print(f"Best model        : {best_name}")
print(f"Best ROC-AUC      : {results_df.loc[best_name, 'ROC-AUC']:.4f}")
print(SEP)