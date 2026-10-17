"""
Healthcare Data Science Experiment #2
Exploratory Data Analysis (EDA) on Heart Disease Dataset
"""

import os
import io
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency, shapiro, skew, kurtosis

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
SEP = "=" * 78

def header(title):
    print(f"\n{SEP}\n>>> {title}\n{SEP}")

def sub(title):
    print(f"\n--- {title} ---")


# =====================================================================
# STEP 0: LOAD DATA (reuse from Exp #1 or auto-download)
# =====================================================================
header("STEP 0: LOADING DATA")

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "heart-disease/processed.cleveland.data")
COLS = ["age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal","target"]

def load_data():
    proc = "data/processed_heart.csv"
    if os.path.exists(proc):
        print(f"[+] Loading processed data: {proc}")
        df = pd.read_csv(proc)
        return df

    raw = "data/raw_heart.csv"
    if os.path.exists(raw):
        print(f"[+] Loading raw data: {raw}")
        df = pd.read_csv(raw, names=COLS, na_values="?")
    else:
        print(f"[+] Downloading: {URL}")
        try:
            r = requests.get(URL, timeout=30); r.raise_for_status()
            with open(raw, "w") as f: f.write(r.text)
            df = pd.read_csv(io.StringIO(r.text), names=COLS, na_values="?")
        except Exception as e:
            print(f"[!] Download failed ({e}); using synthetic sample")
            np.random.seed(42); n = 303
            df = pd.DataFrame({
                "age": np.random.randint(29,77,n), "sex": np.random.randint(0,2,n),
                "cp": np.random.randint(0,4,n), "trestbps": np.random.randint(94,200,n),
                "chol": np.random.randint(126,564,n), "fbs": np.random.randint(0,2,n),
                "restecg": np.random.randint(0,3,n), "thalach": np.random.randint(71,202,n),
                "exang": np.random.randint(0,2,n),
                "oldpeak": np.round(np.random.uniform(0,6.2,n),1),
                "slope": np.random.randint(0,3,n), "ca": np.random.randint(0,4,n),
                "thal": np.random.randint(1,4,n), "target": np.random.randint(0,5,n),
            })

    # Minimal clean so downstream EDA works
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].fillna(df[num].median())
    df["target"] = (df["target"] > 0).astype(int)
    df.to_csv(proc, index=False)
    print(f"[+] Cleaned & cached to {proc}")
    return df


df = load_data()
print(f"[+] Loaded shape: {df.shape}")


# =====================================================================
# STEP 1: DATA STRUCTURE OVERVIEW
# =====================================================================
header("STEP 1: DATA STRUCTURE OVERVIEW")

sub("First 5 rows")
print(df.head().to_string())

sub("Last 5 rows")
print(df.tail().to_string())

sub("Shape & Memory")
print(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")
print(f"Memory usage: {df.memory_usage(deep=True).sum()/1024:.2f} KB")

sub("Data Types")
dtype_df = pd.DataFrame({
    "dtype": df.dtypes,
    "non_null": df.notnull().sum(),
    "null": df.isnull().sum(),
    "null_%": (df.isnull().sum()/len(df)*100).round(2),
    "unique": df.nunique(),
    "sample": [df[c].dropna().iloc[0] if df[c].notnull().any() else None for c in df.columns]
})
print(dtype_df.to_string())


# =====================================================================
# STEP 2: DATA QUALITY CHECKS
# =====================================================================
header("STEP 2: DATA QUALITY CHECKS")

sub("Missing Values")
missing = df.isnull().sum().sum()
print(f"Total missing cells: {missing}")
if missing == 0:
    print("[✓] No missing values.")

sub("Duplicate Rows")
dups = df.duplicated().sum()
print(f"Duplicate rows: {dups}")
if dups == 0:
    print("[✓] No duplicates.")

sub("Constant / Zero-Variance Features")
const_cols = [c for c in df.columns if df[c].nunique() <= 1]
print(f"Constant columns: {const_cols if const_cols else 'None'}")

sub("Zero-Value Audit (physiologically suspicious)")
zero_suspects = ["chol", "trestbps", "thalach"]
for c in zero_suspects:
    n_zero = (df[c] == 0).sum()
    print(f"  {c:10s} -> zeros: {n_zero}")

sub("Target Class Balance")
tc = df["target"].value_counts().sort_index()
pct = (tc / len(df) * 100).round(2)
bal = pd.DataFrame({"count": tc, "percent": pct})
bal.index = ["No Disease (0)", "Disease (1)"]
print(bal.to_string())
imb = tc.max() / tc.min()
print(f"Imbalance ratio: {imb:.2f}  ({'balanced' if imb < 1.5 else 'imbalanced'})")


# =====================================================================
# STEP 3: UNIVARIATE ANALYSIS
# =====================================================================
header("STEP 3: UNIVARIATE ANALYSIS")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in num_cols if df[c].nunique() <= 5 and c != "target"]
cont_cols = [c for c in num_cols if c not in cat_cols + ["target"]]

sub(f"Continuous Features ({len(cont_cols)})")
desc = df[cont_cols].describe().T
desc["skew"] = df[cont_cols].skew()
desc["kurtosis"] = df[cont_cols].kurtosis()
desc["cv%"] = (desc["std"] / desc["mean"] * 100)
print(desc.round(3).to_string())

sub("Normality Test (Shapiro-Wilk, H0=normal)")
print(f"{'Feature':12s} {'W-stat':>10s} {'p-value':>12s} {'Normal?':>10s}")
for c in cont_cols:
    w, p = shapiro(df[c].dropna())
    verdict = "Yes" if p > 0.05 else "No"
    print(f"{c:12s} {w:10.4f} {p:12.4e} {verdict:>10s}")

sub(f"Categorical Features ({len(cat_cols)})")
for c in cat_cols:
    vc = df[c].value_counts().sort_index()
    print(f"\n  {c}:")
    for k, v in vc.items():
        print(f"    value={k}  count={v:4d}  ({v/len(df)*100:5.2f}%)")

sub("Skewness Interpretation")
for c in cont_cols:
    s = df[c].skew()
    if abs(s) < 0.5:   label = "approx symmetric"
    elif abs(s) < 1:   label = "moderately skewed"
    else:              label = "highly skewed"
    direction = "left" if s < 0 else "right"
    print(f"  {c:12s} skew={s:+.3f}  -> {label} ({direction})")


# =====================================================================
# STEP 4: OUTLIER ANALYSIS
# =====================================================================
header("STEP 4: OUTLIER ANALYSIS")

out_summary = []
for c in cont_cols:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    iqr_out = ((df[c] < lo) | (df[c] > hi)).sum()

    z = np.abs((df[c] - df[c].mean()) / df[c].std())
    z_out = (z > 3).sum()

    out_summary.append({
        "feature": c, "IQR_outliers": iqr_out,
        "Z>3_outliers": z_out,
        "IQR_%": round(iqr_out/len(df)*100, 2)
    })

out_df = pd.DataFrame(out_summary)
print(out_df.to_string(index=False))
print(f"\nTotal IQR-based outliers: {out_df['IQR_outliers'].sum()}")


# =====================================================================
# STEP 5: BIVARIATE ANALYSIS (vs TARGET)
# =====================================================================
header("STEP 5: BIVARIATE ANALYSIS vs TARGET")

g0 = df[df["target"] == 0]   # No disease
g1 = df[df["target"] == 1]   # Disease

sub("Group Means (Continuous Features)")
means = pd.DataFrame({
    "No_Disease": g0[cont_cols].mean(),
    "Disease":    g1[cont_cols].mean(),
})
means["diff"] = means["Disease"] - means["No_Disease"]
means["diff_%"] = (means["diff"] / means["No_Disease"] * 100).round(2)
print(means.round(3).to_string())

sub("Independent t-test (Continuous vs Target)")
print(f"{'Feature':12s} {'t-stat':>10s} {'p-value':>12s} {'Sig?':>6s} {'Effect(d)':>10s}")
for c in cont_cols:
    t, p = ttest_ind(g0[c], g1[c], equal_var=False)
    pooled = np.sqrt((g0[c].var() + g1[c].var()) / 2)
    d = (g1[c].mean() - g0[c].mean()) / pooled if pooled else 0
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"{c:12s} {t:10.4f} {p:12.4e} {sig:>6s} {d:+10.3f}")

sub("Chi-Square Test (Categorical vs Target)")
print(f"{'Feature':12s} {'chi2':>10s} {'p-value':>12s} {'Sig?':>6s}")
for c in cat_cols:
    ct = pd.crosstab(df[c], df["target"])
    chi2, p, dof, _ = chi2_contingency(ct)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"{c:12s} {chi2:10.4f} {p:12.4e} {sig:>6s}")

sub("Correlation with Target")
corr = df[num_cols].corr()["target"].drop("target").sort_values(key=abs, ascending=False)
print(corr.round(4).to_string())
print("\nTop positive correlates:", list(corr[corr > 0].head(3).index))
print("Top negative correlates:", list(corr[corr < 0].head(3).index))


# =====================================================================
# STEP 6: MULTIVARIATE ANALYSIS
# =====================================================================
header("STEP 6: MULTIVARIATE ANALYSIS")

sub("Full Correlation Matrix (top correlated pairs)")
corr_matrix = df[num_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
pairs = corr_matrix.where(mask).stack().reset_index()
pairs.columns = ["feat_1", "feat_2", "corr"]
pairs["abs_corr"] = pairs["corr"].abs()
top_pairs = pairs.sort_values("abs_corr", ascending=False).head(10)
print(top_pairs[["feat_1","feat_2","corr"]].to_string(index=False))

sub("Multicollinearity Check (|r| > 0.7)")
high_corr = pairs[pairs["abs_corr"] > 0.7]
if len(high_corr):
    print(high_corr[["feat_1","feat_2","corr"]].to_string(index=False))
else:
    print("[✓] No severe multicollinearity detected (threshold: 0.7).")

sub("Mutual Information (Feature -> Target)")
from sklearn.feature_selection import mutual_info_classif
X = df.drop(columns=["target"])
mi = mutual_info_classif(X, df["target"], random_state=42)
mi_s = pd.Series(mi, index=X.columns).sort_values(ascending=False)
print(mi_s.round(4).to_string())


# =====================================================================
# STEP 7: VISUALIZATIONS
# =====================================================================
header("STEP 7: GENERATING VISUALIZATIONS")

# ---------- Dashboard 1: Univariate ----------
fig, axes = plt.subplots(3, 4, figsize=(20, 13))
axes = axes.flatten()

for i, c in enumerate(cont_cols):
    axes[i].hist(df[c], bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[i].axvline(df[c].mean(),   color="red",    ls="--", lw=1.5, label="mean")
    axes[i].axvline(df[c].median(), color="orange", ls=":",  lw=1.5, label="median")
    axes[i].set_title(f"{c}\nskew={df[c].skew():.2f}", fontsize=10)
    axes[i].legend(fontsize=7)

# Categorical bar charts
for j, c in enumerate(cat_cols[: (len(axes) - len(cont_cols))]):
    vc = df[c].value_counts().sort_index()
    axes[len(cont_cols) + j].bar(vc.index.astype(str), vc.values,
                                 color="#55A868", edgecolor="white")
    axes[len(cont_cols) + j].set_title(f"{c} (categorical)", fontsize=10)

# Hide any extra subplots
for k in range(len(cont_cols) + len(cat_cols), len(axes)):
    axes[k].axis("off")

plt.suptitle("Dashboard 1 — Univariate Distributions", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("outputs/eda_dashboard_1.png", dpi=110, bbox_inches="tight")
plt.close()
print("[+] Saved outputs/eda_dashboard_1.png")

# ---------- Dashboard 2: Bivariate (boxplots vs target) ----------
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for i, c in enumerate(cont_cols):
    sns.boxplot(x="target", y=c, data=df, ax=axes[i],
                palette=["#4C72B0", "#C44E52"])
    axes[i].set_title(f"{c} vs target", fontsize=10)
    axes[i].set_xticklabels(["No Disease", "Disease"])
for k in range(len(cont_cols), len(axes)):
    axes[k].axis("off")
plt.suptitle("Dashboard 2 — Continuous Features vs Target", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("outputs/eda_dashboard_2.png", dpi=110, bbox_inches="tight")
plt.close()
print("[+] Saved outputs/eda_dashboard_2.png")

# ---------- Dashboard 3: Multivariate ----------
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# (a) Correlation heatmap
sns.heatmap(df[num_cols].corr(), cmap="coolwarm", center=0,
            annot=True, fmt=".2f", annot_kws={"size":7},
            cbar_kws={"shrink": 0.7}, ax=axes[0])
axes[0].set_title("Correlation Matrix")

# (b) Pairplot substitute: top-4 features vs target
top_feats = corr.abs().head(4).index.tolist()
sns.scatterplot(x=df[top_feats[0]], y=df[top_feats[1]],
                hue=df["target"], palette=["#4C72B0","#C44E52"],
                alpha=0.7, ax=axes[1])
axes[1].set_title(f"{top_feats[0]} vs {top_feats[1]}")

# (c) Target distribution pie
tc = df["target"].value_counts().sort_index()
axes[2].pie(tc.values, labels=["No Disease","Disease"], autopct="%1.1f%%",
            colors=["#4C72B0","#C44E52"], startangle=90,
            wedgeprops={"edgecolor":"white"})
axes[2].set_title("Target Balance")

plt.suptitle("Dashboard 3 — Multivariate View", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("outputs/eda_dashboard_3.png", dpi=110, bbox_inches="tight")
plt.close()
print("[+] Saved outputs/eda_dashboard_3.png")

# ---------- Bonus: Feature vs Target bar (MI scores) ----------
fig, ax = plt.subplots(figsize=(11, 6))
mi_s.head(12).iloc[::-1].plot(kind="barh", color="#8172B2", ax=ax)
ax.set_title("Top 12 Features by Mutual Information with Target")
ax.set_xlabel("Mutual Information Score")
plt.tight_layout()
plt.savefig("outputs/eda_dashboard_4_mi.png", dpi=110, bbox_inches="tight")
plt.close()
print("[+] Saved outputs/eda_dashboard_4_mi.png")


# =====================================================================
# STEP 8: EDA SUMMARY REPORT
# =====================================================================
header("STEP 8: EDA SUMMARY REPORT")

lines = []
def w(s=""):
    print(s); lines.append(s)

w(f"Dataset Shape         : {df.shape[0]} rows x {df.shape[1]} cols")
w(f"Numeric features      : {len(num_cols)}")
w(f"Continuous features   : {len(cont_cols)} -> {cont_cols}")
w(f"Categorical features  : {len(cat_cols)} -> {cat_cols}")
w(f"Missing values        : {missing}")
w(f"Duplicate rows        : {dups}")
w(f"Total IQR outliers    : {out_df['IQR_outliers'].sum()}")
w(f"Class imbalance ratio : {imb:.2f}")
w(f"Highly skewed feats   : {[c for c in cont_cols if abs(df[c].skew())>1]}")
w(f"Top MI feature        : {mi_s.index[0]} ({mi_s.iloc[0]:.4f})")
w(f"Top |corr| w/ target  : {corr.abs().idxmax()} ({corr.abs().max():.4f})")

with open("outputs/eda_summary.txt", "w") as f:
    f.write("\n".join(lines))
print(f"\n[+] Saved summary to: outputs/eda_summary.txt")

header("EDA COMPLETE ✅")
print("Files generated:")
print("  - outputs/eda_dashboard_1.png         (univariate)")
print("  - outputs/eda_dashboard_2.png         (bivariate)")
print("  - outputs/eda_dashboard_3.png         (multivariate)")
print("  - outputs/eda_dashboard_4_mi.png      (mutual info)")
print("  - outputs/eda_summary.txt             (text summary)")
print(SEP)