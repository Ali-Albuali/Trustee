import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc
)
import lightgbm as lgb
import joblib


# Load the dataset
df = pd.read_csv("Base.csv", sep=None, engine="python")

# Clean column names
df.columns = df.columns.str.strip()

print("Columns:")
print(df.columns.tolist())


# -----------------------------
# Keep all fraud rows and only
# 50% of non-fraud rows
# -----------------------------
fraud_df = df[df["fraud_bool"] == 1]
nonfraud_df = df[df["fraud_bool"] == 0]

nonfraud_df = nonfraud_df.sample(
    frac=0.16,
    random_state=42
)

df = pd.concat([fraud_df, nonfraud_df])

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nNew class distribution:")
print(df["fraud_bool"].value_counts())

print("\nNew class percentages:")
print(df["fraud_bool"].value_counts(normalize=True) * 100)


# -----------------------------
# Target column
# -----------------------------
target_col = "fraud_bool"

X = df.drop(columns=[target_col])
y = df[target_col]


# Replace -1 with NaN
X = X.replace(-1, np.nan)


# Detect feature types
categorical_cols = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

numeric_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Numerical preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)


# Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

preprocessor.set_output(transform="pandas")


# LightGBM model
model = lgb.LGBMClassifier(
    random_state=42,
    n_estimators=250,
    learning_rate=0.05,
    max_depth=6,
    verbosity=-1
)


# Pipeline
clf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ]
)


# Train
print("\nTraining model...")
clf_pipeline.fit(X_train, y_train)


# Predictions
y_pred = clf_pipeline.predict(X_test)
y_pred_proba = clf_pipeline.predict_proba(X_test)[:, 1]


# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))


# PR-AUC
precision, recall, _ = precision_recall_curve(
    y_test,
    y_pred_proba
)

pr_auc = auc(recall, precision)

print(f"\nPR-AUC Score: {pr_auc:.4f}")


# Save model
joblib.dump(
    clf_pipeline,
    "lightgbm_fraud_model.pkl"
)

print("\nModel saved successfully as lightgbm_fraud_model.pkl")