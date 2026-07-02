import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import lightgbm as lgb
import joblib



df = pd.read_csv('Base_sample_40k.csv', delimiter=';')


target_col = 'fraud_bool'
X = df.drop(columns=[target_col])
y = df[target_col]


X = X.replace(-1, np.nan)

categorical_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

print(f"Dataset Shape: {df.shape}")
print(f"Class Distribution:\n{y.value_counts(normalize=True)}\n")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)
preprocessor.set_output(transform="pandas") 


fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()

model = lgb.LGBMClassifier(
   
    scale_pos_weight=np.sqrt(fraud_ratio), 
    random_state=42,
    n_estimators=250,     
    learning_rate=0.05,
    max_depth=6,          
    verbosity=-1
)

clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])


print("Training the Precision-Optimized LightGBM model...")
clf_pipeline.fit(X_train, y_train)


y_pred_proba = clf_pipeline.predict_proba(X_test)[:, 1]


DESIRED_THRESHOLD = 0.70
y_pred_tuned = (y_pred_proba >= DESIRED_THRESHOLD).astype(int)

print(f"\n--- Results using custom Threshold: {DESIRED_THRESHOLD} ---")
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred_tuned))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_tuned))


precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
print(f"PR-AUC Score: {pr_auc:.4f}")


joblib.dump(clf_pipeline, "lightgbm_fraud_model.pkl")

print("Model saved successfully as lightgbm_fraud_model.pkl")