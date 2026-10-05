import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier


DATA_PATH = "data/processed/nhanes_ml_dataset.csv"

FEATURES = [
    "age",
    "bun",
    "systolic_bp",
    "glucose",
    "diabetes",
    "uric_acid",
    "waist"
]

TARGET = "ckd_target"

MODEL_PATH = "models/nhanes_final_xgboost_7_features.joblib"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=" * 60)
print("NHANES 7-FEATURE XGBOOST MODEL")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y = df[TARGET].astype(int)

print(f"\nDataset rows: {len(df)}")
print(f"Features: {FEATURES}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", model)
])

print("\nRunning 5-Fold Cross Validation...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=[
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ],
    n_jobs=-1
)

print("\n5-Fold CV Results")

print(f"Accuracy : {cv_results['test_accuracy'].mean():.4f}")
print(f"Precision: {cv_results['test_precision'].mean():.4f}")
print(f"Recall   : {cv_results['test_recall'].mean():.4f}")
print(f"F1       : {cv_results['test_f1'].mean():.4f}")
print(f"ROC-AUC  : {cv_results['test_roc_auc'].mean():.4f}")

print("\nTraining final 7-feature model...")

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
specificity = tn / (tn + fp)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nFinal Test Results")
print("-" * 40)

print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1          : {f1:.4f}")
print(f"ROC-AUC     : {roc_auc:.4f}")

results = pd.DataFrame([{
    "Model": "XGBoost_7_Features",
    "Features": ", ".join(FEATURES),
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "Specificity": specificity,
    "F1": f1,
    "ROC-AUC": roc_auc
}])

results.to_csv(
    "results/nhanes_7feature_test_results.csv",
    index=False
)

cv_summary = pd.DataFrame([{
    "Model": "XGBoost_7_Features",
    "Accuracy": cv_results["test_accuracy"].mean(),
    "Precision": cv_results["test_precision"].mean(),
    "Recall": cv_results["test_recall"].mean(),
    "F1": cv_results["test_f1"].mean(),
    "ROC-AUC": cv_results["test_roc_auc"].mean()
}])

cv_summary.to_csv(
    "results/nhanes_7feature_cv_results.csv",
    index=False
)

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\nModel saved:")
print(MODEL_PATH)

print("\nConfusion Matrix:")
print(cm)

print("\n" + "=" * 60)
print("7-FEATURE MODEL TRAINING COMPLETED")
print("=" * 60)