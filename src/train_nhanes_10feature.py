import os
import pandas as pd
import numpy as np

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
import joblib


DATA_PATH = "data/processed/nhanes_ml_dataset.csv"
MODEL_PATH = "models/nhanes_final_xgboost_10_features.joblib"
TEST_RESULTS_PATH = "results/nhanes_10feature_test_results.csv"
CV_RESULTS_PATH = "results/nhanes_10feature_cv_results.csv"


FEATURES = [
    "age",
    "bun",
    "systolic_bp",
    "glucose",
    "diabetes",
    "uric_acid",
    "waist",
    "albumin",
    "bmi",
    "total_cholesterol"
]

TARGET = "ckd_target"


print("=" * 60)
print("NHANES 10-FEATURE XGBOOST MODEL")
print("=" * 60)


df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y = df[TARGET]

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

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

cv_results = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)

print("\n5-Fold CV Results")
print(f"Accuracy : {cv_results['test_accuracy'].mean():.4f}")
print(f"Precision: {cv_results['test_precision'].mean():.4f}")
print(f"Recall   : {cv_results['test_recall'].mean():.4f}")
print(f"F1       : {cv_results['test_f1'].mean():.4f}")
print(f"ROC-AUC  : {cv_results['test_roc_auc'].mean():.4f}")


print("\nTraining final 10-feature model...")

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

specificity = tn / (tn + fp)


print("\nFinal Test Results")
print("-" * 40)
print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1          : {f1:.4f}")
print(f"ROC-AUC     : {roc_auc:.4f}")


os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

joblib.dump(pipeline, MODEL_PATH)


test_results = pd.DataFrame([{
    "model": "XGBoost 10 Features",
    "features": len(FEATURES),
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "specificity": specificity,
    "f1": f1,
    "roc_auc": roc_auc
}])

test_results.to_csv(TEST_RESULTS_PATH, index=False)


cv_results_df = pd.DataFrame([{
    "model": "XGBoost 10 Features",
    "cv_accuracy": cv_results["test_accuracy"].mean(),
    "cv_precision": cv_results["test_precision"].mean(),
    "cv_recall": cv_results["test_recall"].mean(),
    "cv_f1": cv_results["test_f1"].mean(),
    "cv_roc_auc": cv_results["test_roc_auc"].mean()
}])

cv_results_df.to_csv(CV_RESULTS_PATH, index=False)


print(f"\nModel saved:")
print(MODEL_PATH)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n" + "=" * 60)
print("10-FEATURE MODEL TRAINING COMPLETED")
print("=" * 60)