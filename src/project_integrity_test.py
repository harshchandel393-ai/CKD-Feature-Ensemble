import os
import joblib
import pandas as pd
import numpy as np

print("=" * 60)
print("CKD NHANES PROJECT INTEGRITY TEST")
print("=" * 60)

errors = []

def check_file(path):
    if os.path.exists(path):
        print(f"[OK] {path}")
        return True
    else:
        print(f"[MISSING] {path}")
        errors.append(path)
        return False


print("\n1. DATASET FILES")
print("-" * 60)

data_files = [
    "data/processed/nhanes_1999_2018_merged.csv",
    "data/processed/nhanes_1999_2018_merged_v2.csv",
    "data/processed/nhanes_ckd_cohort.csv",
    "data/processed/nhanes_ml_dataset.csv",
    "data/processed/nhanes_X_train.csv",
    "data/processed/nhanes_X_test.csv",
    "data/processed/nhanes_y_train.csv",
    "data/processed/nhanes_y_test.csv"
]

for file in data_files:
    check_file(file)


print("\n2. FINAL MODEL FILES")
print("-" * 60)

model_files = [
    "models/nhanes_final_xgboost_all_25.joblib",
    "models/nhanes_final_voting_all_25.joblib",
    "models/nhanes_final_stacking_all_25.joblib"
]

for file in model_files:
    check_file(file)


print("\n3. RESULT FILES")
print("-" * 60)

result_files = [
    "results/nhanes_feature_frequency.csv",
    "results/nhanes_feature_selection_results.csv",
    "results/nhanes_final_test_results.csv",
    "results/nhanes_model_results.csv",
    "results/nhanes_repeated_cv_results.csv",
    "results/nhanes_xgb_feature_importance.csv",
    "results/nhanes_shap_feature_importance.csv",
    "results/nhanes_roc_curves.png",
    "results/nhanes_precision_recall_curves.png",
    "results/nhanes_shap_summary.png",
    "results/nhanes_shap_bar.png",
    "results/nhanes_xgb_feature_importance.png"
]

for file in result_files:
    check_file(file)


print("\n4. ML DATASET TEST")
print("-" * 60)

try:
    df = pd.read_csv("data/processed/nhanes_ml_dataset.csv")

    print(f"[OK] Dataset loaded")
    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    if len(df) == 51713:
        print("[OK] Expected 51,713 rows found")
    else:
        print(f"[WARNING] Expected 51,713 rows but found {len(df)}")

    if "ckd_target" in df.columns:
        print("[OK] ckd_target exists")
    else:
        errors.append("ckd_target")
        print("[ERROR] ckd_target missing")

except Exception as e:
    print(f"[ERROR] Dataset loading failed: {e}")
    errors.append("ML dataset loading")


print("\n5. FEATURE TEST")
print("-" * 60)

features = [
    "age",
    "sex",
    "bmi",
    "weight",
    "waist",
    "systolic_bp",
    "diastolic_bp",
    "glucose",
    "bun",
    "total_cholesterol",
    "uric_acid",
    "albumin",
    "total_protein",
    "calcium",
    "phosphorus",
    "sodium",
    "potassium",
    "chloride",
    "bilirubin",
    "triglycerides",
    "ast",
    "alt",
    "alp",
    "ggt",
    "diabetes"
]

try:
    missing_features = [f for f in features if f not in df.columns]

    if not missing_features:
        print("[OK] All 25 model features exist")
    else:
        print("[ERROR] Missing features:")
        print(missing_features)
        errors.extend(missing_features)

except Exception as e:
    print(f"[ERROR] Feature test failed: {e}")
    errors.append("Feature test")


print("\n6. TARGET DISTRIBUTION")
print("-" * 60)

try:
    counts = df["ckd_target"].value_counts()

    print(f"Non-CKD (0): {counts.get(0, 0)}")
    print(f"CKD (1)    : {counts.get(1, 0)}")

    if set(counts.index).issubset({0, 1}):
        print("[OK] Target contains valid classes")
    else:
        print("[ERROR] Invalid target values")
        errors.append("Target values")

except Exception as e:
    print(f"[ERROR] Target test failed: {e}")
    errors.append("Target test")


print("\n7. FINAL MODEL LOAD TEST")
print("-" * 60)

loaded_models = {}

for model_path in model_files:

    if not os.path.exists(model_path):
        continue

    try:
        model = joblib.load(model_path)
        loaded_models[model_path] = model

        print(f"[OK] Loaded: {model_path}")
        print(f"     Type: {type(model).__name__}")

    except Exception as e:
        print(f"[ERROR] Failed: {model_path}")
        print(f"        {e}")
        errors.append(model_path)


print("\n8. MODEL PREDICTION TEST")
print("-" * 60)

try:
    X = df[features].head(5)

    for model_path, model in loaded_models.items():

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        print(f"\n[OK] {os.path.basename(model_path)}")
        print(f"Predictions : {predictions}")
        print(f"Probabilities: {np.round(probabilities, 4)}")

except Exception as e:
    print(f"[ERROR] Prediction test failed: {e}")
    errors.append("Model prediction test")


print("\n9. FINAL TEST RESULTS")
print("-" * 60)

try:
    results = pd.read_csv(
        "results/nhanes_final_test_results.csv"
    )

    print("[OK] Final results loaded")
    print(results.to_string(index=False))

except Exception as e:
    print(f"[ERROR] Final results failed: {e}")
    errors.append("Final results")


print("\n" + "=" * 60)

if errors:
    print("TEST STATUS: FAILED")
    print("=" * 60)
    print("\nProblems found:")
    for error in errors:
        print(f"- {error}")
else:
    print("TEST STATUS: ALL SYSTEMS PASSED")
    print("=" * 60)
    print("\nDataset       : PASS")
    print("Features      : PASS")
    print("Models        : PASS")
    print("Prediction    : PASS")
    print("Results       : PASS")
    print("Project files : PASS")

print("=" * 60)