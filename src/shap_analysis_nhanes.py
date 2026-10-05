import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")

os.makedirs("results", exist_ok=True)

DATA_PATH = "data/processed/nhanes_ml_dataset.csv"
MODEL_PATH = "models/nhanes_final_xgboost_all_25.joblib"

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

df = pd.read_csv(DATA_PATH)

X = df[features]

model = joblib.load(MODEL_PATH)

preprocessor = model[:-1]
xgb_model = model.named_steps["model"]

X_processed = preprocessor.transform(X)

X_processed = pd.DataFrame(
    X_processed,
    columns=features,
    index=X.index
)

sample_size = min(3000, len(X_processed))

X_sample = X_processed.sample(
    n=sample_size,
    random_state=42
)

print(f"SHAP samples: {len(X_sample)}")

explainer = shap.TreeExplainer(xgb_model)

shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_importance = pd.DataFrame({
    "Feature": features,
    "Mean_Absolute_SHAP": mean_abs_shap
})

shap_importance = shap_importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

shap_importance.to_csv(
    "results/nhanes_shap_feature_importance.csv",
    index=False
)

print("\nTop SHAP Features:")
print(
    shap_importance.head(15).to_string(index=False)
)


plt.figure()

shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=features,
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    "results/nhanes_shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


plt.figure()

shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=features,
    plot_type="bar",
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    "results/nhanes_shap_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nSHAP analysis completed successfully.")

print("\nFiles created:")
print("results/nhanes_shap_feature_importance.csv")
print("results/nhanes_shap_summary.png")
print("results/nhanes_shap_bar.png")