import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

DATA_PATH = "data/processed/nhanes_ml_dataset.csv"

df = pd.read_csv(DATA_PATH)

target = "ckd_target"

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

X = df[features]
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

xgb = XGBClassifier(
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

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

lr = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

voting_model = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("rf", rf),
        ("xgb", xgb)
    ],
    voting="soft"
)

stacking_model = StackingClassifier(
    estimators=[
        ("lr", lr),
        ("rf", rf),
        ("xgb", xgb)
    ],
    final_estimator=LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),
    stack_method="predict_proba",
    n_jobs=-1
)

models = {
    "XGBoost_All_25": xgb,
    "Voting_All_25": voting_model,
    "Stacking_All_25": stacking_model
}

results = []
predictions = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    print(f"\nTraining {name}...")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    specificity = confusion_matrix(y_test, y_pred).ravel()[0] / (
        confusion_matrix(y_test, y_pred).ravel()[0] +
        confusion_matrix(y_test, y_pred).ravel()[1]
    )
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "ROC-AUC": roc_auc
    })

    predictions[name] = {
        "pipeline": pipeline,
        "y_pred": y_pred,
        "y_prob": y_prob
    }

    model_filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(
        pipeline,
        f"models/nhanes_final_{model_filename}"
    )

    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"Specificity : {specificity:.4f}")
    print(f"F1          : {f1:.4f}")
    print(f"ROC-AUC     : {roc_auc:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv(
    "results/nhanes_final_test_results.csv",
    index=False
)

print("\nFinal Results")
print(results_df.round(4))


# --------------------------------------------------
# CONFUSION MATRICES
# --------------------------------------------------

for name, data in predictions.items():

    cm = confusion_matrix(y_test, data["y_pred"])

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.imshow(cm)

    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-CKD", "CKD"])
    ax.set_yticklabels(["Non-CKD", "CKD"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j],
                    ha="center",
                    va="center",
                    fontsize=14)

    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(
        f"results/confusion_matrix_{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# --------------------------------------------------
# ROC CURVE
# --------------------------------------------------

plt.figure(figsize=(8, 6))

for name, data in predictions.items():

    fpr, tpr, _ = roc_curve(
        y_test,
        data["y_prob"]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={roc_auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - NHANES CKD Classification")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/nhanes_roc_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# PRECISION-RECALL CURVE
# --------------------------------------------------

plt.figure(figsize=(8, 6))

for name, data in predictions.items():

    precision, recall, _ = precision_recall_curve(
        y_test,
        data["y_prob"]
    )

    pr_auc = auc(recall, precision)

    plt.plot(
        recall,
        precision,
        label=f"{name} (AUC={pr_auc:.3f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves - NHANES CKD Classification")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/nhanes_precision_recall_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# XGBOOST FEATURE IMPORTANCE
# --------------------------------------------------

xgb_pipeline = predictions["XGBoost_All_25"]["pipeline"]

xgb_model = xgb_pipeline.named_steps["model"]

importance = xgb_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)

feature_importance.to_csv(
    "results/nhanes_xgb_feature_importance.csv",
    index=False
)

top_features = feature_importance.head(15)

plt.figure(figsize=(9, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 XGBoost Feature Importance")

plt.tight_layout()

plt.savefig(
    "results/nhanes_xgb_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nTop 15 Features")
print(top_features.to_string(index=False))

print("\nAll analysis completed successfully.")