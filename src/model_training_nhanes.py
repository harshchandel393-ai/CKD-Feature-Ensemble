import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier


TRAIN = "data/processed/nhanes_X_train.csv"
TEST = "data/processed/nhanes_X_test.csv"
Y_TRAIN = "data/processed/nhanes_y_train.csv"
Y_TEST = "data/processed/nhanes_y_test.csv"

RESULTS = "results/nhanes_model_results.csv"

Path("results").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

X_train = pd.read_csv(TRAIN)
X_test = pd.read_csv(TEST)

y_train = pd.read_csv(Y_TRAIN).squeeze()
y_test = pd.read_csv(Y_TEST).squeeze()

X_train.columns = [
    c.replace("numeric__", "")
    for c in X_train.columns
]

X_test.columns = X_train.columns

core_features = [
    "bun",
    "age",
    "systolic_bp",
    "glucose",
    "uric_acid",
    "diabetes",
    "waist"
]

extended_features = [
    "bun",
    "age",
    "systolic_bp",
    "glucose",
    "uric_acid",
    "diabetes",
    "waist",
    "albumin",
    "bmi",
    "total_cholesterol"
]

all_features = X_train.columns.tolist()

feature_sets = {
    "Core_7": core_features,
    "Extended_10": extended_features,
    "All_25": all_features
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    return model.decision_function(X)


def calculate_metrics(y_true, y_pred, y_score):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    specificity = tn / (tn + fp)

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "Specificity": specificity,
        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_true,
            y_score
        )
    }


def create_models():

    lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

    svm = LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=5000
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=2
    )

    voting = VotingClassifier(
        estimators=[
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced"
                )
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=2
                )
            ),
            (
                "xgb",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=2
                )
            )
        ],
        voting="soft"
    )

    stacking = StackingClassifier(
        estimators=[
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced"
                )
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=2
                )
            ),
            (
                "xgb",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=2
                )
            )
        ],
        final_estimator=LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),
        stack_method="predict_proba",
        cv=3,
        n_jobs=2
    )

    return {
        "Logistic_Regression": lr,
        "SVM": svm,
        "Random_Forest": rf,
        "Gradient_Boosting": gb,
        "XGBoost": xgb,
        "Voting": voting,
        "Stacking": stacking
    }


results = []

for feature_set_name, features in feature_sets.items():

    print("\n" + "=" * 70)
    print(f"FEATURE SET: {feature_set_name}")
    print("=" * 70)

    Xtr = X_train[features]
    Xte = X_test[features]

    models = create_models()

    for model_name, model in models.items():

        print(f"\nTraining: {model_name}")

        if model_name == "SVM":
            method = "decision_function"
        else:
            method = "predict_proba"

        cv_scores = cross_val_predict(
            model,
            Xtr,
            y_train,
            cv=cv,
            method=method,
            n_jobs=1
        )

        if cv_scores.ndim == 2:
            cv_score = cv_scores[:, 1]
            cv_pred = (
                cv_score >= 0.5
            ).astype(int)
        else:
            cv_score = cv_scores
            cv_pred = (
                cv_score >= 0
            ).astype(int)

        cv_metrics = calculate_metrics(
            y_train,
            cv_pred,
            cv_score
        )

        model.fit(
            Xtr,
            y_train
        )

        test_score = get_scores(
            model,
            Xte
        )

        if model_name == "SVM":
            test_pred = (
                test_score >= 0
            ).astype(int)
        else:
            test_pred = (
                test_score >= 0.5
            ).astype(int)

        test_metrics = calculate_metrics(
            y_test,
            test_pred,
            test_score
        )

        print(
            "CV:",
            {
                k: round(float(v), 4)
                for k, v in cv_metrics.items()
            }
        )

        print(
            "TEST:",
            {
                k: round(float(v), 4)
                for k, v in test_metrics.items()
            }
        )

        results.append({
            "Feature_Set": feature_set_name,
            "Model": model_name,

            "CV_Accuracy": cv_metrics["Accuracy"],
            "CV_Precision": cv_metrics["Precision"],
            "CV_Recall": cv_metrics["Recall"],
            "CV_Specificity": cv_metrics["Specificity"],
            "CV_F1": cv_metrics["F1"],
            "CV_ROC_AUC": cv_metrics["ROC_AUC"],

            "Test_Accuracy": test_metrics["Accuracy"],
            "Test_Precision": test_metrics["Precision"],
            "Test_Recall": test_metrics["Recall"],
            "Test_Specificity": test_metrics["Specificity"],
            "Test_F1": test_metrics["F1"],
            "Test_ROC_AUC": test_metrics["ROC_AUC"]
        })

        filename = (
            "models/nhanes_"
            f"{feature_set_name.lower()}_"
            f"{model_name.lower().replace(' ', '_')}.joblib"
        )

        joblib.dump(
            model,
            filename
        )

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Test_ROC_AUC",
    ascending=False
)

results_df.to_csv(
    RESULTS,
    index=False
)

print("\n" + "=" * 70)
print("FINAL MODEL RANKING")
print("=" * 70)

print(
    results_df[
        [
            "Feature_Set",
            "Model",
            "CV_ROC_AUC",
            "Test_ROC_AUC",
            "Test_Accuracy",
            "Test_Precision",
            "Test_Recall",
            "Test_Specificity",
            "Test_F1"
        ]
    ].to_string(index=False)
)

print(f"\nResults saved to: {RESULTS}")