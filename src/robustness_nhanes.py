import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
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


INPUT = "data/processed/nhanes_ml_dataset.csv"
OUTPUT = "results/nhanes_repeated_cv_results.csv"

Path("results").mkdir(exist_ok=True)

df = pd.read_csv(INPUT)

df["ckd_target"] = df["ckd_target"].astype(int)

X = df.drop(
    columns=[
        "participant_id",
        "cycle",
        "ckd_target"
    ]
)

y = df["ckd_target"]


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

all_features = X.columns.tolist()


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


feature_sets = {
    "XGBoost_Core_7": core_features,
    "XGBoost_Extended_10": extended_features,
    "XGBoost_All_25": all_features,
    "Voting_All_25": all_features,
    "Stacking_All_25": all_features
}


def create_xgb():

    return XGBClassifier(
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


def create_voting():

    lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=2
    )

    xgb = create_xgb()

    return VotingClassifier(
        estimators=[
            ("lr", lr),
            ("rf", rf),
            ("xgb", xgb)
        ],
        voting="soft"
    )


def create_stacking():

    lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=2
    )

    xgb = create_xgb()

    return StackingClassifier(
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
        cv=3,
        n_jobs=2
    )


def create_model(name):

    if name.startswith("XGBoost"):
        model = create_xgb()

    elif name == "Voting_All_25":
        model = create_voting()

    elif name == "Stacking_All_25":
        model = create_stacking()

    else:
        raise ValueError(
            f"Unknown model: {name}"
        )

    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            model
        )
    ])


def calculate_metrics(y_true, scores):

    predictions = (
        scores >= 0.5
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions
    ).ravel()

    specificity = tn / (tn + fp)

    return {
        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "Specificity": specificity,
        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_true,
            scores
        )
    }


cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)


results = []


for model_name, features in feature_sets.items():

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    X_selected = X_train[features]

    fold_results = []

    for fold_number, (train_index, valid_index) in enumerate(
        cv.split(X_selected, y_train),
        start=1
    ):

        X_fold_train = X_selected.iloc[
            train_index
        ]

        X_fold_valid = X_selected.iloc[
            valid_index
        ]

        y_fold_train = y_train.iloc[
            train_index
        ]

        y_fold_valid = y_train.iloc[
            valid_index
        ]

        model = create_model(
            model_name
        )

        model.fit(
            X_fold_train,
            y_fold_train
        )

        scores = model.predict_proba(
            X_fold_valid
        )[:, 1]

        metrics = calculate_metrics(
            y_fold_valid,
            scores
        )

        fold_results.append(
            metrics
        )

        print(
            f"Fold {fold_number:02d}/25 | "
            f"AUC: {metrics['ROC_AUC']:.4f} | "
            f"F1: {metrics['F1']:.4f}"
        )

    fold_df = pd.DataFrame(
        fold_results
    )

    print("\nSummary:")

    for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1",
        "ROC_AUC"
    ]:

        mean = fold_df[metric].mean()
        std = fold_df[metric].std(
            ddof=1
        )

        results.append({
            "Model": model_name,
            "Metric": metric,
            "Mean": mean,
            "Std": std,
            "Mean_Percent": mean * 100,
            "Std_Percent": std * 100
        })

        print(
            f"{metric}: "
            f"{mean * 100:.2f} ± "
            f"{std * 100:.2f}"
        )


results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT,
    index=False
)


summary = results_df.pivot(
    index="Model",
    columns="Metric",
    values="Mean_Percent"
)

print("\n" + "=" * 70)
print("ROBUSTNESS SUMMARY")
print("=" * 70)

print(
    summary[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "Specificity",
            "F1",
            "ROC_AUC"
        ]
    ].round(2).to_string()
)

print(
    f"\nSaved: {OUTPUT}"
)