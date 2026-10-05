import pandas as pd
import numpy as np

from sklearn.feature_selection import (
    mutual_info_classif,
    RFE
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

X_train = pd.read_csv(
    "data/processed/nhanes_X_train.csv"
)

y_train = pd.read_csv(
    "data/processed/nhanes_y_train.csv"
).squeeze()

feature_names = X_train.columns.tolist()

X = X_train.values
y = y_train.values

results = {}

correlations = []

for i, feature in enumerate(feature_names):
    corr = np.corrcoef(X[:, i], y)[0, 1]
    correlations.append(
        (feature, abs(corr))
    )

correlations.sort(
    key=lambda x: x[1],
    reverse=True
)

results["Pearson"] = [
    x[0] for x in correlations[:15]
]

mi_scores = mutual_info_classif(
    X,
    y,
    random_state=42
)

mi_ranked = sorted(
    zip(feature_names, mi_scores),
    key=lambda x: x[1],
    reverse=True
)

results["Mutual_Information"] = [
    x[0] for x in mi_ranked[:15]
]

lr = LogisticRegression(
    max_iter=3000,
    solver="liblinear"
)

rfe = RFE(
    estimator=lr,
    n_features_to_select=15
)

rfe.fit(X, y)

rfe_features = [
    feature_names[i]
    for i, selected in enumerate(rfe.support_)
    if selected
]

results["RFE"] = rfe_features

lasso = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    max_iter=3000
)

lasso.fit(X, y)

lasso_scores = np.abs(
    lasso.coef_[0]
)

lasso_ranked = sorted(
    zip(feature_names, lasso_scores),
    key=lambda x: x[1],
    reverse=True
)

results["LASSO"] = [
    x[0] for x in lasso_ranked[:15]
]

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf.fit(X, y)

rf_ranked = sorted(
    zip(feature_names, rf.feature_importances_),
    key=lambda x: x[1],
    reverse=True
)

results["Random_Forest"] = [
    x[0] for x in rf_ranked[:15]
]

print("\nFEATURE SELECTION RESULTS\n")

for method, features in results.items():
    print(f"\n{method}:")
    for rank, feature in enumerate(features, 1):
        print(f"{rank:2}. {feature}")

all_selected = []

for features in results.values():
    all_selected.extend(features)

frequency = pd.Series(
    all_selected
).value_counts()

print("\n\nFEATURE FREQUENCY ACROSS METHODS\n")

print(frequency)

frequency.to_csv(
    "results/nhanes_feature_frequency.csv"
)

feature_table = pd.DataFrame(
    dict([
        (method, pd.Series(features))
        for method, features in results.items()
    ])
)

feature_table.to_csv(
    "results/nhanes_feature_selection_results.csv",
    index=False
)

print(
    "\nSaved:"
    "\nresults/nhanes_feature_frequency.csv"
    "\nresults/nhanes_feature_selection_results.csv"
)