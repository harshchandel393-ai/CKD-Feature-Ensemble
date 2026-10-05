import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

INPUT = "data/processed/nhanes_ml_dataset.csv"

X_TRAIN = "data/processed/nhanes_X_train.csv"
X_TEST = "data/processed/nhanes_X_test.csv"
Y_TRAIN = "data/processed/nhanes_y_train.csv"
Y_TEST = "data/processed/nhanes_y_test.csv"

df = pd.read_csv(INPUT)

df["ckd_target"] = df["ckd_target"].astype(int)

X = df.drop(
    columns=["participant_id", "cycle", "ckd_target"]
)

y = df["ckd_target"]

numeric_features = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    )
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)

X_train_processed.to_csv(
    X_TRAIN,
    index=False
)

X_test_processed.to_csv(
    X_TEST,
    index=False
)

y_train.to_csv(
    Y_TRAIN,
    index=False
)

y_test.to_csv(
    Y_TEST,
    index=False
)

joblib.dump(
    preprocessor,
    "models/nhanes_preprocessor.joblib"
)

print("Original dataset:", df.shape)

print("Training set:", X_train.shape)
print("Test set:", X_test.shape)

print("\nProcessed training:", X_train_processed.shape)
print("Processed test:", X_test_processed.shape)

print("\nTraining target:")
print(y_train.value_counts())

print("\nTest target:")
print(y_test.value_counts())

print("\nProcessed missing values:")
print("Train:", X_train_processed.isna().sum().sum())
print("Test:", X_test_processed.isna().sum().sum())

print("\nPreprocessor saved:")
print("models/nhanes_preprocessor.joblib")