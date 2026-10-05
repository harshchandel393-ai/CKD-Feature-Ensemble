import pandas as pd
import numpy as np

INPUT = "data/processed/nhanes_ckd_cohort.csv"
OUTPUT = "data/processed/nhanes_ml_dataset.csv"

df = pd.read_csv(INPUT)

features = {
    "age": "RIDAGEYR",
    "sex": "RIAGENDR",
    "bmi": "BMXBMI",
    "weight": "BMXWT",
    "waist": "BMXWAIST",
    "systolic_bp": "BPXSY1",
    "diastolic_bp": "BPXDI1",
    "glucose": "LBXSGL",
    "bun": "LBXSBU",
    "total_cholesterol": "LBXSCH",
    "uric_acid": "LBXSUA",
    "albumin": "LBXSAL",
    "total_protein": "LBXSTP",
    "calcium": "LBXSCA",
    "phosphorus": "LBXSPH",
    "sodium": "LBXSNASI",
    "potassium": "LBXSKSI",
    "chloride": "LBXSCLSI",
    "bilirubin": "LBXSTB",
    "triglycerides": "LBXSTR",
    "ast": "LBXSATSI",
    "alt": "LBXSASSI",
    "alp": "LBXSAPSI",
    "ggt": "LBXSGTSI",
    "diabetes": "DIQ010",
    "bp_told_high": "BPQ020"
}

available = {
    new: old
    for new, old in features.items()
    if old in df.columns
}

print("Available features:")
print(list(available.keys()))

columns = [
    "participant_id",
    "cycle",
    "ckd_target"
] + list(available.values())

ml = df[columns].copy()

ml = ml.rename(columns={
    old: new for new, old in available.items()
})

for col in ml.columns:
    if col not in ["participant_id", "cycle"]:
        ml[col] = pd.to_numeric(
            ml[col],
            errors="coerce"
        )

missing_codes = [
    7, 9,
    77, 99,
    777, 999,
    7777, 9999
]

for col in ml.columns:
    if col not in ["participant_id", "cycle"]:
        ml.loc[
            ml[col].isin(missing_codes),
            col
        ] = np.nan

ml["sex"] = ml["sex"].replace({
    1: 1,
    2: 0
})

ml["diabetes"] = ml["diabetes"].replace({
    1: 1,
    2: 0
})



leakage_columns = [
    "creatinine",
    "egfr",
    "uacr",
    "albumin_urine",
    "creatinine_urine"
]

ml = ml.drop(
    columns=[
        c for c in leakage_columns
        if c in ml.columns
    ]
)

print("\nFinal ML dataset shape:", ml.shape)

print("\nTarget distribution:")
print(ml["ckd_target"].value_counts())

print("\nTarget percentage:")
print(
    ml["ckd_target"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nMissing values:")
print(
    ml.isna()
    .sum()
    .sort_values(ascending=False)
)

ml.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:", OUTPUT)