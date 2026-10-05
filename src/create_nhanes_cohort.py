import pandas as pd
import numpy as np

INPUT = "data/processed/nhanes_1999_2018_merged.csv"
OUTPUT = "data/processed/nhanes_ckd_cohort.csv"

df = pd.read_csv(INPUT)

print("Initial rows:", len(df))

df["participant_id"] = (
    df["cycle"].astype(str) + "_" + df["SEQN"].astype(str)
)

df["age"] = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
df["sex"] = pd.to_numeric(df["RIAGENDR"], errors="coerce")
df["creatinine"] = pd.to_numeric(df["LBXSCR"], errors="coerce")
df["albumin_urine"] = pd.to_numeric(df["URXUMA"], errors="coerce")
df["creatinine_urine"] = pd.to_numeric(df["URXUCR"], errors="coerce")
df["bmi"] = pd.to_numeric(df["BMXBMI"], errors="coerce")
df["glucose"] = pd.to_numeric(df["LBXSGL"], errors="coerce")
df["bun"] = pd.to_numeric(df["LBXSBU"], errors="coerce")
df["systolic_bp"] = pd.to_numeric(df["BPXSY1"], errors="coerce")
df["diastolic_bp"] = pd.to_numeric(df["BPXDI1"], errors="coerce")

df = df[df["age"] >= 20].copy()

print("Adults >=20:", len(df))

df["uacr"] = (
    df["albumin_urine"] / df["creatinine_urine"]
) * 100

df.loc[
    (df["albumin_urine"] <= 0) |
    (df["creatinine_urine"] <= 0),
    "uacr"
] = np.nan

df["ckd_target"] = np.nan

valid = df["creatinine"].notna() & df["age"].notna() & df["sex"].notna()

female = df["sex"] == 2
male = df["sex"] == 1

df.loc[valid & female, "egfr"] = (
    142
    * np.minimum(df.loc[valid & female, "creatinine"] / 0.7, 1) ** -0.241
    * np.maximum(df.loc[valid & female, "creatinine"] / 0.7, 1) ** -1.200
    * 0.9938 ** df.loc[valid & female, "age"]
    * 1.012
)

df.loc[valid & male, "egfr"] = (
    142
    * np.minimum(df.loc[valid & male, "creatinine"] / 0.9, 1) ** -0.302
    * np.maximum(df.loc[valid & male, "creatinine"] / 0.9, 1) ** -1.200
    * 0.9938 ** df.loc[valid & male, "age"]
)

df.loc[
    df["egfr"].notna() | df["uacr"].notna(),
    "ckd_target"
] = 0

df.loc[
    (df["egfr"] < 60) | (df["uacr"] >= 30),
    "ckd_target"
] = 1

df = df[df["ckd_target"].notna()].copy()

print("\nFinal cohort:", len(df))
print("\nCKD distribution:")
print(df["ckd_target"].value_counts())
print("\nCKD percentage:")
print(df["ckd_target"].value_counts(normalize=True) * 100)

print("\nMissingness:")
print(
    df[
        [
            "age",
            "sex",
            "creatinine",
            "uacr",
            "egfr",
            "bmi",
            "glucose",
            "bun",
            "systolic_bp",
            "diastolic_bp",
            "ckd_target"
        ]
    ].isna().sum()
)

df.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)