import pandas as pd
from pathlib import Path

# ============================================================
# 1. DATASET PATH
# ============================================================

DATA_PATH = Path("data/raw/chronic_kidney_disease.arff")

# ============================================================
# 2. READ ARFF FILE MANUALLY
# ============================================================

with open(DATA_PATH, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

# Find @data section
data_start = None

for i, line in enumerate(lines):
    if line.strip().lower() == "@data":
        data_start = i + 1
        break

if data_start is None:
    raise ValueError("@data section not found in ARFF file")

raw_rows = []

for line in lines[data_start:]:
    line = line.strip()

    if not line:
        continue

    # Ignore comments
    if line.startswith("%"):
        continue

    values = [x.strip() for x in line.split(",")]
    raw_rows.append(values)

# ============================================================
# 3. CHECK ROW LENGTHS
# ============================================================

print("\n========== DATASET STRUCTURE ==========")

print("Total raw rows:", len(raw_rows))

lengths = {}

for row in raw_rows:
    lengths[len(row)] = lengths.get(len(row), 0) + 1

print("Row length distribution:", lengths)

# Show problematic rows
for index, row in enumerate(raw_rows):
    if len(row) != 25:
        print("\nProblematic row:", index + 1)
        print("Number of values:", len(row))
        print(row)

# ============================================================
# 4. KEEP ONLY 25 EXPECTED COLUMNS
# ============================================================

columns = [
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "rbc",
    "pc",
    "pcc",
    "ba",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wc",
    "rc",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane",
    "class"
]

print("\nExpected columns:", len(columns))

# ============================================================
# 5. FIX ROWS
# ============================================================

rows = []

for row in raw_rows:

    if len(row) == 25:
        rows.append(row)

    elif len(row) > 25:
        # Keep first 24 features + final target
        fixed_row = row[:24] + [row[-1]]
        rows.append(fixed_row)

    elif len(row) < 25:
        # Pad missing values
        fixed_row = row + [None] * (25 - len(row))
        rows.append(fixed_row)

# ============================================================
# 6. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(rows, columns=columns)

# ============================================================
# 7. CLEAN VALUES
# ============================================================

df = df.map(
    lambda x: x.strip() if isinstance(x, str) else x
)

# Convert missing values
df = df.replace(
    ["?", "", "nan", "NaN", "None", "none"],
    pd.NA
)

# ============================================================
# 8. NUMERICAL COLUMNS
# ============================================================

numeric_columns = [
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wc",
    "rc"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ============================================================
# 9. DISPLAY BASIC INFORMATION
# ============================================================

print("\n========== DATASET INFORMATION ==========")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

# ============================================================
# 10. MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")

missing = df.isnull().sum()

print(missing)

print("\nTotal missing values:", missing.sum())

# ============================================================
# 11. TARGET DISTRIBUTION
# ============================================================

print("\n========== TARGET DISTRIBUTION ==========")

print(df["class"].value_counts(dropna=False))

print("\nUnique target values:")
print(df["class"].unique())

# ============================================================
# 12. DUPLICATES
# ============================================================

print("\n========== DUPLICATES ==========")

print("Duplicate rows:", df.duplicated().sum())

# ============================================================
# 13. CATEGORICAL VALUES
# ============================================================

categorical_columns = [
    "rbc",
    "pc",
    "pcc",
    "ba",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane"
]

print("\n========== CATEGORICAL VALUES ==========")

for col in categorical_columns:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))

# ============================================================
# 14. NUMERICAL SUMMARY
# ============================================================

print("\n========== NUMERICAL SUMMARY ==========")

print(df[numeric_columns].describe())

print("\n========================================")
print("DATASET INSPECTION COMPLETED")
print("========================================")