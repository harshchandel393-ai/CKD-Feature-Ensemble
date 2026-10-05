import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/nhanes_1999_2018_merged.csv")
OUTPUT = Path("data/processed/nhanes_1999_2018_merged_v2.csv")

cycles = [
    "1999-2000",
    "2001-2002",
    "2003-2004",
    "2005-2006",
    "2007-2008",
    "2009-2010",
    "2011-2012",
    "2013-2014",
    "2015-2016",
    "2017-2018"
]

df = pd.read_csv(INPUT)

parts = []

for cycle in cycles:
    base = df[df["cycle"] == cycle].copy()

    bpq = pd.read_sas(
        f"data/raw/nhanes/{cycle}/bpq.XPT"
    )

    base = base.merge(
        bpq,
        on="SEQN",
        how="left",
        suffixes=("", "_bpq")
    )

    parts.append(base)

    print(cycle, ":", len(base), "rows")

result = pd.concat(parts, ignore_index=True)

result.to_csv(OUTPUT, index=False)

print("\nFinal shape:", result.shape)
print("Saved:", OUTPUT)