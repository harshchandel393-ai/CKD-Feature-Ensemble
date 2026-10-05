from pathlib import Path
import pandas as pd

base = Path("data/raw/nhanes")
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

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

all_cycles = []

for cycle in cycles:
    folder = base / cycle

    print(f"\nProcessing {cycle}...")

    demo = pd.read_sas(folder / "demo.XPT")
    biochem = pd.read_sas(folder / "biochem.XPT")
    urine = pd.read_sas(folder / "urine.XPT")
    bmx = pd.read_sas(folder / "bmx.XPT")
    bpx = pd.read_sas(folder / "bpx.XPT")
    diq = pd.read_sas(folder / "diq.XPT")

    data = demo.merge(biochem, on="SEQN", how="left", suffixes=("", "_bio"))
    data = data.merge(urine, on="SEQN", how="left", suffixes=("", "_urine"))
    data = data.merge(bmx, on="SEQN", how="left", suffixes=("", "_bmx"))
    data = data.merge(bpx, on="SEQN", how="left", suffixes=("", "_bpx"))
    data = data.merge(diq, on="SEQN", how="left", suffixes=("", "_diq"))

    data["cycle"] = cycle

    print(f"Rows after merge: {len(data):,}")
    print(f"Columns after merge: {len(data.columns):,}")

    all_cycles.append(data)

final_data = pd.concat(all_cycles, ignore_index=True)

final_data.to_csv(
    output_dir / "nhanes_1999_2018_merged.csv",
    index=False
)

print("\n===================================")
print("NHANES MERGE COMPLETED")
print("===================================")
print(f"Total rows: {len(final_data):,}")
print(f"Total columns: {len(final_data.columns):,}")
print(f"Saved to: {output_dir / 'nhanes_1999_2018_merged.csv'}")