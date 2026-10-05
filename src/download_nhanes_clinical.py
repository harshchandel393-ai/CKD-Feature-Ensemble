from pathlib import Path
import requests
import pandas as pd

files = {
    "1999-2000": {
        "biochem": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/LAB18.XPT",
        "urine": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/LAB16.XPT"
    },
    "2001-2002": {
        "biochem": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/L40_B.XPT",
        "urine": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/L16_B.XPT"
    },
    "2003-2004": {
        "biochem": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/L40_C.XPT",
        "urine": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/L16_C.XPT"
    },
    "2005-2006": {
        "biochem": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/BIOPRO_D.XPT",
        "urine": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/ALB_CR_D.XPT"
    },
    "2007-2008": {
        "biochem": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/BIOPRO_E.XPT",
        "urine": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/ALB_CR_E.XPT"
    },
    "2009-2010": {
        "biochem": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/BIOPRO_F.XPT",
        "urine": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/ALB_CR_F.XPT"
    },
    "2011-2012": {
        "biochem": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/BIOPRO_G.XPT",
        "urine": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/ALB_CR_G.XPT"
    },
    "2013-2014": {
        "biochem": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/BIOPRO_H.XPT",
        "urine": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/ALB_CR_H.XPT"
    },
    "2015-2016": {
        "biochem": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BIOPRO_I.XPT",
        "urine": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/ALB_CR_I.XPT"
    },
    "2017-2018": {
        "biochem": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BIOPRO_J.XPT",
        "urine": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/ALB_CR_J.XPT"
    }
}

base = Path("data/raw/nhanes")

for cycle, cycle_files in files.items():
    folder = base / cycle
    folder.mkdir(parents=True, exist_ok=True)

    for name, url in cycle_files.items():
        output = folder / f"{name}.XPT"

        print(f"\nDownloading {cycle} - {name}...")

        response = requests.get(url, timeout=60)
        response.raise_for_status()

        content = response.content

        if b"<html" in content[:1000].lower() or b"<!doctype" in content[:1000].lower():
            raise RuntimeError(f"Invalid HTML response for {cycle} - {name}")

        output.write_bytes(content)

        df = pd.read_sas(output)

        print(f"Saved: {output}")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")

print("\nClinical NHANES files downloaded and verified successfully.")