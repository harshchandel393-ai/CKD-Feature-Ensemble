from pathlib import Path
import requests
import pandas as pd

files = {
    "1999-2000": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/DEMO.XPT",
    "2001-2002": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/DEMO_B.XPT",
    "2003-2004": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/DEMO_C.XPT",
    "2005-2006": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/DEMO_D.XPT",
    "2007-2008": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/DEMO_E.XPT",
    "2009-2010": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/DEMO_F.XPT",
    "2011-2012": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/DEMO_G.XPT",
    "2013-2014": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DEMO_H.XPT",
    "2015-2016": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DEMO_I.XPT",
    "2017-2018": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.XPT",
}

base = Path("data/raw/nhanes")
base.mkdir(parents=True, exist_ok=True)

for cycle, url in files.items():
    folder = base / cycle
    folder.mkdir(parents=True, exist_ok=True)

    output = folder / "demo.XPT"

    print(f"\nDownloading {cycle}...")

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    content = response.content

    if b"<html" in content[:1000].lower() or b"<!doctype" in content[:1000].lower():
        raise RuntimeError(f"Invalid HTML response received for {cycle}")

    output.write_bytes(content)

    df = pd.read_sas(output)

    print(f"Saved: {output}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

print("\nAll NHANES DEMO files downloaded and verified successfully.")