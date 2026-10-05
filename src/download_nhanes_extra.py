from pathlib import Path
import requests
import pandas as pd

files = {
    "1999-2000": {
        "bmx": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/BMX.XPT",
        "bpx": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/BPX.XPT",
        "diq": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/DIQ.XPT",
    },
    "2001-2002": {
        "bmx": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/BMX_B.XPT",
        "bpx": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/BPX_B.XPT",
        "diq": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/DIQ_B.XPT",
    },
    "2003-2004": {
        "bmx": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/BMX_C.XPT",
        "bpx": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/BPX_C.XPT",
        "diq": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/DIQ_C.XPT",
    },
    "2005-2006": {
        "bmx": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/BMX_D.XPT",
        "bpx": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/BPX_D.XPT",
        "diq": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/DIQ_D.XPT",
    },
    "2007-2008": {
        "bmx": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/BMX_E.XPT",
        "bpx": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/BPX_E.XPT",
        "diq": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/DIQ_E.XPT",
    },
    "2009-2010": {
        "bmx": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/BMX_F.XPT",
        "bpx": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/BPX_F.XPT",
        "diq": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/DIQ_F.XPT",
    },
    "2011-2012": {
        "bmx": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/BMX_G.XPT",
        "bpx": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/BPX_G.XPT",
        "diq": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/DIQ_G.XPT",
    },
    "2013-2014": {
        "bmx": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/BMX_H.XPT",
        "bpx": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/BPX_H.XPT",
        "diq": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DIQ_H.XPT",
    },
    "2015-2016": {
        "bmx": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BMX_I.XPT",
        "bpx": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BPX_I.XPT",
        "diq": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DIQ_I.XPT",
    },
    "2017-2018": {
        "bmx": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.XPT",
        "bpx": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPX_J.XPT",
        "diq": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DIQ_J.XPT",
    },
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
            raise RuntimeError(f"Invalid HTML response: {cycle} - {name}")

        output.write_bytes(content)

        df = pd.read_sas(output)

        print(f"Saved: {output}")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")

print("\nBMI, BP and diabetes files downloaded successfully.")