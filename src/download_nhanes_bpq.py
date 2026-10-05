import requests
from pathlib import Path

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"

cycles = {
    "1999-2000": "1999",
    "2001-2002": "2001",
    "2003-2004": "2003",
    "2005-2006": "2005",
    "2007-2008": "2007",
    "2009-2010": "2009",
    "2011-2012": "2011",
    "2013-2014": "2013",
    "2015-2016": "2015",
    "2017-2018": "2017"
}

files = {
    "1999": "BPQ.XPT",
    "2001": "BPQ_B.XPT",
    "2003": "BPQ_C.XPT",
    "2005": "BPQ_D.XPT",
    "2007": "BPQ_E.XPT",
    "2009": "BPQ_F.XPT",
    "2011": "BPQ_G.XPT",
    "2013": "BPQ_H.XPT",
    "2015": "BPQ_I.XPT",
    "2017": "BPQ_J.XPT"
}

out_dir = Path("data/raw/nhanes")

for cycle, year in cycles.items():
    filename = files[year]
    url = f"{BASE}/{year}/DataFiles/{filename}"
    output = out_dir / cycle / "bpq.XPT"

    output.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    output.write_bytes(response.content)

    print(f"{cycle}: downloaded {len(response.content):,} bytes")

print("Done.")