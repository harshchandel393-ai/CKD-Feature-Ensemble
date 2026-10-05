from pathlib import Path
import requests
import pandas as pd

url = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/DEMO.XPT"

output_dir = Path("data/raw/nhanes/1999-2000")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "demo.XPT"

response = requests.get(url, timeout=60)
response.raise_for_status()

content_type = response.headers.get("Content-Type", "")
print("Status:", response.status_code)
print("Content-Type:", content_type)
print("Size:", len(response.content), "bytes")

if response.content.startswith(b"<!DOCTYPE html") or b"<html" in response.content[:500].lower():
    raise RuntimeError("CDC returned HTML instead of an XPT file.")

output_file.write_bytes(response.content)

df = pd.read_sas(output_file)

print("\nSUCCESS")
print("Shape:", df.shape)
print("Columns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())