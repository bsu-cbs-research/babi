import os
import pandas as pd

def parse_preprocessed_capnostream_file(file_path: str):
    raise ValueError("Preprocessed files are not supported yet")

def parse_raw_capnostream_file(file_path: str):
    """Parse a single raw capnostream file into predictable pkl format"""
    with open(file_path, "r", encoding="utf-16") as f:
        lines = f.readlines()

    metadata: dict[str, str] = {}
    columns: list[str] = []
    
    starting_row = 0
    for i, line in enumerate(lines):
        if i == 0: continue # skip first info row
        if line.startswith("Date"):
            columns = [c.strip() for c in line.split("\t")]
            starting_row = i + 2
            break
        else:
            metadata[line.split("\t")[0].strip()] = line.split("\t")[1].strip()

    df = pd.read_csv(file_path, encoding="utf-16", sep="\t", names=columns, skiprows=starting_row)
    df.attrs = {} | metadata
    df = df[["Date", "Time", "CO₂ Wave"]].rename(columns={"Date": "date", "Time": "time", "CO₂ Wave": "co2_wave"})
    df = df[df["co2_wave"] != "--"].copy()
    df["co2_wave"] = df["co2_wave"].astype(float)
    return df #? handle streaks (continuous data that stops then starts again)?

if __name__ == "__main__":
    print(os.getcwd())
    raw_example = os.path.join("./examples/raw-capnostream-example.csv")
    df = parse_raw_capnostream_file(raw_example)

    print(df.attrs)
    print(df.head())