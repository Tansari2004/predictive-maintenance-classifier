from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


DATA_URL = "https://archive.ics.uci.edu/static/public/601/data.csv"
FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]
TARGET = "Machine failure"


def fetch_data(path: Path = Path("data/ai4i2020.csv")) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urlretrieve(DATA_URL, path)
    frame = pd.read_csv(path)
    required = set(FEATURES + [TARGET])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    if len(frame) != 10_000:
        raise ValueError(f"Expected 10,000 rows, found {len(frame):,}")
    return frame
