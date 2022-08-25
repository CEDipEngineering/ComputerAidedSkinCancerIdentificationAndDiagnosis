from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parents[1]/'data'
DB_FILE = DATA_DIR / "metadata.csv"

def get_db() -> pd.DataFrame:
    return pd.read_csv(DB_FILE)