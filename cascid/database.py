#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from cascid.configs import pad_ufes
from cascid import install

def get_db() -> pd.DataFrame:
    try:
        df = pd.read_csv(pad_ufes.METADATA)
    except FileNotFoundError as e:
        print("File not found: ", e)
        print("Downloading dataset now:")
        install.install_data_ufes(True)
    return df

if __name__ == "__main__":
    # Test function
    print(get_db().head(5).transpose())