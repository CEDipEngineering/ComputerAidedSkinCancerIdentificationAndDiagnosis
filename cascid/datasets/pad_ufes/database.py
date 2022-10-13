#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from cascid.configs import pad_ufes_cnf
from cascid.datasets.pad_ufes import install

def get_df() -> pd.DataFrame:
    try:
        df = pd.read_csv(pad_ufes_cnf.METADATA)
    except FileNotFoundError as e:
        print("File not found: ", e)
        print("Downloading dataset now:")
        install.install_data_ufes(True)
        return pd.read_csv(pad_ufes_cnf.METADATA) 
    return df 

if __name__ == "__main__":
    # Test function
    print(get_df().head(5).transpose())