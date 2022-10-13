#!/usr/bin/env python3
# Base imports
from typing import Tuple
import os
import pickle as pk
# External imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Cascid imports
from cascid.configs import pad_ufes_cnf
from cascid.datasets.pad_ufes import install, images

TRAIN_TEST_CACHE_RAW = pad_ufes_cnf.PAD_UFES_DIR / 'train_test_cache_raw.pkl'
TRAIN_TEST_CACHE_HAIRLESS = pad_ufes_cnf.PAD_UFES_DIR / 'train_test_cache_hairless.pkl'

def get_df() -> pd.DataFrame:
    """
    Read dataframe from metadata for this dataset.
    If dataset has not been installed yet, will install it, then return the dataframe as normal.
    """
    try:
        df = pd.read_csv(pad_ufes_cnf.METADATA)
    except FileNotFoundError as e:
        print("File not found: ", e)
        print("Downloading dataset now:")
        install.install_data_ufes(True)
        return pd.read_csv(pad_ufes_cnf.METADATA) 
    return df 

def _load_cache(path):
    with open(TRAIN_TEST_CACHE_RAW, 'rb') as fl:
        split = pk.load(fl)
    return (
        split['x_train'],
        split['x_test'],
        split['y_train'],
        split['y_test'],
    )

def _save_cache(path, x_train, x_test, y_train, y_test):
    split = dict()
    split['x_train'], = x_train
    split['x_test'], = x_test
    split['y_train'], = y_train
    split['y_test'], = y_test
    with open(TRAIN_TEST_CACHE_RAW, 'wb') as fl:
        split = pk.dump(split,fl)

def get_train_test_images_raw(test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Automated caching implementation of sklearn's train_test_split.
    This function uses raw images from the dataset, loaded at 256x256, in RGB.
    Args:
    test_size: Percent size of test split, if 0.2, 20% of data will be in test, and 80% in training.
    random_state: Passed directly to train_test_split, random seed to ensure reproducibility.

    Returns:
    x_train, x_test, y_train, y_test 
    """
    try:
        x_train, x_test, y_train, y_test = _load_cache(TRAIN_TEST_CACHE_RAW)
    except FileNotFoundError:
        df = get_df()
        x = df['img_id'].apply(lambda x: images.get_raw_image(x, (256,256))).to_numpy()
        x = np.array([x[i] for i in range(len(x))])
        y=df['diagnostic'].to_numpy().reshape(-1,1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state, test_size=test_size)
        _save_cache(TRAIN_TEST_CACHE_RAW)
        return x_train, x_test, y_train, y_test

def get_train_test_images_hairless(test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Automated caching implementation of sklearn's train_test_split.
    This function uses preprocessed images (reduced hair) from the dataset, loaded at 256x256, in RGB.
    Args:
    test_size: Percent size of test split, if 0.2, 20% of data will be in test, and 80% in training.
    random_state: Passed directly to train_test_split, random seed to ensure reproducibility.

    Returns:
    x_train, x_test, y_train, y_test 
    """
    try:
        x_train, x_test, y_train, y_test = _load_cache(TRAIN_TEST_CACHE_HAIRLESS)
    except FileNotFoundError:
        df = get_df()
        x = df['img_id'].apply(lambda x: images.get_hairless_image(x, (256,256))).to_numpy()
        x = np.array([x[i] for i in range(len(x))])
        y=df['diagnostic'].to_numpy().reshape(-1,1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state, test_size=test_size)
        _save_cache(TRAIN_TEST_CACHE_HAIRLESS)
        return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    # Test function
    print(get_df().head(5).transpose())