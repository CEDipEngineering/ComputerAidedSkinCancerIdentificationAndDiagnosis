#!/usr/bin/env python3
# Base imports
from typing import Tuple
import os
import pickle as pk
from multiprocessing.pool import ThreadPool
# External imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Cascid imports
from cascid.configs import isic_cnf
from cascid.datasets.isic import fetcher, images

TRAIN_TEST_CACHE_RAW = isic_cnf.IMAGES_DIR / 'train_test_cache_raw.pkl'
TRAIN_TEST_CACHE_HAIRLESS = isic_cnf.IMAGES_DIR / 'train_test_cache_hairless.pkl'

def get_df() -> pd.DataFrame:
    """
    Read dataframe from metadata for this dataset.
    """
    df = pd.read_csv(isic_cnf.METADATA, index_col=0)
    df['diagnostic'] = df['diagnostic'].replace({
        'seborrheic keratosis' : 'SEK',
        'actinic keratosis' : 'ACK',
        'nevus' : 'NEV',
        'squamous cell carcinoma' : 'SCC',
        'basal cell carcinoma' : 'BCC',
        'melanoma' : 'MEL', 
    })
    return df

def update_all_files(df: pd.DataFrame) -> None:
    """
    Function to download files for isic dataset. Supply with dataframe containing columns 'isic_id' and 'image_url', 
    having strings containing the data that would be returned in the same name fields from the ISIC API.

    Will only download images if a file with the target name (<ISIC_ID>.jpg) cannot be found in the directory.
    Images are downloaded to a directory that can be found by checking cascid.configs.isic.IMAGE_DIR variable.

    Example:

    # Verify download for every image in dataset currently.
    df = get_db()
    update_all_files(df)

    """
    def check_and_download(isic_id: str):
        if not os.path.exists(isic_cnf.IMAGES_DIR / (isic_id + ".jpg")):
            fetcher.download_image(isic_id=isic_id)
    print("Beginning image downloads...")
    with ThreadPool(8) as p:
        p.map(check_and_download, df["isic_id"].to_list())
    print("Done")
    return

def _load_cache(path):
    with open(path, 'rb') as fl:
        split = pk.load(fl)

    if (split['x_train'].shape[0] + split['x_test'].shape[0]) != get_df().shape[0]:
        raise FileNotFoundError("Cache is outdated!")

    return (
        split['x_train'],
        split['x_test'],
        split['y_train'],
        split['y_test'],
    )

def _save_cache(path, x_train, x_test, y_train, y_test):
    split = dict()
    split['x_train'] = x_train
    split['x_test'] = x_test
    split['y_train'] = y_train
    split['y_test'] = y_test
    with open(path, 'wb') as fl:
        split = pk.dump(split,fl)

def get_train_test_images_raw(test_size: float = 0.2, random_state: int = 42, image_shape=(256, 256)) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        return x_train, x_test, y_train, y_test
    except FileNotFoundError:
        df = get_df()
        x = df['img_id'].apply(lambda x: images.get_raw_image(x, image_shape)).to_numpy()
        x = np.array([x[i] for i in range(len(x))])
        y=df['diagnostic'].to_numpy().reshape(-1,1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state, test_size=test_size)
        _save_cache(TRAIN_TEST_CACHE_RAW, x_train, x_test, y_train, y_test)
        return x_train, x_test, y_train, y_test

def get_train_test_images_hairless(test_size: float = 0.2, random_state: int = 42, image_shape=(256,256)) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        return x_train, x_test, y_train, y_test
    except FileNotFoundError:
        df = get_df()
        x = df['img_id'].apply(lambda x: images.get_hairless_image(x, image_shape)).to_numpy()
        x = np.array([x[i] for i in range(len(x))])
        y=df['diagnostic'].to_numpy().reshape(-1,1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state, test_size=test_size)
        _save_cache(TRAIN_TEST_CACHE_HAIRLESS, x_train, x_test, y_train, y_test)
        return x_train, x_test, y_train, y_test