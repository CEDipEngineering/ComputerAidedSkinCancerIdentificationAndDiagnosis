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
TRAIN_TEST_CACHE_HAIRLESS_QUANT = pad_ufes_cnf.PAD_UFES_DIR / 'train_test_cache_hairless_quant.pkl'

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

def get_train_test_images_hairless(test_size: float = 0.2, random_state: int = 42, image_shape=(256, 256)) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

def get_train_test_images_hairless_quantized(test_size: float = 0.2, random_state: int = 42, image_shape=(256, 256)) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        x_train, x_test, y_train, y_test = _load_cache(TRAIN_TEST_CACHE_HAIRLESS_QUANT)
        return x_train, x_test, y_train, y_test
    except FileNotFoundError:
        df = get_df()
        x = df['img_id'].apply(lambda x: images.get_hq_image(x, image_shape)).to_numpy()
        x = np.array([x[i] for i in range(len(x))])
        y=df['diagnostic'].to_numpy().reshape(-1,1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state, test_size=test_size)
        _save_cache(TRAIN_TEST_CACHE_HAIRLESS_QUANT, x_train, x_test, y_train, y_test)
        return x_train, x_test, y_train, y_test
    
def get_train_test_metadata(test_size: float = 0.2, random_state: int = 42, return_img_id: bool = False):
    """
    x_train_metadata, x_train_stacked, x_test, y_train_metadata, y_train_stacked, y_test = get_train_test_metadata()
    """
    database = get_df()
    
    database[['smoke','drink','pesticide','skin_cancer_history','cancer_history','has_piped_water','has_sewage_system','itch','grew','hurt','changed','bleed','elevation','biopsed']] = database[['smoke','drink','pesticide','skin_cancer_history','cancer_history','has_piped_water','has_sewage_system','itch','grew','hurt','changed','bleed','elevation','biopsed']].astype("bool")

    database = database.drop_duplicates()
    
    df = database.copy()

    df = df.sort_values('diameter_1', ascending=False).groupby('patient_id').first().reset_index()
    
    df['is_cancer'] = df['diagnostic'].apply(lambda x: 'Not' if x in ['ACK','NEV','SEK'] else 'Cancer')
    
    selected_columns = ['smoke', 'drink', 'skin_cancer_history', 'cancer_history', 'age','pesticide','is_cancer', 'img_id']

    df = df[selected_columns].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(['is_cancer'], axis=1),
        df['is_cancer'].to_numpy(),
        test_size = test_size,
        random_state=random_state,
    )
    
    x_train_metadata, x_train_stacked, y_train_metadata, y_train_stacked = train_test_split(
        x_train,
        y_train,
        test_size = 0.5,
        random_state=random_state,
    )
    if return_img_id:
        return x_train_metadata, x_train_stacked, x_test, y_train_metadata, y_train_stacked, y_test
    return x_train_metadata.drop('img_id',axis=1).to_numpy().astype('float64'), x_train_stacked.drop('img_id',axis=1).to_numpy().astype('float64'), x_test.drop('img_id',axis=1).to_numpy().astype('float64'), y_train_metadata, y_train_stacked, y_test

if __name__ == "__main__":
    # Test function
    print(get_df().head(5).transpose())