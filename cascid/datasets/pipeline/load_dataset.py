#!/usr/bin/env python3
import numpy as np
from typing import Tuple
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide tf warnings
from cascid.datasets.pad_ufes import database as pad_ufes_db
from cascid.datasets.isic import database as isic_db

def train_test_split(dataset: str, prepro: str, test_size: float = 0.2, random_state: int = 42, image_shape: Tuple[int, int] = (256,256)) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Function wrapper to load train test split for either dataset with any of the 3 available preprocessing types.

    Raises:
    - NameError: Invalid dataset/prepro name
    - Exception: Should never happen, avoids silent errors

    Args:
    - dataset: str -> String indicating which dataset to use (PAD-UFES or ISIC). Required. Possible values include: "ISIC", "PAD", "PAD-UFES". Not case sensitive.
    - prepro: str -> String indicating which type of preprocessing to use. Defaults to 'raw'. Possible values include: "raw", "hairless", "hairless_quantized". Not case sensitive.
    - test_size: float -> Float indicating ratio of data used for testing. 1.0 means all data is in test, 0.0 means no data in test. Default 0.2.
    - random_state: int -> Random seed passed to sklearn's train_test_split function. Deafults to 42, set to None for random behaviour each execution.
    - image_shape: Tuple[int, int] -> Tuple of ints used to size images being loaded. Defaults to 256,256 (all loaded images are in RGB format).

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], x_train, x_test, y_train, y_test

    Example:
        # Loading ISIC's raw images, using explicitly set default values.
        x_train, x_test, y_train, y_test = train_test_split(
            dataset = "ISIC", 
            prepro = "raw", 
            test_size = 0.2, 
            random_state = 42, 
            image_shape = (256,256)
        )

    """

    # Case insensitive
    dataset = dataset.lower()
    prepro = prepro.lower()

    # Input validation
    available_datasets = ['isic', 'pad', 'pad-ufes']
    if dataset not in available_datasets:
        raise NameError("Invalid dataset passed! {} is not in available datasets! Please use one of {}".format(dataset, available_datasets))
    available_prepro = ['raw', 'hairless', 'hairless_quantized', 'hq']
    if prepro not in available_prepro:
        raise NameError("Invalid preprocessing passed! {} is not in available preprocessings! Please use one of {}".format(prepro, available_prepro))
    
    # Select and load dataset, then return
    print("Loading {} {} dataset, this may take a minute, but caching is done automatically, so the next time it should be much faster.".format(prepro, dataset))
    if dataset == 'isic':
        if prepro == 'raw':
            return isic_db.get_train_test_images_raw(test_size=test_size, random_state=random_state, image_shape=image_shape)
        if prepro == 'hairless':
            return isic_db.get_train_test_images_hairless(test_size=test_size, random_state=random_state, image_shape=image_shape)
        if prepro in ['hairless_quantized', 'hq']:
            return isic_db.get_train_test_images_hairless_quantized(test_size=test_size, random_state=random_state, image_shape=image_shape)
    if dataset in ['pad', 'pad-ufes']:
        if prepro == 'raw':
            return pad_ufes_db.get_train_test_images_raw(test_size=test_size, random_state=random_state, image_shape=image_shape)
        if prepro == 'hairless':
            return pad_ufes_db.get_train_test_images_hairless(test_size=test_size, random_state=random_state, image_shape=image_shape)
        if prepro in ['hairless_quantized', 'hq']:
            return pad_ufes_db.get_train_test_images_hairless_quantized(test_size=test_size, random_state=random_state, image_shape=image_shape)
    raise Exception("Something bad has happened, this exception should never trigger!")

if __name__ == "__main__":
    '''
    Basic test in case you run this script directly for whatever reason.
    '''
    x_train, x_test, y_train, y_test = train_test_split(
        dataset = "ISIC", 
        prepro = "raw", 
        test_size = 0.2, 
        random_state = 42, 
        image_shape = (256,256)
    )

    print("x_train shape: \t{0}".format(x_train.shape))
    print("x_test shape: \t{0}".format(x_test.shape))
    print("y_train shape: \t{0}".format(y_train.shape))
    print("y_test shape: \t{0}".format(y_test.shape))