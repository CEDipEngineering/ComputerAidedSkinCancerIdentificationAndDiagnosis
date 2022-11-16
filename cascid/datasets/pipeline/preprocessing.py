#!/usr/bin/env python3

import numpy as np
from typing import Tuple
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide tf warnings
from cascid.datasets.isic import images as isic_images, database as isic_db
from cascid.datasets.pad_ufes import images as pad_ufes_images, database as pad_ufes_db

def preprocess_dataset(dataset: str, prepro: str, image_shape: Tuple[int, int] = (256,256)) -> None:
    """
    Function wrapper to apply preprocessing steps for either dataset with any of the 3 available preprocessing types.

    Raises:
    - NameError: Invalid dataset/prepro name
    - Exception: Should never happen, avoids silent errors

    Args:
    - dataset: str -> String indicating which dataset to use (PAD-UFES or ISIC). Required. Possible values include: "ISIC", "PAD", "PAD-UFES", "All". Not case sensitive.
    - prepro: str -> String indicating which type of preprocessing to use. Required. Possible values include: "hairless", "hairless_quantized", "All". Not case sensitive.
    Note: Indicating "All" on both prepro and dataset will preprocess and cache all images, and can take several minutes.
    - image_shape: Tuple[int, int] -> Tuple of ints used to size images being loaded (Determines size that preprocessed images will be saved). Defaults to 256,256 (all loaded images are in RGB format).

    Example:

    # Apply all available forms of preprocessing to pad-ufes, and save images in 512x512 RGB resolution.
    preprocess_dataset('pad-ufes', 'all', image_shape=(512,512)) 

    """

    # Case insensitive
    dataset = dataset.lower()
    prepro = prepro.lower()

    # Keyword all for dataset means to apply preprocessing to both datasets.
    if dataset == "all":
        preprocess_dataset('isic', prepro, image_shape)
        preprocess_dataset('pad', prepro, image_shape)
        return
    
    # Keyword all for prepro means to apply all preprocessings available to indicated dataset
    if prepro == "all":
        preprocess_dataset(dataset, 'hairless', image_shape)
        preprocess_dataset(dataset, 'hairless_quantized', image_shape)
        return

    # Input validation
    available_datasets = ['isic', 'pad', 'pad-ufes']
    if dataset not in available_datasets:
        raise NameError("Invalid dataset passed! {} is not in available datasets! Please use one of {} or 'All'".format(dataset, available_datasets))
    available_prepro = ['raw', 'hairless', 'hairless_quantized', 'hq']
    if prepro not in available_prepro:
        raise NameError("Invalid preprocessing passed! {} is not in available preprocessings! Please use one of {} or 'All'".format(prepro, available_prepro))
    
    # Select and load dataset, then return
    print("Applying {} preprocessing to {} dataset, this may take a few minutes, but caching is done automatically, so the next time it should be much faster.".format(prepro, dataset))
    if dataset == 'isic':
        df = isic_db.get_df()
        imlist = df['img_id'].to_list()
        if prepro == 'hairless':
            return isic_images.remove_hair(imlist, image_shape=image_shape)
        if prepro in ['hairless_quantized', 'hq']:
            return isic_images.remove_hair_and_quantize(imlist, image_shape=image_shape)
    if dataset in ['pad', 'pad-ufes']:
        df = pad_ufes_db.get_df()
        imlist = df['img_id'].to_list()
        if prepro == 'hairless':
            return pad_ufes_images.remove_hair(imlist, image_shape=image_shape)
        if prepro in ['hairless_quantized', 'hq']:
            return pad_ufes_images.remove_hair_and_quantize(imlist, image_shape=image_shape)
    raise Exception("Something bad has happened, this exception should never trigger!")
