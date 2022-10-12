from pathlib import Path
from typing import Tuple
from cascid.configs import pad_ufes_cnf
from cascid.datasets.pad_ufes import database
import numpy as np
from tensorflow import keras
from keras.utils import load_img, img_to_array

def _load_image(img_name: str, prefix: Path, shape: Tuple[int, int] = None) -> np.ndarray:
    if shape is None:
        print("Image shape not specified, defaulting to (128, 128)")
        shape = (128, 128)
    pil_img = load_img(
        str(prefix / img_name),
        grayscale=False,
        color_mode='rgb',
        target_size=(shape[0], shape[1]),
        interpolation='nearest',
    )
    return img_to_array(pil_img, dtype=np.uint8)

def get_raw_image(img_name: str, image_shape: Tuple[int, int] = None) -> np.ndarray:
    """
    ## Function used to read image from disk. Used to abstract directory structure from user.
    Args:
    - img_name: string, as found in 'img_id' column in metadata, such as 'PAT_2046_4323_394.png'.
    - image_shape: Tuple of two integers, size of image array output. Defaults to (128, 128).

    Example:

    ## Load first image from dataset:    
    df = datasets.pad_ufes.database.get_df() # Get metadata
    
    img_raw = get_raw_image(df.iloc[0]['img_id'], (128, 128)) # Get first image raw 

    """
    return _load_image(img_name, pad_ufes_cnf.IMAGES_DIR, image_shape)


def get_hairless_image(img_name: str, image_shape: Tuple[int, int] = None):
    """
    ## Function used to read preprocessed image from disk. Used to abstract directory structure from user.
    
    Args:
    - img_name: string, as found in 'img_id' column in metadata, such as 'PAT_2046_4323_394.png'.
    - image_shape: Tuple of two integers, size of image array output. Defaults to (128, 128).

    Raises: FileNotFoundError if img_name has not been preprocessed yet

    Example:

    ## Load first image from dataset:    
    df = datasets.pad_ufes.database.get_df() # Get metadata
    
    img_raw = get_hairless_image(df.iloc[0]['img_id'], (128, 128)) # Get first image raw 

    """
    if (pad_ufes_cnf.HAIRLESS_DIR / img_name).exists():
        return _load_image(img_name, pad_ufes_cnf.HAIRLESS_DIR, image_shape)
    print("Supplied image name '{}' has no preprocessed file found".format(img_name))
    print("Please, apply preprocessing to all images that will be used beforehand")
    print("Preprocessing a list of images can be done using cascid.image.apply_preprocessing.remove_hair(img_list)")
    print("\n"*3)
    raise FileNotFoundError

if __name__ == "__main__":
    df = database.get_df() # Get metadata
    img_raw = get_raw_image(df.iloc[0]['img_id'], (128, 128)) # Get first image raw