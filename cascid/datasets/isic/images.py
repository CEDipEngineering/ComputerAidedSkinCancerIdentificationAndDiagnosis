from pathlib import Path
from typing import Tuple
from cascid.configs import isic_cnf
from cascid.datasets.isic import database, fetcher
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array

_warning_load_image_without_shape = False

def _load_image(img_name: str, prefix: Path, shape: Tuple[int, int] = None) -> np.ndarray:
    global _warning_load_image_without_shape
    if shape is None:
        if not _warning_load_image_without_shape:
            print("Image shape not specified in image getter, will always default to (128, 128). This warning will only appear once after loading the module.")
            _warning_load_image_without_shape = True # used to print only once, prevents spamming when loading multiple images
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
    df = datasets.isic.database.get_df() # Get metadata
    
    img_raw = get_raw_image(df.iloc[0]['img_id'], (128, 128)) # Get first image raw 

    """
    try:
        return _load_image(img_name, isic_cnf.IMAGES_DIR, image_shape)
    except Exception as e:
        print("Image {} seems corrupted, trying to delete and redownload...".format(img_name))
        os.remove(isic_cnf.IMAGES_DIR / img_name)
        fetcher.download_image(Path(img_name).stem)
        return _load_image(img_name, isic_cnf.IMAGES_DIR, image_shape)

def get_hairless_image(img_name: str, image_shape: Tuple[int, int] = None):
    """
    ## Function used to read preprocessed image from disk. Used to abstract directory structure from user.
    
    Args:
    - img_name: string, as found in 'img_id' column in metadata, such as 'PAT_2046_4323_394.png'.
    - image_shape: Tuple of two integers, size of image array output. Defaults to (128, 128).

    Raises: FileNotFoundError if img_name has not been preprocessed yet

    Example:

    ## Load first image from dataset:    
    df = datasets.isic.database.get_df() # Get metadata
    
    img_raw = get_hairless_image(df.iloc[0]['img_id'], (128, 128)) # Get first image raw 

    """
    if (isic_cnf.HAIRLESS_DIR / img_name).exists():
        return _load_image(img_name, isic_cnf.HAIRLESS_DIR, image_shape)
    print("Supplied image name '{}' has no preprocessed file found".format(img_name))
    print("Please, apply preprocessing to all images that will be used beforehand")
    print("Preprocessing a list of images can be done using cascid.image.apply_preprocessing.remove_hair(img_list)")
    print("\n"*3)
    raise FileNotFoundError("Supplied image does not have a preprocessed file, read above error")

if __name__ == "__main__":
    df = database.get_df() # Get metadata
    img_raw = get_raw_image(df.iloc[0]['img_id'], (128, 128)) # Get first image raw