import numpy as np
from pathlib import Path
from typing import Tuple, List, Callable
from itertools import repeat
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
import time
from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array
import cv2
import os

from cascid.configs import isic_cnf
from cascid.datasets.isic import database, fetcher, images
from cascid.image.image_preprocessing import adaptive_hair_removal
from cascid.image.image_preprocessing import adaptive_hair_removal2, color_quantization

_warning_load_image_without_shape = False

def remove_and_quantize(img):
    return color_quantization(adaptive_hair_removal2(img))

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
    
    img_raw = get_hairless_image(df.iloc[0]['img_id'], (128, 128)) # Get first image hairless 

    """
    if (isic_cnf.HAIRLESS_DIR / img_name).exists():
        return _load_image(img_name, isic_cnf.HAIRLESS_DIR, image_shape)
    print("Supplied image name '{}' has no preprocessed file found".format(img_name))
    print("Please, apply preprocessing to all images that will be used beforehand")
    print("Preprocessing a list of images can be done using cascid.image.apply_preprocessing.remove_hair(img_list)")
    print("\n"*3)
    raise FileNotFoundError("Supplied image does not have a preprocessed file, read above error")

def get_hq_image(img_name: str, image_shape: Tuple[int, int] = None):
    """
    ## Function used to read preprocessed image from disk. Used to abstract directory structure from user.
    
    Args:
    - img_name: string, as found in 'img_id' column in metadata, such as 'PAT_2046_4323_394.png'.
    - image_shape: Tuple of two integers, size of image array output. Defaults to (128, 128).

    Raises: FileNotFoundError if img_name has not been preprocessed yet

    Example:

    ## Load first image from dataset:    
    df = datasets.isic.database.get_df() # Get metadata
    
    img_raw = get_hairless_image(df.iloc[0]['img_id'], (128, 128)) # Get first image hq 

    """
    if (isic_cnf.HAIRLESS_QUANTIZED_DIR / img_name).exists():
        return _load_image(img_name, isic_cnf.HAIRLESS_QUANTIZED_DIR, image_shape)
    print("Supplied image name '{}' has no preprocessed file found".format(img_name))
    print("Please, apply preprocessing to all images that will be used beforehand")
    print("Preprocessing a list of images can be done using cascid.image.apply_preprocessing.remove_hair(img_list)")
    print("\n"*3)
    raise FileNotFoundError("Supplied image does not have a preprocessed file, read above error")

def remove_hair(img_list: List[str]) -> None:
    """
    Preprocessing function, used to remove hair from list of images, and save preprocessed results in isic_cnf.HAIRLESS_DIR.
    Warning, this processing is done with mutiple threads, and as such should be done with larger numbers of images. 
    Calling this function repeatedly with a single image on the list will result in extremely slow performance.

    Args:
    img_list: List of strings of image names, as found in metadata, such as ['PAT_2046_4323_394.png'].
    """
    prepend_output_dir = lambda x: str(isic_cnf.HAIRLESS_DIR / x)
    # Arg 1
    orig_names = np.array(img_list).reshape(-1,1)
    # Arg 2
    target_names = np.array([prepend_output_dir(i) for i in img_list]).reshape(-1,1)
    # Arg 3
    func = np.array(list(repeat(adaptive_hair_removal,len(orig_names))), dtype=object).reshape(-1,1)
    # Stack into list
    args = np.hstack([orig_names, target_names, func])
    # Run
    print("Beginning transformations, this may take a while...")
    start = time.perf_counter()
    result = _apply_params_async(_process_and_save, args, nthreads=8)
    elapsed = time.perf_counter()-start
    hour=int(elapsed//3600)
    minute=int((elapsed%3600)//60)
    seconds=float((elapsed%3600)%60)
    print("Finished transformations after {:d}h{:02d}min{:.02f}s".format(hour,minute,seconds))

def remove_hair_and_quantize(img_list: List[str]) -> None:
    """
    Preprocessing function, used to remove hair from list of images, and save preprocessed results in isic_cnf.HAIRLESS_QUANTIZED_DIR.
    Warning, this processing is done with mutiple threads, and as such should be done with larger numbers of images. 
    Calling this function repeatedly with a single image on the list will result in extremely slow performance.

    Args:
    img_list: List of strings of image names, as found in metadata, such as ['PAT_2046_4323_394.png'].
    """

    prepend_output_dir = lambda x: str(isic_cnf.HAIRLESS_QUANTIZED_DIR / x)
    # Arg 1
    orig_names = np.array(img_list).reshape(-1,1)
    # Arg 2
    target_names = np.array([prepend_output_dir(i) for i in img_list]).reshape(-1,1)
    # Arg 3
    func = np.array(list(repeat(remove_and_quantize,len(orig_names))), dtype=object).reshape(-1,1)
    # Stack into list
    args = np.hstack([orig_names, target_names, func])
    # Run
    print("Beginning transformations, this may take a while...")
    start = time.perf_counter()
    result = _apply_params_async(_process_and_save, args, nthreads=8)
    elapsed = time.perf_counter()-start
    hour=int(elapsed//3600)
    minute=int((elapsed%3600)//60)
    seconds=float((elapsed%3600)%60)
    print("Finished transformations after {:d}h{:02d}min{:.02f}s".format(hour,minute,seconds))


def _apply_params_async(transform: Callable, args: np.ndarray, nthreads: int = cpu_count()//2) -> List:
    """
    Apply callable to 2D list of args using a ThreadPool. Args are unpacked using star operator, 
    and must be supplied in the correct order the function requires its positional arguments.

    Args:
    transform: function with some set of positional arguments, to be applied in parallel using the ThreadPool
    args: List of lists of args for each fucntion call. Given a function f(x,y), args might be: [[1,2],[3,4]].

    If transform returns values, these values will be captured and returned in a list, in an order corresponding to
    args.
    """    
    with ThreadPool(nthreads) as pool:
        results = pool.starmap(transform, args)
    return results

def _process_and_save(img_name: str, target_path: str, transform: Callable, *, force_transform: bool= False) -> None:
    """
    Function to apply processing to images, and save results.
    Args:
    orig_path: str or Path-Like of iamge to be transformed
    target_path: Destination of transformed image
    transform: Callable that alters and returns modified image, callable is not expected to receive any arguments beside original image array
    Kwargs:
    force_transform: Must be keyworded, boolean, indicating whether to skip already present target_path images. Uses Path.exists() method to verify existence.
    """
    if not force_transform:
        if Path(target_path).exists():
            # print(target_path, "exists")
            return True
    # print("called with {}".format(img_name))
    img = images.get_raw_image(img_name, (512,512))
    processed = transform(img)
    ret = cv2.imwrite(str(target_path), processed)
    return ret

if __name__ == "__main__":
    df = database.get_df() # Get metadata
    img_raw = get_raw_image(df.iloc[0]['img_id'], (128, 128)) # Get first image raw