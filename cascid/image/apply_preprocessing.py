
# Native imports
from multiprocessing.pool import ThreadPool
from pathlib import Path
from multiprocessing import cpu_count
from typing import Callable, List

# Package imports
import numpy as np
import cv2
# from tqdm import tqdm

# Cascid imports
from cascid.datasets.pad_ufes import images

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
    def starmap_decorator(args):
        return transform(*args)
    
    with ThreadPool(nthreads) as pool:
        # results = list(tqdm(pool.imap(starmap_decorator, args), total=len(args), miniters=1)) # Use tqdm iterator to update progress bar automatically
        # pool.close() # Inform pool there will be no further job submissions
        # pool.join() # Await pool jobs
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