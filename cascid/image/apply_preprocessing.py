
# Native imports
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
import time
from multiprocessing import cpu_count
from typing import Callable, List

# Package imports
import numpy as np
import cv2
# from tqdm import tqdm

# Cascid imports
from cascid.datasets.pad_ufes import database, images
from cascid.configs import pad_ufes_cnf
from cascid.image.image_preprocessing import adaptive_hair_removal

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

def remove_hair(img_list: List[str]) -> None:
    """
    Preprocessing function, used to remove hair from list of images, and save preprocessed results in pad_ufes_cnf.HAIRLESS_DIR.
    Warning, this processing is done with mutiple threads, and as such should be done with larger numbers of images. 
    Calling this function repeatedly with a single image on the list will result in extremely slow performance.

    Args:
    img_list: List of strings of image names, as found in metadata, such as ['PAT_2046_4323_394.png'].
    """
    prepend_output_dir = lambda x: str(pad_ufes_cnf.HAIRLESS_DIR / x)
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

if __name__ == "__main__":
    df = database.get_df()
    img_names = df['img_id'].to_list()
    remove_hair(img_names)
