from itertools import repeat
from multiprocessing.pool import ThreadPool
from typing import Callable
import numpy as np
from tqdm import tqdm
import cv2
from pathlib import Path
from cascid.datasets.pad_ufes import database
from cascid.configs import pad_ufes_cnf
import time
from cascid.image_preprocessing import adaptive_hair_removal

def apply_params_async(transform: Callable, args: np.ndarray):
    def starmap_decorator(args):
        return transform(*args)
    
    with ThreadPool(6) as TP:
        results = list(tqdm(TP.imap(starmap_decorator, args), total=len(args))) # Use tqdm iterator to update progress bar automatically
        TP.close() # Inform pool there will be no further job submissions
        TP.join() # Await pool jobs
    return results

def sample_func(*args):
    # Sample function, can receive any amount of positional arguments, will return their sum
    return sum(args)

def process_and_save(orig_path: str, target_path: str, transform: Callable, *, force_transform: bool= False) -> None:
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
            return
    img = cv2.imread(str(orig_path))
    processed = transform(img)
    cv2.imwrite(str(target_path), processed)

def main():
    df = database.get_df()
    df = df.sample(25) # testing

    # Arg 1
    orig_names = df['img_id'].apply(lambda x: str(pad_ufes_cnf.IMAGES_DIR / x)).to_numpy().reshape(-1,1)
    # Arg 2
    target_names = df['img_id'].apply(lambda x: str(pad_ufes_cnf.HAIRLESS_DIR / x)).to_numpy().reshape(-1,1)
    # Arg 3
    func = np.array(list(repeat(adaptive_hair_removal,len(orig_names))), dtype=object).reshape(-1,1)
    # Stack into list
    args = np.hstack([orig_names, target_names, func])
    # Run
    print("Beginning transformations, this may take a while...")
    start = time.perf_counter()
    result = apply_params_async(process_and_save, args)
    print("Finished transformations after {:.03f}s".format(time.perf_counter()-start))

    return 

if __name__ == "__main__":
    main()