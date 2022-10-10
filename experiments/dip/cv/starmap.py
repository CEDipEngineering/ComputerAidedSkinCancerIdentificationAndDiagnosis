from multiprocessing.pool import ThreadPool
from typing import Callable
import numpy as np
from tqdm import tqdm

def apply_params_async(transform: Callable, args: np.ndarray):
    with ThreadPool(16) as TP:
        results = list(tqdm(TP.imap(transform, args), total=len(args))) # Use tqdm iterator to update progress bar automatically
        TP.close() # Inform pool there will be no further job submissions
        TP.join() # Await pool jobs
    return results

def sample_func(*args):
    # Sample function, can receive any amount of positional arguments, will return their sum
    return sum(args)

def main():
    # Example of starmap use
    sample_data = np.random.normal(0,1, (100000,2000))
    result = apply_params_async(sample_func, sample_data)
    # print(sample_data)
    # print(result)
    return

if __name__ == "__main__":
    main()