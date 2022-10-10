from multiprocessing.pool import ThreadPool
from turtle import update
from typing import List, Callable
import numpy as np
from tqdm import tqdm

def apply_params_async(transform: Callable, args: np.ndarray):
    with ThreadPool(16) as TP:
        results = list(tqdm(TP.imap(transform, args), total=len(args)))
        TP.close()
        TP.join()
    return results

def sample_func(*args):
    return sum(args)

def main():
    sample_data = np.random.randint(0,5, (100000,2000))
    result = apply_params_async(sample_func, sample_data)
    # print(sample_data)
    # print(result)
    return

if __name__ == "__main__":
    main()