from multiprocessing.pool import ThreadPool
from typing import List, Callable
import numpy as np

def apply_params_async(transform: Callable, args: np.ndarray):
    with ThreadPool(16) as TP:
        result = TP.starmap(transform, args)
    return result

def sample_func(*args):
    return sum(args)

def main():
    sample_data = np.random.randint(0,5, (5,5))
    result = apply_params_async(sample_func, sample_data)
    print(sample_data)
    print(result)
    return

if __name__ == "__main__":
    main()