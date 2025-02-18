import os
import pickle
import logging as logger


def cache_data(func, cache_dir, *args, **kwargs):
    """
    Caches the result of a function to a specified directory. If the cache exists, it loads the result from the cache.
    Otherwise, it executes the function, caches the result, and returns it.
    Args:
        func (callable): The function to be cached.
        cache_dir (str): The directory where the cache file will be stored.
        *args: Variable length argument list to be passed to the function.
        **kwargs: Arbitrary keyword arguments to be passed to the function.
    Returns:
        The result of the function, either from the cache or from executing the function.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{func.__name__}_cache.pkl")

    if os.path.exists(cache_file):
        logger.info(f"Loading cached data from {cache_file}")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
    else:
        logger.info(f"Cache not found. Executing {func.__name__}")
        data = func(*args, **kwargs)
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)

    return data
