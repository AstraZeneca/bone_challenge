import numpy as np


def moving_average(array: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        return array

    return np.convolve(array, np.ones(window), "valid") / window
