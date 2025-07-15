import numpy as np
def normalize(array):
    """
    Safely normalize a NumPy array to the range [0, 1].

    Args:
        array (np.ndarray): Input array (1D, 2D, or 3D)

    Returns:
        np.ndarray: Normalized array in same shape, scaled to [0, 1]
    """
    array = array.astype(np.float32)
    min_val = np.min(array)
    max_val = np.max(array)
    range_val = max_val - min_val

    if range_val == 0:
        # If all values are the same, return zeros
        return np.zeros_like(array, dtype=np.float32)
    else:
        return (array - min_val) / range_val
