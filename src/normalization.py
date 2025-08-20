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





def normalize_per_band(array, per_band=True, band_axis=None):
    """
    Normalize a NumPy array to the range [0, 1].

    Supports both global normalization and per-band normalization.

    Args:
        array (np.ndarray): Input array (2D or 3D)
        per_band (bool): If True, normalize each band separately
        band_axis (int or None): Axis of bands (0, 1, or 2). If None, will auto-detect

    Returns:
        np.ndarray: Normalized array in the same shape
    """
    array = array.astype(np.float32)

    # Auto-detect band axis if not given
    if band_axis is None and array.ndim == 3:
        if array.shape[0] in [3, 12]:
            band_axis = 0
        elif array.shape[2] in [3, 12]:
            band_axis = 2
        else:
            raise ValueError("Can't determine band axis automatically. Please specify `band_axis`.")

    # Global normalization
    if array.ndim == 2 or (not per_band):
        min_val = np.min(array)
        max_val = np.max(array)
        range_val = max_val - min_val
        return np.zeros_like(array) if range_val == 0 else (array - min_val) / range_val

    # Per-band normalization
    if array.ndim == 3 and per_band:
        array_moved = np.moveaxis(array, band_axis, 0)
        for i in range(array_moved.shape[0]):
            band = array_moved[i]
            min_val = np.min(band)
            max_val = np.max(band)
            range_val = max_val - min_val
            array_moved[i] = np.zeros_like(band) if range_val == 0 else (band - min_val) / range_val
        return np.moveaxis(array_moved, 0, band_axis)

# ------------------------------------------------------------------------------
#  Function Purpose:
# This function is used to normalize a 2D or 3D NumPy array to the range [0, 1].
# It's useful for image data or multispectral data (e.g. 12-band satellite image).
#
#  Supports two modes:
#   1. Global normalization (whole array at once) → per_band = False
#   2. Per-band normalization (normalize each channel independently) → per_band = True (default)
#
#  Parameters:
#   - array: your input NumPy array (can be 2D or 3D)
#   - per_band: if True → normalize each band (channel) individually
#   - band_axis: use it if you know where the band dimension is (0, 1, or 2)
#                e.g., (bands, H, W) → band_axis=0, or (H, W, bands) → band_axis=2
#                If not given, the function will try to auto-detect it.
#
#  Example Use:
#   normalize_per_band(image_array)  # for automatic mode
#   normalize_per_band(image_array, per_band=True, band_axis=2)  # manually
#
#  Special Cases:
#   - If a band has all pixels equal → it returns zeros for that band (to avoid division by zero)
#   - If 2D array → it just normalizes the whole image as one band
#
#  Logic Flow:
#   1. Converts the input to float32
#   2. If per-band mode:
#       - Moves the band axis to front
#       - Loops through each band, normalizes it to [0, 1]
#       - Moves the bands back to original position
#   3. Else:
#       - Normalizes the whole array once
#
#  Output:
#   - Normalized array (same shape as input), values in [0, 1]
# ------------------------------------------------------------------------------
