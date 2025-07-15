import rasterio
import matplotlib.pyplot as plt
import numpy as np


def read_image(path):
    """
    Reads an image from the specified path using rasterio.
    
    Args:
        path (str): The file path to the image.
        
    Returns:
        numpy.ndarray: The image data as a NumPy array.
    """
    with rasterio.open(path) as src:
        image = src.read()
    return image



def visualize_image_and_label(image, label, band_idx=0):
    """
    Visualize a single band from the input image alongside its binary label.
    
    Parameters:
    - image: numpy array of shape (H, W, 12), representing the input image
    - label: numpy array of shape (H, W, 1) or (H, W), representing the binary label
    - band_idx: integer, index of the band to visualize (0 to 11)
    """

    # If label is 3D (H, W, 1), squeeze it to 2D
    if label.ndim == 3:
        label = np.squeeze(label)

    # Plot image band and label side by side
    plt.figure(figsize=(10, 5))

    # Show the selected image band
    plt.subplot(1, 2, 1)
    plt.imshow(image[:, :, band_idx], cmap='gray')
    plt.title(f'Image - Band {band_idx + 1}')
    plt.axis('off')

    # Show the corresponding label
    plt.subplot(1, 2, 2)
    plt.imshow(label, cmap='gray')
    plt.title('Label (Water Area)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
