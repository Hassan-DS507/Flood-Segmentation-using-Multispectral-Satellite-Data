import numpy as np
import matplotlib.pyplot as plt

def show_image_all_bands(img_array, mask_array, index=0, title="Multispectral Image + Mask", figsize=(16, 8)):
    """
    Display all 12 bands of a multispectral image along with its mask.

    Parameters:
    - img_array: array of shape (N, H, W, 12)
    - mask_array: array of shape (N, H, W, 1)
    - index: which image to display
    - title: optional title to show above the whole figure
    - figsize: size of the plot
    """
    img = img_array[index]
    lbl = mask_array[index]
    if lbl.ndim == 3:
        lbl = lbl[:, :, 0]


    fig, axes = plt.subplots(3, 5, figsize=figsize)
    fig.suptitle(title, fontsize=16)

    # Show mask
    axes[0, 0].imshow(lbl, cmap='gray')
    axes[0, 0].set_title("Mask")
    axes[0, 0].axis('off')

    # Show all 12 bands
    band_idx = 0
    for row in range(3):
        for col in range(5):
            if row == 0 and col == 0:
                continue
            if band_idx >= 12:
                break
            ax = axes[row, col]
            ax.imshow(img[:, :, band_idx], cmap='gray')
            ax.set_title(f"Band {band_idx + 1}")
            ax.axis('off')
            band_idx += 1

    plt.tight_layout()
    plt.show()




def show_single_image_all_bands(img, lbl, title="One Image + Mask", figsize=(16, 8)):
    """
    Display all 12 bands of a single multispectral image + mask.

    Parameters:
    - img: shape (128, 128, 12)
    - lbl: shape (128, 128) or (128, 128, 1)
    """
    if lbl.ndim == 3:
        lbl = lbl[:, :, 0]

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 5, figsize=figsize)
    fig.suptitle(title, fontsize=16)

    # Show mask
    axes[0, 0].imshow(lbl, cmap='gray')
    axes[0, 0].set_title("Mask")
    axes[0, 0].axis('off')

    # Show 12 bands
    band_idx = 0
    for row in range(3):
        for col in range(5):
            if row == 0 and col == 0:
                continue
            if band_idx >= img.shape[-1]:
                break
            ax = axes[row, col]
            ax.imshow(img[:, :, band_idx], cmap='gray')
            ax.set_title(f"Band {band_idx + 1}")
            ax.axis('off')
            band_idx += 1

    plt.tight_layout()
    plt.show()



import numpy as np

from matplotlib.colors import ListedColormap

def normalize_image(img):
    """Normalize image bands to [0,1] for display."""
    p2, p98 = np.percentile(img, (2, 98))
    return np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1)

def plot_predictions(X_test, y_test, y_pred, num_images=5, title="Segmentation Results"):
    """
    Display a comparison of input images, ground truth masks, predicted masks, and overlays.
    
    Parameters:
        X_test (np.ndarray): Array of test images (N, H, W, C)
        y_test (np.ndarray): Ground truth masks (N, H, W)
        y_pred (np.ndarray): Predicted masks (N, H, W)
        num_images (int): Number of samples to display
        title (str): Title for the entire figure
    """
    assert len(X_test) >= num_images, "Not enough images in X_test"

    indices = np.random.choice(len(X_test), num_images, replace=False)
    fig, axes = plt.subplots(num_images, 4, figsize=(24, 6 * num_images))
    fig.suptitle(title, fontsize=20)

    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, idx in enumerate(indices):
        image = X_test[idx]
        true_mask = y_test[idx]
        pred_mask = y_pred[idx]

        # Normalize and prepare RGB image
        if image.shape[-1] >= 3:
            rgb_image = normalize_image(image[:, :, :3])
        else:
            raise ValueError("Image must have at least 3 channels for RGB display.")

        # Column 1: Input image
        axes[i, 0].imshow(rgb_image)
        axes[i, 0].set_title(f"Image [{idx}]")
        axes[i, 0].axis('off')

        # Column 2: Ground Truth Mask
        axes[i, 1].imshow(true_mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')

        # Column 3: Predicted Mask
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis('off')

        # Column 4: Overlay
        axes[i, 3].imshow(rgb_image)
        axes[i, 3].imshow(pred_mask, cmap=ListedColormap(['none', 'cyan']), alpha=0.5)
        axes[i, 3].set_title("Overlay")
        axes[i, 3].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()





def visualize_image_NDWI(images, masks, number_images=4, title_prefix="Sample"):
    """
    Visualize multiple (True Color | NDWI | Mask) plots for a batch of images.

    Parameters:
    - images: numpy array of shape (N, H, W, 12) or (N, 12, H, W)
    - masks: numpy array of shape (N, H, W) or (N, H, W, 1)
    - number_images: how many images to display
    - title_prefix: optional string to prepend to titles
    """
    total_images = min(number_images, len(images))
    fig, axes = plt.subplots(total_images, 3, figsize=(15, 5 * total_images))

    if total_images == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(total_images):
        try:
            image_np = images[i+15]
            mask_np = masks[i+15]

            # ----- معالجة شكل الصورة -----
            if image_np.ndim == 2:
                raise ValueError(f"[{i}] Image is 2D. Shape: {image_np.shape}")

            elif image_np.ndim == 3:
                # إذا كانت (12, H, W)
                if image_np.shape[0] == 12:
                    image_np = np.transpose(image_np, (1, 2, 0))  # (H, W, 12)
                elif image_np.shape[2] != 12:
                    raise ValueError(f"[{i}] Image has invalid number of channels. Shape: {image_np.shape}")
            else:
                raise ValueError(f"[{i}] Image has unsupported shape: {image_np.shape}")

            # ----- معالجة شكل القناع -----
            if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
                mask_np = mask_np.squeeze(-1)

            # ----- True Color -----
            true_color_img = image_np[:, :, [3, 2, 1]]  # Bands 4, 3, 2
            min_val, max_val = true_color_img.min(), true_color_img.max()
            if max_val > min_val:
                true_color_img = (true_color_img - min_val) / (max_val - min_val)

            axes[i, 0].imshow(true_color_img)
            axes[i, 0].set_title(f"{title_prefix} {i+1} - True Color")
            axes[i, 0].axis('off')

            # ----- NDWI -----
            green = image_np[:, :, 2].astype(np.float32)  # Band 3
            nir = image_np[:, :, 7].astype(np.float32)    # Band 8
            ndwi = (green - nir) / (green + nir + 1e-10)

            im = axes[i, 1].imshow(ndwi, cmap='Blues')
            axes[i, 1].set_title(f"{title_prefix} {i+1} - NDWI")
            axes[i, 1].axis('off')
            fig.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

            # ----- Mask -----
            axes[i, 2].imshow(mask_np, cmap='gray')
            axes[i, 2].set_title(f"{title_prefix} {i+1} - Ground Truth Mask")
            axes[i, 2].axis('off')

        except Exception as e:
            print(f"Skipping image {i} due to error: {e}")
            axes[i, 0].text(0.5, 0.5, "Error", ha='center', va='center')
            axes[i, 1].axis('off')
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


