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
