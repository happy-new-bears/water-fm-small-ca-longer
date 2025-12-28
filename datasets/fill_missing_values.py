"""
Fill missing values (-99999) in land areas using spatial interpolation

Strategy:
- For the 391 permanent missing pixels in land areas (UK coastal borders)
- Use spatial interpolation from nearby valid land pixels
- Methods: nearest neighbor, bilinear, or median of neighbors
"""

import numpy as np
from scipy import ndimage


def fill_missing_with_nearest(img: np.ndarray, land_mask: np.ndarray, missing_value: float = -99999) -> np.ndarray:
    """
    Fill missing values in land areas using nearest valid neighbor

    Args:
        img: [H, W] image array
        land_mask: [H, W] binary mask (1=land, 0=ocean)
        missing_value: value indicating missing data (default: -99999)

    Returns:
        filled_img: [H, W] image with missing values filled
    """
    filled_img = img.copy()

    # Find missing pixels in land areas
    missing_in_land = (land_mask == 1) & (img < -1000)

    if not missing_in_land.any():
        return filled_img  # No missing values to fill

    # Find valid land pixels
    valid_land = (land_mask == 1) & (img > -1000)

    # Create a mask for distance transform
    # Distance transform finds distance to nearest valid pixel
    distance, indices = ndimage.distance_transform_edt(
        ~valid_land,  # True where we need to find nearest valid pixel
        return_indices=True
    )

    # For missing pixels, use value from nearest valid pixel
    missing_coords = np.where(missing_in_land)
    for i in range(len(missing_coords[0])):
        y, x = missing_coords[0][i], missing_coords[1][i]
        # Get coordinates of nearest valid pixel
        nearest_y = indices[0, y, x]
        nearest_x = indices[1, y, x]
        filled_img[y, x] = img[nearest_y, nearest_x]

    return filled_img


def fill_missing_with_median(img: np.ndarray, land_mask: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Fill missing values using median of nearby valid pixels

    Args:
        img: [H, W] image array
        land_mask: [H, W] binary mask (1=land, 0=ocean)
        window_size: size of neighborhood window (default: 5)

    Returns:
        filled_img: [H, W] image with missing values filled
    """
    filled_img = img.copy()

    # Find missing pixels in land areas
    missing_in_land = (land_mask == 1) & (img < -1000)

    if not missing_in_land.any():
        return filled_img

    # Get coordinates of missing pixels
    missing_coords = np.where(missing_in_land)

    half_window = window_size // 2
    H, W = img.shape

    for i in range(len(missing_coords[0])):
        y, x = missing_coords[0][i], missing_coords[1][i]

        # Define neighborhood bounds
        y_min = max(0, y - half_window)
        y_max = min(H, y + half_window + 1)
        x_min = max(0, x - half_window)
        x_max = min(W, x + half_window + 1)

        # Get neighborhood
        neighborhood = img[y_min:y_max, x_min:x_max]
        land_neighborhood = land_mask[y_min:y_max, x_min:x_max]

        # Get valid values in neighborhood (land pixels with valid data)
        valid_in_neighborhood = (land_neighborhood == 1) & (neighborhood > -1000)
        valid_values = neighborhood[valid_in_neighborhood]

        if len(valid_values) > 0:
            # Use median of valid neighbors
            filled_img[y, x] = np.median(valid_values)
        else:
            # Fallback: expand search radius
            # Use nearest neighbor from entire image
            valid_land = (land_mask == 1) & (img > -1000)
            if valid_land.any():
                valid_coords = np.where(valid_land)
                distances = (valid_coords[0] - y)**2 + (valid_coords[1] - x)**2
                nearest_idx = np.argmin(distances)
                nearest_y = valid_coords[0][nearest_idx]
                nearest_x = valid_coords[1][nearest_idx]
                filled_img[y, x] = img[nearest_y, nearest_x]
            else:
                # Ultimate fallback: use 0
                filled_img[y, x] = 0.0

    return filled_img


def fill_missing_with_zero(img: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    """
    Fill missing values with zero (conservative approach for precipitation)

    For precipitation data, zero is a valid value (no rain), so this is safe.

    Args:
        img: [H, W] image array
        land_mask: [H, W] binary mask (1=land, 0=ocean)

    Returns:
        filled_img: [H, W] image with missing values filled with 0
    """
    filled_img = img.copy()

    # Find missing pixels in land areas
    missing_in_land = (land_mask == 1) & (img < -1000)

    # Fill with 0
    filled_img[missing_in_land] = 0.0

    return filled_img


if __name__ == '__main__':
    # Test the functions
    import h5py
    import torch

    print("Testing missing value filling methods...")
    print("="*70)

    # Load data
    land_mask = torch.load('/Users/transformer/Desktop/water_data/new_version/gb_temp_valid_mask_290x180.pt')
    land_mask_np = land_mask.numpy()

    precip_file = '/Users/transformer/Desktop/water_data/new_version/precipitation_train_1989_2010.h5'

    with h5py.File(precip_file, 'r') as f:
        img = f['data'][0][:]

        print(f"Original image:")
        print(f"  Missing in land: {((land_mask_np == 1) & (img < -1000)).sum()}")
        print(f"  Min: {img.min():.2f}, Max: {img.max():.2f}")

        # Method 1: Fill with zero
        print(f"\nMethod 1: Fill with zero")
        filled_zero = fill_missing_with_zero(img, land_mask_np)
        valid_land = (land_mask_np == 1) & (filled_zero > -1000)
        print(f"  Missing after filling: {((land_mask_np == 1) & (filled_zero < -1000)).sum()}")
        print(f"  Mean (valid land): {filled_zero[valid_land].mean():.4f}")
        print(f"  Std (valid land): {filled_zero[valid_land].std():.4f}")

        # Method 2: Fill with nearest neighbor
        print(f"\nMethod 2: Fill with nearest neighbor")
        filled_nearest = fill_missing_with_nearest(img, land_mask_np)
        valid_land = (land_mask_np == 1) & (filled_nearest > -1000)
        print(f"  Missing after filling: {((land_mask_np == 1) & (filled_nearest < -1000)).sum()}")
        print(f"  Mean (valid land): {filled_nearest[valid_land].mean():.4f}")
        print(f"  Std (valid land): {filled_nearest[valid_land].std():.4f}")

        # Method 3: Fill with median of neighbors
        print(f"\nMethod 3: Fill with median of neighbors (window=5)")
        filled_median = fill_missing_with_median(img, land_mask_np, window_size=5)
        valid_land = (land_mask_np == 1) & (filled_median > -1000)
        print(f"  Missing after filling: {((land_mask_np == 1) & (filled_median < -1000)).sum()}")
        print(f"  Mean (valid land): {filled_median[valid_land].mean():.4f}")
        print(f"  Std (valid land): {filled_median[valid_land].std():.4f}")

    print("\n" + "="*70)
    print("✓ All methods tested successfully!")
    print("\nRecommendation:")
    print("  - For precipitation: Use Method 3 (median) - most realistic")
    print("  - For soil/temp: Use Method 2 (nearest) - smoother")
