# src/preprocessing/augmentation.py

import numpy as np
import random
from scipy.ndimage import rotate, gaussian_filter

def random_flip(volume: np.ndarray, axes=(0, 1, 2), p: float = 0.5) -> np.ndarray:
    """
    Randomly flip the 3D volume along selected axes.
    """
    for axis in axes:
        if random.random() < p:
            volume = np.flip(volume, axis=axis)
    return volume


def random_rotate(volume: np.ndarray, max_angle: float = 15.0, axes=(1, 2)) -> np.ndarray:
    """
    Apply a small random rotation to the 3D volume.
    
    Parameters
    ----------
    max_angle : float
        Maximum absolute rotation angle in degrees.
    axes : tuple
        Axes of rotation (default: axial plane Y-X).
    """
    angle = random.uniform(-max_angle, max_angle)
    rotated = rotate(volume, angle, axes=axes, reshape=False, order=1, mode='nearest')
    return rotated


def random_gaussian_noise(volume: np.ndarray, sigma_range=(0.0, 0.05)) -> np.ndarray:
    """
    Add Gaussian noise to the volume.
    """
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, size=volume.shape)
    return np.clip(volume + noise, 0.0, 1.0)


def random_blur(volume: np.ndarray, sigma_range=(0.5, 1.5), p: float = 0.3) -> np.ndarray:
    """
    Random Gaussian blur (simulates scanner differences).
    """
    if random.random() < p:
        sigma = random.uniform(*sigma_range)
        volume = gaussian_filter(volume, sigma=sigma)
    return volume


def random_intensity_shift(volume: np.ndarray, shift_range=(-0.1, 0.1), scale_range=(0.9, 1.1), p: float = 0.5) -> np.ndarray:
    """
    Random brightness/contrast change.
    """
    if random.random() < p:
        shift = random.uniform(*shift_range)
        scale = random.uniform(*scale_range)
        volume = volume * scale + shift
        volume = np.clip(volume, 0.0, 1.0)
    return volume


def augment_volume(volume: np.ndarray) -> np.ndarray:
    """
    Apply a series of random augmentations.
    Designed for normalized volumes in [0,1].
    """
    volume = random_flip(volume, axes=(1, 2), p=0.5)          # horizontal/vertical flip
    volume = random_rotate(volume, max_angle=10.0, axes=(1, 2)) # small rotation
    volume = random_intensity_shift(volume)                   # brightness/contrast
    volume = random_gaussian_noise(volume)                    # additive noise
    volume = random_blur(volume)                              # optional blur
    return volume.astype(np.float32)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dicom_loader import load_dicom_series
    from normalization import normalize_volume
    from resampling import resample_isotropic

    # Load one example
    series_dir = "/media/aboukhelifa/HDD2/dataset_rsna/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10118104902601294641571465174067732646"  # à adapter
    volume, meta = load_dicom_series(series_dir)
    volume = normalize_volume(volume, meta["Modality"])
    spacing = (meta["SliceSpacing"], *meta["PixelSpacing"])
    volume = resample_isotropic(volume, spacing)

    aug = augment_volume(volume)

    mid = volume.shape[0] // 2
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(volume[mid], cmap="gray")
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(aug[mid], cmap="gray")
    plt.title("Augmented")
    plt.show()
