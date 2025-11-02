# src/preprocessing/normalization.py

import numpy as np

def window_intensity(volume: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """
    Apply HU windowing (for CT / CTA) to focus on relevant intensity range.

    Parameters
    ----------
    volume : np.ndarray
        3D array (num_slices, H, W)
    window_center : float
        HU center value (e.g., 40 for soft tissue, 300 for bone)
    window_width : float
        HU window width (e.g., 80 for soft tissue, 1500 for bone)

    Returns
    -------
    np.ndarray : windowed and clipped volume
    """
    lower = window_center - (window_width / 2)
    upper = window_center + (window_width / 2)
    volume = np.clip(volume, lower, upper)
    return volume


def normalize_cta(volume: np.ndarray) -> np.ndarray:
    """
    Normalize a CTA volume (HU values) to [0, 1] range after windowing.

    Typical HU ranges:
    - Brain: center=40, width=80
    - Bone:  center=300, width=1500
    We use a brain window by default.
    """
    windowed = window_intensity(volume, window_center=40, window_width=80)
    normed = (windowed - windowed.min()) / (windowed.max() - windowed.min() + 1e-8)
    return normed.astype(np.float32)


def normalize_mra(volume: np.ndarray) -> np.ndarray:
    """
    Normalize an MRA (Magnetic Resonance Angiography) volume.

    MRA intensities vary a lot, so we use robust percentile-based clipping + z-score normalization.
    """
    lower, upper = np.percentile(volume, (1, 99))
    volume = np.clip(volume, lower, upper)
    mean, std = volume.mean(), volume.std() + 1e-8
    zscore = (volume - mean) / std
    zscore = (zscore - zscore.min()) / (zscore.max() - zscore.min() + 1e-8)
    return zscore.astype(np.float32)


def normalize_mri(volume: np.ndarray) -> np.ndarray:
    """
    Normalize MRI (T1, T2, etc.) volume using z-score + min-max scaling.
    """
    mean, std = volume.mean(), volume.std() + 1e-8
    normed = (volume - mean) / std
    normed = (normed - normed.min()) / (normed.max() - normed.min() + 1e-8)
    return normed.astype(np.float32)


def normalize_volume(volume: np.ndarray, modality: str = "CTA") -> np.ndarray:
    """
    Normalize the volume according to imaging modality.

    Parameters
    ----------
    volume : np.ndarray
        3D volume (num_slices, H, W)
    modality : str
        DICOM Modality tag (e.g., 'CT', 'MR', 'MRA', 'CTA', 'T1', 'T2')

    Returns
    -------
    np.ndarray : normalized 3D volume in [0, 1]
    """
    modality = modality.upper().strip()

    # group common synonyms
    if modality in ["CT", "CTA"]:
        return normalize_cta(volume)
    elif modality in ["MR", "MRI", "MRA", "T1", "T2"]:
        return normalize_mri(volume)
    else:
        print(f"⚠️ Unknown modality '{modality}', using generic z-score normalization.")
        return normalize_mri(volume)


if __name__ == "__main__":
    # test
    import matplotlib.pyplot as plt
    from dicom_loader import load_dicom_series

    series_dir = "/media/aboukhelifa/HDD2/dataset_rsna/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10118104902601294641571465174067732646"  # à adapter
    volume, meta = load_dicom_series(series_dir)

    normed = normalize_volume(volume, meta["Modality"])

    mid = normed.shape[0] // 2
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(volume[mid], cmap="gray")
    plt.title("Original")
    plt.subplot(1,2,2)
    plt.imshow(normed[mid], cmap="gray")
    plt.title("Normalized")
    plt.show()
