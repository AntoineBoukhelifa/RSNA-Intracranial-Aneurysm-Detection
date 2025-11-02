# src/preprocessing/pipeline.py

import os
import numpy as np
from typing import Dict, Any

from src.preprocessing.dicom_loader import load_dicom_series
from src.preprocessing.normalization import normalize_volume
from src.preprocessing.resampling import resample_isotropic

from src.utils.helpers import ensure_dir, log


def preprocess_series(
    series_dir: str,
    new_spacing: tuple = (1.0, 1.0, 1.0),
    cache_dir: str = "cache",
    use_cache: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline for a single DICOM series.
    Uses automatic caching to speed up repeated loading.

    Steps:
    1. Check if cached .npy volume exists
    2. If not, preprocess and save it
    3. Return normalized + resampled volume

    Parameters
    ----------
    series_dir : str
        Path to the DICOM series folder (contains multiple .dcm)
    new_spacing : tuple
        Target isotropic spacing in mm (default: (1,1,1))
    cache_dir : str
        Path to store cached volumes (.npy files)
    use_cache : bool
        If True, use and save cached volumes
    verbose : bool
        Print progress

    Returns
    -------
    dict : {
        "volume": np.ndarray,
        "metadata": dict
    }
    """
    if not os.path.exists(series_dir):
        raise FileNotFoundError(f"Series directory not found: {series_dir}")

    # Create cache folder if not exists
    ensure_dir(cache_dir)

    # Get unique series ID
    series_id = os.path.basename(series_dir.rstrip("/"))
    cache_path = os.path.join(cache_dir, f"{series_id}.npy")

    # Load from cache if available
    if use_cache and os.path.exists(cache_path):
        if verbose:
            print(f"⚡ Loaded from cache: {cache_path}")
        volume = np.load(cache_path, mmap_mode="r")
        return {"volume": volume, "metadata": {"SeriesInstanceUID": series_id}}

    # Otherwise preprocess normally
    volume, metadata = load_dicom_series(series_dir)

    if verbose:
        print(f"📂 Loaded series {metadata.get('SeriesInstanceUID', 'unknown')} "
              f"({metadata['Modality']}) | shape={volume.shape}")

    # normalize
    volume = normalize_volume(volume, modality=metadata["Modality"])

    # resample
    orig_spacing = (
        metadata["SliceSpacing"],
        metadata["PixelSpacing"][0],
        metadata["PixelSpacing"][1],
    )

    volume = resample_isotropic(volume, spacing=orig_spacing, new_spacing=new_spacing, verbose=False)

    # save to cache
    if use_cache:
        np.save(cache_path, volume.astype(np.float32))
        if verbose:
            log(f"Cached preprocessed volume to {cache_path}")

    return {"volume": volume.astype(np.float32), "metadata": metadata}


if __name__ == "__main__":
    #test
    test_series = "series/1.2.826.0.1.3680043.2055"  
    data = preprocess_series(test_series)

    volume = data["volume"]
    print("Final shape:", volume.shape)

    mid = volume.shape[0] // 2

