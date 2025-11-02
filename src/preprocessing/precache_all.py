# src/preprocessing/precache_all.py

import os
import pandas as pd
from tqdm import tqdm

from src.preprocessing.pipeline import preprocess_series
from src.utils.helpers import list_series_dirs, ensure_dir, log, load_config


def precache_all(series_root: str, csv_path: str = None, cache_dir: str = "cache"):
    """
    Preprocess all DICOM series in the dataset and save them as cached .npy volumes.

    Parameters
    ----------
    series_root : str
        Path to the folder containing subfolders for each SeriesInstanceUID.
    csv_path : str, optional
        Path to train.csv (used to filter only train series if provided)
    cache_dir : str
        Directory where cached .npy volumes will be stored.
    """
    ensure_dir(cache_dir)

    # If a CSV is provided → we filter the known SeriesInstanceUIDs
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        series_ids = df["SeriesInstanceUID"].astype(str).unique().tolist()
        log(f"Found {len(series_ids)} series in {csv_path}")
        series_dirs = [os.path.join(series_root, s) for s in series_ids if os.path.isdir(os.path.join(series_root, s))]
    else:
        # Otherwise, we process all the series in the folder
        series_dirs = list_series_dirs(series_root)
        log(f"Found {len(series_dirs)} series in {series_root}")

    # Main loop with progress bar
    log(f"Starting preprocessing for {len(series_dirs)} series...")
    for series_dir in tqdm(series_dirs, desc="Preprocessing", ncols=100):
        try:
            preprocess_series(series_dir, cache_dir=cache_dir, use_cache=True, verbose=False)
        except Exception as e:
            log(f"❌ Error processing {series_dir}: {e}")

    log(f"✅ Cached volumes saved to '{cache_dir}/'.")


if __name__ == "__main__":
    config = load_config("src/training/config.yaml")

    series_root = config["dataset"]["series_root"]
    csv_path = config["dataset"]["csv_path"]
    cache_dir = "cache"

    precache_all(series_root=series_root, csv_path=csv_path, cache_dir=cache_dir)
