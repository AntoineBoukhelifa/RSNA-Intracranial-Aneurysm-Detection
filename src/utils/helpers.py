# src/utils/helpers.py

import os, glob, json, yaml, numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


# --- files and folders ---
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_series_dirs(series_root: str):
    return sorted([d for d in glob.glob(os.path.join(series_root, "*")) if os.path.isdir(d)])


# --- Cache and JSON ---
def get_cache_path(series_id, cache_dir="cache"):
    return os.path.join(cache_dir, f"{series_id}.npy")

def load_cached_volume(series_id, cache_dir="cache"):
    path = get_cache_path(series_id, cache_dir)
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")
    else:
        raise FileNotFoundError(f"No cached volume found at {path}")

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# --- Logs and config ---
def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{timestamp()}] {msg}")

def load_config(path="src/training/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# --- Visualization ---
def show_volume_slices(volume, n=5):
    step = max(1, volume.shape[0] // n)
    fig, axes = plt.subplots(1, n, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(volume[i * step], cmap="gray")
        ax.axis("off")
    plt.show()
