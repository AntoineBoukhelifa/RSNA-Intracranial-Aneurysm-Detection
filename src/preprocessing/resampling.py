# src/preprocessing/resampling.py

from tabnanny import verbose
import numpy as np
from scipy.ndimage import zoom

def resample_isotropic(
    volume: np.ndarray,
    spacing: tuple,
    new_spacing: tuple = (1.0, 1.0, 1.0),
    order: int = 1,
    verbose:bool = False
) -> np.ndarray:
    """
    Resample a 3D volume to isotropic spacing (default 1x1x1 mm³).

    Parameters
    ----------
    volume : np.ndarray
        3D array of shape (num_slices, H, W)
    spacing : tuple
        Original spacing in mm, e.g. (z_spacing, y_spacing, x_spacing)
    new_spacing : tuple, optional
        Target spacing in mm, default (1.0, 1.0, 1.0)
    order : int, optional
        Interpolation order (0=nearest, 1=linear, 3=cubic)

    Returns
    -------
    resampled_volume : np.ndarray
        Volume resampled to isotropic spacing.
    """

 #--- Normalization ---
    # (H, W)  → (1, H, W)
    # (1, Z, H, W) → (Z, H, W)
    # Other case → error
    if volume.ndim == 2:
        volume = volume[np.newaxis, :, :]
    elif volume.ndim == 4 and volume.shape[0] == 1:
        volume = volume.squeeze(0)  # remove the empty channel
    elif volume.ndim != 3:
        raise ValueError(f"Unexpected volume shape: {volume.shape}")

    z_spacing, y_spacing, x_spacing = spacing
    new_z, new_y, new_x = new_spacing

    # Compute zoom factors (old_spacing / new_spacing)
    zoom_factors = [
        z_spacing / new_z,
        y_spacing / new_y,
        x_spacing / new_x
    ]
    #print(f"[DEBUG] volume.shape = {volume.shape}")
    #print(f"[DEBUG] spacing = {spacing}")
    resampled = zoom(volume, zoom=zoom_factors, order=order)

    if len(zoom_factors) != volume.ndim:
        raise ValueError(
            f"[ERROR] Zoom factors ({len(zoom_factors)}) != input dims ({volume.ndim}) | shape={volume.shape}"
        )

    return resampled.astype(np.float32)


if __name__ == "__main__":
    # test
    volume = np.random.rand(1, 23, 512, 512)
    spacing = (5.0, 1.0, 1.0)
    #if verbose:
       # print(f"[DEBUG] volume.shape = {volume.shape}")
        #print(f"[DEBUG] spacing = {spacing}")
    resampled = resample_isotropic(volume, spacing)
