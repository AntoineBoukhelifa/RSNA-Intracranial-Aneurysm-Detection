# src/preprocessing/dicom_loader.py

import os
from typing import Tuple, List, Dict, Any

import numpy as np
import pydicom

def load_dicom_series(series_dir: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a DICOM series (multiple .dcm files in a folder) and return a 3D volume
    sorted along the z-axis.

    Parameters
    ----------
    series_dir : str
        Path to a folder containing a single DICOM series
        e.g. "series/1.2.826.0.1.3680043.2055/"

    Returns
    -------
    volume : np.ndarray
        3D array of shape (num_slices, height, width)
    metadata : dict
        Useful DICOM metadata:
        - "PixelSpacing" : (row_spacing_mm, col_spacing_mm)
        - "SliceThickness" or "SpacingBetweenSlices"
        - "SliceLocations" : list of z positions
        - "SOPInstanceUIDs" : list of SOPInstanceUID in the same order as volume
        - "Modality"
    """
    # 1. collect all dicom files
    dicom_files = [
        os.path.join(series_dir, f)
        for f in os.listdir(series_dir)
        if f.lower().endswith(".dcm")
    ]
    if len(dicom_files) == 0:
        raise FileNotFoundError(f"No DICOM files found in: {series_dir}")

    # 2. read all headers
    instances = []
    for fp in dicom_files:
        ds = pydicom.dcmread(fp)
        # Some series don't have ImagePositionPatient -> we fallback on InstanceNumber
        z = None
        if hasattr(ds, "ImagePositionPatient"):
            # z = position along patient axis
            z = float(ds.ImagePositionPatient[2])
        elif hasattr(ds, "SliceLocation"):
            z = float(ds.SliceLocation)
        elif hasattr(ds, "InstanceNumber"):
            # less reliable but common
            z = float(ds.InstanceNumber)
        else:
            # final fallback: alphabetical order
            z = 0.0

        instances.append(
            {
                "dataset": ds,
                "z": z,
                "path": fp
            }
        )

    # 3. sort by z (descending or ascending depending on acquisition)
    # we choose ascending so volume[0] is the most inferior slice
    instances = sorted(instances, key=lambda x: x["z"])

    # 4. stack into numpy array
    slices = []
    sop_uids = []
    for inst in instances:
        arr = inst["dataset"].pixel_array.astype(np.int16)
        slices.append(arr)
        sop_uids.append(inst["dataset"].SOPInstanceUID)

    volume = np.stack(slices, axis=0)  # (num_slices, H, W)

    # 5. get spacing
    ds0 = instances[0]["dataset"]
    # pixel spacing: (row, col)
    pixel_spacing = getattr(ds0, "PixelSpacing", [1.0, 1.0])
    # slice spacing
    slice_thickness = getattr(ds0, "SliceThickness", None)
    spacing_between_slices = getattr(ds0, "SpacingBetweenSlices", None)

    if slice_thickness is not None:
        z_spacing = float(slice_thickness)
    elif spacing_between_slices is not None:
        z_spacing = float(spacing_between_slices)
    else:
        # fallback
        z_spacing = 1.0

    metadata = {
        "PixelSpacing": (float(pixel_spacing[0]), float(pixel_spacing[1])),
        "SliceSpacing": z_spacing,
        "SliceLocations": [inst["z"] for inst in instances],
        "SOPInstanceUIDs": sop_uids,
        "Modality": getattr(ds0, "Modality", "UNKNOWN"),
        "SeriesInstanceUID": getattr(ds0, "SeriesInstanceUID", None),
        "PatientID": getattr(ds0, "PatientID", None),
    }

    return volume, metadata


if __name__ == "__main__":
    # petit test manuel
    test_dir = "/media/aboukhelifa/HDD2/dataset_rsna/rsna-intracranial-aneurysm-detection/series/"  
    if os.path.exists(test_dir):
        vol, meta = load_dicom_series(test_dir)
        print("volume shape:", vol.shape)
        print("spacing:", meta["PixelSpacing"], meta["SliceSpacing"])
        print("modality:", meta["Modality"])
        print("first 3 SOPs:", meta["SOPInstanceUIDs"][:3])
    else:
        print("⚠️ Test dir does not exist, change the path in __main__")
