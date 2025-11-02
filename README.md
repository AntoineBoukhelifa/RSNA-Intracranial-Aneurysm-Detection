# 🧠 RSNA Intracranial Aneurysm Detection


> **Goal:**  
> Develop a robust deep learning system for the automatic **detection and localization of intracranial aneurysms** from medical imaging (CTA, MRA, MRI).  
> This project is built for the [RSNA Intracranial Aneurysm Detection Challenge](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/overview).

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Preprocessing Pipeline](#preprocessing-pipeline)
4. [Modeling & Training](#modeling--training)
5. [Evaluation Metric](#evaluation-metric)
6. [Results](#results)
7. [Project Structure](#project-structure)
8. [Setup](#setup)
9. [Usage](#usage)
10. [License](#license)

---

## 🧩 Overview

Intracranial aneurysms affect around **3% of the global population**, and early detection can save lives.  
This project builds a complete ML pipeline — from **raw DICOMs to model predictions** — capable of handling diverse scanner protocols, imaging modalities, and institutions.

Key goals:
- Automate aneurysm presence detection across 13 vessel locations  
- Normalize heterogeneous imaging data (CTA, MRA, MRI)  
- Optimize for RSNA’s **Mean Weighted Columnwise AUCROC** metric

---

## 🧬 Dataset

| File | Description |
|------|--------------|
| `train.csv` | Contains 14 labels (13 vessel locations + `Aneurysm Present`) |
| `train_localizers.csv` | Provides pixel coordinates of annotated aneurysms |
| `series/{SeriesInstanceUID}/` | Folder containing DICOM images for each patient |
| `segmentations/` | Vessel segmentation masks for a subset of cases |

Example structure:

data/
├── train.csv
├── train_localizers.csv
└── series/
├── 1.2.826.0.1.3680043.2055/
│ ├── 1.2.826.0.1.3680043.2055.1.dcm
│ └── ...
└── 1.2.826.0.1.3680043.2056/


---

## 🧼 Preprocessing Pipeline

The preprocessing module standardizes all imaging data to ensure uniform input for CNN models.

| Step | Script | Purpose |
|------|---------|----------|
| **1. Load DICOM** | `dicom_loader.py` | Reads and stacks DICOM slices into 3D volumes |
| **2. Normalize Intensities** | `normalization.py` | Standardizes intensities by modality (CTA, MRI...) |
| **3. Resample Spatially** | `resampling.py` | Converts voxel spacing to isotropic 1×1×1 mm³ |
| **4. Cache Volumes** | `pipeline.py` | preprocessing pipeline for a single DICOM series |
| **5. Parallel Preprocessing** | `precache_all.py` | Preprocess all DICOM series in the dataset and save them as cached .npy volumes |

```bash
python -m src.preprocessing.precache_all
'''


