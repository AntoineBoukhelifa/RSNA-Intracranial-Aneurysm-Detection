                ┌──────────────────────────────┐
                │  train.csv + series/ DICOMs  │
                └────────────┬─────────────────┘
                             │
                             ▼
                 [1] precache_all.py
                  ├─> appelle pipeline.py
                  ├─> crée cache/*.npy
                  └─> gain de temps énorme

                             │
                             ▼
                 [2] train.py (Entraînement)
                  ├─> lit config.yaml
                  ├─> utilise cache/
                  ├─> calcule AUCs
                  └─> sauvegarde best_model.pth

                             │
                             ▼
                 [3] (optionnel) inference.py
                  └─> prédiction + soumission Kaggle


python src/preprocessing/precache_all.py

python src/preprocessing/pipeline.py (optionnel)

python src/training/train.py

