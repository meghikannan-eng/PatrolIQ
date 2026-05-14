# PatrolIQ — Smart Safety Analytics Platform

Chicago crime intelligence using unsupervised ML. Built per the PatrolIQ project brief.

## Folder layout

```
patroliq/
├── PatrolIQ.ipynb            # Main notebook (clean copy, generic paths)
├── PatrolIQ_executed.ipynb   # Same notebook executed end-to-end on the uploaded CSV
├── step1_preprocess.py       # Load + sample + clean
├── step2_eda.py              # EDA + figures
├── step3_features.py         # Feature engineering
├── step4_clustering.py       # KMeans, DBSCAN, Hierarchical + temporal KMeans
├── step5_dimreduction.py     # PCA + t-SNE
├── step7_mlflow.py           # MLflow experiment tracking
├── app/app.py                # Streamlit Cloud entry point (multi-page)
├── data/                     # clean.parquet, features.parquet, clustered.parquet, etc.
├── figures/                  # All PNGs from EDA + clustering + DR
├── mlruns/                   # MLflow tracking store
└── requirements.txt
```

## Run order

```bash
python step1_preprocess.py
python step2_eda.py
python step3_features.py
python step4_clustering.py
python step5_dimreduction.py
python step7_mlflow.py
streamlit run app/app.py
```

## Key results on uploaded CSV (76,415 rows)

| Algorithm     | k / params      | Silhouette | Davies-Bouldin |
|---------------|-----------------|-----------:|---------------:|
| KMeans (deploy) | k=8           |       0.41 |          0.78 |
| KMeans (sil-best) | k=2         |       0.50 |          0.75 |
| DBSCAN        | eps=0.15        |       0.17 |          0.55 |
| Hierarchical  | k=8, ward       |       0.37 |          0.71 |

PCA: 8 components needed for ≥70% variance (lat/lon-related features dominate PC1).
Top 5 drivers of PC1: District, Beat, Latitude, Ward, Community Area.

## Streamlit Cloud deploy

1. Push this folder to a public GitHub repo.
2. Streamlit Cloud → New app → main file `app/app.py`.
3. `requirements.txt` is at the repo root.

## Optional Docker (+10% bonus)

Add a `Dockerfile` with `FROM python:3.11-slim`, `pip install -r requirements.txt`, `CMD streamlit run app/app.py --server.port=$PORT --server.address=0.0.0.0`.
