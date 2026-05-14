"""Step 7: MLflow experiment tracking — log clustering + DR runs."""
import mlflow, json, os
import pandas as pd, numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

DATA = "/sessions/laughing-bold-gauss/mnt/outputs/patroliq/data/features.parquet"
TRACK = "file:" + "/sessions/laughing-bold-gauss/mnt/outputs/patroliq/mlruns"
mlflow.set_tracking_uri(TRACK)
mlflow.set_experiment("PatrolIQ_Clustering")

df = pd.read_parquet(DATA)
rng = np.random.RandomState(42)
idx = rng.choice(len(df), min(10_000, len(df)), replace=False)
geo = df[["Latitude","Longitude"]].values
geo_z = StandardScaler().fit_transform(geo)

# KMeans sweep
for k in [3, 5, 8, 10]:
    with mlflow.start_run(run_name=f"KMeans_k{k}"):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(geo)
        sil = silhouette_score(geo[idx], km.predict(geo[idx]), sample_size=5000, random_state=42)
        dbi = davies_bouldin_score(geo[idx], km.predict(geo[idx]))
        mlflow.log_params({"algorithm":"KMeans","k":k,"random_state":42})
        mlflow.log_metrics({"silhouette":float(sil),"davies_bouldin":float(dbi),
                            "inertia":float(km.inertia_)})

# DBSCAN sweep
for eps in [0.10, 0.15, 0.25]:
    with mlflow.start_run(run_name=f"DBSCAN_eps{eps}"):
        db = DBSCAN(eps=eps, min_samples=30).fit(geo_z[idx])
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        noise = float((db.labels_ == -1).mean())
        mask = db.labels_ != -1
        if mask.sum() > 100 and n_clusters > 1:
            sil = silhouette_score(geo_z[idx][mask], db.labels_[mask], sample_size=5000, random_state=42)
            dbi = davies_bouldin_score(geo_z[idx][mask], db.labels_[mask])
        else:
            sil, dbi = float("nan"), float("nan")
        mlflow.log_params({"algorithm":"DBSCAN","eps":eps,"min_samples":30})
        mlflow.log_metrics({"silhouette":float(sil) if sil==sil else 0.0,
                            "davies_bouldin":float(dbi) if dbi==dbi else 0.0,
                            "n_clusters":n_clusters,"noise_pct":noise})

# Hierarchical
for k in [5, 8]:
    with mlflow.start_run(run_name=f"Hierarchical_k{k}"):
        hc = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(geo_z[idx])
        sil = silhouette_score(geo_z[idx], hc.labels_, sample_size=5000, random_state=42)
        dbi = davies_bouldin_score(geo_z[idx], hc.labels_)
        mlflow.log_params({"algorithm":"Hierarchical","k":k,"linkage":"ward"})
        mlflow.log_metrics({"silhouette":float(sil),"davies_bouldin":float(dbi)})

# PCA experiment
mlflow.set_experiment("PatrolIQ_DimReduction")
features = ["Latitude","Longitude","Hour_sin","Hour_cos","DOW_sin","DOW_cos",
            "Month_sin","Month_cos","Crime_Severity_Score","PrimaryType_Code"]
X = StandardScaler().fit_transform(df[features].fillna(0).values)
for n in [2, 3, 5]:
    with mlflow.start_run(run_name=f"PCA_n{n}"):
        pca = PCA(n_components=n).fit(X)
        mlflow.log_params({"technique":"PCA","n_components":n})
        mlflow.log_metric("cum_explained_variance", float(pca.explained_variance_ratio_.sum()))

print("MLflow runs written to", TRACK)
# Quick summary
client = mlflow.tracking.MlflowClient()
for exp_name in ["PatrolIQ_Clustering", "PatrolIQ_DimReduction"]:
    exp = client.get_experiment_by_name(exp_name)
    runs = client.search_runs([exp.experiment_id], order_by=["metrics.silhouette DESC"])
    print(f"\n[{exp_name}] {len(runs)} runs")
    for r in runs[:5]:
        print(f"  {r.info.run_name}: {r.data.metrics}")
