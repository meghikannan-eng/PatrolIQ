"""Step 4: Clustering — KMeans + DBSCAN + Hierarchical (geographic), KMeans (temporal)."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, json
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram

DATA = "/sessions/laughing-bold-gauss/mnt/outputs/patroliq/data/features.parquet"
FIG  = "/sessions/laughing-bold-gauss/mnt/outputs/patroliq/figures"
DAT  = "/sessions/laughing-bold-gauss/mnt/outputs/patroliq/data"

df = pd.read_parquet(DATA)
print(f"Data: {len(df):,} rows")

# ---------- Geographic clustering ----------
geo = df[["Latitude","Longitude"]].values
# For DBSCAN/Hierarchical we sample to keep it tractable
rng = np.random.RandomState(42)
idx = rng.choice(len(geo), size=min(15_000, len(geo)), replace=False)
geo_s = geo[idx]

scaler = StandardScaler()
geo_s_z = scaler.fit_transform(geo_s)

results = {}

# Elbow + silhouette for KMeans
inertias, sils = [], []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(geo_s_z)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(geo_s_z, km.labels_, sample_size=5000, random_state=42))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(range(2,11), inertias, "o-"); axes[0].set_title("Elbow"); axes[0].set_xlabel("k")
axes[1].plot(range(2,11), sils, "o-", color="orange"); axes[1].set_title("Silhouette by k"); axes[1].set_xlabel("k")
plt.tight_layout(); plt.savefig(f"{FIG}/10_elbow_silhouette.png", dpi=110); plt.close()
best_k = int(np.argmax(sils) + 2)
print(f"KMeans best k = {best_k}, silhouette = {max(sils):.3f}")

# Fit best KMeans on full geo
km_full = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(geo)
df["geo_kmeans"] = km_full.labels_
km_sil = silhouette_score(geo[idx], km_full.predict(geo[idx]), sample_size=5000, random_state=42)
km_db  = davies_bouldin_score(geo[idx], km_full.predict(geo[idx]))
results["KMeans"] = {"k": best_k, "silhouette": float(km_sil), "davies_bouldin": float(km_db)}

# DBSCAN on sampled scaled geo
db = DBSCAN(eps=0.15, min_samples=30).fit(geo_s_z)
mask = db.labels_ != -1
if mask.sum() > 100 and len(set(db.labels_[mask])) > 1:
    db_sil = silhouette_score(geo_s_z[mask], db.labels_[mask], sample_size=5000, random_state=42)
    db_db  = davies_bouldin_score(geo_s_z[mask], db.labels_[mask])
else:
    db_sil = db_db = float("nan")
results["DBSCAN"] = {"clusters": int(len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)),
                     "noise_pct": float((db.labels_ == -1).mean()*100),
                     "silhouette": float(db_sil), "davies_bouldin": float(db_db)}

# Hierarchical
hc = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit(geo_s_z)
hc_sil = silhouette_score(geo_s_z, hc.labels_, sample_size=5000, random_state=42)
hc_db  = davies_bouldin_score(geo_s_z, hc.labels_)
results["Hierarchical"] = {"k": best_k, "silhouette": float(hc_sil), "davies_bouldin": float(hc_db)}

# Dendrogram (small subsample)
sub = geo_s_z[rng.choice(len(geo_s_z), 500, replace=False)]
Z = linkage(sub, method="ward")
fig, ax = plt.subplots(figsize=(10, 4))
dendrogram(Z, truncate_mode="lastp", p=20, ax=ax)
ax.set_title("Hierarchical Dendrogram (500-pt subsample)")
plt.tight_layout(); plt.savefig(f"{FIG}/11_dendrogram.png", dpi=110); plt.close()

# Visualize KMeans clusters geographically
fig, ax = plt.subplots(figsize=(7, 8))
sc = ax.scatter(df["Longitude"], df["Latitude"], c=df["geo_kmeans"], s=2, cmap="tab10", alpha=0.5)
ax.scatter(km_full.cluster_centers_[:,1], km_full.cluster_centers_[:,0],
           c="black", marker="X", s=120, label="centers")
ax.set_title(f"Geographic KMeans Hotspots (k={best_k})"); ax.legend()
plt.tight_layout(); plt.savefig(f"{FIG}/12_geo_kmeans.png", dpi=110); plt.close()

# DBSCAN viz
fig, ax = plt.subplots(figsize=(7, 8))
colors = np.where(db.labels_ == -1, "lightgray", "crimson")
ax.scatter(geo_s[:,1], geo_s[:,0], c=colors, s=3, alpha=0.5)
ax.set_title(f"DBSCAN — {results['DBSCAN']['clusters']} clusters, {results['DBSCAN']['noise_pct']:.1f}% noise")
plt.tight_layout(); plt.savefig(f"{FIG}/13_dbscan.png", dpi=110); plt.close()

# ---------- Temporal clustering ----------
T_feats = ["Hour_sin","Hour_cos","DOW_sin","DOW_cos","Month_sin","Month_cos","Crime_Severity_Score"]
T = StandardScaler().fit_transform(df[T_feats].values)
sils_t = []
for k in range(2, 8):
    km_t = KMeans(n_clusters=k, n_init=10, random_state=42).fit(T[idx])
    sils_t.append(silhouette_score(T[idx], km_t.labels_, sample_size=5000, random_state=42))
best_kt = int(np.argmax(sils_t) + 2)
km_t = KMeans(n_clusters=best_kt, n_init=10, random_state=42).fit(T)
df["temporal_kmeans"] = km_t.labels_
results["Temporal_KMeans"] = {"k": best_kt, "silhouette": float(max(sils_t))}
print(f"Temporal KMeans best k = {best_kt}, sil = {max(sils_t):.3f}")

# Temporal cluster profile
prof = df.groupby("temporal_kmeans").agg(
    n=("ID","size"),
    avg_hour=("Hour","mean"),
    weekend_pct=("Is_Weekend","mean"),
    severity=("Crime_Severity_Score","mean"),
).round(2)
print("Temporal cluster profile:\n", prof)

# Save
df.to_parquet(f"{DAT}/clustered.parquet", index=False)
with open(f"{DAT}/cluster_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))

# ---------- Deployment KMeans with 8 hotspots (project requires 5-10) ----------
DEPLOY_K = 8
km_deploy = KMeans(n_clusters=DEPLOY_K, n_init=10, random_state=42).fit(geo)
df["geo_kmeans_8"] = km_deploy.labels_
deploy_sil = silhouette_score(geo[idx], km_deploy.predict(geo[idx]), sample_size=5000, random_state=42)
fig, ax = plt.subplots(figsize=(7, 8))
ax.scatter(df["Longitude"], df["Latitude"], c=df["geo_kmeans_8"], s=2, cmap="tab10", alpha=0.6)
ax.scatter(km_deploy.cluster_centers_[:,1], km_deploy.cluster_centers_[:,0],
           c="black", marker="X", s=120)
ax.set_title(f"Deployment Hotspots (k={DEPLOY_K}, sil={deploy_sil:.2f})")
plt.tight_layout(); plt.savefig(f"{FIG}/14_geo_kmeans_8.png", dpi=110); plt.close()

# Hotspot summary
hot = df.groupby("geo_kmeans_8").agg(
    n=("ID","size"),
    top_crime=("Primary Type", lambda s: s.value_counts().index[0]),
    arrest_rate=("Arrest","mean"),
    severity=("Crime_Severity_Score","mean"),
    lat=("Latitude","mean"),
    lon=("Longitude","mean"),
).round(3).sort_values("n", ascending=False)
hot.to_csv(f"{DAT}/hotspot_summary.csv")
print("\nHotspot summary (k=8):")
print(hot)
df.to_parquet(f"{DAT}/clustered.parquet", index=False)
