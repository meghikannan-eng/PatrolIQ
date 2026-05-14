"""Step 5: Dimensionality Reduction — PCA + t-SNE."""
import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

DATA = "/sessions/laughing-bold-gauss/mnt/outputs/patroliq/data/clustered.parquet"
FIG  = "/sessions/laughing-bold-gauss/mnt/outputs/patroliq/figures"
DAT  = "/sessions/laughing-bold-gauss/mnt/outputs/patroliq/data"

df = pd.read_parquet(DATA)
features = ["Latitude","Longitude","Hour_sin","Hour_cos","DOW_sin","DOW_cos",
            "Month_sin","Month_cos","Crime_Severity_Score","Is_Weekend",
            "PrimaryType_Code","LocDesc_Code","District","Ward","Community Area",
            "Beat","Arrest","Domestic","Year"]
X = df[features].copy()
X["Is_Weekend"] = X["Is_Weekend"].astype(int)
X["Arrest"] = X["Arrest"].astype(int)
X["Domestic"] = X["Domestic"].astype(int)
X = X.fillna(0).values
Xz = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=min(10, Xz.shape[1])).fit(Xz)
var = pca.explained_variance_ratio_
cum = var.cumsum()
print("PCA explained variance:", np.round(var, 3))
print("Cumulative:", np.round(cum, 3))
n_for_70 = int(np.searchsorted(cum, 0.70) + 1)
print(f"Components for >=70% variance: {n_for_70}")

fig, ax = plt.subplots(figsize=(8,4))
ax.bar(range(1, len(var)+1), var, alpha=0.6, label="Per component")
ax.plot(range(1, len(cum)+1), cum, "o-", color="crimson", label="Cumulative")
ax.axhline(0.70, ls="--", color="gray"); ax.set_xlabel("Component"); ax.set_ylabel("Variance ratio")
ax.set_title("PCA Scree"); ax.legend(); plt.tight_layout()
plt.savefig(f"{FIG}/20_pca_scree.png", dpi=110); plt.close()

# PCA 2D projection colored by hotspot cluster
pcs = PCA(n_components=2).fit_transform(Xz)
fig, ax = plt.subplots(figsize=(8,6))
sc = ax.scatter(pcs[:,0], pcs[:,1], c=df["geo_kmeans_8"], cmap="tab10", s=2, alpha=0.4)
ax.set_title("PCA 2D — colored by geo hotspot"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
plt.tight_layout(); plt.savefig(f"{FIG}/21_pca_2d.png", dpi=110); plt.close()

# Feature importance from PC1 + PC2 loadings
loadings = pd.DataFrame(pca.components_[:3].T, index=features,
                        columns=["PC1","PC2","PC3"])
loadings["abs_PC1"] = loadings["PC1"].abs()
top_drivers = loadings.sort_values("abs_PC1", ascending=False).head(5)
print("\nTop 5 features driving PC1:\n", top_drivers[["PC1","PC2","PC3"]])
loadings.to_csv(f"{DAT}/pca_loadings.csv")

# t-SNE on a sample (expensive)
rng = np.random.RandomState(42)
idx = rng.choice(len(Xz), min(5000, len(Xz)), replace=False)
print(f"\nRunning t-SNE on {len(idx)} points...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
emb = tsne.fit_transform(Xz[idx])
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
axes[0].scatter(emb[:,0], emb[:,1], c=df["geo_kmeans_8"].values[idx], cmap="tab10", s=4, alpha=0.6)
axes[0].set_title("t-SNE — by geographic hotspot")
axes[1].scatter(emb[:,0], emb[:,1], c=df["Crime_Severity_Score"].values[idx], cmap="YlOrRd", s=4, alpha=0.6)
axes[1].set_title("t-SNE — by crime severity")
plt.tight_layout(); plt.savefig(f"{FIG}/22_tsne.png", dpi=110); plt.close()

# Save reduced datasets
np.save(f"{DAT}/pca_2d.npy", pcs)
np.save(f"{DAT}/tsne_2d.npy", emb)
np.save(f"{DAT}/tsne_idx.npy", idx)
import json
with open(f"{DAT}/dimreduction_results.json","w") as f:
    json.dump({"pca_var": var.tolist(), "cum": cum.tolist(),
               "n_for_70pct": n_for_70,
               "top_drivers": top_drivers.index.tolist()}, f, indent=2)
print("Saved PCA + t-SNE artefacts.")
