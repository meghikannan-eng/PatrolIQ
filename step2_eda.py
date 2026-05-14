"""Step 2: Exploratory Data Analysis."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA = "/sessions/laughing-bold-gauss/mnt/outputs/patroliq/data/clean.parquet"
FIG = "/sessions/laughing-bold-gauss/mnt/outputs/patroliq/figures"
os.makedirs(FIG, exist_ok=True)
sns.set_style("whitegrid")

df = pd.read_parquet(DATA)
df["Date"] = pd.to_datetime(df["Date"])
df["Hour"] = df["Date"].dt.hour
df["DOW"] = df["Date"].dt.day_name()
df["Month"] = df["Date"].dt.month

# 1. Top crime types
top = df["Primary Type"].value_counts().head(15)
fig, ax = plt.subplots(figsize=(9, 5))
top.plot(kind="barh", ax=ax, color="steelblue")
ax.invert_yaxis(); ax.set_title("Top 15 Crime Types"); ax.set_xlabel("Count")
plt.tight_layout(); plt.savefig(f"{FIG}/01_top_crime_types.png", dpi=110); plt.close()

# 2. Hourly distribution
fig, ax = plt.subplots(figsize=(9, 4))
df["Hour"].value_counts().sort_index().plot(kind="bar", ax=ax, color="tomato")
ax.set_title("Crimes by Hour of Day"); ax.set_xlabel("Hour"); ax.set_ylabel("Count")
plt.tight_layout(); plt.savefig(f"{FIG}/02_hourly.png", dpi=110); plt.close()

# 3. Day of week
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
fig, ax = plt.subplots(figsize=(8, 4))
df["DOW"].value_counts().reindex(dow_order).plot(kind="bar", ax=ax, color="mediumseagreen")
ax.set_title("Crimes by Day of Week"); plt.xticks(rotation=30)
plt.tight_layout(); plt.savefig(f"{FIG}/03_dow.png", dpi=110); plt.close()

# 4. Monthly
fig, ax = plt.subplots(figsize=(8, 4))
df["Month"].value_counts().sort_index().plot(kind="bar", ax=ax, color="slateblue")
ax.set_title("Crimes by Month"); plt.tight_layout()
plt.savefig(f"{FIG}/04_monthly.png", dpi=110); plt.close()

# 5. Hour x DOW heatmap
piv = df.pivot_table(index="DOW", columns="Hour", values="ID", aggfunc="count").reindex(dow_order)
fig, ax = plt.subplots(figsize=(11, 4))
sns.heatmap(piv, cmap="rocket_r", ax=ax)
ax.set_title("Crime Density: Day x Hour"); plt.tight_layout()
plt.savefig(f"{FIG}/05_heatmap_day_hour.png", dpi=110); plt.close()

# 6. Geographic scatter (sample for speed)
samp = df.sample(min(20_000, len(df)), random_state=1)
fig, ax = plt.subplots(figsize=(7, 8))
ax.scatter(samp["Longitude"], samp["Latitude"], s=1, alpha=0.25, c="crimson")
ax.set_title("Chicago Crime Locations (sample)"); ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
plt.tight_layout(); plt.savefig(f"{FIG}/06_geo_scatter.png", dpi=110); plt.close()

print("Records:", len(df))
print("Unique crime types:", df["Primary Type"].nunique())
print(f"Arrest rate: {df['Arrest'].mean()*100:.1f}%")
print(f"Domestic rate: {df['Domestic'].mean()*100:.1f}%")
print(f"Date range: {df['Date'].min()} -> {df['Date'].max()}")
print("Top 5 types:")
print(top.head().to_string())
print(f"Figures saved to {FIG}/")
