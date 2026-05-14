"""Step 1: Data acquisition, sampling, and preprocessing."""
import pandas as pd
import numpy as np
import os

RAW_PATH = "/sessions/laughing-bold-gauss/mnt/uploads/6a4c3cd9-2e3d-46ff-a26a-ee92dab8c8f1-1778786136572_Crimes_-_2001_to_Present_20260515.csv"
OUT = "/sessions/laughing-bold-gauss/mnt/outputs/patroliq/data/clean.parquet"

def load_and_sample(path, n=500_000, seed=42):
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} cols")
    # Use most recent records first (project asks for "recent 500k")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date", ascending=False)
    if len(df) > n:
        df = df.head(n).copy()
    return df.reset_index(drop=True)

def clean(df):
    # Drop rows missing critical fields
    before = len(df)
    df = df.dropna(subset=["Latitude", "Longitude", "Date", "Primary Type"])
    # Filter Chicago bounding box (project spec)
    df = df[(df["Latitude"].between(41.6, 42.05)) & (df["Longitude"].between(-87.95, -87.5))]
    # Fill non-critical
    for c in ["Location Description", "Description"]:
        if c in df.columns:
            df[c] = df[c].fillna("UNKNOWN")
    for c in ["District", "Ward", "Community Area", "Beat"]:
        if c in df.columns:
            df[c] = df[c].fillna(-1)
    # Cast booleans
    for c in ["Arrest", "Domestic"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    print(f"Cleaned: {before:,} -> {len(df):,} rows ({before - len(df):,} dropped)")
    return df.reset_index(drop=True)

if __name__ == "__main__":
    df = load_and_sample(RAW_PATH)
    df = clean(df)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"Saved -> {OUT}")
    print(df[["Date", "Primary Type", "Latitude", "Longitude", "Arrest"]].head())
