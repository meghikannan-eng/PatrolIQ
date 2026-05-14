"""Step 3: Feature engineering."""
import pandas as pd
import numpy as np

DATA = "/sessions/laughing-bold-gauss/mnt/outputs/patroliq/data/clean.parquet"
OUT = "/sessions/laughing-bold-gauss/mnt/outputs/patroliq/data/features.parquet"

# Crime severity score (1=low, 5=most severe). Customize as needed.
SEVERITY = {
    "HOMICIDE": 5, "CRIM SEXUAL ASSAULT": 5, "CRIMINAL SEXUAL ASSAULT": 5,
    "KIDNAPPING": 5, "ARSON": 5,
    "ROBBERY": 4, "ASSAULT": 4, "BATTERY": 4, "WEAPONS VIOLATION": 4,
    "OFFENSE INVOLVING CHILDREN": 4, "HUMAN TRAFFICKING": 5,
    "BURGLARY": 3, "MOTOR VEHICLE THEFT": 3, "NARCOTICS": 3,
    "STALKING": 3, "INTIMIDATION": 3, "SEX OFFENSE": 4,
    "THEFT": 2, "CRIMINAL DAMAGE": 2, "CRIMINAL TRESPASS": 2,
    "DECEPTIVE PRACTICE": 2, "PUBLIC PEACE VIOLATION": 2,
    "INTERFERENCE WITH PUBLIC OFFICER": 2,
    "GAMBLING": 1, "LIQUOR LAW VIOLATION": 1, "OBSCENITY": 1,
    "PUBLIC INDECENCY": 1, "OTHER OFFENSE": 1, "OTHER NARCOTIC VIOLATION": 2,
    "CONCEALED CARRY LICENSE VIOLATION": 2, "NON-CRIMINAL": 1,
}

def season_from_month(m):
    return {12:"Winter",1:"Winter",2:"Winter",
            3:"Spring",4:"Spring",5:"Spring",
            6:"Summer",7:"Summer",8:"Summer",
            9:"Fall",10:"Fall",11:"Fall"}[m]

df = pd.read_parquet(DATA)
df["Date"] = pd.to_datetime(df["Date"])

# Temporal features
df["Hour"] = df["Date"].dt.hour
df["Day_of_Week"] = df["Date"].dt.day_name()
df["DOW_Num"] = df["Date"].dt.dayofweek      # 0=Mon
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year
df["Season"] = df["Month"].map(season_from_month)
df["Is_Weekend"] = df["DOW_Num"].isin([5, 6])

# Cyclical encodings (useful for distance-based clustering)
df["Hour_sin"] = np.sin(2*np.pi*df["Hour"]/24)
df["Hour_cos"] = np.cos(2*np.pi*df["Hour"]/24)
df["DOW_sin"] = np.sin(2*np.pi*df["DOW_Num"]/7)
df["DOW_cos"] = np.cos(2*np.pi*df["DOW_Num"]/7)
df["Month_sin"] = np.sin(2*np.pi*(df["Month"]-1)/12)
df["Month_cos"] = np.cos(2*np.pi*(df["Month"]-1)/12)

# Severity
df["Crime_Severity_Score"] = df["Primary Type"].map(SEVERITY).fillna(2).astype(int)

# Geographic binning (helps coarse cluster summaries)
df["Lat_bin"] = (df["Latitude"]*100).round().astype(int)
df["Lon_bin"] = (df["Longitude"]*100).round().astype(int)

# Categorical codes (cheap encoding for ML)
df["PrimaryType_Code"] = df["Primary Type"].astype("category").cat.codes
df["LocDesc_Code"] = df["Location Description"].astype("category").cat.codes

df.to_parquet(OUT, index=False)
print(f"Saved -> {OUT}")
print("Engineered cols:", [c for c in df.columns if c not in pd.read_parquet(DATA).columns])
print(df[["Hour","Day_of_Week","Season","Is_Weekend","Crime_Severity_Score"]].head())
