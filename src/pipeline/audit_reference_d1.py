
import pandas as pd
from pathlib import Path

df = pd.read_csv("data/sample/allnodes_enhancement.csv")

print("===== ADDRESS ANALYSIS =====")

address_stats = df.groupby("address_id").agg(
    rows=("address_id", "size"),
    label_count=("label", "nunique"),
    label=("label", "first"),
    from_count=("From", "nunique"),
    to_count=("To", "nunique")
)

print("Unique addresses:", len(address_stats))
print("Addresses appearing more than once:",
      (address_stats["rows"] > 1).sum())
print("Addresses with inconsistent labels:",
      (address_stats["label_count"] > 1).sum())

print("\n===== LABEL DISTRIBUTION BY ADDRESS =====")
print(address_stats["label"].value_counts().sort_index())

print("\n===== ROWS PER ADDRESS =====")
print(address_stats["rows"].describe())

print("\n===== FROM / TO / ADDRESS RELATION =====")
print("From equals sequence + 1:",
      (df["From"] == df["sequence"] + 1).all())

print("Unique From:", df["From"].nunique())
print("Unique To:", df["To"].nunique())
print("Unique address_id:", df["address_id"].nunique())

print("\n===== ZERO RATIO =====")
for col in [
    "value1", "value2", "value3", "value4",
    "timestamps1", "timestamps2",
    "timestamps3", "timestamps4"
]:
    print(f"{col}: {(df[col] == 0).mean():.2%}")

print("\n===== LABEL BY ADDRESS =====")
print(
    address_stats.groupby("label")["rows"]
    .agg(["count", "mean", "median", "max"])
)

print("\nAudit completed.")
