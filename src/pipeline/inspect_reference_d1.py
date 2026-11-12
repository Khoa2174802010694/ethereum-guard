
import pandas as pd
from pathlib import Path

path = Path("data/sample/allnodes_enhancement.csv")

df = pd.read_csv(path)

print("===== D1 DATASET SUMMARY =====")
print("Shape:", df.shape)

print("\n===== COLUMNS =====")
for i, col in enumerate(df.columns):
    print(i, col, "|", df[col].dtype)

print("\n===== FIRST 5 ROWS =====")
print(df.head().to_string())

print("\n===== MISSING VALUES =====")
print(df.isna().sum().to_string())

print("\n===== DUPLICATES =====")
print("Duplicate rows:", df.duplicated().sum())

print("\n===== POSSIBLE LABEL COLUMN =====")
for col in df.columns:
    if df[col].nunique() <= 10:
        print(
            col,
            "unique:", df[col].unique()[:10]
        )

print("\n===== SAVED FILE =====")
print("D1 rows:", len(df))
