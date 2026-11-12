
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data/raw/mdbchain")

files = list(DATA_DIR.rglob("*.csv"))

summary = []
all_timestamps = []
all_values = []

for file in files:
    try:
        df = pd.read_csv(file)

        if not set([
            "BlockHeight",
            "TimeStamp",
            "From",
            "To",
            "Value"
        ]).issubset(df.columns):
            continue

        df["TimeStamp"] = pd.to_numeric(
            df["TimeStamp"], errors="coerce"
        )
        df["Value"] = pd.to_numeric(
            df["Value"], errors="coerce"
        )

        summary.append({
            "wallet": file.stem,
            "transactions": len(df),
            "unique_from": df["From"].nunique(),
            "unique_to": df["To"].nunique(),
            "missing": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "min_timestamp": df["TimeStamp"].min(),
            "max_timestamp": df["TimeStamp"].max(),
            "zero_value": int((df["Value"] == 0).sum()),
            "positive_value": int((df["Value"] > 0).sum()),
            "negative_value": int((df["Value"] < 0).sum())
        })

        all_timestamps.extend(
            df["TimeStamp"].dropna().tolist()
        )
        all_values.extend(
            df["Value"].dropna().tolist()
        )

    except Exception as e:
        print(f"Error reading {file.name}: {e}")

result = pd.DataFrame(summary)

print("\n===== WALLET SUMMARY =====")
print("Wallet files:", len(result))
print("Total transactions:", result["transactions"].sum())
print("Transactions per wallet:")
print(result["transactions"].describe())

print("\n===== DATA QUALITY =====")
print("Total missing:", result["missing"].sum())
print("Total duplicate rows:", result["duplicate_rows"].sum())

print("\n===== TRANSACTION VALUES =====")
print("Zero values:", result["zero_value"].sum())
print("Positive values:", result["positive_value"].sum())
print("Negative values:", result["negative_value"].sum())

print("\n===== TIME RANGE =====")
print("Min timestamp:", min(all_timestamps))
print("Max timestamp:", max(all_timestamps))

print("\n===== TOP 10 WALLETS BY TRANSACTIONS =====")
print(
    result.nlargest(10, "transactions")[
        ["wallet", "transactions"]
    ].to_string(index=False)
)

Path("results").mkdir(exist_ok=True)
result.to_csv(
    "results/mdbchain_wallet_summary.csv",
    index=False
)

print("\nSaved: results/mdbchain_wallet_summary.csv")
