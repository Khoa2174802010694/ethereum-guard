
import pandas as pd

df = pd.read_csv("data/sample/allnodes_enhancement.csv")

print("===== ID RELATIONSHIP =====")
print("To equals address_id:", (df["To"] == df["address_id"]).all())
print("sequence + 1 equals From:",
      (df["sequence"] + 1 == df["From"]).all())

print("\n===== DUPLICATE ADDRESS FEATURE PATTERNS =====")

feature_cols = [
    "total_transactions",
    "transactions_initiated",
    "transactions_received",
    "average_transaction_value",
    "total_transaction_value",
    "average_outgoing_value",
    "average_incoming_value",
    "total_outgoing_value",
    "total_incoming_value",
    "last_outgoing_transaction_value",
    "last_incoming_transaction_value",
    "value1", "value2", "value3", "value4",
    "timestamps1", "timestamps2", "timestamps3", "timestamps4"
]

print("Unique feature vectors:", df[feature_cols].drop_duplicates().shape[0])
print("Total rows:", len(df))

print("\n===== LABEL DISTRIBUTION BY ADDRESS =====")

address_df = (
    df.groupby("address_id")
    .agg(
        label=("label", "first"),
        row_count=("label", "size")
    )
    .reset_index()
)

print(address_df.groupby("label")["row_count"].describe())

print("\n===== MOST FREQUENT ADDRESSES =====")
print(
    address_df.sort_values("row_count", ascending=False)
    .head(15)
    .to_string(index=False)
)

print("\n===== FEATURE STATISTICS BY LABEL =====")

for col in feature_cols:
    print(f"\n{col}")
    print(df.groupby("label")[col].agg(
        ["mean", "median", "std", "min", "max"]
    ))

print("\nDeep audit completed.")
