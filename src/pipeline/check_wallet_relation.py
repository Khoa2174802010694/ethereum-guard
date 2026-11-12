
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/mdbchain")

files = list(DATA_DIR.rglob("*.csv"))

total_rows = 0
from_matches = 0
to_matches = 0
both_matches = 0
neither_matches = 0

examples = []

for file in files:
    wallet = file.stem.lower()

    df = pd.read_csv(
        file,
        dtype={"From": "string", "To": "string"}
    )

    from_col = df["From"].fillna("").str.lower()
    to_col = df["To"].fillna("").str.lower()

    from_mask = from_col == wallet
    to_mask = to_col == wallet

    from_count = int(from_mask.sum())
    to_count = int(to_mask.sum())
    both_count = int((from_mask & to_mask).sum())
    neither_count = int((~from_mask & ~to_mask).sum())

    total_rows += len(df)
    from_matches += from_count
    to_matches += to_count
    both_matches += both_count
    neither_matches += neither_count

    if len(examples) < 10:
        examples.append({
            "wallet": wallet,
            "rows": len(df),
            "as_sender": from_count,
            "as_receiver": to_count,
            "neither": neither_count
        })

print("\n===== WALLET RELATION SUMMARY =====")
print("Total transactions:", total_rows)
print("Wallet appears as From:", from_matches)
print("Wallet appears as To:", to_matches)
print("Wallet appears as both:", both_matches)
print("Wallet appears in neither:", neither_matches)

print("\n===== FIRST 10 WALLETS =====")
print(pd.DataFrame(examples).to_string(index=False))

print("\n===== PERCENTAGES =====")
print(f"From: {from_matches / total_rows * 100:.2f}%")
print(f"To: {to_matches / total_rows * 100:.2f}%")
print(f"Neither: {neither_matches / total_rows * 100:.2f}%")
