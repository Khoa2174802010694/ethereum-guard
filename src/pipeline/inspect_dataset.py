
from pathlib import Path
import json

import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
RESULT_DIR = PROJECT_ROOT / "artifacts" / "results"

TRANSACTION_PATTERN = "ethereum__transactions__*.parquet"
BLOCK_PATTERN = "ethereum__blocks__*.parquet"

REQUIRED_TX_COLUMNS = {
    "block_number",
    "transaction_index",
    "transaction_hash",
    "from_address",
    "to_address",
    "value_f64",
    "gas_used",
    "gas_price",
    "success",
}

REQUIRED_BLOCK_COLUMNS = {
    "block_number",
    "timestamp",
}


# ============================================================
# HELPERS
# ============================================================

def find_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"Không tìm thấy file theo mẫu: {pattern}"
        )

    if len(files) > 1:
        print(
            f"Cảnh báo: tìm thấy {len(files)} file. "
            f"Đang kiểm tra file đầu tiên: {files[0].name}"
        )

    return files[0]


def inspect_parquet(path: Path) -> dict:
    parquet = pq.ParquetFile(path)

    print(f"\nFILE: {path.name}")
    print(f"ROWS: {parquet.metadata.num_rows}")
    print(f"ROW GROUPS: {parquet.num_row_groups}")
    print("SCHEMA:")
    print(parquet.schema)

    return {
        "path": path,
        "parquet": parquet,
        "rows": parquet.metadata.num_rows,
        "columns": parquet.schema.names,
    }


# ============================================================
# TRANSACTION AUDIT
# ============================================================

def audit_transactions(info: dict) -> dict:
    parquet = info["parquet"]
    columns = set(info["columns"])

    missing_columns = sorted(
        REQUIRED_TX_COLUMNS - columns
    )

    print("\n===== TRANSACTION AUDIT =====")
    print("Total rows:", info["rows"])
    print("Missing required columns:", missing_columns)

    required_to_check = [
        col for col in REQUIRED_TX_COLUMNS
        if col in columns
    ]

    total_nulls = {col: 0 for col in required_to_check}
    unique_hashes = set()
    duplicate_hashes = 0
    unique_senders = set()
    unique_receivers = set()

    success_counts = {}
    block_min = None
    block_max = None
    value_min = None
    value_max = None

    # Đọc từng batch, không nạp toàn bộ Parquet vào RAM.
    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=required_to_check,
    ):
        df = batch.to_pandas()

        for col in required_to_check:
            total_nulls[col] += int(df[col].isna().sum())

        if "transaction_hash" in df:
            hashes = df["transaction_hash"].dropna()

            for tx_hash in hashes:
                if tx_hash in unique_hashes:
                    duplicate_hashes += 1
                else:
                    unique_hashes.add(tx_hash)

        if "from_address" in df:
            unique_senders.update(
                df["from_address"].dropna().unique()
            )

        if "to_address" in df:
            unique_receivers.update(
                df["to_address"].dropna().unique()
            )

        if "success" in df:
            counts = df["success"].value_counts(dropna=False)
            for key, value in counts.items():
                success_counts[str(key)] = (
                    success_counts.get(str(key), 0)
                    + int(value)
                )

        if "block_number" in df:
            current_min = df["block_number"].min()
            current_max = df["block_number"].max()

            block_min = (
                current_min if block_min is None
                else min(block_min, current_min)
            )
            block_max = (
                current_max if block_max is None
                else max(block_max, current_max)
            )

        if "value_f64" in df:
            values = pd.to_numeric(
                df["value_f64"], errors="coerce"
            ).dropna()

            if not values.empty:
                current_min = values.min()
                current_max = values.max()

                value_min = (
                    current_min if value_min is None
                    else min(value_min, current_min)
                )
                value_max = (
                    current_max if value_max is None
                    else max(value_max, current_max)
                )

    result = {
        "file": info["path"].name,
        "rows": info["rows"],
        "missing_columns": missing_columns,
        "null_counts": total_nulls,
        "unique_transaction_hashes": len(unique_hashes),
        "duplicate_hashes": duplicate_hashes,
        "unique_senders": len(unique_senders),
        "unique_receivers": len(unique_receivers),
        "success_distribution": success_counts,
        "block_min": block_min,
        "block_max": block_max,
        "value_min": value_min,
        "value_max": value_max,
    }

    print("Null counts:", total_nulls)
    print("Unique transaction hashes:", len(unique_hashes))
    print("Duplicate hashes:", duplicate_hashes)
    print("Unique senders:", len(unique_senders))
    print("Unique receivers:", len(unique_receivers))
    print("Success distribution:", success_counts)
    print("Block range:", block_min, "->", block_max)
    print("Value range:", value_min, "->", value_max)

    return result


# ============================================================
# BLOCK AUDIT
# ============================================================

def audit_blocks(info: dict) -> dict:
    parquet = info["parquet"]
    columns = set(info["columns"])

    missing_columns = sorted(
        REQUIRED_BLOCK_COLUMNS - columns
    )

    print("\n===== BLOCK AUDIT =====")
    print("Total rows:", info["rows"])
    print("Missing required columns:", missing_columns)

    if missing_columns:
        return {
            "file": info["path"].name,
            "rows": info["rows"],
            "missing_columns": missing_columns,
        }

    block_numbers = set()
    duplicate_blocks = 0
    timestamps = []
    block_intervals = {}

    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=["block_number", "timestamp"],
    ):
        df = batch.to_pandas()

        for number in df["block_number"].dropna():
            if number in block_numbers:
                duplicate_blocks += 1
            else:
                block_numbers.add(number)

        timestamps.extend(
            df["timestamp"].dropna().tolist()
        )

    sorted_blocks = sorted(block_numbers)

    if len(sorted_blocks) > 1:
        differences = pd.Series(sorted_blocks).diff()
        block_intervals = {
            str(int(key)): int(value)
            for key, value in differences.value_counts().items()
        }

    result = {
        "file": info["path"].name,
        "rows": info["rows"],
        "unique_blocks": len(block_numbers),
        "duplicate_blocks": duplicate_blocks,
        "block_min": min(block_numbers) if block_numbers else None,
        "block_max": max(block_numbers) if block_numbers else None,
        "timestamp_min": min(timestamps) if timestamps else None,
        "timestamp_max": max(timestamps) if timestamps else None,
        "block_intervals": block_intervals,
    }

    print("Unique blocks:", result["unique_blocks"])
    print("Duplicate blocks:", duplicate_blocks)
    print("Block range:", result["block_min"], "->", result["block_max"])
    print("Timestamp range:", result["timestamp_min"], "->", result["timestamp_max"])
    print("Block intervals:", block_intervals)

    return result


# ============================================================
# JOIN AUDIT
# ============================================================

def audit_join(tx_info: dict, block_info: dict) -> dict:
    print("\n===== TRANSACTION-BLOCK JOIN AUDIT =====")

    block_numbers = set()

    for batch in block_info["parquet"].iter_batches(
        batch_size=100_000,
        columns=["block_number"],
    ):
        block_numbers.update(
            batch.column(0).to_pylist()
        )

    total_transactions = 0
    matched_transactions = 0
    transaction_blocks = set()

    for batch in tx_info["parquet"].iter_batches(
        batch_size=100_000,
        columns=["block_number"],
    ):
        numbers = batch.column(0).to_pylist()
        total_transactions += len(numbers)

        for number in numbers:
            if number is None:
                continue

            transaction_blocks.add(number)

            if number in block_numbers:
                matched_transactions += 1

    unmatched_transactions = (
        total_transactions - matched_transactions
    )

    match_rate = (
        matched_transactions / total_transactions * 100
        if total_transactions else 0
    )

    result = {
        "total_transactions": total_transactions,
        "matched_transactions": matched_transactions,
        "unmatched_transactions": unmatched_transactions,
        "match_rate_percent": round(match_rate, 4),
        "unique_transaction_blocks": len(transaction_blocks),
        "unique_block_records": len(block_numbers),
    }

    print("Total transactions:", total_transactions)
    print("Matched:", matched_transactions)
    print("Unmatched:", unmatched_transactions)
    print("Match rate:", result["match_rate_percent"], "%")
    print("Unique transaction blocks:", len(transaction_blocks))
    print("Unique block records:", len(block_numbers))

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("ETHEREUM GUARD - VNEGI10 DATA AUDIT")
    print("=" * 60)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    tx_path = find_file(SAMPLE_DIR, TRANSACTION_PATTERN)
    block_path = find_file(SAMPLE_DIR, BLOCK_PATTERN)

    tx_info = inspect_parquet(tx_path)
    block_info = inspect_parquet(block_path)

    tx_result = audit_transactions(tx_info)
    block_result = audit_blocks(block_info)
    join_result = audit_join(tx_info, block_info)

    report = {
        "dataset": "vnegi10 Ethereum Blockchain Parquet",
        "transactions": tx_result,
        "blocks": block_result,
        "join": join_result,
    }

    output_path = RESULT_DIR / "vnegi10_sample_audit.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 60)
    print("AUDIT COMPLETED")
    print("Report saved to:", output_path)
    print("=" * 60)


if __name__ == "__main__":
    main()
