
"""
ETHEREUM GUARD
TV1-02: Transaction ingestion and cleaning.

- Reads transaction and block Parquet files in batches.
- Joins transactions with block timestamps.
- Normalizes addresses and transaction values.
- Writes canonical transaction Parquet files.
- Does not assign account labels.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "sample"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "transactions"
DEFAULT_REPORT = PROJECT_ROOT / "artifacts" / "results" / "ingestion_report.json"

BATCH_SIZE = 100_000
WEI_PER_ETH = 10**18

TX_PATTERN = "ethereum__transactions__*.parquet"
BLOCK_PATTERN = "ethereum__blocks__*.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ethereum_guard.ingest")


# ============================================================
# CANONICAL OUTPUT SCHEMA
# ============================================================

OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("transaction_hash", pa.string()),
        pa.field("block_number", pa.uint32()),
        pa.field("transaction_index", pa.uint64()),
        pa.field("timestamp", pa.timestamp("s", tz="UTC")),
        pa.field("from_address", pa.string()),
        pa.field("to_address", pa.string()),
        pa.field("value_wei", pa.string()),
        pa.field("value_eth", pa.float64()),
        pa.field("gas_used", pa.uint64()),
        pa.field("gas_price", pa.uint64()),
        pa.field("success", pa.bool_()),
        pa.field("chain_id", pa.uint64()),
    ]
)


# ============================================================
# NORMALIZATION
# ============================================================

def binary_to_hex(value: Any, field_name: str) -> str | None:
    """Convert binary Ethereum data into lowercase 0x-prefixed hex."""

    if value is None:
        return None

    if isinstance(value, memoryview):
        value = value.tobytes()

    if not isinstance(value, bytes):
        raise TypeError(
            f"{field_name} must be bytes, got {type(value).__name__}"
        )

    return "0x" + value.hex()


def normalize_value(
    value_string: str | None,
    value_binary: bytes | None,
) -> tuple[str | None, float | None, bool]:
    """
    Return:
        value_wei: exact decimal integer string
        value_eth: derived floating-point ETH
        is_consistent: binary/string consistency check
    """

    if value_string is None:
        return None, None, value_binary is None

    value_string = str(value_string).strip()

    if not value_string:
        return None, None, value_binary is None

    value_int = int(value_string)

    if value_int < 0:
        raise ValueError(f"Negative transaction value: {value_string}")

    consistent = True

    if value_binary is not None:
        if isinstance(value_binary, memoryview):
            value_binary = value_binary.tobytes()

        binary_int = int.from_bytes(value_binary, byteorder="big")
        consistent = binary_int == value_int

    value_eth = value_int / WEI_PER_ETH

    return value_string, value_eth, consistent


# ============================================================
# BLOCK TIMESTAMP INDEX
# ============================================================

def build_block_timestamp_index(
    input_dir: Path,
) -> dict[int, int]:
    """
    Build block_number -> timestamp index.

    Block files are expected to be much smaller than transaction
    files, so the timestamp index is loaded once.
    """

    block_files = sorted(input_dir.glob(BLOCK_PATTERN))

    if not block_files:
        raise FileNotFoundError(
            f"No block Parquet files found in: {input_dir}"
        )

    timestamp_index: dict[int, int] = {}

    logger.info("Building block timestamp index...")

    for file_path in block_files:
        logger.info("Reading block file: %s", file_path.name)

        table = pq.read_table(
            file_path,
            columns=["block_number", "timestamp"],
        )

        block_numbers = table["block_number"].to_pylist()
        timestamps = table["timestamp"].to_pylist()

        for block_number, timestamp in zip(
            block_numbers, timestamps
        ):
            if block_number is None or timestamp is None:
                continue

            block_number = int(block_number)
            timestamp = int(timestamp)

            previous = timestamp_index.get(block_number)

            if previous is not None and previous != timestamp:
                raise ValueError(
                    f"Conflicting timestamps for block {block_number}: "
                    f"{previous} vs {timestamp}"
                )

            timestamp_index[block_number] = timestamp

    logger.info(
        "Block timestamp index ready: %s unique blocks",
        len(timestamp_index),
    )

    return timestamp_index


# ============================================================
# TRANSACTION BATCH PROCESSING
# ============================================================

def process_transaction_file(
    file_path: Path,
    timestamp_index: dict[int, int],
    output_dir: Path,
) -> dict[str, Any]:

    output_path = output_dir / file_path.name

    columns = [
        "transaction_hash",
        "block_number",
        "transaction_index",
        "from_address",
        "to_address",
        "value_string",
        "value_binary",
        "gas_used",
        "gas_price",
        "success",
        "chain_id",
    ]

    parquet_file = pq.ParquetFile(file_path)
    available_columns = set(parquet_file.schema_arrow.names)

    missing = set(columns) - available_columns

    if missing:
        raise ValueError(
            f"{file_path.name} is missing required columns: "
            f"{sorted(missing)}"
        )

    stats = {
        "source_file": file_path.name,
        "output_file": output_path.name,
        "rows": 0,
        "null_to_address": 0,
        "missing_block_timestamp": 0,
        "value_mismatches": 0,
    }

    logger.info("Processing transaction file: %s", file_path.name)

    with pq.ParquetWriter(
        output_path,
        OUTPUT_SCHEMA,
        compression="zstd",
    ) as writer:

        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=columns,
        ):
            data = batch.to_pydict()

            output = {
                field.name: []
                for field in OUTPUT_SCHEMA
            }

            for i in range(batch.num_rows):
                block_number = int(data["block_number"][i])
                timestamp = timestamp_index.get(block_number)

                if timestamp is None:
                    stats["missing_block_timestamp"] += 1

                tx_hash = binary_to_hex(
                    data["transaction_hash"][i],
                    "transaction_hash",
                )

                from_address = binary_to_hex(
                    data["from_address"][i],
                    "from_address",
                )

                to_address = binary_to_hex(
                    data["to_address"][i],
                    "to_address",
                )

                if to_address is None:
                    stats["null_to_address"] += 1

                value_wei, value_eth, consistent = normalize_value(
                    data["value_string"][i],
                    data["value_binary"][i],
                )

                if not consistent:
                    stats["value_mismatches"] += 1

                output["transaction_hash"].append(tx_hash)
                output["block_number"].append(block_number)
                output["transaction_index"].append(
                    data["transaction_index"][i]
                )
                output["timestamp"].append(
                    None if timestamp is None else
                    timestamp
                )
                output["from_address"].append(from_address)
                output["to_address"].append(to_address)
                output["value_wei"].append(value_wei)
                output["value_eth"].append(value_eth)
                output["gas_used"].append(data["gas_used"][i])
                output["gas_price"].append(data["gas_price"][i])
                output["success"].append(data["success"][i])
                output["chain_id"].append(data["chain_id"][i])

            table = pa.Table.from_pydict(
                output,
                schema=OUTPUT_SCHEMA,
            )

            writer.write_table(table)
            stats["rows"] += batch.num_rows

    if stats["missing_block_timestamp"] > 0:
        output_path.unlink(missing_ok=True)
        raise ValueError(
            f"{file_path.name}: "
            f"{stats['missing_block_timestamp']} transactions "
            "have no matching block timestamp. "
            "Incomplete output was removed."
        )

    if stats["value_mismatches"] > 0:
        output_path.unlink(missing_ok=True)
        raise ValueError(
            f"{file_path.name}: "
            f"{stats['value_mismatches']} value mismatches. "
            "Incomplete output was removed."
        )

    logger.info(
        "Completed %s: %s rows",
        file_path.name,
        stats["rows"],
    )

    return stats


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_ingestion(
    input_dir: Path,
    output_dir: Path,
    report_path: Path,
) -> dict[str, Any]:

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    report_path = report_path.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    transaction_files = sorted(input_dir.glob(TX_PATTERN))

    if not transaction_files:
        raise FileNotFoundError(
            f"No transaction Parquet files found in: {input_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp_index = build_block_timestamp_index(input_dir)

    file_reports = []

    total_rows = 0
    total_null_to = 0

    for file_path in transaction_files:
        stats = process_transaction_file(
            file_path,
            timestamp_index,
            output_dir,
        )

        file_reports.append(stats)

        total_rows += stats["rows"]
        total_null_to += stats["null_to_address"]

    report = {
        "pipeline": "ETHEREUM GUARD TV1-02",
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "transaction_files": len(transaction_files),
        "indexed_blocks": len(timestamp_index),
        "total_transactions": total_rows,
        "null_to_address": total_null_to,
        "files": file_reports,
        "label_assignment": "Not performed during ingestion",
    }

    report_path.write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("=" * 60)
    logger.info("INGESTION COMPLETED")
    logger.info("Transaction files: %s", len(transaction_files))
    logger.info("Total transactions: %s", total_rows)
    logger.info("Null to_address: %s", total_null_to)
    logger.info("Report: %s", report_path)

    return report


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ethereum Guard transaction ingestion"
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing transaction and block Parquet files",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for canonical transaction Parquet files",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="JSON ingestion report path",
    )

    args = parser.parse_args()

    run_ingestion(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
