"""
Build an address-level label registry from canonical Ethereum transactions.

Input:
  data/processed/transactions/*.parquet
  data/raw/mdbchain/**/*.csv  (CSV filenames are known phishing addresses)

Output:
  data/processed/address_registry/address_labels.parquet
  artifacts/results/address_registry_report.json

Label policy:
  1  = address is in the MDBChain known-phishing address list
 -1  = address has not been verified as benign or phishing

IMPORTANT:
  Do not interpret -1 as benign. Absence from the phishing list is not proof
  that an address is normal.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSACTIONS_DIR = PROJECT_ROOT / "data" / "processed" / "transactions"
MDBCHAIN_DIR = PROJECT_ROOT / "data" / "raw" / "mdbchain"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "address_registry"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "results" / "address_registry_report.json"

BATCH_SIZE = 100_000
LABEL_NORMAL = 0
LABEL_PHISHING = 1
LABEL_UNLABELED = -1
# Optional trusted negative-label source. One address per line; blank lines and
# lines beginning with # are ignored. Absence of this file means no label 0.
VERIFIED_BENIGN_FILE = PROJECT_ROOT / "data" / "raw" / "verified_benign_addresses.txt"


def normalize_address(value) -> str | None:
    """Normalize an address to lowercase 0x-prefixed hexadecimal text."""
    if value is None:
        return None

    if isinstance(value, bytes):
        value = "0x" + value.hex()

    value = str(value).strip().lower()
    if not value or value in {"none", "nan", "null"}:
        return None

    if not value.startswith("0x"):
        value = "0x" + value

    return value


def to_epoch_seconds(value) -> int | None:
    """Accept Arrow/Python datetime values or numeric Unix timestamps."""
    if value is None:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return int(value.timestamp())

    # pandas/numpy scalar values, if encountered
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return int(value.timestamp())

    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def load_known_phishing_addresses() -> set[str]:
    if not MDBCHAIN_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục MDBChain: {MDBCHAIN_DIR}"
        )

    csv_files = list(MDBCHAIN_DIR.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Không tìm thấy CSV nào trong: {MDBCHAIN_DIR}"
        )

    addresses = set()
    for csv_file in csv_files:
        address = normalize_address(csv_file.stem)
        if address:
            addresses.add(address)

    return addresses


def load_verified_benign_addresses() -> set[str]:
    """Load only explicitly verified benign addresses; never infer from absence."""
    if not VERIFIED_BENIGN_FILE.exists():
        return set()

    addresses: set[str] = set()
    for line_number, raw in enumerate(
        VERIFIED_BENIGN_FILE.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        address = normalize_address(value)
        if address is None:
            raise ValueError(
                f"Địa chỉ normal không hợp lệ tại "
                f"{VERIFIED_BENIGN_FILE}:{line_number}"
            )
        addresses.add(address)
    return addresses


def main() -> None:
    parquet_files = sorted(TRANSACTIONS_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"Không tìm thấy Parquet trong: {TRANSACTIONS_DIR}"
        )

    known_phishing = load_known_phishing_addresses()
    verified_benign = load_verified_benign_addresses()
    conflicts = known_phishing & verified_benign
    if conflicts:
        examples = sorted(conflicts)[:5]
        raise ValueError(
            "Có địa chỉ đồng thời nằm trong danh sách phishing và normal "
            f"đã xác minh. Cần giải quyết xung đột trước: {examples}"
        )

    # Per-address aggregates. Label 0 is assigned only from the explicit,
    # trusted verified-benign file. Missing from MDBChain is NOT enough.
    stats = defaultdict(
        lambda: {
            "tx_sent_count": 0,
            "tx_received_count": 0,
            "first_seen_ts": None,
            "last_seen_ts": None,
        }
    )

    total_transactions = 0
    transactions_involving_known_phishing = 0
    unique_transaction_hashes_not_checked = True
    files_processed = []

    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        required_columns = {
            "from_address",
            "to_address",
            "timestamp",
        }
        available_columns = set(parquet_file.schema_arrow.names)
        missing = required_columns - available_columns
        if missing:
            raise ValueError(
                f"{parquet_path.name} thiếu cột bắt buộc: {sorted(missing)}"
            )

        file_rows = 0
        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=["from_address", "to_address", "timestamp"],
        ):
            columns = batch.to_pydict()
            senders = columns["from_address"]
            receivers = columns["to_address"]
            timestamps = columns["timestamp"]

            for sender_raw, receiver_raw, timestamp_raw in zip(
                senders, receivers, timestamps
            ):
                sender = normalize_address(sender_raw)
                receiver = normalize_address(receiver_raw)
                timestamp = to_epoch_seconds(timestamp_raw)

                total_transactions += 1
                file_rows += 1

                if (
                    (sender is not None and sender in known_phishing)
                    or (receiver is not None and receiver in known_phishing)
                ):
                    transactions_involving_known_phishing += 1

                if sender is not None:
                    item = stats[sender]
                    item["tx_sent_count"] += 1
                    if timestamp is not None:
                        first = item["first_seen_ts"]
                        last = item["last_seen_ts"]
                        item["first_seen_ts"] = (
                            timestamp if first is None else min(first, timestamp)
                        )
                        item["last_seen_ts"] = (
                            timestamp if last is None else max(last, timestamp)
                        )

                if receiver is not None:
                    item = stats[receiver]
                    item["tx_received_count"] += 1
                    if timestamp is not None:
                        first = item["first_seen_ts"]
                        last = item["last_seen_ts"]
                        item["first_seen_ts"] = (
                            timestamp if first is None else min(first, timestamp)
                        )
                        item["last_seen_ts"] = (
                            timestamp if last is None else max(last, timestamp)
                        )

        files_processed.append(
            {"file": parquet_path.name, "rows": file_rows}
        )
        print(f"[OK] {parquet_path.name}: {file_rows:,} rows")

    rows = []
    for address, item in stats.items():
        is_phishing = address in known_phishing
        is_verified_benign = address in verified_benign
        label = (
            LABEL_PHISHING if is_phishing
            else LABEL_NORMAL if is_verified_benign
            else LABEL_UNLABELED
        )
        label_source = (
            "MDBChain_filename" if is_phishing
            else "verified_benign_addresses.txt" if is_verified_benign
            else None
        )
        rows.append(
            {
                "address": address,
                "label": label,
                "label_source": label_source,
                "tx_sent_count": item["tx_sent_count"],
                "tx_received_count": item["tx_received_count"],
                "total_tx_count": (
                    item["tx_sent_count"] + item["tx_received_count"]
                ),
                "first_seen_ts": item["first_seen_ts"],
                "last_seen_ts": item["last_seen_ts"],
            }
        )

    rows.sort(key=lambda row: row["address"])

    schema = pa.schema(
        [
            pa.field("address", pa.string(), nullable=False),
            pa.field("label", pa.int8(), nullable=False),
            pa.field("label_source", pa.string(), nullable=True),
            pa.field("tx_sent_count", pa.int64(), nullable=False),
            pa.field("tx_received_count", pa.int64(), nullable=False),
            pa.field("total_tx_count", pa.int64(), nullable=False),
            pa.field("first_seen_ts", pa.int64(), nullable=True),
            pa.field("last_seen_ts", pa.int64(), nullable=True),
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(rows, schema=schema)
    output_path = OUTPUT_DIR / "address_labels.parquet"
    pq.write_table(table, output_path, compression="zstd")

    phishing_in_sample = sum(
        1 for address in stats if address in known_phishing
    )
    normal_in_sample = sum(
        1 for address in stats if address in verified_benign
    )
    unlabeled_in_sample = len(stats) - phishing_in_sample - normal_in_sample

    report = {
        "status": "success",
        "label_policy": {
            "1": "Known phishing address found in MDBChain filename list",
            "0": "Address explicitly listed in verified_benign_addresses.txt",
            "-1": "Unlabeled; not verified benign or phishing",
        },
        "inputs": {
            "transaction_files": files_processed,
            "mdbchain_directory": str(MDBCHAIN_DIR),
            "known_phishing_addresses_in_source": len(known_phishing),
            "verified_benign_file": (
                str(VERIFIED_BENIGN_FILE) if VERIFIED_BENIGN_FILE.exists() else None
            ),
            "verified_benign_addresses_in_source": len(verified_benign),
        },
        "summary": {
            "total_transactions_scanned": total_transactions,
            "unique_addresses_in_sample": len(stats),
            "known_phishing_addresses_found_in_sample": phishing_in_sample,
            "verified_benign_addresses_found_in_sample": normal_in_sample,
            "unlabeled_addresses_in_sample": unlabeled_in_sample,
            "binary_metrics_ready": bool(phishing_in_sample > 0 and normal_in_sample > 0),
            "transactions_involving_known_phishing_addresses": (
                transactions_involving_known_phishing
            ),
            "known_phishing_addresses_not_seen_in_sample": (
                len(known_phishing) - phishing_in_sample
            ),
        },
        "output": str(output_path),
        "note": (
            "The registry is limited to addresses observed in the processed "
            "transaction samples. Label 0 is assigned only from the explicit "
            "verified-benign file. Label -1 means unlabeled, not benign. "
            "The transaction count involving known phishing addresses is "
            "not a count of independently verified phishing transactions."
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== ADDRESS REGISTRY SUMMARY ===")
    print(f"Transactions scanned: {total_transactions:,}")
    print(f"Unique addresses: {len(stats):,}")
    print(f"Known phishing found: {phishing_in_sample:,}")
    print(f"Unlabeled: {unlabeled_in_sample:,}")
    print(
        "Transactions involving known phishing addresses: "
        f"{transactions_involving_known_phishing:,}"
    )
    print(f"\nParquet output: {output_path}")
    print(f"JSON report:    {REPORT_PATH}")


if __name__ == "__main__":
    main()
