from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "data" / "sample"
MDBCHAIN_DIR = ROOT / "data" / "raw" / "mdbchain"
RESULTS_DIR = ROOT / "artifacts" / "results"

BATCH_SIZE = 100_000

TX_PREFIX = "ethereum__transactions__"
BLOCK_PREFIX = "ethereum__blocks__"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("mdbchain_overlap_audit")


# ============================================================
# HELPERS
# ============================================================

def normalize_address(value: Any) -> str | None:
    """Convert Parquet binary/string address values to lowercase 0x hex."""
    if value is None:
        return None

    if isinstance(value, memoryview):
        value = value.tobytes()

    if isinstance(value, (bytes, bytearray)):
        if len(value) == 0:
            return None
        return "0x" + bytes(value).hex()

    if isinstance(value, str):
        value = value.strip().lower()
        if not value:
            return None
        if value.startswith("0x"):
            return value
        return "0x" + value

    return None


def load_phishing_addresses() -> set[str]:
    """MDBChain labels are encoded in the CSV filenames."""
    if not MDBCHAIN_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục MDBChain: {MDBCHAIN_DIR}"
        )

    files = list(MDBCHAIN_DIR.rglob("*.csv"))
    addresses = {
        file.stem.strip().lower()
        for file in files
        if file.stem.strip().lower().startswith("0x")
    }

    if not addresses:
        raise RuntimeError(
            f"Không tìm thấy file CSV địa chỉ phishing trong {MDBCHAIN_DIR}"
        )

    return addresses


def find_matching_block_file(tx_path: Path) -> Path:
    """Find the block Parquet having the same block-range suffix."""
    if not tx_path.name.startswith(TX_PREFIX):
        raise ValueError(f"Tên transaction file không hợp lệ: {tx_path.name}")

    block_name = BLOCK_PREFIX + tx_path.name[len(TX_PREFIX):]
    return SAMPLE_DIR / block_name


def inspect_block_file(block_path: Path) -> dict[str, Any]:
    """Load block numbers and derive the UTC timestamp range."""
    table = pq.read_table(
        block_path,
        columns=["block_number", "timestamp"],
    )
    blocks = table.to_pandas()

    block_numbers = set(
        blocks["block_number"].dropna().astype("int64").tolist()
    )

    timestamps = pd.to_datetime(
        blocks["timestamp"],
        unit="s",
        utc=True,
        errors="coerce",
    ).dropna()

    return {
        "block_rows": int(len(blocks)),
        "unique_blocks": int(len(block_numbers)),
        "block_numbers": block_numbers,
        "timestamp_min": (
            timestamps.min().isoformat() if not timestamps.empty else None
        ),
        "timestamp_max": (
            timestamps.max().isoformat() if not timestamps.empty else None
        ),
    }


def audit_transaction_file(
    tx_path: Path,
    block_numbers: set[int] | None,
    phishing_addresses: set[str],
) -> dict[str, Any]:
    """Audit one transaction Parquet incrementally."""
    parquet = pq.ParquetFile(tx_path)

    required_columns = {"block_number", "from_address", "to_address"}
    available_columns = set(parquet.schema_arrow.names)
    missing = required_columns - available_columns
    if missing:
        raise ValueError(
            f"{tx_path.name} thiếu cột bắt buộc: {sorted(missing)}"
        )

    total = 0
    matched_blocks = 0
    null_senders = 0
    null_receivers = 0

    sender_rows = 0
    receiver_rows = 0
    both_rows = 0

    phishing_senders: set[str] = set()
    phishing_receivers: set[str] = set()

    for batch in parquet.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["block_number", "from_address", "to_address"],
    ):
        df = batch.to_pandas()
        total += len(df)

        if block_numbers is not None:
            matched_blocks += int(
                df["block_number"].isin(block_numbers).sum()
            )

        for sender_raw, receiver_raw in zip(
            df["from_address"],
            df["to_address"],
        ):
            sender = normalize_address(sender_raw)
            receiver = normalize_address(receiver_raw)

            if sender is None:
                null_senders += 1
            if receiver is None:
                null_receivers += 1

            sender_is_phishing = (
                sender is not None and sender in phishing_addresses
            )
            receiver_is_phishing = (
                receiver is not None and receiver in phishing_addresses
            )

            if sender_is_phishing:
                sender_rows += 1
                phishing_senders.add(sender)

            if receiver_is_phishing:
                receiver_rows += 1
                phishing_receivers.add(receiver)

            if sender_is_phishing and receiver_is_phishing:
                both_rows += 1

    involving_rows = sender_rows + receiver_rows - both_rows
    matched_addresses = phishing_senders | phishing_receivers

    return {
        "transaction_file": tx_path.name,
        "rows": int(total),
        "null_senders": int(null_senders),
        "null_receivers": int(null_receivers),
        "matched_block_rows": (
            int(matched_blocks) if block_numbers is not None else None
        ),
        "unmatched_block_rows": (
            int(total - matched_blocks) if block_numbers is not None else None
        ),
        "join_rate_percent": (
            round(matched_blocks / total * 100, 6)
            if block_numbers is not None and total
            else None
        ),
        "transactions_sent_by_phishing": int(sender_rows),
        "transactions_received_by_phishing": int(receiver_rows),
        "transactions_involving_both_phishing_endpoints": int(both_rows),
        "transactions_involving_phishing": int(involving_rows),
        "unique_phishing_senders": int(len(phishing_senders)),
        "unique_phishing_receivers": int(len(phishing_receivers)),
        "unique_phishing_addresses": int(len(matched_addresses)),
        "phishing_transaction_rate_percent": (
            round(involving_rows / total * 100, 6) if total else 0.0
        ),
        # Kept internally for exact overall unique-address aggregation.
        "_phishing_senders": phishing_senders,
        "_phishing_receivers": phishing_receivers,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tx_files = sorted(SAMPLE_DIR.glob(f"{TX_PREFIX}*.parquet"))
    if not tx_files:
        raise FileNotFoundError(
            f"Không tìm thấy {TX_PREFIX}*.parquet trong {SAMPLE_DIR}"
        )

    phishing_addresses = load_phishing_addresses()

    logger.info("Transaction Parquet files found: %s", len(tx_files))
    logger.info("MDBChain phishing addresses: %s", len(phishing_addresses))
    logger.info("Batch size: %s", BATCH_SIZE)

    file_reports: list[dict[str, Any]] = []

    total_transactions = 0
    total_matched_blocks = 0
    total_sender_rows = 0
    total_receiver_rows = 0
    total_both_rows = 0
    total_null_senders = 0
    total_null_receivers = 0

    all_phishing_senders: set[str] = set()
    all_phishing_receivers: set[str] = set()
    missing_block_files: list[str] = []

    for index, tx_path in enumerate(tx_files, start=1):
        logger.info(
            "[%s/%s] Auditing %s",
            index,
            len(tx_files),
            tx_path.name,
        )

        block_path = find_matching_block_file(tx_path)
        block_info = None

        if block_path.exists():
            block_info = inspect_block_file(block_path)
            block_numbers = block_info["block_numbers"]
        else:
            logger.warning("Missing matching block file: %s", block_path.name)
            missing_block_files.append(block_path.name)
            block_numbers = None

        report = audit_transaction_file(
            tx_path,
            block_numbers,
            phishing_addresses,
        )

        report["block_file"] = (
            block_path.name if block_info is not None else None
        )
        report["block_rows"] = (
            block_info["block_rows"] if block_info is not None else None
        )
        report["unique_blocks"] = (
            block_info["unique_blocks"] if block_info is not None else None
        )
        report["timestamp_min_utc"] = (
            block_info["timestamp_min"] if block_info is not None else None
        )
        report["timestamp_max_utc"] = (
            block_info["timestamp_max"] if block_info is not None else None
        )

        total_transactions += report["rows"]
        total_sender_rows += report["transactions_sent_by_phishing"]
        total_receiver_rows += report["transactions_received_by_phishing"]
        total_both_rows += report[
            "transactions_involving_both_phishing_endpoints"
        ]
        total_null_senders += report["null_senders"]
        total_null_receivers += report["null_receivers"]

        if report["matched_block_rows"] is not None:
            total_matched_blocks += report["matched_block_rows"]

        all_phishing_senders.update(report.pop("_phishing_senders"))
        all_phishing_receivers.update(report.pop("_phishing_receivers"))

        file_reports.append(report)

        logger.info(
            "Rows=%s | phishing_tx=%s | phishing_addresses=%s",
            report["rows"],
            report["transactions_involving_phishing"],
            report["unique_phishing_addresses"],
        )

    total_involving = (
        total_sender_rows + total_receiver_rows - total_both_rows
    )
    all_matched_addresses = all_phishing_senders | all_phishing_receivers
    files_with_blocks = len(tx_files) - len(missing_block_files)

    overall = {
        "transaction_files_found": len(tx_files),
        "transaction_files_with_matching_blocks": files_with_blocks,
        "transaction_files_missing_blocks": len(missing_block_files),
        "missing_block_files": missing_block_files,
        "mdbchain_phishing_addresses": len(phishing_addresses),
        "unique_phishing_senders": len(all_phishing_senders),
        "unique_phishing_receivers": len(all_phishing_receivers),
        "unique_phishing_addresses_found": len(all_matched_addresses),
        "phishing_address_coverage_percent": round(
            len(all_matched_addresses) / len(phishing_addresses) * 100, 6
        ),
        "total_transactions": total_transactions,
        "transactions_sent_by_phishing": total_sender_rows,
        "transactions_received_by_phishing": total_receiver_rows,
        "transactions_involving_both_phishing_endpoints": total_both_rows,
        "transactions_involving_phishing": total_involving,
        "phishing_transaction_rate_percent": round(
            total_involving / total_transactions * 100, 6
        ) if total_transactions else 0.0,
        "null_senders": total_null_senders,
        "null_receivers": total_null_receivers,
        "matched_block_rows": (
            total_matched_blocks if files_with_blocks else None
        ),
        "join_rate_percent": (
            round(
                total_matched_blocks / total_transactions * 100, 6
            )
            if files_with_blocks == len(tx_files) and total_transactions
            else None
        ),
    }

    # CSV: one row per transaction Parquet.
    csv_path = RESULTS_DIR / "mdbchain_overlap_by_file.csv"
    pd.DataFrame(file_reports).to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    # JSON: detailed per-file and overall report.
    json_path = RESULTS_DIR / "mdbchain_overlap_report.json"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_directory": str(SAMPLE_DIR),
        "mdbchain_directory": str(MDBCHAIN_DIR),
        "batch_size": BATCH_SIZE,
        "overall": overall,
        "files": file_reports,
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print("MDBCHAIN OVERLAP AUDIT - COMPLETE")
    print("=" * 64)
    print("Transaction files:", overall["transaction_files_found"])
    print(
        "Files with matching block files:",
        overall["transaction_files_with_matching_blocks"],
    )
    print(
        "Files missing block files:",
        overall["transaction_files_missing_blocks"],
    )
    print("MDBChain phishing addresses:", overall["mdbchain_phishing_addresses"])
    print("Total transactions:", overall["total_transactions"])
    print(
        "Transactions involving phishing:",
        overall["transactions_involving_phishing"],
    )
    print(
        "Phishing transaction rate:",
        f'{overall["phishing_transaction_rate_percent"]:.6f}%',
    )
    print(
        "Unique phishing addresses found:",
        overall["unique_phishing_addresses_found"],
    )
    print(
        "MDBChain address coverage:",
        f'{overall["phishing_address_coverage_percent"]:.6f}%',
    )
    print("Null senders:", overall["null_senders"])
    print("Null receivers:", overall["null_receivers"])
    print("Join rate:", overall["join_rate_percent"])
    print("\nReports saved:")
    print("CSV :", csv_path)
    print("JSON:", json_path)

    if missing_block_files:
        print("\nWARNING: Some transaction files have no matching block file.")
        for name in missing_block_files:
            print(" -", name)


if __name__ == "__main__":
    main()
