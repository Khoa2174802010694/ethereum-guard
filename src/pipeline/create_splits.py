"""
ETHEREUM GUARD - TV1-03
Temporal split (70/15/15) + explicit inductive address holdout.

Outputs:
  data/processed/splits/train_transactions.parquet
      Raw chronological train-period partition. Kept for audit/reproducibility.
  data/processed/splits/val_transactions.parquet
  data/processed/splits/test_transactions.parquet
  data/processed/splits/train_graph_transactions.parquet
      Train-period transactions with every edge touching a held-out address
      removed. THIS is the transaction file to use for the train graph.
  data/processed/splits/address_splits.parquet
      Address-level split/holdout manifest.
  artifacts/results/split_report.json

Important:
- 70/15/15 is applied to transaction rows using global timestamp quantiles.
- The three period files are disjoint chronological partitions.
- `train_graph_transactions.parquet` is a filtered derivative of the raw
  train-period partition. It excludes edges where sender OR receiver is in
  the deterministic 10% inductive holdout set.
- Validation/test period files remain unfiltered. Later snapshots are built
  cumulatively up to their own cutoff, so held-out addresses can be evaluated
  when they appear in later periods.
- `is_inductive_test` means a held-out address is active in the test period.
  `is_new_after_train_cutoff` separately records addresses whose first
  observed sample timestamp is after T_train.
- Labels remain account-level: 1 = known phishing, -1 = unlabeled. No
  unlabeled address is silently converted to normal (0).
- The 10% holdout is sampled from all unique addresses in the available
  sample using the configured random seed. It is an experimental node
  holdout, not a claim that these addresses first appeared later on-chain.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

TX_COLUMNS = [
    "transaction_hash",
    "block_number",
    "transaction_index",
    "timestamp",
    "from_address",
    "to_address",
    "value_wei",
    "value_eth",
    "gas_used",
    "gas_price",
    "success",
    "chain_id",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ethereum_guard.create_splits")


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    project = config.get("project", {})
    data = config.get("data", {})
    split = config.get("split", {})
    pipeline = config.get("pipeline", {})
    output = config.get("output", {})

    cfg = {
        "seed": int(project.get("random_seed", project.get("seed", 42))),
        "processed_dir": data.get("processed_dir", "data/processed"),
        "train_ratio": float(split.get("train_ratio", 0.70)),
        "val_ratio": float(split.get("val_ratio", 0.15)),
        "test_ratio": float(split.get("test_ratio", 0.15)),
        "inductive_holdout_ratio": float(split.get("inductive_holdout_ratio", 0.10)),
        "batch_size": int(pipeline.get("batch_size", 100_000)),
        "results_dir": output.get("results_dir", "artifacts/results"),
    }

    ratios = (
        cfg["train_ratio"],
        cfg["val_ratio"],
        cfg["test_ratio"],
    )
    if any(value <= 0 or value >= 1 for value in ratios):
        raise ValueError(f"Các split ratio phải nằm trong (0, 1): {ratios}")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Train/Val/Test ratios phải có tổng bằng 1: {ratios}")
    if not 0.0 <= cfg["inductive_holdout_ratio"] < 1.0:
        raise ValueError("inductive_holdout_ratio phải nằm trong [0, 1).")
    if cfg["batch_size"] <= 0:
        raise ValueError("batch_size phải lớn hơn 0.")

    return cfg


def to_epoch_seconds(value: Any) -> int:
    if hasattr(value, "timestamp"):
        return int(value.timestamp())
    return int(value)


def iso_utc(timestamp: int) -> str:
    return datetime.fromtimestamp(
        int(timestamp), timezone.utc
    ).isoformat()


def load_timestamps(files: list[Path], batch_size: int) -> np.ndarray:
    """Read only timestamp columns and sort globally."""
    chunks: list[np.ndarray] = []
    total = 0

    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=["timestamp"],
        ):
            values = batch.column(0).to_pylist()
            values = [
                to_epoch_seconds(value)
                for value in values
                if value is not None
            ]
            if values:
                arr = np.asarray(values, dtype=np.int64)
                chunks.append(arr)
                total += len(arr)
        logger.info("[SCAN] %s", path.name)

    if not chunks:
        raise ValueError("Không tìm thấy timestamp hợp lệ.")

    timestamps = np.concatenate(chunks)
    if len(timestamps) != total:
        raise RuntimeError("Timestamp count mismatch.")

    timestamps.sort()
    return timestamps


def write_period_partitions(
    files: list[Path],
    output_dir: Path,
    train_cutoff: int,
    val_cutoff: int,
    test_cutoff: int,
    batch_size: int,
) -> dict[str, Any]:
    """Write disjoint chronological transaction partitions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "train": output_dir / "train_transactions.parquet",
        "val": output_dir / "val_transactions.parquet",
        "test": output_dir / "test_transactions.parquet",
    }

    source_schema = pq.ParquetFile(files[0]).schema_arrow
    missing_columns = set(TX_COLUMNS) - set(source_schema.names)
    if missing_columns:
        raise ValueError(
            f"Canonical transaction columns missing: {sorted(missing_columns)}"
        )
    output_schema = pa.schema(
        [source_schema.field(name) for name in TX_COLUMNS]
    )

    writers = {}
    for split, path in output_paths.items():
        path.unlink(missing_ok=True)
        writers[split] = pq.ParquetWriter(
            path,
            output_schema,
            compression="zstd",
        )

    counts: dict[str, int] = defaultdict(int)

    try:
        for path in files:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=TX_COLUMNS,
            ):
                table = pa.Table.from_batches([batch])
                ts = table["timestamp"]

                train_scalar = pa.scalar(
                    datetime.fromtimestamp(train_cutoff, timezone.utc),
                    type=ts.type,
                )
                val_scalar = pa.scalar(
                    datetime.fromtimestamp(val_cutoff, timezone.utc),
                    type=ts.type,
                )
                test_scalar = pa.scalar(
                    datetime.fromtimestamp(test_cutoff, timezone.utc),
                    type=ts.type,
                )

                masks = {
                    "train": pc.less_equal(ts, train_scalar),
                    "val": pc.and_(
                        pc.greater(ts, train_scalar),
                        pc.less_equal(ts, val_scalar),
                    ),
                    "test": pc.and_(
                        pc.greater(ts, val_scalar),
                        pc.less_equal(ts, test_scalar),
                    ),
                }

                for split, mask in masks.items():
                    part = table.filter(mask)
                    if part.num_rows:
                        writers[split].write_table(part)
                        counts[split] += part.num_rows

            logger.info("[WRITE] %s", path.name)
    finally:
        for writer in writers.values():
            writer.close()

    return {
        split: {
            "path": str(path.relative_to(ROOT)),
            "rows": int(counts[split]),
        }
        for split, path in output_paths.items()
    }


def build_address_splits(
    files: list[Path],
    registry_path: Path,
    output_dir: Path,
    train_cutoff: int,
    val_cutoff: int,
    test_cutoff: int,
    holdout_ratio: float,
    seed: int,
    batch_size: int,
) -> tuple[pa.Table, dict[str, Any], set[str]]:
    """
    Build address-level temporal activity and deterministic holdout manifest.

    Holdout selection is performed over unique addresses in the full available
    sample. The selected addresses are excluded from the train graph later.
    """
    first_seen: dict[str, int] = {}
    last_seen: dict[str, int] = {}
    active_periods: dict[str, set[str]] = defaultdict(set)

    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=["timestamp", "from_address", "to_address"],
        ):
            data = batch.to_pydict()

            for ts_raw, sender, receiver in zip(
                data["timestamp"],
                data["from_address"],
                data["to_address"],
            ):
                if ts_raw is None:
                    continue

                ts = to_epoch_seconds(ts_raw)
                if ts <= train_cutoff:
                    period = "train"
                elif ts <= val_cutoff:
                    period = "val"
                elif ts <= test_cutoff:
                    period = "test"
                else:
                    continue

                for address in (sender, receiver):
                    if not address:
                        continue

                    old_first = first_seen.get(address)
                    old_last = last_seen.get(address)

                    if old_first is None or ts < old_first:
                        first_seen[address] = ts
                    if old_last is None or ts > old_last:
                        last_seen[address] = ts

                    active_periods[address].add(period)

        logger.info("[NODES] %s", path.name)

    if not registry_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy Address Registry: {registry_path}"
        )

    registry = pq.read_table(
        registry_path,
        columns=["address", "label"],
    )
    positive_mask = pc.equal(registry["label"], 1)
    known_phishing_addresses = set(
        registry.filter(positive_mask)["address"].to_pylist()
    )

    addresses = sorted(first_seen)
    rng = np.random.default_rng(seed)
    holdout_count = (
        int(np.ceil(len(addresses) * holdout_ratio))
        if holdout_ratio > 0 and addresses
        else 0
    )
    if holdout_count >= len(addresses) and addresses:
        holdout_count = max(0, len(addresses) - 1)

    if holdout_count:
        selected_indices = rng.choice(
            len(addresses),
            size=holdout_count,
            replace=False,
        )
        holdout_addresses = {
            addresses[int(index)] for index in selected_indices
        }
    else:
        holdout_addresses = set()

    records = []
    for address in addresses:
        periods = active_periods[address]
        seen_train = "train" in periods
        seen_val = "val" in periods
        seen_test = "test" in periods
        is_holdout = address in holdout_addresses
        first_ts = first_seen[address]

        records.append(
            {
                "address": address,
                "label": (
                    1 if address in known_phishing_addresses else -1
                ),
                "label_source": (
                    "MDBChain_filename"
                    if address in known_phishing_addresses
                    else None
                ),
                "first_seen_ts": int(first_ts),
                "last_seen_ts": int(last_seen[address]),
                "seen_in_train_period": seen_train,
                "seen_in_val_period": seen_val,
                "seen_in_test_period": seen_test,
                "is_inductive_holdout": is_holdout,
                "seen_in_train_graph": bool(seen_train and not is_holdout),
                "is_new_after_train_cutoff": bool(
                    seen_test and first_ts > train_cutoff
                ),
                "is_new_in_test_period": bool(
                    seen_test and first_ts > val_cutoff
                ),
                "is_inductive_test": bool(is_holdout and seen_test),
            }
        )

    schema = pa.schema(
        [
            pa.field("address", pa.string(), nullable=False),
            pa.field("label", pa.int8(), nullable=False),
            pa.field("label_source", pa.string()),
            pa.field("first_seen_ts", pa.int64(), nullable=False),
            pa.field("last_seen_ts", pa.int64(), nullable=False),
            pa.field("seen_in_train_period", pa.bool_(), nullable=False),
            pa.field("seen_in_val_period", pa.bool_(), nullable=False),
            pa.field("seen_in_test_period", pa.bool_(), nullable=False),
            pa.field("is_inductive_holdout", pa.bool_(), nullable=False),
            pa.field("seen_in_train_graph", pa.bool_(), nullable=False),
            pa.field("is_new_after_train_cutoff", pa.bool_(), nullable=False),
            pa.field("is_new_in_test_period", pa.bool_(), nullable=False),
            pa.field("is_inductive_test", pa.bool_(), nullable=False),
        ]
    )

    table = pa.Table.from_pylist(records, schema=schema)
    output_path = output_dir / "address_splits.parquet"
    pq.write_table(table, output_path, compression="zstd")

    summary = {
        "unique_addresses": len(records),
        "inductive_holdout_ratio_requested": holdout_ratio,
        "inductive_holdout_addresses": len(holdout_addresses),
        "inductive_holdout_ratio_actual": (
            len(holdout_addresses) / len(records) if records else 0.0
        ),
        "inductive_holdout_active_in_test": sum(
            row["is_inductive_test"] for row in records
        ),
        "inductive_holdout_active_in_train_period": sum(
            row["is_inductive_holdout"] and row["seen_in_train_period"]
            for row in records
        ),
        "labels": {
            "known_phishing": sum(row["label"] == 1 for row in records),
            "verified_normal": sum(row["label"] == 0 for row in records),
            "unlabeled": sum(row["label"] == -1 for row in records),
        },
        "active_addresses_by_period": {
            split: sum(split in active_periods[address] for address in active_periods)
            for split in ("train", "val", "test")
        },
        "known_phishing_by_period": {
            split: sum(
                split in active_periods[address]
                and address in known_phishing_addresses
                for address in active_periods
            )
            for split in ("train", "val", "test")
        },
        "note": (
            "The holdout set is a deterministic random sample of unique "
            "addresses from the available sample. Every train-period edge "
            "touching one of these addresses is removed from the train graph. "
            "is_inductive_test marks held-out addresses active in the test "
            "period. is_new_after_train_cutoff is a separate temporal property."
        ),
    }

    return table, summary, holdout_addresses


def write_train_graph_transactions(
    train_path: Path,
    output_path: Path,
    holdout_addresses: set[str],
    batch_size: int,
) -> dict[str, int | str]:
    """
    Create a train-graph-only transaction file, excluding any edge incident
    to a held-out address. The raw train period partition is not modified.
    """
    parquet = pq.ParquetFile(train_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    rows_in = 0
    rows_out = 0
    writer = pq.ParquetWriter(
        output_path,
        parquet.schema_arrow,
        compression="zstd",
    )

    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            rows_in += table.num_rows

            if not holdout_addresses:
                filtered = table
            else:
                sender_is_holdout = pc.is_in(
                    table["from_address"],
                    value_set=pa.array(sorted(holdout_addresses), type=pa.string()),
                )
                receiver_is_holdout = pc.is_in(
                    table["to_address"],
                    value_set=pa.array(sorted(holdout_addresses), type=pa.string()),
                )
                incident = pc.or_(
                    pc.fill_null(sender_is_holdout, False),
                    pc.fill_null(receiver_is_holdout, False),
                )
                filtered = table.filter(pc.invert(incident))

            if filtered.num_rows:
                writer.write_table(filtered)
                rows_out += filtered.num_rows
    finally:
        writer.close()

    return {
        "path": str(output_path.relative_to(ROOT)),
        "input_rows": rows_in,
        "retained_rows": rows_out,
        "removed_rows": rows_in - rows_out,
        "holdout_addresses": len(holdout_addresses),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ethereum Guard temporal + inductive split"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to configs/default.yaml",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg = load_config(config_path)

    processed_dir = ROOT / cfg["processed_dir"]
    tx_dir = processed_dir / "transactions"
    registry_path = processed_dir / "address_registry" / "address_labels.parquet"
    output_dir = processed_dir / "splits"
    report_path = ROOT / cfg["results_dir"] / "split_report.json"

    files = sorted(tx_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy Parquet trong {tx_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Config: %s", config_path)
    logger.info(
        "Ratios train/val/test = %.2f/%.2f/%.2f | holdout = %.2f | seed = %d",
        cfg["train_ratio"],
        cfg["val_ratio"],
        cfg["test_ratio"],
        cfg["inductive_holdout_ratio"],
        cfg["seed"],
    )

    timestamps = load_timestamps(files, cfg["batch_size"])
    n = len(timestamps)

    train_idx = int(np.ceil(n * cfg["train_ratio"])) - 1
    val_idx = int(np.ceil(
        n * (cfg["train_ratio"] + cfg["val_ratio"])
    )) - 1

    train_cutoff = int(timestamps[train_idx])
    val_cutoff = int(timestamps[val_idx])
    test_cutoff = int(timestamps[-1])

    if not train_cutoff < val_cutoff < test_cutoff:
        raise ValueError(
            "Timestamp cutoffs are not strictly increasing. "
            "The sampled data may have too many identical timestamps."
        )

    period_report = write_period_partitions(
        files=files,
        output_dir=output_dir,
        train_cutoff=train_cutoff,
        val_cutoff=val_cutoff,
        test_cutoff=test_cutoff,
        batch_size=cfg["batch_size"],
    )

    _, address_report, holdout_addresses = build_address_splits(
        files=files,
        registry_path=registry_path,
        output_dir=output_dir,
        train_cutoff=train_cutoff,
        val_cutoff=val_cutoff,
        test_cutoff=test_cutoff,
        holdout_ratio=cfg["inductive_holdout_ratio"],
        seed=cfg["seed"],
        batch_size=cfg["batch_size"],
    )

    train_graph_report = write_train_graph_transactions(
        train_path=output_dir / "train_transactions.parquet",
        output_path=output_dir / "train_graph_transactions.parquet",
        holdout_addresses=holdout_addresses,
        batch_size=cfg["batch_size"],
    )

    actual_rows = {
        split: int(period_report[split]["rows"])
        for split in ("train", "val", "test")
    }
    actual_total = sum(actual_rows.values())
    if actual_total != n:
        raise RuntimeError(
            f"Partition row count mismatch: input={n:,}, output={actual_total:,}. "
            "Check null timestamps or timestamp boundaries."
        )

    report = {
        "status": "success",
        "method": (
            "Global chronological transaction-row split. Timestamps are "
            "sorted across all input Parquet files; cutoffs are selected "
            "at cumulative train and train+validation row quantiles."
        ),
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "random_seed": cfg["seed"],
            "train_ratio": cfg["train_ratio"],
            "val_ratio": cfg["val_ratio"],
            "test_ratio": cfg["test_ratio"],
            "inductive_holdout_ratio": cfg["inductive_holdout_ratio"],
            "batch_size": cfg["batch_size"],
        },
        "input": {
            "directory": str(tx_dir.relative_to(ROOT)),
            "files": [path.name for path in files],
            "total_transactions": n,
        },
        "cutoffs": {
            "T_train_unix": train_cutoff,
            "T_train_utc": iso_utc(train_cutoff),
            "T_val_unix": val_cutoff,
            "T_val_utc": iso_utc(val_cutoff),
            "T_test_unix": test_cutoff,
            "T_test_utc": iso_utc(test_cutoff),
        },
        "period_partitions": period_report,
        "actual_ratios": {
            split: actual_rows[split] / actual_total
            for split in ("train", "val", "test")
        },
        "train_graph_partition": train_graph_report,
        "address_split_summary": address_report,
        "anti_leakage_rules": [
            "Raw train/val/test period partitions are disjoint and chronological.",
            "Train graph uses train_graph_transactions.parquet, not the raw train partition.",
            "Every train-period transaction touching a held-out address is excluded from the train graph.",
            "Validation/test period partitions are retained unfiltered for later cumulative snapshots.",
            "Build each snapshot only from transactions with timestamp <= its cutoff.",
            "Fit feature scalers on train only; transform validation/test.",
            "Do not convert label -1 into verified normal label 0.",
            "is_inductive_test identifies held-out addresses active in the test period.",
        ],
        "limitations": [
            "The input consists of four sparse time samples, not continuous Ethereum history.",
            "The 70/15/15 ratio applies to transaction rows, not unique addresses.",
            "Random holdout is an experimental node holdout over addresses present in the available sample; it is not a true future-arrival cohort.",
            "Because the samples are sparse, empty temporal windows cannot automatically be interpreted as confirmed zero activity.",
        ],
        "outputs": {
            "address_splits": str(
                (output_dir / "address_splits.parquet").relative_to(ROOT)
            ),
            "train_graph_transactions": str(
                (output_dir / "train_graph_transactions.parquet").relative_to(ROOT)
            ),
            "report": str(report_path.relative_to(ROOT)),
        },
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== TEMPORAL & INDUCTIVE SPLIT SUMMARY ===")
    print(f"Total transactions: {n:,}")
    print(f"T_train: {iso_utc(train_cutoff)}")
    print(f"T_val:   {iso_utc(val_cutoff)}")
    print(f"T_test:  {iso_utc(test_cutoff)}")

    print("\nPeriod partitions:")
    for split in ("train", "val", "test"):
        rows = actual_rows[split]
        ratio = report["actual_ratios"][split] * 100
        print(f"  {split:5}: {rows:>12,} rows ({ratio:5.2f}%)")

    print("\nInductive holdout:")
    print(f"  Selected addresses: {len(holdout_addresses):,}")
    print(
        "  Active in test: "
        f"{address_report['inductive_holdout_active_in_test']:,}"
    )
    print(
        "  Train graph edges removed: "
        f"{train_graph_report['removed_rows']:,}"
    )
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
