"""
ETHEREUM GUARD - TV1-04 Feature Engineering (vectorized version)

Faster implementation:
- Replaces Python loops over millions of transactions with pandas groupby/agg.
- Handles Parquet timestamps with s/ms/us/ns resolution.
- Builds 12 tabular + 7 temporal-statistical features.
- Builds temporal sequence (N, num_windows, 8) using vectorized groupby.
- Reserves 6 graph-topology features for TV1-05.
- Uses stable node_id values from the complete address manifest.
- Fits RobustScaler on Train only.

Run from repository root:
    python src/pipeline/features.py

Important:
The source consists of sparse time samples. Empty temporal windows mean
"no sampled transaction observed", not necessarily zero real-world activity.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

FEATURE_COLUMNS = [
    "total_eth_sent", "total_eth_received", "net_balance_diff",
    "avg_eth_sent", "max_eth_sent", "avg_eth_received",
    "max_eth_received", "tx_sent_count", "tx_recv_count",
    "sent_recv_ratio", "zero_val_tx_ratio", "failed_tx_ratio",
    "in_degree", "out_degree", "unique_in_neighbors",
    "unique_out_neighbors", "neighbor_repeat_ratio",
    "local_ego_density", "active_duration_hours",
    "tx_frequency_per_day", "mean_inter_tx_time",
    "std_inter_tx_time", "burstiness_index",
    "recent_volume_spike_ratio", "recent_tx_spike_ratio",
]

TABULAR = FEATURE_COLUMNS[:12]
GRAPH = FEATURE_COLUMNS[12:18]
TEMPORAL_STATS = FEATURE_COLUMNS[18:]

SEQUENCE_COLUMNS = [
    "tx_in_count", "tx_out_count", "val_in_sum", "val_out_sum",
    "val_max", "unique_counterparties", "avg_gas_price",
    "mean_inter_arrival_time",
]

LOG = logging.getLogger("ethereum_guard.features")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy config: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    project = raw.get("project", {})
    data = raw.get("data", {})
    temporal = raw.get("temporal", {})
    pipeline = raw.get("pipeline", {})
    output = raw.get("output", {})

    cfg = {
        "seed": int(project.get("random_seed", project.get("seed", 42))),
        "processed_dir": data.get("processed_dir", "data/processed"),
        "num_windows": int(temporal.get("num_windows", 10)),
        "window_hours": int(temporal.get("window_size_hours", 24)),
        "batch_size": int(pipeline.get("batch_size", 100_000)),
        "results_dir": output.get("results_dir", "artifacts/results"),
    }
    if cfg["num_windows"] <= 0 or cfg["window_hours"] <= 0:
        raise ValueError("num_windows và window_size_hours phải > 0")
    return cfg


def timestamp_to_unix_seconds(series: pd.Series) -> pd.Series:
    """Convert datetime64[s/ms/us/ns, UTC] or parseable values to Unix seconds."""
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    unit = getattr(ts.dtype, "unit", "ns")
    divisors = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}
    if unit not in divisors:
        raise ValueError(f"Không hỗ trợ timestamp resolution: {unit}")
    raw = ts.astype("int64").to_numpy(dtype=np.int64)
    valid = ts.notna().to_numpy()
    seconds = raw // divisors[unit]
    out = pd.Series(seconds.astype(np.float64), index=series.index)
    out.loc[~valid] = np.nan
    return out


def read_transactions(paths: list[Path], columns: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy dữ liệu: {path}")
        LOG.info("Reading %s", path.name)
        # Read only required columns. Parquet column pruning avoids loading
        # unused transaction hashes, input data, and other large fields.
        table = pq.read_table(path, columns=columns)
        frames.append(table.to_pandas())
        del table
    if not frames:
        return pd.DataFrame(columns=columns)
    frame = pd.concat(frames, ignore_index=True)
    del frames
    return frame


def normalize_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "timestamp", "from_address", "to_address",
        "value_eth", "gas_price", "success",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Thiếu cột giao dịch: {sorted(missing)}")

    frame = frame.copy()
    frame["timestamp"] = timestamp_to_unix_seconds(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp", "from_address"])
    frame["timestamp"] = frame["timestamp"].astype(np.int64)

    # Categoricals reduce repeated wallet-address memory during groupby.
    frame["from_address"] = (
        frame["from_address"].astype("string").str.lower()
    )
    frame["to_address"] = (
        frame["to_address"].astype("string").str.lower()
    )
    frame["value_eth"] = pd.to_numeric(
        frame["value_eth"], errors="coerce"
    ).fillna(0.0).astype(np.float64)
    frame["gas_price"] = pd.to_numeric(
        frame["gas_price"], errors="coerce"
    )
    frame["gas_gwei"] = frame["gas_price"] / 1e9
    frame["success"] = frame["success"].fillna(False).astype(bool)
    frame = frame[frame["from_address"].notna()]
    frame = frame[frame["from_address"].str.len() > 0]
    return frame


def make_event_tables(tx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create outgoing/incoming event frames with one row per wallet-side event."""
    outgoing = pd.DataFrame({
        "address": tx["from_address"].to_numpy(),
        "counterparty": tx["to_address"].to_numpy(),
        "timestamp": tx["timestamp"].to_numpy(),
        "value_eth": tx["value_eth"].to_numpy(),
        "gas_gwei": tx["gas_gwei"].to_numpy(),
        "success": tx["success"].to_numpy(),
        "direction": "out",
    })

    valid_receiver = tx["to_address"].notna()
    incoming_tx = tx.loc[valid_receiver]
    incoming = pd.DataFrame({
        "address": incoming_tx["to_address"].to_numpy(),
        "counterparty": incoming_tx["from_address"].to_numpy(),
        "timestamp": incoming_tx["timestamp"].to_numpy(),
        "value_eth": incoming_tx["value_eth"].to_numpy(),
        "gas_gwei": incoming_tx["gas_gwei"].to_numpy(),
        "success": incoming_tx["success"].to_numpy(),
        "direction": "in",
    })

    return outgoing, incoming


def compute_tabular_and_temporal(
    tx: pd.DataFrame,
    cutoff: int,
    num_windows: int,
    window_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute 19 non-topology features using vectorized aggregations."""
    tx = tx.loc[tx["timestamp"] <= cutoff].copy()
    if tx.empty:
        raise ValueError("Không có giao dịch trước cutoff.")

    outgoing, incoming = make_event_tables(tx)
    events = pd.concat([outgoing, incoming], ignore_index=True)
    events["is_zero"] = events["value_eth"].eq(0).astype(np.int8)
    events["is_failed"] = (~events["success"]).astype(np.int8)

    LOG.info("Aggregating wallet-side events: %s rows", f"{len(events):,}")
    grouped = events.groupby(["address", "direction"], sort=False, observed=True)
    agg = grouped.agg(
        tx_count=("value_eth", "size"),
        total_value=("value_eth", "sum"),
        avg_value=("value_eth", "mean"),
        max_value=("value_eth", "max"),
        zero_count=("is_zero", "sum"),
        failed_count=("is_failed", "sum"),
    )
    wide = agg.unstack("direction")
    wide = wide.fillna(0.0)
    wide.columns = [f"{metric}_{direction}" for metric, direction in wide.columns]
    wide = wide.fillna(0.0)

    result = pd.DataFrame(index=wide.index)
    result["total_eth_sent"] = wide.get("total_value_out", 0.0)
    result["total_eth_received"] = wide.get("total_value_in", 0.0)
    result["net_balance_diff"] = (
        result["total_eth_received"] - result["total_eth_sent"]
    )
    result["avg_eth_sent"] = wide.get("avg_value_out", 0.0)
    result["max_eth_sent"] = wide.get("max_value_out", 0.0)
    result["avg_eth_received"] = wide.get("avg_value_in", 0.0)
    result["max_eth_received"] = wide.get("max_value_in", 0.0)
    result["tx_sent_count"] = wide.get("tx_count_out", 0.0)
    result["tx_recv_count"] = wide.get("tx_count_in", 0.0)
    result["sent_recv_ratio"] = (
        result["tx_sent_count"] + 1.0
    ) / (result["tx_recv_count"] + 1.0)

    event_total = result["tx_sent_count"] + result["tx_recv_count"]
    zero_total = (
        wide.get("zero_count_out", 0.0) + wide.get("zero_count_in", 0.0)
    )
    failed_total = (
        wide.get("failed_count_out", 0.0) + wide.get("failed_count_in", 0.0)
    )
    result["zero_val_tx_ratio"] = zero_total / event_total.replace(0, np.nan)
    result["failed_tx_ratio"] = failed_total / event_total.replace(0, np.nan)

    # Per-address inter-arrival statistics. Sort once, then use groupby.diff.
    event_times = events[["address", "timestamp"]].sort_values(
        ["address", "timestamp"], kind="mergesort"
    )
    event_times["delta"] = (
        event_times.groupby("address", sort=False)["timestamp"].diff()
    )
    time_agg = event_times.groupby("address", sort=False)["delta"].agg(
        mean_inter_tx_time="mean",
        std_inter_tx_time="std",
    )
    first_last = event_times.groupby("address", sort=False)["timestamp"].agg(
        first_seen="min", last_seen="max", event_count="size"
    )
    result["active_duration_hours"] = (
        (first_last["last_seen"] - first_last["first_seen"]) / 3600.0
    ).reindex(result.index).fillna(0.0)
    active_days = (
        (first_last["last_seen"] - first_last["first_seen"]) / 86400.0
    ).clip(lower=1.0 / 24.0)
    result["tx_frequency_per_day"] = (
        first_last["event_count"] / active_days
    ).reindex(result.index).fillna(0.0)

    result["mean_inter_tx_time"] = (
        time_agg["mean_inter_tx_time"].reindex(result.index).fillna(0.0)
    )
    result["std_inter_tx_time"] = (
        time_agg["std_inter_tx_time"].fillna(0.0).reindex(result.index)
    )
    mean_delta = result["mean_inter_tx_time"]
    std_delta = result["std_inter_tx_time"]
    denom = std_delta + mean_delta
    result["burstiness_index"] = (
        (std_delta - mean_delta) / denom.replace(0, np.nan)
    ).fillna(0.0)

    # Recent spike features: last 24h vs average of preceding (W-1) 24h bins.
    window_seconds = window_hours * 3600
    history_start = cutoff - num_windows * window_seconds
    recent_start = cutoff - window_seconds
    recent = tx.loc[tx["timestamp"] > recent_start]
    history = tx.loc[
        (tx["timestamp"] > history_start)
        & (tx["timestamp"] <= recent_start)
    ]

    recent_agg = recent.groupby("from_address", sort=False).agg(
        recent_value=("value_eth", "sum"),
        recent_count=("value_eth", "size"),
    )
    history_agg = history.groupby("from_address", sort=False).agg(
        history_value=("value_eth", "sum"),
        history_count=("value_eth", "size"),
    )
    spike = recent_agg.join(history_agg, how="outer").fillna(0.0)
    n_history_windows = max(num_windows - 1, 1)
    volume_base = spike["history_value"] / n_history_windows
    count_base = spike["history_count"] / n_history_windows
    spike["recent_volume_spike_ratio"] = np.where(
        volume_base > 0, spike["recent_value"] / volume_base, 0.0
    )
    spike["recent_tx_spike_ratio"] = np.where(
        count_base > 0, spike["recent_count"] / count_base, 0.0
    )
    result["recent_volume_spike_ratio"] = (
        spike["recent_volume_spike_ratio"].reindex(result.index).fillna(0.0)
    )
    result["recent_tx_spike_ratio"] = (
        spike["recent_tx_spike_ratio"].reindex(result.index).fillna(0.0)
    )

    for col in GRAPH:
        result[col] = np.nan

    result = result.reindex(columns=FEATURE_COLUMNS)
    result = result.replace([np.inf, -np.inf], np.nan)

    # Temporal sequence is computed with vectorized groupby on the last W bins.
    start = history_start
    seq_events = events.loc[
        (events["timestamp"] > start) & (events["timestamp"] <= cutoff)
    ].copy()
    seq_events["window_idx"] = (
        (seq_events["timestamp"] - start) // window_seconds
    ).astype(np.int16)
    seq_events["window_idx"] = seq_events["window_idx"].clip(
        0, num_windows - 1
    )

    keys = ["address", "window_idx"]
    seq_group = seq_events.groupby(keys, sort=False, observed=True)

    seq_agg = seq_group.agg(
        event_count=("value_eth", "size"),
        value_sum=("value_eth", "sum"),
        value_max=("value_eth", "max"),
        unique_counterparties=("counterparty", "nunique"),
        avg_gas_price=("gas_gwei", "mean"),
    )

    # Directional counts and value sums.
    directional = (
        seq_events.groupby(
            ["address", "window_idx", "direction"],
            sort=False,
            observed=True,
        )
        .agg(
            count=("value_eth", "size"),
            value_sum=("value_eth", "sum"),
        )
        .unstack("direction")
        .fillna(0.0)
    )
    directional.columns = [
        f"{metric}_{direction}" for metric, direction in directional.columns
    ]

    # Inter-arrival time per wallet per window.
    seq_sorted = seq_events.sort_values(
        ["address", "window_idx", "timestamp"],
        kind="mergesort",
    )
    seq_sorted["delta"] = seq_sorted.groupby(
        ["address", "window_idx"], sort=False
    )["timestamp"].diff()
    inter_window = seq_sorted.groupby(
        ["address", "window_idx"], sort=False
    )["delta"].mean()

    # Build compact MultiIndex table then scatter into dense tensor.
    seq_table = pd.DataFrame(index=seq_agg.index)
    # Columns were flattened above to strings such as "count_in".
    # Looking them up with ("count", "in") silently returned scalar 0.0,
    # which erased all directional temporal counts and value sums.
    seq_table["tx_in_count"] = directional.get("count_in", 0.0)
    seq_table["tx_out_count"] = directional.get("count_out", 0.0)
    seq_table["val_in_sum"] = directional.get("value_sum_in", 0.0)
    seq_table["val_out_sum"] = directional.get("value_sum_out", 0.0)
    seq_table["val_max"] = seq_agg["value_max"]
    seq_table["unique_counterparties"] = seq_agg["unique_counterparties"]
    seq_table["avg_gas_price"] = seq_agg["avg_gas_price"]
    seq_table["mean_inter_arrival_time"] = inter_window

    # Stable address order is assigned by caller from the global manifest.
    metadata = {
        "transactions_in_history": int(len(tx)),
        "wallet_side_events": int(len(events)),
        "temporal_events": int(len(seq_events)),
        "temporal_window_coverage": [],
    }
    for wi in range(num_windows):
        bin_start = start + wi * window_seconds
        bin_end = bin_start + window_seconds
        observed = bool(
            ((tx["timestamp"] > bin_start) & (tx["timestamp"] <= bin_end)).any()
        )
        metadata["temporal_window_coverage"].append({
            "window_index": wi,
            "start_utc": datetime.fromtimestamp(
                bin_start, timezone.utc
            ).isoformat(),
            "end_utc": datetime.fromtimestamp(
                bin_end, timezone.utc
            ).isoformat(),
            "has_observed_transactions": observed,
        })

    # Keep the sparse sequence table; caller creates dense tensor using address map.
    return result, seq_table, metadata


def attach_manifest_and_ids(
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    global_id_map: dict[str, int],
) -> pd.DataFrame:
    cols = [
        "address", "label", "label_source", "is_inductive_holdout",
        "seen_in_train_period", "seen_in_val_period", "seen_in_test_period",
        "is_inductive_test", "is_new_after_train_cutoff", "is_new_in_test_period",
    ]
    meta = manifest[cols].drop_duplicates("address")
    out = features.reset_index().rename(columns={"index": "address"})
    out = out.merge(meta, on="address", how="left", sort=False)
    out["label"] = out["label"].fillna(-1).astype(np.int8)
    out["is_inductive_holdout"] = (
        out["is_inductive_holdout"].fillna(False).astype(bool)
    )
    out["node_id"] = out["address"].map(global_id_map).astype(np.int64)
    return out


def make_dense_sequence(
    sparse_sequence: pd.DataFrame,
    addresses: pd.Series,
    global_id_map: dict[str, int],
    num_nodes: int,
    num_windows: int,
) -> np.ndarray:
    """Create (N,W,8) float32 tensor without Python loops over transactions."""
    seq = np.zeros(
        (num_nodes, num_windows, len(SEQUENCE_COLUMNS)),
        dtype=np.float32,
    )
    if sparse_sequence.empty:
        return seq

    address_values = sparse_sequence.index.get_level_values("address")
    window_values = sparse_sequence.index.get_level_values("window_idx").to_numpy(
        dtype=np.int64
    )
    node_values = pd.Index(addresses).get_indexer(address_values)
    valid = node_values >= 0
    if not valid.any():
        return seq

    subset = sparse_sequence.iloc[np.flatnonzero(valid)]
    node_values = node_values[valid]
    window_values = window_values[valid]

    seq[node_values, window_values, 0] = subset["tx_in_count"].to_numpy(np.float32)
    seq[node_values, window_values, 1] = subset["tx_out_count"].to_numpy(np.float32)
    seq[node_values, window_values, 2] = subset["val_in_sum"].to_numpy(np.float32)
    seq[node_values, window_values, 3] = subset["val_out_sum"].to_numpy(np.float32)
    seq[node_values, window_values, 4] = subset["val_max"].to_numpy(np.float32)
    seq[node_values, window_values, 5] = subset["unique_counterparties"].to_numpy(np.float32)
    seq[node_values, window_values, 6] = subset["avg_gas_price"].to_numpy(np.float32)
    seq[node_values, window_values, 7] = subset["mean_inter_arrival_time"].fillna(0).to_numpy(np.float32)
    return seq


def fit_scaler(train: pd.DataFrame, path: Path) -> dict[str, Any]:
    columns = TABULAR + TEMPORAL_STATS
    values = (
        train[columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    scaler = RobustScaler()
    scaler.fit(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "columns": columns}, path)
    return {
        "path": str(path.relative_to(ROOT)),
        "fit_split": "train",
        "method": "RobustScaler",
        "columns": columns,
    }


def transform_features(
    frame: pd.DataFrame,
    scaler_bundle: dict[str, Any],
) -> pd.DataFrame:
    cols = scaler_bundle["columns"]
    values = (
        frame[cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    out = frame.copy()
    out.loc[:, cols] = scaler_bundle["scaler"].transform(values)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ethereum Guard vectorized feature engineering"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg = load_config(args.config.resolve())
    processed = ROOT / cfg["processed_dir"]
    split_dir = processed / "splits"
    feature_dir = processed / "features"
    results_dir = ROOT / cfg["results_dir"]

    manifest_path = split_dir / "address_splits.parquet"
    split_report_path = results_dir / "split_report.json"
    if not manifest_path.exists() or not split_report_path.exists():
        raise FileNotFoundError(
            "Thiếu address_splits.parquet hoặc split_report.json. "
            "Hãy chạy create_splits.py trước."
        )

    manifest = pq.read_table(manifest_path).to_pandas()
    manifest["address"] = manifest["address"].astype(str).str.lower()
    # A single global mapping keeps node IDs consistent across snapshots.
    all_addresses = sorted(manifest["address"].unique())
    global_id_map = {address: idx for idx, address in enumerate(all_addresses)}

    split_report = json.loads(split_report_path.read_text(encoding="utf-8"))
    cutoffs = {
        "train": int(split_report["cutoffs"]["T_train_unix"]),
        "val": int(split_report["cutoffs"]["T_val_unix"]),
        "test": int(split_report["cutoffs"]["T_test_unix"]),
    }

    sources = {
        "train": [split_dir / "train_graph_transactions.parquet"],
        "val": [
            split_dir / "train_transactions.parquet",
            split_dir / "val_transactions.parquet",
        ],
        "test": [
            split_dir / "train_transactions.parquet",
            split_dir / "val_transactions.parquet",
            split_dir / "test_transactions.parquet",
        ],
    }

    scaler_path = ROOT / "artifacts" / "scalers" / "feature_robust_scaler.joblib"
    report: dict[str, Any] = {
        "status": "success",
        "implementation": "vectorized_pandas_groupby",
        "feature_count": 25,
        "implemented_now": {
            "tabular": TABULAR,
            "temporal_statistics": TEMPORAL_STATS,
            "temporal_sequence": SEQUENCE_COLUMNS,
        },
        "deferred_to_tv1_05_graph_builder": GRAPH,
        "global_node_count": len(all_addresses),
        "splits": {},
        "limitations": [
            "Graph topology features are pending TV1-05 and remain NaN.",
            "The dataset consists of sparse time samples; empty windows mean no sampled transaction was observed.",
            "Train uses train_graph_transactions.parquet to exclude held-out addresses from the training graph/features.",
            "Validation/Test use cumulative chronological history.",
        ],
    }

    columns = [
        "timestamp", "from_address", "to_address",
        "value_eth", "gas_price", "success",
    ]

    for split in ("train", "val", "test"):
        tx = read_transactions(sources[split], columns)
        tx = normalize_transactions(tx)

        features, sparse_sequence, metadata = compute_tabular_and_temporal(
            tx=tx,
            cutoff=cutoffs[split],
            num_windows=cfg["num_windows"],
            window_hours=cfg["window_hours"],
        )
        del tx

        features = attach_manifest_and_ids(
            features, manifest, global_id_map
        )
        addresses = features["address"].reset_index(drop=True)

        dense_sequence = make_dense_sequence(
            sparse_sequence=sparse_sequence,
            addresses=addresses,
            global_id_map=global_id_map,
            num_nodes=len(features),
            num_windows=cfg["num_windows"],
        )
        del sparse_sequence

        split_out = feature_dir / split
        split_out.mkdir(parents=True, exist_ok=True)
        nodes_path = split_out / "nodes.parquet"
        temporal_path = split_out / "temporal.pt"

        if split == "train":
            scaler_info = fit_scaler(features, scaler_path)
            scaler_bundle = joblib.load(scaler_path)
            report["scaler"] = scaler_info
        else:
            if not scaler_path.exists():
                raise FileNotFoundError("Scaler Train chưa tồn tại.")
            scaler_bundle = joblib.load(scaler_path)

        scaled = transform_features(features, scaler_bundle)
        scaled.to_parquet(
            nodes_path, index=False, compression="zstd"
        )

        torch.save(
            {
                "node_ids": torch.tensor(
                    scaled["node_id"].to_numpy(), dtype=torch.long
                ),
                "addresses": scaled["address"].astype(str).tolist(),
                "features": torch.tensor(
                    scaled[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
                    dtype=torch.float32,
                ),
                "feature_columns": FEATURE_COLUMNS,
                "sequence": torch.from_numpy(dense_sequence),
                "sequence_columns": SEQUENCE_COLUMNS,
                "labels": torch.tensor(
                    scaled["label"].to_numpy(), dtype=torch.int8
                ),
                "is_inductive_holdout": torch.tensor(
                    scaled["is_inductive_holdout"].to_numpy(),
                    dtype=torch.bool,
                ),
            },
            temporal_path,
        )

        report["splits"][split] = {
            "nodes_path": str(nodes_path.relative_to(ROOT)),
            "temporal_path": str(temporal_path.relative_to(ROOT)),
            "node_rows": int(len(scaled)),
            "sequence_shape": list(dense_sequence.shape),
            "cutoff_utc": datetime.fromtimestamp(
                cutoffs[split], timezone.utc
            ).isoformat(),
            **metadata,
        }

        LOG.info(
            "[%s] nodes=%s | sequence=%s | tx=%s",
            split.upper(),
            f"{len(scaled):,}",
            dense_sequence.shape,
            f"{metadata['transactions_in_history']:,}",
        )

        del features, scaled, dense_sequence

    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "feature_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== FEATURE ENGINEERING SUMMARY ===")
    for split, info in report["splits"].items():
        print(
            f"{split:5}: {info['node_rows']:,} nodes | "
            f"sequence={info['sequence_shape']} | "
            f"history_tx={info['transactions_in_history']:,}"
        )
        observed = sum(
            item["has_observed_transactions"]
            for item in info["temporal_window_coverage"]
        )
        print(f"       observed temporal windows: {observed}/{cfg['num_windows']}")

    print(f"Report: {report_path}")
    print("Graph topology features are intentionally pending TV1-05.")


if __name__ == "__main__":
    main()
