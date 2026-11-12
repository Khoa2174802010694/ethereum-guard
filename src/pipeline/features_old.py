"""
ETHEREUM GUARD - TV1-04 Feature Engineering

Implemented:
- 12 Tabular Features
- 7 Temporal Statistics
- Temporal Sequence: (N, 10, 8)
- 6 Graph Topology Features: reserved for TV1-05

Timestamp conversion supports s, ms, us, ns.
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
import pyarrow.parquet as pq
import torch
import yaml
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

FEATURE_COLUMNS = [
    "total_eth_sent",
    "total_eth_received",
    "net_balance_diff",
    "avg_eth_sent",
    "max_eth_sent",
    "avg_eth_received",
    "max_eth_received",
    "tx_sent_count",
    "tx_recv_count",
    "sent_recv_ratio",
    "zero_val_tx_ratio",
    "failed_tx_ratio",
    "in_degree",
    "out_degree",
    "unique_in_neighbors",
    "unique_out_neighbors",
    "neighbor_repeat_ratio",
    "local_ego_density",
    "active_duration_hours",
    "tx_frequency_per_day",
    "mean_inter_tx_time",
    "std_inter_tx_time",
    "burstiness_index",
    "recent_volume_spike_ratio",
    "recent_tx_spike_ratio",
]

TABULAR = FEATURE_COLUMNS[:12]
GRAPH = FEATURE_COLUMNS[12:18]
TEMPORAL_STATS = FEATURE_COLUMNS[18:]

SEQUENCE_COLUMNS = [
    "tx_in_count",
    "tx_out_count",
    "val_in_sum",
    "val_out_sum",
    "val_max",
    "unique_counterparties",
    "avg_gas_price",
    "mean_inter_arrival_time",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ethereum_guard.features")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    project = cfg.get("project", {})
    data = cfg.get("data", {})
    temporal = cfg.get("temporal", {})
    pipeline = cfg.get("pipeline", {})
    output = cfg.get("output", {})

    return {
        "seed": int(project.get("random_seed", project.get("seed", 42))),
        "processed_dir": data.get("processed_dir", "data/processed"),
        "num_windows": int(
            temporal.get("num_windows", data.get("temporal_windows", 10))
        ),
        "window_size_hours": int(
            temporal.get(
                "window_size_hours",
                data.get("window_size_hours", 24),
            )
        ),
        "batch_size": int(pipeline.get("batch_size", 100_000)),
        "results_dir": output.get("results_dir", "artifacts/results"),
    }


def read_parquet(
    paths: list[Path],
    columns: list[str],
) -> pd.DataFrame:
    frames = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy dữ liệu: {path}"
            )

        logger.info("Reading %s", path.name)

        table = pq.read_table(path, columns=columns)
        frames.append(table.to_pandas())

    if not frames:
        return pd.DataFrame(columns=columns)

    return pd.concat(frames, ignore_index=True)


def normalize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "timestamp",
        "from_address",
        "to_address",
        "value_eth",
        "gas_price",
        "success",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Thiếu cột giao dịch: {sorted(missing)}"
        )

    df = df.copy()

    # Convert timestamp to UTC.
    ts = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    # Detect actual timestamp resolution.
    # Parquet timestamp[ms] must be divided by 1,000,
    # not 1,000,000,000.
    unit = getattr(ts.dtype, "unit", "ns")

    divisor_by_unit = {
        "s": 1,
        "ms": 1_000,
        "us": 1_000_000,
        "ns": 1_000_000_000,
    }

    divisor = divisor_by_unit.get(unit)

    if divisor is None:
        raise ValueError(
            f"Unsupported timestamp resolution: {unit}"
        )

    timestamp_values = ts.astype("int64").to_numpy(
        dtype=np.int64
    )

    valid_timestamp = ts.notna().to_numpy()

    unix_seconds = timestamp_values // divisor

    df["timestamp"] = unix_seconds.astype(np.float64)

    df.loc[~valid_timestamp, "timestamp"] = np.nan

    df["value_eth"] = (
        pd.to_numeric(df["value_eth"], errors="coerce")
        .fillna(0.0)
    )

    df["gas_price"] = pd.to_numeric(
        df["gas_price"],
        errors="coerce",
    )

    df["success"] = (
        df["success"]
        .fillna(False)
        .astype(bool)
    )

    df["from_address"] = (
        df["from_address"]
        .astype("string")
        .str.lower()
    )

    df["to_address"] = (
        df["to_address"]
        .astype("string")
        .str.lower()
    )

    df = df.dropna(
        subset=["timestamp", "from_address"]
    )

    df = df[df["from_address"].str.len() > 0]

    return df


def build_node_features(
    transactions: pd.DataFrame,
    address_manifest: pd.DataFrame,
    cutoff: int,
    num_windows: int,
    window_hours: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:

    """Compute 19 non-topology features and (N, W, 8) sequence."""

    df = transactions[
        transactions["timestamp"] <= cutoff
    ].copy()

    if df.empty:
        raise ValueError(
            "Không có giao dịch trước cutoff."
        )

    # Convert Wei/gas to Gwei/gas.
    df["gas_gwei"] = df["gas_price"] / 1e9

    # Outgoing events.
    send = df[
        [
            "from_address",
            "timestamp",
            "value_eth",
            "gas_gwei",
            "success",
            "to_address",
        ]
    ].copy()

    send = send.rename(
        columns={"from_address": "address"}
    )

    send["direction"] = "out"

    # Incoming events.
    recv = df[
        df["to_address"].notna()
    ][
        [
            "to_address",
            "timestamp",
            "value_eth",
            "gas_gwei",
            "success",
            "from_address",
        ]
    ].copy()

    recv = recv.rename(
        columns={
            "to_address": "address",
            "from_address": "counterparty",
        }
    )

    send = send.rename(
        columns={"to_address": "counterparty"}
    )

    recv["direction"] = "in"

    events = pd.concat(
        [send, recv],
        ignore_index=True,
    )

    events["counterparty"] = (
        events["counterparty"]
        .astype("string")
        .str.lower()
    )

    events["is_zero"] = events["value_eth"].eq(0)
    events["is_failed"] = ~events["success"]

    grouped = events.groupby(
        ["address", "direction"],
        sort=False,
    )

    agg = grouped.agg(
        tx_count=("value_eth", "size"),
        total_value=("value_eth", "sum"),
        avg_value=("value_eth", "mean"),
        max_value=("value_eth", "max"),
        zero_count=("is_zero", "sum"),
        failed_count=("is_failed", "sum"),
    ).reset_index()

    wide = agg.pivot(
        index="address",
        columns="direction",
    )

    wide.columns = [
        f"{metric}_{direction}"
        for metric, direction in wide.columns
    ]

    wide = wide.fillna(0.0)

    # Timestamp statistics.
    tx_times = df[
        [
            "from_address",
            "to_address",
            "timestamp",
            "value_eth",
        ]
    ].copy()

    outgoing_times = tx_times[
        [
            "from_address",
            "timestamp",
            "value_eth",
        ]
    ].rename(
        columns={"from_address": "address"}
    )

    incoming_times = tx_times[
        tx_times["to_address"].notna()
    ][
        [
            "to_address",
            "timestamp",
            "value_eth",
        ]
    ].rename(
        columns={"to_address": "address"}
    )

    wallet_times = pd.concat(
        [outgoing_times, incoming_times],
        ignore_index=True,
    )

    wallet_times = wallet_times.sort_values(
        ["address", "timestamp"]
    )

    time_groups = wallet_times.groupby(
        "address",
        sort=False,
    )

    first = time_groups["timestamp"].min()
    last = time_groups["timestamp"].max()

    duration = (last - first) / 3600.0

    def inter_stats(values: pd.Series) -> pd.Series:

        diffs = (
            values.diff()
            .dropna()
            .to_numpy(dtype=np.float64)
        )

        if len(diffs) == 0:
            return pd.Series({
                "mean_inter_tx_time": 0.0,
                "std_inter_tx_time": 0.0,
                "burstiness_index": 0.0,
            })

        mean = float(np.mean(diffs))
        std = float(np.std(diffs, ddof=0))

        denom = std + mean

        burst = (
            (std - mean) / denom
            if denom > 0
            else 0.0
        )

        return pd.Series({
            "mean_inter_tx_time": mean,
            "std_inter_tx_time": std,
            "burstiness_index": burst,
        })

    inter = (
        time_groups["timestamp"]
        .apply(inter_stats)
        .unstack()
    )

    total_tx = time_groups.size()

    active_days = (
        (last - first) / 86400.0
    ).clip(lower=1.0 / 24.0)

    frequency = total_tx / active_days

    # Recent spike features.
    lookback_seconds = (
        num_windows * window_hours * 3600
    )

    recent_start = cutoff - window_hours * 3600
    history_start = cutoff - lookback_seconds

    recent = df[
        df["timestamp"] > recent_start
    ]

    history = df[
        (df["timestamp"] > history_start)
        & (df["timestamp"] <= recent_start)
    ]

    recent_sent_value = (
        recent.groupby("from_address")["value_eth"].sum()
    )

    history_sent_value = (
        history.groupby("from_address")["value_eth"].sum()
    )

    recent_sent_count = (
        recent.groupby("from_address").size()
    )

    history_sent_count = (
        history.groupby("from_address").size()
    )

    hist_windows = max(num_windows - 1, 1)

    recent_volume_ratio = (
        recent_sent_value
        / (
            history_sent_value / hist_windows
        ).replace(0, np.nan)
    )

    recent_count_ratio = (
        recent_sent_count
        / (
            history_sent_count / hist_windows
        ).replace(0, np.nan)
    )

    # Build 12 tabular features.
    result = pd.DataFrame(index=wide.index)

    result["total_eth_sent"] = wide.get(
        "total_value_out", 0.0
    )

    result["total_eth_received"] = wide.get(
        "total_value_in", 0.0
    )

    result["net_balance_diff"] = (
        result["total_eth_received"]
        - result["total_eth_sent"]
    )

    result["avg_eth_sent"] = wide.get(
        "avg_value_out", 0.0
    )

    result["max_eth_sent"] = wide.get(
        "max_value_out", 0.0
    )

    result["avg_eth_received"] = wide.get(
        "avg_value_in", 0.0
    )

    result["max_eth_received"] = wide.get(
        "max_value_in", 0.0
    )

    result["tx_sent_count"] = wide.get(
        "tx_count_out", 0.0
    )

    result["tx_recv_count"] = wide.get(
        "tx_count_in", 0.0
    )

    result["sent_recv_ratio"] = (
        result["tx_sent_count"] + 1.0
    ) / (
        result["tx_recv_count"] + 1.0
    )

    zero_total = (
        wide.get("zero_count_out", 0.0)
        + wide.get("zero_count_in", 0.0)
    )

    failed_total = (
        wide.get("failed_count_out", 0.0)
        + wide.get("failed_count_in", 0.0)
    )

    event_total = (
        result["tx_sent_count"]
        + result["tx_recv_count"]
    )

    result["zero_val_tx_ratio"] = (
        zero_total / event_total.replace(0, np.nan)
    )

    result["failed_tx_ratio"] = (
        failed_total / event_total.replace(0, np.nan)
    )

    # Seven temporal statistics.
    result["active_duration_hours"] = (
        duration.reindex(result.index).fillna(0.0)
    )

    result["tx_frequency_per_day"] = (
        frequency.reindex(result.index).fillna(0.0)
    )

    for col in (
        "mean_inter_tx_time",
        "std_inter_tx_time",
        "burstiness_index",
    ):
        result[col] = (
            inter[col]
            .reindex(result.index)
            .fillna(0.0)
        )

    result["recent_volume_spike_ratio"] = (
        recent_volume_ratio.reindex(result.index)
    )

    result["recent_tx_spike_ratio"] = (
        recent_count_ratio.reindex(result.index)
    )

    result[
        [
            "recent_volume_spike_ratio",
            "recent_tx_spike_ratio",
        ]
    ] = (
        result[
            [
                "recent_volume_spike_ratio",
                "recent_tx_spike_ratio",
            ]
        ]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # Graph features are reserved for TV1-05.
    for col in GRAPH:
        result[col] = np.nan

    result = result.reindex(
        columns=FEATURE_COLUMNS
    )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Merge labels and split metadata.
    manifest_cols = [
        "address",
        "label",
        "label_source",
        "is_inductive_holdout",
        "seen_in_train_period",
        "seen_in_val_period",
        "seen_in_test_period",
        "is_inductive_test",
        "is_new_after_train_cutoff",
        "is_new_in_test_period",
    ]

    manifest = (
        address_manifest[manifest_cols]
        .drop_duplicates("address")
    )

    result = (
        result.reset_index()
        .rename(columns={"index": "address"})
    )

    result = result.merge(
        manifest,
        on="address",
        how="left",
    )

    result["label"] = (
        result["label"]
        .fillna(-1)
        .astype("int8")
    )

    result["is_inductive_holdout"] = (
        result["is_inductive_holdout"]
        .fillna(False)
        .astype(bool)
    )

    # Build temporal sequence.
    window_seconds = window_hours * 3600

    window_starts = [
        cutoff - (num_windows - i) * window_seconds
        for i in range(num_windows)
    ]

    addr_index = result["address"].tolist()

    addr_to_idx = {
        address: i
        for i, address in enumerate(addr_index)
    }

    seq = np.zeros(
        (
            len(addr_index),
            num_windows,
            len(SEQUENCE_COLUMNS),
        ),
        dtype=np.float32,
    )

    temporal_rows = df[
        (df["timestamp"] > history_start)
        & (df["timestamp"] <= cutoff)
    ].copy()

    temporal_rows["window_idx"] = (
        (
            temporal_rows["timestamp"] - history_start
        ) // window_seconds
    ).astype(int)

    temporal_rows["window_idx"] = (
        temporal_rows["window_idx"]
        .clip(0, num_windows - 1)
    )

    # Aggregate directional counts, values and gas price.
    for row in temporal_rows.itertuples(index=False):

        sender = row.from_address
        receiver = row.to_address
        wi = int(row.window_idx)

        value = float(row.value_eth)

        gas = (
            float(row.gas_gwei)
            if pd.notna(row.gas_gwei)
            else 0.0
        )

        si = addr_to_idx.get(sender)

        if si is not None:
            seq[si, wi, 1] += 1.0
            seq[si, wi, 3] += value
            seq[si, wi, 4] = max(
                seq[si, wi, 4],
                value,
            )
            seq[si, wi, 6] += gas

        ri = (
            addr_to_idx.get(receiver)
            if pd.notna(receiver)
            else None
        )

        if ri is not None:
            seq[ri, wi, 0] += 1.0
            seq[ri, wi, 2] += value
            seq[ri, wi, 4] = max(
                seq[ri, wi, 4],
                value,
            )
            seq[ri, wi, 6] += gas

    # Unique counterparties and inter-arrival times.
    counterparty_sets: dict[
        tuple[str, int], set[str]
    ] = {}

    timestamp_lists: dict[
        tuple[str, int], list[int]
    ] = {}

    # Avoid creating a second huge Python event list.
    for sender, receiver, ts in temporal_rows[
        [
            "from_address",
            "to_address",
            "timestamp",
        ]
    ].itertuples(index=False, name=None):

        ts_int = int(ts)

        wi = min(
            max(
                (ts_int - history_start) // window_seconds,
                0,
            ),
            num_windows - 1,
        )

        sender_key = (str(sender), int(wi))

        if pd.notna(receiver):
            counterparty_sets.setdefault(
                sender_key,
                set(),
            ).add(str(receiver))

        timestamp_lists.setdefault(
            sender_key,
            [],
        ).append(ts_int)

        if pd.notna(receiver):

            receiver_key = (
                str(receiver),
                int(wi),
            )

            counterparty_sets.setdefault(
                receiver_key,
                set(),
            ).add(str(sender))

            timestamp_lists.setdefault(
                receiver_key,
                [],
            ).append(ts_int)

    for (address, wi), counterparties in (
        counterparty_sets.items()
    ):

        idx = addr_to_idx.get(address)

        if idx is not None:
            seq[idx, wi, 5] = float(
                len(counterparties)
            )

    for (address, wi), times in timestamp_lists.items():

        idx = addr_to_idx.get(address)

        if idx is not None:
            times.sort()

            seq[idx, wi, 7] = (
                float(np.mean(np.diff(times)))
                if len(times) > 1
                else 0.0
            )

    # Average gas price per window, in Gwei.
    for i in range(len(addr_index)):
        for wi in range(num_windows):

            count = (
                seq[i, wi, 0]
                + seq[i, wi, 1]
            )

            if count > 0:
                seq[i, wi, 6] /= count

    # Coverage report.
    coverage = []

    for i, start in enumerate(window_starts):

        end = start + window_seconds

        observed = bool(
            (
                (df["timestamp"] > start)
                & (df["timestamp"] <= end)
            ).any()
        )

        coverage.append({
            "window_index": i,
            "start_utc": datetime.fromtimestamp(
                start,
                timezone.utc,
            ).isoformat(),
            "end_utc": datetime.fromtimestamp(
                end,
                timezone.utc,
            ).isoformat(),
            "has_observed_transactions": observed,
        })

    result["node_id"] = np.arange(
        len(result),
        dtype=np.int64,
    )

    metadata = {
        "cutoff_utc": datetime.fromtimestamp(
            cutoff,
            timezone.utc,
        ).isoformat(),
        "nodes": len(result),
        "transactions_in_history": int(len(df)),
        "temporal_window_coverage": coverage,
        "graph_features_pending": GRAPH,
    }

    return result, seq, metadata


def fit_and_save_scaler(
    train_features: pd.DataFrame,
    scaler_path: Path,
) -> dict[str, Any]:

    numeric_cols = TABULAR + TEMPORAL_STATS

    x = (
        train_features[numeric_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    scaler = RobustScaler()
    scaler.fit(x)

    scaler_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "scaler": scaler,
            "columns": numeric_cols,
        },
        scaler_path,
    )

    return {
        "path": str(scaler_path.relative_to(ROOT)),
        "fit_split": "train",
        "columns": numeric_cols,
        "method": "RobustScaler",
    }


def apply_scaler(
    frame: pd.DataFrame,
    scaler_bundle: dict[str, Any],
) -> pd.DataFrame:

    cols = scaler_bundle["columns"]
    scaler = scaler_bundle["scaler"]

    values = (
        frame[cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    frame = frame.copy()

    frame.loc[:, cols] = scaler.transform(values)

    return frame


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Ethereum Guard feature engineering"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    args = parser.parse_args()

    cfg = load_config(args.config.resolve())

    processed = ROOT / cfg["processed_dir"]

    split_dir = processed / "splits"
    feature_dir = processed / "features"

    report_path = (
        ROOT
        / cfg["results_dir"]
        / "feature_report.json"
    )

    manifest_path = (
        split_dir / "address_splits.parquet"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Chưa có address_splits.parquet. "
            "Chạy create_splits.py trước."
        )

    manifest = pq.read_table(
        manifest_path
    ).to_pandas()

    split_sources = {
        "train": [
            split_dir / "train_graph_transactions.parquet"
        ],
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

    split_report_path = (
        ROOT
        / cfg["results_dir"]
        / "split_report.json"
    )

    with split_report_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        split_report = json.load(f)

    cutoffs = {
        "train": int(
            split_report["cutoffs"]["T_train_unix"]
        ),
        "val": int(
            split_report["cutoffs"]["T_val_unix"]
        ),
        "test": int(
            split_report["cutoffs"]["T_test_unix"]
        ),
    }

    outputs = {}

    report = {
        "status": "success",
        "feature_count": 25,
        "implemented_now": {
            "tabular": TABULAR,
            "temporal_statistics": TEMPORAL_STATS,
            "temporal_sequence": SEQUENCE_COLUMNS,
        },
        "deferred_to_tv1_05_graph_builder": GRAPH,
        "splits": {},
        "limitations": [
            "Graph topology features are reserved as NaN until graph_builder.py constructs the corresponding snapshot.",
            "The source dataset consists of sparse temporal samples; zero-filled sequence bins do not prove zero real-world activity.",
            "Train history uses train_graph_transactions.parquet to exclude inductive holdout nodes from the training graph/features.",
            "Validation/Test histories are cumulative chronological partitions and retain holdout addresses for evaluation.",
        ],
    }

    for split in ("train", "val", "test"):

        columns = [
            "timestamp",
            "from_address",
            "to_address",
            "value_eth",
            "gas_price",
            "success",
        ]

        tx = read_parquet(
            split_sources[split],
            columns,
        )

        tx = normalize_transactions(tx)

        features, temporal, metadata = build_node_features(
            transactions=tx,
            address_manifest=manifest,
            cutoff=cutoffs[split],
            num_windows=cfg["num_windows"],
            window_hours=cfg["window_size_hours"],
        )

        split_out = feature_dir / split

        split_out.mkdir(
            parents=True,
            exist_ok=True,
        )

        nodes_path = split_out / "nodes.parquet"
        temporal_path = split_out / "temporal.pt"

        scaler_path = (
            ROOT
            / "artifacts"
            / "scalers"
            / "feature_robust_scaler.joblib"
        )

        if split == "train":

            scaler_info = fit_and_save_scaler(
                features,
                scaler_path,
            )

            scaler_bundle = joblib.load(
                scaler_path
            )

            report["scaler"] = scaler_info

        else:

            if not scaler_path.exists():
                raise FileNotFoundError(
                    "Scaler Train chưa tồn tại."
                )

            scaler_bundle = joblib.load(
                scaler_path
            )

        scaled = apply_scaler(
            features,
            scaler_bundle,
        )

        scaled.to_parquet(
            nodes_path,
            index=False,
            compression="zstd",
        )

        torch.save(
            {
                "node_ids": torch.tensor(
                    scaled["node_id"].to_numpy(),
                    dtype=torch.long,
                ),

                "addresses": (
                    scaled["address"]
                    .astype(str)
                    .tolist()
                ),

                "features": torch.tensor(
                    scaled[FEATURE_COLUMNS].to_numpy(
                        dtype=np.float32
                    ),
                    dtype=torch.float32,
                ),

                "feature_columns": FEATURE_COLUMNS,

                "sequence": torch.tensor(
                    temporal,
                    dtype=torch.float32,
                ),

                "sequence_columns": SEQUENCE_COLUMNS,

                "labels": torch.tensor(
                    scaled["label"].to_numpy(),
                    dtype=torch.int8,
                ),

                "is_inductive_holdout": torch.tensor(
                    scaled["is_inductive_holdout"].to_numpy(),
                    dtype=torch.bool,
                ),
            },
            temporal_path,
        )

        outputs[split] = {
            "nodes_path": str(
                nodes_path.relative_to(ROOT)
            ),

            "temporal_path": str(
                temporal_path.relative_to(ROOT)
            ),

            "node_rows": int(len(scaled)),

            "sequence_shape": list(
                temporal.shape
            ),

            "cutoff_utc": metadata["cutoff_utc"],

            "transactions_in_history": (
                metadata["transactions_in_history"]
            ),

            "temporal_window_coverage": (
                metadata["temporal_window_coverage"]
            ),
        }

        logger.info(
            "[%s] nodes=%s | sequence=%s",
            split.upper(),
            f"{len(scaled):,}",
            temporal.shape,
        )

    report["splits"] = outputs

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== FEATURE ENGINEERING SUMMARY ===")

    for split, info in outputs.items():

        print(
            f"{split:5}: "
            f"{info['node_rows']:,} nodes | "
            f"sequence={info['sequence_shape']} | "
            f"history_tx={info['transactions_in_history']:,}"
        )

    print(f"Report: {report_path}")

    print(
        "Graph topology features are intentionally pending TV1-05."
    )


if __name__ == "__main__":
    main()