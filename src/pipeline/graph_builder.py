"""
ETHEREUM GUARD - TV1-05 Graph Construction

Builds directed, weighted-by-aggregated-transaction PyG snapshots for:
  train: train_graph_transactions.parquet (inductive holdout edges removed)
  val:   train_transactions + val_transactions (cumulative history)
  test:  train_transactions + val_transactions + test_transactions

Node order in each PyG Data.x is the corresponding features/{split}/nodes.parquet
row order. `node_id` preserves the global stable ID from that feature file.
Edges are directed sender -> receiver and aggregated by address pair.

Graph feature definitions used here:
- in_degree/out_degree: transaction multiplicity (sum of tx_count on incident edges)
- unique_in_neighbors/unique_out_neighbors: distinct counterparties
- neighbor_repeat_ratio: 1 - (unique_in + unique_out) / (sent_tx + received_tx + 1)
- local_ego_density: directed density of the induced 1-hop ego subgraph,
  including the center node, denominator k*(k-1), excluding self-loops.

Important:
- No verified-normal labels are invented. Unknown remains -1.
- is_contract remains unavailable/unknown; it is not inferred.
- Graph feature scaler is fit on Train only.
- The local ego density calculation is exact but can be CPU-intensive on a
  graph with millions of nodes. Use --skip-ego-density only for a diagnostic
  run; do not treat that run as final research output.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
import torch
import yaml
from sklearn.preprocessing import RobustScaler
from torch_geometric.data import Data
from numba import njit


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

GRAPH_FEATURES = [
    "in_degree",
    "out_degree",
    "unique_in_neighbors",
    "unique_out_neighbors",
    "neighbor_repeat_ratio",
    "local_ego_density",
]

EDGE_FEATURES = [
    "tx_count_log",
    "total_value_log",
    "mean_value_log",
    "last_tx_age_norm",
    "avg_gas_log",
]

TABULAR_FEATURES = [
    "total_eth_sent", "total_eth_received", "net_balance_diff",
    "avg_eth_sent", "max_eth_sent", "avg_eth_received",
    "max_eth_received", "tx_sent_count", "tx_recv_count",
    "sent_recv_ratio", "zero_val_tx_ratio", "failed_tx_ratio",
]
TEMPORAL_STAT_FEATURES = [
    "active_duration_hours", "tx_frequency_per_day",
    "mean_inter_tx_time", "std_inter_tx_time",
    "burstiness_index", "recent_volume_spike_ratio",
    "recent_tx_spike_ratio",
]
NODE_FEATURES = TABULAR_FEATURES + GRAPH_FEATURES + TEMPORAL_STAT_FEATURES

LOG = logging.getLogger("ethereum_guard.graph_builder")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy config: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data = raw.get("data", {})
    pipeline = raw.get("pipeline", {})
    output = raw.get("output", {})
    return {
        "processed_dir": data.get("processed_dir", "data/processed"),
        "batch_size": int(pipeline.get("batch_size", 100_000)),
        "results_dir": output.get("results_dir", "artifacts/results"),
    }


def read_transactions(paths: list[Path]) -> pd.DataFrame:
    columns = [
        "timestamp", "from_address", "to_address",
        "value_eth", "gas_price",
    ]
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy dữ liệu: {path}")
        LOG.info("Reading transactions: %s", path.name)
        frames.append(pq.read_table(path, columns=columns).to_pandas())
    if not frames:
        return pd.DataFrame(columns=columns)

    tx = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(tx["timestamp"], utc=True, errors="coerce")
    unit = getattr(ts.dtype, "unit", "ns")
    divisors = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}
    if unit not in divisors:
        raise ValueError(f"Timestamp resolution không hỗ trợ: {unit}")
    raw_ts = ts.astype("int64").to_numpy(dtype=np.int64)
    valid_ts = ts.notna().to_numpy()
    tx["timestamp"] = raw_ts // divisors[unit]
    tx = tx.loc[valid_ts].copy()

    tx["from_address"] = tx["from_address"].astype("string").str.lower()
    tx["to_address"] = tx["to_address"].astype("string").str.lower()
    tx["value_eth"] = pd.to_numeric(
        tx["value_eth"], errors="coerce"
    ).fillna(0.0).astype(np.float64)
    tx["gas_price"] = pd.to_numeric(
        tx["gas_price"], errors="coerce"
    ).fillna(0.0).astype(np.float64)
    tx = tx.dropna(subset=["from_address"])
    tx = tx[tx["from_address"].str.len() > 0]
    return tx


def aggregate_edges(
    tx: pd.DataFrame,
    cutoff_unix: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    tx = tx.loc[
        (tx["timestamp"] <= cutoff_unix) & tx["to_address"].notna()
    ].copy()
    tx = tx[tx["to_address"].str.len() > 0]
    tx = tx[tx["from_address"] != tx["to_address"]]

    if tx.empty:
        return pd.DataFrame(columns=[
            "from_address", "to_address", "tx_count", "total_value",
            "mean_value", "last_timestamp", "avg_gas_price",
        ]), {"input_transactions": 0, "valid_edges": 0}

    LOG.info("Aggregating %s directed transactions", f"{len(tx):,}")
    edges = (
        tx.groupby(["from_address", "to_address"], sort=False, observed=True)
        .agg(
            tx_count=("timestamp", "size"),
            total_value=("value_eth", "sum"),
            mean_value=("value_eth", "mean"),
            last_timestamp=("timestamp", "max"),
            avg_gas_price=("gas_price", "mean"),
        )
        .reset_index()
    )
    return edges, {
        "input_transactions": int(len(tx)),
        "valid_edges": int(len(edges)),
    }


def build_local_mapping(node_frame: pd.DataFrame) -> tuple[dict[str, int], np.ndarray]:
    addresses = node_frame["address"].astype(str).tolist()
    mapping = {address: idx for idx, address in enumerate(addresses)}
    stable_ids = node_frame["node_id"].to_numpy(dtype=np.int64)
    if len(mapping) != len(addresses):
        raise ValueError("Node table có địa chỉ trùng lặp.")
    if pd.Series(stable_ids).duplicated().any():
        raise ValueError("node_id bị trùng trong node table.")
    return mapping, stable_ids


@njit(cache=True)
def _accumulate_ego_edge_counts(
    indptr,
    indices,
    src,
    dst,
    counts,
):
    """
    Đếm số cạnh có hướng nằm trong ego graph của từng node.

    Sử dụng giao hai danh sách láng giềng đã sắp xếp
    trong CSR, được Numba biên dịch.
    """
    for edge_pos in range(src.shape[0]):
        u = src[edge_pos]
        w = dst[edge_pos]

        i = indptr[u]
        i_end = indptr[u + 1]

        j = indptr[w]
        j_end = indptr[w + 1]

        while i < i_end and j < j_end:
            left = indices[i]
            right = indices[j]

            if left == right:
                counts[left] += 1
                i += 1
                j += 1

            elif left < right:
                i += 1

            else:
                j += 1


def compute_local_ego_density_exact(
    src,
    dst,
    num_nodes,
    chunk_size=100_000,
):
    """
    Tính chính xác directed 1-hop ego density.

    Ego neighborhood:
    - Các node có cạnh đi vào node trung tâm.
    - Các node có cạnh đi ra từ node trung tâm.
    - Bao gồm chính node trung tâm.

    Công thức:
        density = induced_directed_edges / (k * (k - 1))

    Trong đó k là số node trong ego neighborhood.
    """

    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)

    if src.size != dst.size:
        raise ValueError("src và dst phải cùng kích thước.")

    if src.size == 0:
        return np.zeros(num_nodes, dtype=np.float32), {
            "ego_density_computed": True,
            "ego_nodes_processed": 0,
            "ego_edges_counted": 0,
            "algorithm": "numba_sparse_intersection",
        }

    # Tạo đồ thị vô hướng chỉ để xác định membership
    # của 1-hop neighborhood.
    # Cạnh gốc vẫn giữ nguyên hướng khi tính density.

    rows = np.concatenate((src, dst))
    cols = np.concatenate((dst, src))

    undirected = sp.csr_matrix(
        (
            np.ones(len(rows), dtype=np.uint8),
            (rows, cols),
        ),
        shape=(num_nodes, num_nodes),
        dtype=np.uint8,
    )

    undirected.data[:] = 1
    undirected.setdiag(1)
    undirected.data[:] = 1
    undirected.eliminate_zeros()
    undirected.sort_indices()

    indptr = np.asarray(
        undirected.indptr,
        dtype=np.int64,
    )

    indices = np.asarray(
        undirected.indices,
        dtype=np.int64,
    )

    counts = np.zeros(
        num_nodes,
        dtype=np.int64,
    )

    total_edges = len(src)

    # Xử lý theo batch để kiểm soát bộ nhớ.
    for begin in range(0, total_edges, chunk_size):

        finish = min(
            begin + chunk_size,
            total_edges,
        )

        _accumulate_ego_edge_counts(
            indptr,
            indices,
            src[begin:finish],
            dst[begin:finish],
            counts,
        )

        LOG.info(
            "local_ego_density: processed %s/%s edges",
            f"{finish:,}",
            f"{total_edges:,}",
        )

    # Kích thước closed neighborhood của từng node.
    ego_sizes = np.diff(indptr).astype(np.float64)

    denominators = ego_sizes * (ego_sizes - 1.0)

    density = np.divide(
        counts,
        denominators,
        out=np.zeros(
            num_nodes,
            dtype=np.float64,
        ),
        where=denominators > 0,
    )

    density = np.clip(
        density,
        0.0,
        1.0,
    ).astype(np.float32)

    stats = {
        "ego_density_computed": True,
        "ego_nodes_processed": int(
            np.count_nonzero(ego_sizes > 1)
        ),
        "ego_edges_counted": int(counts.sum()),
        "algorithm": "numba_sparse_intersection",
        "definition": (
            "Directed induced 1-hop ego density; "
            "closed undirected neighborhood membership; "
            "original directed edges as numerator; "
            "denominator k*(k-1); self-loops excluded."
        ),
        "chunk_size": int(chunk_size),
    }

    del undirected

    return density, stats


def graph_topology_features(
    edges,
    address_to_local,
    num_nodes,
    compute_ego_density,
):
    """Tính 6 Graph Topology Features."""

    src = edges["src"].to_numpy(
        dtype=np.int64,
        copy=True,
    )

    dst = edges["dst"].to_numpy(
        dtype=np.int64,
        copy=True,
    )

    tx_count = edges["tx_count"].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    # Degree tính theo số lượng giao dịch.
    out_degree = np.bincount(
        src,
        weights=tx_count,
        minlength=num_nodes,
    )

    in_degree = np.bincount(
        dst,
        weights=tx_count,
        minlength=num_nodes,
    )

    # Số lượng láng giềng khác nhau.
    unique_out = np.bincount(
        src,
        minlength=num_nodes,
    ).astype(np.float64)

    unique_in = np.bincount(
        dst,
        minlength=num_nodes,
    ).astype(np.float64)

    # Neighbor repeat ratio.
    total_tx = out_degree + in_degree

    unique_total = unique_out + unique_in

    repeat_ratio = (
        1.0 - unique_total / (total_tx + 1.0)
    )

    repeat_ratio = np.clip(
        repeat_ratio,
        0.0,
        1.0,
    )

    # Local ego density.
    density = np.zeros(
        num_nodes,
        dtype=np.float32,
    )

    ego_stats = {
        "ego_density_computed": False,
        "ego_nodes_processed": 0,
        "algorithm": "skipped_diagnostic",
    }

    if compute_ego_density:

        density, ego_stats = compute_local_ego_density_exact(
            src=src,
            dst=dst,
            num_nodes=num_nodes,
        )

    features = np.column_stack((
        in_degree,
        out_degree,
        unique_in,
        unique_out,
        repeat_ratio,
        density,
    )).astype(np.float64)

    stats = {
        **ego_stats,
        "degree_definition": (
            "in_degree/out_degree count transaction multiplicity; "
            "unique_in/out_neighbors count distinct counterparties."
        ),
    }

    return features, stats



def build_pyg_snapshot(
    split: str,
    tx_paths: list[Path],
    feature_dir: Path,
    output_dir: Path,
    cutoff_unix: int,
    compute_ego_density: bool,
) -> dict[str, Any]:
    nodes_path = feature_dir / split / "nodes.parquet"
    temporal_path = feature_dir / split / "temporal.pt"
    if not nodes_path.exists() or not temporal_path.exists():
        raise FileNotFoundError(
            f"Thiếu Feature Engineering output của {split}: "
            f"{nodes_path} hoặc {temporal_path}"
        )

    LOG.info("[%s] Loading node features", split.upper())
    node_frame = pd.read_parquet(nodes_path)
    node_frame["address"] = node_frame["address"].astype(str).str.lower()
    mapping, stable_ids = build_local_mapping(node_frame)
    num_nodes = len(node_frame)

    tx = read_transactions(tx_paths)
    edges, edge_stats = aggregate_edges(tx, cutoff_unix)
    del tx

    # Map address endpoints to local node-table row indices.
    edges["src"] = edges["from_address"].map(mapping)
    edges["dst"] = edges["to_address"].map(mapping)
    edges = edges.dropna(subset=["src", "dst"]).copy()
    edges["src"] = edges["src"].astype(np.int64)
    edges["dst"] = edges["dst"].astype(np.int64)

    # Edge aggregation already guarantees one directed edge per pair.
    if edges.duplicated(["src", "dst"]).any():
        raise AssertionError("Duplicate directed edges remain after aggregation.")

    topology, topology_stats = graph_topology_features(
        edges=edges,
        address_to_local=mapping,
        num_nodes=num_nodes,
        compute_ego_density=compute_ego_density,
    )

    # Edge attributes in schema order.
    tx_count = edges["tx_count"].to_numpy(dtype=np.float64)
    total_value = edges["total_value"].to_numpy(dtype=np.float64)
    mean_value = edges["mean_value"].to_numpy(dtype=np.float64)
    last_ts = edges["last_timestamp"].to_numpy(dtype=np.float64)
    avg_gas = edges["avg_gas_price"].to_numpy(dtype=np.float64)

    age_seconds = np.maximum(float(cutoff_unix) - last_ts, 0.0)
    age_norm = age_seconds / max(1.0, float(cutoff_unix - int(edges["last_timestamp"].min()))) if len(edges) else age_seconds

    edge_attr = np.column_stack((
        np.log1p(np.maximum(tx_count, 0.0)),
        np.log1p(np.maximum(total_value, 0.0)),
        np.log1p(np.maximum(mean_value, 0.0)),
        age_norm,
        np.log1p(np.maximum(avg_gas, 0.0)),
    )).astype(np.float32)

    # Update the node feature parquet with graph features.
    for col_idx, col in enumerate(GRAPH_FEATURES):
        node_frame[col] = topology[:, col_idx]

    output_dir.mkdir(parents=True, exist_ok=True)
    updated_nodes_path = output_dir / "nodes.parquet"
    node_frame.to_parquet(
        updated_nodes_path, index=False, compression="zstd"
    )

    # Graph-feature scaler is fitted later, once all split topology is built.
    edge_index = torch.from_numpy(np.vstack((
        edges["src"].to_numpy(dtype=np.int64),
        edges["dst"].to_numpy(dtype=np.int64),
    ))).long()
    edge_attr_tensor = torch.from_numpy(edge_attr)

    # Use the exact 25-column contract order, not incidental Parquet order.
    missing_features = [c for c in NODE_FEATURES if c not in node_frame.columns]
    if missing_features:
        raise ValueError(f"Thiếu node features ({split}): {missing_features}")
    feature_frame = (
        node_frame[NODE_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    feature_array = feature_frame.to_numpy(dtype=np.float32, copy=True)
    feature_tensor = torch.from_numpy(feature_array)

    temporal_payload = torch.load(
        temporal_path, map_location="cpu", weights_only=False
    )

    # Assert temporal sequence row order aligns with the node Parquet.
    payload_addresses = temporal_payload.get("addresses")
    if payload_addresses is not None:
        if list(payload_addresses) != node_frame["address"].astype(str).tolist():
            raise ValueError(
                f"Temporal address order không khớp nodes.parquet ({split})."
            )

    labels = node_frame["label"].to_numpy(dtype=np.int64, copy=True)
    if "eval_mask" in node_frame.columns:
        eval_mask = node_frame["eval_mask"].fillna(False).to_numpy(dtype=bool)
    else:
        eval_mask = labels >= 0

    temporal_sequence = temporal_payload["sequence"].float()
    if temporal_sequence.shape[0] != num_nodes:
        raise ValueError(
            f"Temporal sequence/node count mismatch ({split}): "
            f"{temporal_sequence.shape[0]} != {num_nodes}"
        )

    data = Data(
        x=feature_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr_tensor,
        y=torch.from_numpy(labels),
        num_nodes=num_nodes,
    )
    data.node_id = torch.from_numpy(stable_ids.copy()).long()
    data.addresses = node_frame["address"].astype(str).tolist()
    data.is_inductive_holdout = torch.from_numpy(
        node_frame["is_inductive_holdout"].fillna(False).to_numpy(dtype=bool)
        if "is_inductive_holdout" in node_frame.columns
        else np.zeros(num_nodes, dtype=bool)
    )
    data.eval_mask = torch.from_numpy(eval_mask.copy())
    data.x_tab = torch.from_numpy(feature_array[:, :len(TABULAR_FEATURES)].copy())
    graph_start = len(TABULAR_FEATURES)
    graph_end = graph_start + len(GRAPH_FEATURES)
    data.x_topo = torch.from_numpy(feature_array[:, graph_start:graph_end].copy())
    data.x_temp_stat = torch.from_numpy(feature_array[:, graph_end:].copy())
    data.temporal_seq = temporal_sequence
    data.temporal_sequence = data.temporal_seq
    data.edge_timestamp = torch.from_numpy(
        edges["last_timestamp"].to_numpy(dtype=np.int64, copy=True)
    )
    data.feature_columns = NODE_FEATURES
    data.edge_feature_columns = EDGE_FEATURES
    data.cutoff_unix = int(cutoff_unix)

    graph_path = output_dir / "graph.pt"
    snapshot_path = output_dir / f"snapshot_{split}.pt"
    torch.save(data, graph_path)
    torch.save(data, snapshot_path)

    report = {
        "split": split,
        "nodes": int(num_nodes),
        "directed_edges": int(data.edge_index.shape[1]),
        "edge_attributes_shape": list(data.edge_attr.shape),
        "node_features_shape": list(data.x.shape),
        "temporal_sequence_shape": list(data.temporal_seq.shape),
        "eval_nodes": int(data.eval_mask.sum().item()),
        "x_tab_shape": list(data.x_tab.shape),
        "x_topo_shape": list(data.x_topo.shape),
        "x_temp_stat_shape": list(data.x_temp_stat.shape),
        "edge_timestamp_shape": list(data.edge_timestamp.shape),
        "stable_node_ids_unique": bool(
            len(np.unique(stable_ids)) == len(stable_ids)
        ),
        "input_transactions": edge_stats["input_transactions"],
        "aggregated_directed_edges": edge_stats["valid_edges"],
        "edges_after_node_mapping": int(len(edges)),
        "cutoff_utc": datetime.fromtimestamp(
            cutoff_unix, timezone.utc
        ).isoformat(),
        "topology": topology_stats,
        "graph_path": str(graph_path.relative_to(ROOT)),
        "snapshot_path": str(snapshot_path.relative_to(ROOT)),
        "nodes_path": str(updated_nodes_path.relative_to(ROOT)),
    }
    LOG.info(
        "[%s] nodes=%s | directed_edges=%s | x=%s | edge_attr=%s",
        split.upper(), f"{num_nodes:,}", f"{data.edge_index.shape[1]:,}",
        tuple(data.x.shape), tuple(data.edge_attr.shape),
    )
    return report


def fit_graph_scaler(
    train_nodes_path: Path,
    split_nodes_paths: dict[str, Path],
    scaler_path: Path,
) -> dict[str, Any]:
    train = pd.read_parquet(train_nodes_path)
    values = (
        train[GRAPH_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    scaler = RobustScaler()
    scaler.fit(values)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"scaler": scaler, "columns": GRAPH_FEATURES},
        scaler_path,
    )

    # Apply the Train-fitted scaler to each split and update graph Data.x.
    for split, nodes_path in split_nodes_paths.items():
        frame = pd.read_parquet(nodes_path)
        raw = (
            frame[GRAPH_FEATURES]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        frame.loc[:, GRAPH_FEATURES] = scaler.transform(raw)
        frame.to_parquet(nodes_path, index=False, compression="zstd")

        graph_path = nodes_path.parent / "graph.pt"
        data = torch.load(graph_path, map_location="cpu", weights_only=False)
        # Graph x uses the exact 25-feature schema order.
        scaled_features = (
            frame[NODE_FEATURES]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32, copy=True)
        )
        data.x = torch.from_numpy(scaled_features)
        data.x_tab = torch.from_numpy(
            scaled_features[:, :len(TABULAR_FEATURES)].copy()
        )
        graph_start = len(TABULAR_FEATURES)
        graph_end = graph_start + len(GRAPH_FEATURES)
        data.x_topo = torch.from_numpy(
            scaled_features[:, graph_start:graph_end].copy()
        )
        data.x_temp_stat = torch.from_numpy(
            scaled_features[:, graph_end:].copy()
        )

        torch.save(data, graph_path)
        torch.save(data, nodes_path.parent / f"snapshot_{split}.pt")

    return {
        "path": str(scaler_path.relative_to(ROOT)),
        "fit_split": "train",
        "method": "RobustScaler",
        "columns": GRAPH_FEATURES,
    }


def _self_test_ego_density():
    """So sánh thuật toán Numba với phép tính sparse trực tiếp."""

    n = 7

    src = np.asarray(
        [0, 0, 1, 1, 2, 3, 4, 5, 6],
        dtype=np.int64,
    )

    dst = np.asarray(
        [1, 2, 2, 3, 3, 4, 5, 6, 0],
        dtype=np.int64,
    )

    actual, _ = compute_local_ego_density_exact(
        src,
        dst,
        n,
        chunk_size=3,
    )

    adjacency = sp.csr_matrix(
        (
            np.ones(src.size, dtype=np.uint8),
            (src, dst),
        ),
        shape=(n, n),
    )

    adjacency.data[:] = 1

    undirected = adjacency.maximum(
        adjacency.T
    ).tocsr()

    undirected.setdiag(1)
    undirected.data[:] = 1
    undirected.sort_indices()

    expected = np.zeros(
        n,
        dtype=np.float64,
    )

    for node in range(n):

        ego = undirected.indices[
            undirected.indptr[node]:
            undirected.indptr[node + 1]
        ]

        k = len(ego)

        if k > 1:
            expected[node] = (
                adjacency[ego, :][:, ego].nnz
                / (k * (k - 1))
            )

    if not np.allclose(
        actual,
        expected,
        atol=1e-7,
    ):
        raise AssertionError(
            f"Self-test failed: {actual} != {expected}"
        )

    print("Ego density self-test: PASS")
    print("Optimized:", np.round(actual, 6).tolist())
    print("Reference:", np.round(expected, 6).tolist())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ethereum Guard TV1-05 directed graph construction"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--skip-ego-density",
        action="store_true",
        help="Diagnostic only: set local_ego_density to zero; not final output.",
    )

    parser.add_argument(
        "--self-test-ego-density",
        action="store_true",
        help="Test exact ego density on a tiny graph.",
    )

    args = parser.parse_args()

    if args.self_test_ego_density:
        _self_test_ego_density()
        return


    cfg = load_config(args.config.resolve())
    processed = ROOT / cfg["processed_dir"]
    split_dir = processed / "splits"
    feature_dir = processed / "features"
    graph_root = processed / "graphs"
    results_dir = ROOT / cfg["results_dir"]

    split_report = json.loads(
        (results_dir / "split_report.json").read_text(encoding="utf-8")
    )
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

    reports = {}
    node_paths = {}

    for split in ("train", "val", "test"):
        out_dir = graph_root / split
        reports[split] = build_pyg_snapshot(
            split=split,
            tx_paths=sources[split],
            feature_dir=feature_dir,
            output_dir=out_dir,
            cutoff_unix=cutoffs[split],
            compute_ego_density=not args.skip_ego_density,
        )
        node_paths[split] = out_dir / "nodes.parquet"

    scaler_path = ROOT / "artifacts" / "scalers" / "graph_feature_scaler.joblib"
    scaler_info = fit_graph_scaler(
        train_nodes_path=node_paths["train"],
        split_nodes_paths=node_paths,
        scaler_path=scaler_path,
    )

    report = {
        "status": "success",
        "method": "directed_transaction_graph",
        "node_features": 25,
        "node_feature_groups": {
            "x_tab": TABULAR_FEATURES,
            "x_topo": GRAPH_FEATURES,
            "x_temp_stat": TEMPORAL_STAT_FEATURES,
        },
        "snapshot_fields": [
            "x", "x_tab", "x_topo", "x_temp_stat", "temporal_seq",
            "edge_index", "edge_attr", "edge_timestamp", "y", "eval_mask",
            "node_id", "addresses",
        ],
        "edge_features": EDGE_FEATURES,
        "graph_features": GRAPH_FEATURES,
        "edge_direction": "from_address -> to_address",
        "edge_aggregation": "one directed edge per ordered address pair",
        "degree_definition": (
            "in_degree/out_degree are transaction multiplicities; "
            "unique_in/out_neighbors count distinct counterparties."
        ),
        "neighbor_repeat_ratio_definition": (
            "1 - (unique_in_neighbors + unique_out_neighbors) / "
            "(in_degree + out_degree + 1), clipped to [0,1]."
        ),
        "local_ego_density_definition": (
            "Directed induced 1-hop ego density including center node; "
            "m/(k*(k-1)); self-loops excluded."
        ),
        "inductive_policy": (
            "Train uses train_graph_transactions only. Validation and Test "
            "use cumulative chronological transaction partitions."
        ),
        "graph_feature_scaler": scaler_info,
        "splits": reports,
        "limitations": [
            "Exact local ego density can be CPU-intensive for million-node graphs.",
            "The source data are sparse temporal samples, not continuous chain history.",
            "is_contract is not inferred because the source does not provide it.",
            "Validation/Test cumulative histories include prior transactions for held-out addresses; "
            "the evaluation protocol must explicitly permit this transductive history or be tightened."
        ],
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "graph_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\n=== GRAPH CONSTRUCTION SUMMARY ===")
    for split, item in reports.items():
        print(
            f"{split:5}: nodes={item['nodes']:,} | "
            f"edges={item['directed_edges']:,} | "
            f"x={item['node_features_shape']} | "
            f"edge_attr={item['edge_attributes_shape']}"
        )
    print(f"Report: {report_path}")
    if args.skip_ego_density:
        print("WARNING: local_ego_density skipped; diagnostic output only.")


if __name__ == "__main__":
    main()
