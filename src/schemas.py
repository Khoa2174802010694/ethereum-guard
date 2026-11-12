
"""
Ethereum Guard shared Data Contract.

Source dataset:
    vnegi10/Ethereum_blockchain_parquet

Coordinate schema changes with TV2 and TV3.
"""

from typing import Final

# ============================================================
# PROJECT
# ============================================================

PROJECT_NAME: Final[str] = "ethereum-guard"
RANDOM_SEED: Final[int] = 42
SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")

# ============================================================
# ACCOUNT-LEVEL LABELS
# ============================================================

LABEL_NORMAL: Final[int] = 0
LABEL_ANOMALY: Final[int] = 1
LABEL_UNLABELED: Final[int] = -1

VALID_LABELS: Final[tuple[int, ...]] = (-1, 0, 1)

# 0 must only represent verified normal accounts.
# Missing from phishing lists does NOT imply normal.

# ============================================================
# RAW SOURCE: TRANSACTION PARQUET
# ============================================================

RAW_TX_COLUMNS: Final[tuple[str, ...]] = (
    "block_number",
    "transaction_index",
    "transaction_hash",
    "nonce",
    "from_address",
    "to_address",
    "value_binary",
    "value_string",
    "value_f64",
    "input",
    "gas_limit",
    "gas_used",
    "gas_price",
    "transaction_type",
    "max_priority_fee_per_gas",
    "max_fee_per_gas",
    "success",
    "n_input_bytes",
    "n_input_zero_bytes",
    "n_input_nonzero_bytes",
    "chain_id",
)

RAW_TX_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "block_number",
    "transaction_index",
    "transaction_hash",
    "from_address",
    "value_string",
    "value_binary",
    "gas_used",
    "gas_price",
    "success",
)

RAW_TX_OPTIONAL_COLUMNS: Final[tuple[str, ...]] = (
    "to_address",
    "nonce",
    "value_f64",
    "input",
    "gas_limit",
    "transaction_type",
    "max_priority_fee_per_gas",
    "max_fee_per_gas",
    "n_input_bytes",
    "n_input_zero_bytes",
    "n_input_nonzero_bytes",
    "chain_id",
)

# ============================================================
# RAW SOURCE: BLOCK PARQUET
# ============================================================

RAW_BLOCK_COLUMNS: Final[tuple[str, ...]] = (
    "block_hash",
    "author",
    "block_number",
    "gas_used",
    "extra_data",
    "timestamp",
    "base_fee_per_gas",
    "chain_id",
)

RAW_BLOCK_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "block_number",
    "timestamp",
)

# Join:
# transactions.block_number == blocks.block_number
# Canonical timestamp is sourced from blocks.timestamp.
# Timestamp unit: Unix seconds.

# ============================================================
# CANONICAL TRANSACTION SCHEMA
# ============================================================

CANONICAL_TX_COLUMNS: Final[tuple[str, ...]] = (
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
)

# value_wei:
# Exact decimal integer represented as string.
# Convert to Python int for arithmetic.
# Do not use float64 as the source of truth.
#
# value_eth:
# Derived numeric value = value_wei / 10**18.
#
# Addresses:
# Convert 20-byte binary values to lowercase 0x-prefixed hex.
# to_address may be null.

# ============================================================
# ACCOUNT / NODE METADATA
# ============================================================

NODE_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "node_id",
    "address",
    "is_contract",
    "first_seen_ts",
    "last_seen_ts",
    "eval_mask",
    "label",
)

# Labels are assigned at account level, not transaction level.

# ============================================================
# ENGINEERED FEATURES: EXACTLY 25
# Keep feature order stable.
# ============================================================

TABULAR_TX_COLS: Final[tuple[str, ...]] = (
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
)

GRAPH_TOPO_COLS: Final[tuple[str, ...]] = (
    "in_degree",
    "out_degree",
    "unique_in_neighbors",
    "unique_out_neighbors",
    "neighbor_repeat_ratio",
    "local_ego_density",
)

TEMPORAL_STAT_COLS: Final[tuple[str, ...]] = (
    "active_duration_hours",
    "tx_frequency_per_day",
    "mean_inter_tx_time",
    "std_inter_tx_time",
    "burstiness_index",
    "recent_volume_spike_ratio",
    "recent_tx_spike_ratio",
)

ALL_FEATURE_COLS: Final[tuple[str, ...]] = (
    TABULAR_TX_COLS
    + GRAPH_TOPO_COLS
    + TEMPORAL_STAT_COLS
)

assert len(TABULAR_TX_COLS) == 12
assert len(GRAPH_TOPO_COLS) == 6
assert len(TEMPORAL_STAT_COLS) == 7
assert len(ALL_FEATURE_COLS) == 25
assert len(set(ALL_FEATURE_COLS)) == 25

# ============================================================
# TEMPORAL SEQUENCE
# Shape: (N_nodes, 10_windows, 8_features)
# ============================================================

TEMPORAL_WINDOWS: Final[int] = 10

TEMPORAL_SEQUENCE_COLS: Final[tuple[str, ...]] = (
    "tx_in_count",
    "tx_out_count",
    "val_in_sum",
    "val_out_sum",
    "val_max",
    "unique_counterparties",
    "avg_gas_price",
    "mean_inter_arrival_time",
)

TEMPORAL_FEATURE_DIM: Final[int] = len(TEMPORAL_SEQUENCE_COLS)

# ============================================================
# EDGE ATTRIBUTES
# ============================================================

EDGE_FEATURE_COLS: Final[tuple[str, ...]] = (
    "tx_count_log",
    "total_value_log",
    "mean_value_log",
    "last_tx_age_norm",
    "avg_gas_log",
)

EDGE_FEATURE_DIM: Final[int] = len(EDGE_FEATURE_COLS)

# ============================================================
# TEMPORAL SPLIT
# ============================================================

TRAIN_RATIO: Final[float] = 0.70
VAL_RATIO: Final[float] = 0.15
TEST_RATIO: Final[float] = 0.15

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9
