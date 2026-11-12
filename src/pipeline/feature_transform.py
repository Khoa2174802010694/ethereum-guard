"""Shared feature transforms required by Ethereum Guard Master Plan."""
from __future__ import annotations
import numpy as np
import pandas as pd

LOG1P_COLS = {
    "total_eth_sent", "total_eth_received", "avg_eth_sent", "max_eth_sent",
    "avg_eth_received", "max_eth_received", "tx_sent_count", "tx_recv_count",
    "in_degree", "out_degree", "unique_in_neighbors", "unique_out_neighbors",
    "active_duration_hours", "tx_frequency_per_day", "mean_inter_tx_time",
    "std_inter_tx_time", "recent_volume_spike_ratio", "recent_tx_spike_ratio",
}
SIGNED_LOG_COLS = {"net_balance_diff"}
LOG_COLS = {"sent_recv_ratio"}


def transform_feature_frame(frame: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Return a copy with the Master Plan's per-column log transforms applied."""
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce").astype(float)
        if col in LOG1P_COLS:
            if (values.dropna() < 0).any():
                raise ValueError(f"{col} must be non-negative before log10(1+x)")
            out[col] = np.log10(1.0 + values.clip(lower=0.0))
        elif col in SIGNED_LOG_COLS:
            out[col] = np.sign(values) * np.log10(1.0 + np.abs(values))
        elif col in LOG_COLS:
            if (values.dropna() <= 0).any():
                raise ValueError(f"{col} must be positive before log10(x)")
            out[col] = np.log10(values)
    return out
