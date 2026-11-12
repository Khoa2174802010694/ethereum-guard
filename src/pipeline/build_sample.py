"""
ETHEREUM GUARD - TV1-01
Build a deterministic, compact 1,000-address sample from the canonical
transaction Parquet files and account-level label registry.

Inputs:
  data/processed/transactions/*.parquet
  data/processed/address_registry/address_labels.parquet

Outputs:
  data/sample/transactions.parquet
  data/sample/address_labels.parquet
  data/sample/sample_manifest.json

Policy:
  - Include known phishing addresses first, then fill by transaction activity.
  - Keep unlabeled addresses as -1. Never manufacture normal labels.
  - Keep only transactions whose sender AND receiver are in the selected
    address set, yielding a self-contained induced transaction graph.
  - Null receivers are excluded from the induced graph but are counted.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TX_DIR = ROOT / "data" / "processed" / "transactions"
DEFAULT_LABELS = ROOT / "data" / "processed" / "address_registry" / "address_labels.parquet"
DEFAULT_OUTPUT = ROOT / "data" / "sample"
DEFAULT_SEED = 42
DEFAULT_N_ADDRESSES = 1_000

TX_COLUMNS = [
    "transaction_hash", "block_number", "transaction_index", "timestamp",
    "from_address", "to_address", "value_wei", "value_eth", "gas_used",
    "gas_price", "success", "chain_id",
]


def normalize_address(value) -> str | None:
    if value is None:
        return None
    value = str(value).strip().lower()
    if not value or value in {"none", "nan", "null"}:
        return None
    return value if value.startswith("0x") else "0x" + value


def load_transactions(tx_dir: Path) -> pd.DataFrame:
    paths = sorted(tx_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No canonical transaction Parquet files in {tx_dir}")
    frames = []
    for path in paths:
        table = pq.read_table(path, columns=TX_COLUMNS)
        frames.append(table.to_pandas())
    frame = pd.concat(frames, ignore_index=True)
    frame["from_address"] = frame["from_address"].map(normalize_address)
    frame["to_address"] = frame["to_address"].map(normalize_address)
    frame = frame.dropna(subset=["from_address"])
    return frame


def build_sample(
    tx_dir: Path,
    labels_path: Path,
    output_dir: Path,
    n_addresses: int = DEFAULT_N_ADDRESSES,
    seed: int = DEFAULT_SEED,
) -> dict:
    if n_addresses < 2:
        raise ValueError("n_addresses must be at least 2")
    if not labels_path.exists():
        raise FileNotFoundError(f"Label registry not found: {labels_path}")

    tx = load_transactions(tx_dir)
    labels = pq.read_table(labels_path).to_pandas()
    required = {"address", "label"}
    if not required.issubset(labels.columns):
        raise ValueError(f"Label registry must contain columns {sorted(required)}")

    labels["address"] = labels["address"].map(normalize_address)
    labels = labels.dropna(subset=["address"]).drop_duplicates("address")
    labels["label"] = labels["label"].astype("int8")

    # Rank by observed activity. Include every observed known phishing address
    # first, then use deterministic activity ranking to fill remaining slots.
    activity = pd.concat(
        [
            tx["from_address"].value_counts().rename("sent"),
            tx["to_address"].dropna().value_counts().rename("received"),
        ],
        axis=1,
    ).fillna(0)
    activity["total_tx"] = activity["sent"] + activity["received"]
    activity.index.name = "address"

    observed = set(activity.index)
    label_observed = labels[labels["address"].isin(observed)].copy()
    positives = sorted(
        label_observed.loc[label_observed["label"] == 1, "address"].tolist()
    )
    if len(positives) > n_addresses:
        rng = np.random.default_rng(seed)
        positives = sorted(rng.choice(positives, size=n_addresses, replace=False).tolist())

    selected = list(positives)
    remaining = n_addresses - len(selected)

    candidates = activity.drop(index=selected, errors="ignore")
    candidates = candidates.sort_values(
        ["total_tx", "sent", "received"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    selected.extend(candidates.head(remaining).index.tolist())

    if len(selected) < n_addresses:
        raise ValueError(
            f"Only {len(selected)} observed addresses available; requested {n_addresses}"
        )

    selected_set = set(selected)
    sample_labels = labels[labels["address"].isin(selected)].copy()
    label_map = dict(zip(sample_labels["address"], sample_labels["label"]))
    sample_labels = pd.DataFrame({
        "address": selected,
        "label": [int(label_map.get(address, -1)) for address in selected],
    })
    sample_labels["label_source"] = np.where(
        sample_labels["label"] == 1, "MDBChain", "unlabeled"
    )

    # Induced subgraph: both endpoints must be in the selected 1,000 addresses.
    nonnull_receiver = tx["to_address"].notna()
    keep = (
        tx["from_address"].isin(selected_set)
        & tx["to_address"].isin(selected_set)
        & nonnull_receiver
    )
    sample_tx = tx.loc[keep, TX_COLUMNS].copy()
    sample_tx = sample_tx.sort_values(
        ["timestamp", "block_number", "transaction_index"],
        kind="mergesort",
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    tx_path = output_dir / "transactions.parquet"
    labels_out = output_dir / "address_labels.parquet"
    manifest_path = output_dir / "sample_manifest.json"

    sample_tx.to_parquet(tx_path, index=False)
    sample_labels.to_parquet(labels_out, index=False)

    positive_count = int((sample_labels["label"] == 1).sum())
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "requested_addresses": n_addresses,
        "actual_addresses": len(sample_labels),
        "transactions_in_induced_graph": len(sample_tx),
        "known_phishing_addresses": positive_count,
        "verified_normal_addresses": int((sample_labels["label"] == 0).sum()),
        "unlabeled_addresses": int((sample_labels["label"] == -1).sum()),
        "transactions_with_null_receiver_excluded": int(
            (tx["from_address"].isin(selected_set) & tx["to_address"].isna()).sum()
        ),
        "selection_policy": (
            "All observed known-phishing addresses first (up to requested size), "
            "then highest-activity observed addresses; deterministic tie ordering."
        ),
        "edge_policy": "Keep only transactions with both endpoints in selected address set.",
        "label_policy": "1=known phishing, 0=verified normal only, -1=unlabeled.",
        "files": {
            "transactions": str(tx_path.relative_to(ROOT)),
            "address_labels": str(labels_out.relative_to(ROOT)),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Fail loudly if the sample violates the basic contract.
    assert len(sample_labels) == n_addresses
    assert sample_labels["address"].is_unique
    assert set(sample_labels["label"].unique()).issubset({-1, 0, 1})
    assert not (
        (sample_labels["label"] == 0)
        & (sample_labels["label_source"] == "unlabeled")
    ).any()

    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Ethereum Guard 1,000-address sample")
    parser.add_argument("--tx-dir", type=Path, default=DEFAULT_TX_DIR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-addresses", type=int, default=DEFAULT_N_ADDRESSES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    build_sample(args.tx_dir, args.labels, args.output_dir, args.n_addresses, args.seed)


if __name__ == "__main__":
    main()
