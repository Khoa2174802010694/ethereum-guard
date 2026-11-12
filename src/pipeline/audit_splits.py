"""
ETHEREUM GUARD - TV1-03 Leakage Audit
Run from repository root:
    python src/pipeline/audit_splits.py

Checks:
1. Train/validation/test are chronological, disjoint partitions.
2. train_graph_transactions contains no edge touching a holdout address.
3. Train graph transactions do not exceed T_train.
4. Each PyG snapshot edge_timestamp does not exceed its own cutoff.
5. Address split manifest has consistent holdout flags.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
SPLIT_DIR = PROCESSED / "splits"
GRAPH_DIR = PROCESSED / "graphs"
REPORT_PATH = ROOT / "artifacts" / "results" / "split_report.json"


def read_ts_bounds(path: Path) -> tuple[int, int, int]:
    pf = pq.ParquetFile(path)
    minimum = None
    maximum = None
    count = 0
    for batch in pf.iter_batches(columns=["timestamp"], batch_size=200_000):
        values = pd.to_datetime(batch.column(0).to_pandas(), utc=True)
        if values.isna().any():
            raise AssertionError(f"{path.name}: contains null timestamps")
        lo = int(values.min().timestamp())
        hi = int(values.max().timestamp())
        minimum = lo if minimum is None else min(minimum, lo)
        maximum = hi if maximum is None else max(maximum, hi)
        count += len(values)
    if count == 0:
        raise AssertionError(f"{path.name}: empty partition")
    return minimum, maximum, count


def check_train_graph_holdout(path: Path, holdout: set[str]) -> tuple[int, int]:
    pf = pq.ParquetFile(path)
    total = 0
    violations = 0
    for batch in pf.iter_batches(
        columns=["from_address", "to_address"], batch_size=200_000
    ):
        frame = batch.to_pandas()
        sender = frame["from_address"].astype("string")
        receiver = frame["to_address"].astype("string")
        bad = sender.isin(holdout) | receiver.isin(holdout)
        violations += int(bad.sum())
        total += len(frame)
    return total, violations


def load_graph(path: Path, cutoff: int, split: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing graph snapshot: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)

    edge_ts = getattr(data, "edge_timestamp", None)
    if edge_ts is None:
        raise AssertionError(f"{split}: edge_timestamp missing")
    if edge_ts.numel() and int(edge_ts.max()) > cutoff:
        raise AssertionError(
            f"{split}: future edge timestamp {int(edge_ts.max())} > cutoff {cutoff}"
        )
    if data.edge_index.shape[1] != edge_ts.numel():
        raise AssertionError(
            f"{split}: edge_index count {data.edge_index.shape[1]} "
            f"!= edge_timestamp count {edge_ts.numel()}"
        )
    return {
        "nodes": int(data.num_nodes),
        "edges": int(data.edge_index.shape[1]),
        "max_edge_timestamp": int(edge_ts.max()) if edge_ts.numel() else None,
    }


def main() -> None:
    if not REPORT_PATH.exists():
        raise FileNotFoundError(REPORT_PATH)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    cutoffs = {
        "train": int(report["cutoffs"]["T_train_unix"]),
        "val": int(report["cutoffs"]["T_val_unix"]),
        "test": int(report["cutoffs"]["T_test_unix"]),
    }

    partition_files = {
        "train": SPLIT_DIR / "train_transactions.parquet",
        "val": SPLIT_DIR / "val_transactions.parquet",
        "test": SPLIT_DIR / "test_transactions.parquet",
    }

    bounds = {}
    for split, path in partition_files.items():
        if not path.exists():
            raise FileNotFoundError(path)
        bounds[split] = read_ts_bounds(path)

    # Strict chronological separation of period partitions.
    assert bounds["train"][1] <= cutoffs["train"]
    assert bounds["val"][0] > cutoffs["train"]
    assert bounds["val"][1] <= cutoffs["val"]
    assert bounds["test"][0] > cutoffs["val"]
    assert bounds["test"][1] <= cutoffs["test"]
    assert cutoffs["train"] < cutoffs["val"] < cutoffs["test"]

    # Confirm that the filtered training graph has no holdout endpoints.
    address_path = SPLIT_DIR / "address_splits.parquet"
    addresses = pd.read_parquet(address_path)
    required = {"address", "is_inductive_holdout", "seen_in_train_graph"}
    if not required.issubset(addresses.columns):
        raise AssertionError(f"address_splits missing columns: {required - set(addresses.columns)}")

    holdout = set(
        addresses.loc[addresses["is_inductive_holdout"].astype(bool), "address"]
        .astype(str)
        .str.lower()
    )
    declared_train_graph = set(
        addresses.loc[addresses["seen_in_train_graph"].astype(bool), "address"]
        .astype(str)
        .str.lower()
    )
    assert not (holdout & declared_train_graph)

    train_graph_path = SPLIT_DIR / "train_graph_transactions.parquet"
    train_graph_rows, holdout_edge_violations = check_train_graph_holdout(
        train_graph_path, holdout
    )
    assert holdout_edge_violations == 0, (
        f"Train graph contains {holdout_edge_violations} edges touching holdout nodes"
    )

    # Train graph timestamps must be bounded by T_train.
    train_graph_min, train_graph_max, _ = read_ts_bounds(train_graph_path)
    assert train_graph_max <= cutoffs["train"]

    graph_reports = {}
    for split in ("train", "val", "test"):
        graph_path = GRAPH_DIR / split / f"snapshot_{split}.pt"
        graph_reports[split] = load_graph(graph_path, cutoffs[split], split)

    result = {
        "status": "PASS",
        "partition_bounds_unix": {
            name: {"min": lo, "max": hi, "rows": rows}
            for name, (lo, hi, rows) in bounds.items()
        },
        "cutoffs_unix": cutoffs,
        "holdout_addresses": len(holdout),
        "train_graph_rows": train_graph_rows,
        "train_graph_holdout_edge_violations": holdout_edge_violations,
        "train_graph_max_timestamp": train_graph_max,
        "snapshots": graph_reports,
        "notes": [
            "This audit verifies chronology, train holdout-edge exclusion, and snapshot edge timestamp cutoffs.",
            "It does not prove that every engineered node feature is free of future information; feature-history windows require a separate audit.",
            "Validation/Test snapshots intentionally use cumulative chronological history under the current protocol.",
        ],
    }

    out = ROOT / "artifacts" / "results" / "leakage_audit_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    main()
