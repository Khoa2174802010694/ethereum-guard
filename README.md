# ETHEREUM GUARD

Ethereum account-level anomaly/phishing detection using transaction, temporal,
and graph representations.

## Current data sources

- **VNEGI10 Ethereum Blockchain Parquet**: transaction and block data.
- **MDBChain Phishing Detection Dataset**: known phishing addresses used for
  account-level positive labels.

Labels are account-level:

- `1`: known phishing
- `0`: verified normal only
- `-1`: unlabeled

An address absent from the phishing list is **not** automatically normal.

## Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

PyTorch installation can depend on the machine's Python, operating system,
and CUDA configuration. If the pinned wheel is unavailable, install a
compatible PyTorch build first, then install the remaining requirements.

## Pipeline order

Run commands from the repository root:

```powershell
python src/pipeline/ingest.py
python src/pipeline/build_label_registry.py
python src/pipeline/create_splits.py
python src/pipeline/features.py
python src/pipeline/graph_builder.py
```

Raw source files are intentionally excluded from Git. Place the VNEGI10
transaction and block Parquet files in `data/sample/`, and MDBChain CSV files
under `data/raw/mdbchain/`, following the filenames and layout expected by
the scripts.

## Build a compact 1,000-address sample

After ingestion and label-registry generation:

```powershell
python src/pipeline/build_sample.py
```

This creates:

- `data/sample/transactions.parquet`
- `data/sample/address_labels.parquet`
- `data/sample/sample_manifest.json`

The sample contains an induced transaction graph over 1,000 observed
addresses. Known phishing addresses are included first, then remaining
addresses are selected by observed activity. Unlabeled addresses remain `-1`.

**Important:** this sample is a compact integration fixture. It is not an
independent or statistically representative evaluation dataset.

## Generated outputs

Processed datasets, graph snapshots, model artifacts, scalers, and raw data
are local generated files and are not committed to Git. Small JSON/CSV reports
under `artifacts/results/` are tracked for reproducibility and documentation.
