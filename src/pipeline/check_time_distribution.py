
from pathlib import Path
from datetime import datetime, timezone

import pyarrow.parquet as pq


DATA_DIR = Path("data/processed/transactions")


def format_timestamp(value):
    if value is None:
        return "NULL"

    if hasattr(value, "timestamp"):
        timestamp = value.timestamp()
    else:
        timestamp = int(value)

    return datetime.fromtimestamp(
        timestamp, timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")


def main():
    files = sorted(DATA_DIR.glob("*.parquet"))

    if not files:
        print("Không tìm thấy file Parquet!")
        return

    print("=" * 110)
    print(f"{'FILE':55} | {'ROWS':>12} | {'MIN TIME':22} | {'MAX TIME':22}")
    print("=" * 110)

    for file in files:
        parquet = pq.ParquetFile(file)

        min_time = None
        max_time = None
        row_count = 0

        for batch in parquet.iter_batches(
            batch_size=100_000,
            columns=["timestamp"]
        ):
            values = batch.column(0).to_pylist()
            values = [v for v in values if v is not None]

            if not values:
                continue

            current_min = min(values)
            current_max = max(values)

            if min_time is None or current_min < min_time:
                min_time = current_min

            if max_time is None or current_max > max_time:
                max_time = current_max

            row_count += len(values)

        print(
            f"{file.name:55} | "
            f"{row_count:12,} | "
            f"{format_timestamp(min_time):22} | "
            f"{format_timestamp(max_time):22}"
        )

    print("=" * 110)


if __name__ == "__main__":
    main()
