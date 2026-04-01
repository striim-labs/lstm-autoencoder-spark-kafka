"""
Step 1: Data Ingestion
Load the NYC taxi demand dataset, parse it, and display summary statistics.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import ANOMALY_WINDOWS, SAMPLES_PER_DAY, SAMPLES_PER_WEEK

# Descriptions for each known anomaly
ANOMALY_DESCRIPTIONS = [
    "NYC Marathon",
    "Thanksgiving",
    "Christmas",
    "New Year's",
    "January Blizzard",
]


def main():
    data_path = PROJECT_ROOT / "data" / "nyc_taxi.csv"

    print("=" * 60)
    print("DATA INGESTION: NYC Taxi Demand Dataset")
    print("=" * 60)

    # Load and parse
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Summary statistics
    print(f"\nRecords: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    total_days = (df["timestamp"].max() - df["timestamp"].min()).days
    print(f"Duration: {total_days} days (~{total_days // 7} weeks)")
    print(f"Sampling: 30-minute intervals ({SAMPLES_PER_DAY}/day, {SAMPLES_PER_WEEK}/week)")

    print(f"\nValue statistics:")
    print(f"  Min:    {df['value'].min():,.0f}")
    print(f"  Max:    {df['value'].max():,.0f}")
    print(f"  Mean:   {df['value'].mean():,.0f}")
    print(f"  Median: {df['value'].median():,.0f}")
    print(f"  Std:    {df['value'].std():,.0f}")

    # Known anomaly windows
    print(f"\nKnown anomaly windows ({len(ANOMALY_WINDOWS)}):")
    for (start, end), desc in zip(ANOMALY_WINDOWS, ANOMALY_DESCRIPTIONS):
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        duration_hours = (end_ts - start_ts).total_seconds() / 3600
        print(f"  {desc}: {start} to {end} ({duration_hours:.0f}h)")

    # Sample data
    print(f"\nFirst 5 records:")
    print(df.head().to_string(index=False))

    print(f"\nLast 5 records:")
    print(df.tail().to_string(index=False))

    print("\n" + "=" * 60)
    print("Data ingestion complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
