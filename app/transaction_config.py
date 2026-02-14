"""
Configuration for Transaction Frequency Anomaly Detection

Defines configuration dataclasses and constants for processing
synthetic transaction data into hourly count time series for
4 independent LSTM-AE models (one per network/type combination).
"""

from dataclasses import dataclass
from typing import Tuple, List

# Constants for transaction data
SAMPLES_PER_DAY = 24  # Hourly aggregation
SAMPLES_PER_WINDOW = SAMPLES_PER_DAY  # Daily windows (24 hours)

# Network/transaction type combinations (4 independent models)
COMBO_KEYS: List[Tuple[str, str]] = [
    ("Accel", "CMP"),
    ("Accel", "no-pin"),
    ("Star", "CMP"),
    ("Star", "no-pin"),
]


@dataclass
class TransactionPreprocessorConfig:
    """
    Configuration for transaction data preprocessing.

    Supports two split strategies:

    1. CSV-based splits (when 'split' column exists):
       Uses the split column values directly ("train", "val", "test")
       This is the preferred method for the 60-day dataset.

    2. Day-based splits (legacy, for 30-day dataset):
       - Train: Days 0-17 (18 days)
       - Val: Days 18-21 (4 days)
       - Threshold: Days 22-24 (3 days)
       - Test: Days 25-29 (5 days)
    """
    sequence_length: int = SAMPLES_PER_WINDOW  # 24 (daily windows)
    train_days: int = 18           # Days 0-17 (legacy)
    val_days: int = 4              # Days 18-21 (legacy)
    threshold_days: int = 3        # Days 22-24 (legacy)
    # Remaining days for test (legacy)

    # Whether to use split column from CSV if available
    use_csv_splits: bool = True


# Transaction-specific anomaly scorer parameter adjustments
# These override defaults from anomaly_scorer.py for the transaction domain
TRANSACTION_SCORER_DEFAULTS = {
    "hard_criterion_k": 3,         # 3/24 = 12.5% of day must be anomalous
    "threshold_percentile": 97.0,  # More calibration windows available (vs NYC taxi)
    "samples_per_hour": 1,         # Hourly aggregation (vs 30-min for taxi)
}
