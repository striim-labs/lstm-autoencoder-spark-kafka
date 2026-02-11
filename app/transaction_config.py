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

    The synthetic transaction dataset has 30 days starting on a Monday.
    Split strategy:
    - Train: Days 0-17 (18 days, ~2.5 instances of each weekday)
    - Val: Days 18-21 (4 days, Thu-Sun for early stopping + error distribution)
    - Threshold: Days 22-24 (3 days, Mon-Wed for threshold calibration)
    - Test: Days 25-29 (5 days, Thu-Mon for evaluation)
    """
    sequence_length: int = SAMPLES_PER_WINDOW  # 24 (daily windows)
    train_days: int = 18           # Days 0-17
    val_days: int = 4              # Days 18-21
    threshold_days: int = 3        # Days 22-24
    # Remaining 5 days (25-29) for test


# Transaction-specific anomaly scorer parameter adjustments
# These override defaults from anomaly_scorer.py for the transaction domain
TRANSACTION_SCORER_DEFAULTS = {
    "hard_criterion_k": 3,         # 3/24 = 12.5% of day must be anomalous
    "threshold_percentile": 97.0,  # More calibration windows available (vs NYC taxi)
    "samples_per_hour": 1,         # Hourly aggregation (vs 30-min for taxi)
}
