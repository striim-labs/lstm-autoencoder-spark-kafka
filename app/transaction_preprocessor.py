"""
Transaction Preprocessor for LSTM Encoder-Decoder Anomaly Detection

Handles loading, aggregating, segmenting, normalizing, and splitting
synthetic transaction data for 4 independent LSTM autoencoder models
(one per network/transaction type combination).

Key differences from NYCTaxiPreprocessor:
- Aggregates raw event transactions to hourly counts (vs pre-aggregated 30-min data)
- Handles 4 combo groups independently (vs single time series)
- Uses daily windows (24 hours) vs weekly windows (336 samples)
- Different split: 18/4/3/5 days for train/val/threshold/test
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

from app.transaction_config import (
    TransactionPreprocessorConfig,
    COMBO_KEYS,
    SAMPLES_PER_DAY,
    SAMPLES_PER_WINDOW,
)

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for daily time series sequences.

    Each sample is a complete day of hourly transaction counts with DoW features,
    shaped as (sequence_length, num_features) for LSTM input.

    With DoW conditioning: (num_days, 24, 3) where features are:
        - Channel 0: normalized transaction count
        - Channel 1: sin(2π * day_of_week / 7)
        - Channel 2: cos(2π * day_of_week / 7)
    """

    def __init__(self, sequences: np.ndarray):
        """
        Args:
            sequences: Array of shape (num_days, 24) or (num_days, 24, num_features)
        """
        self.sequences = torch.FloatTensor(sequences)
        # Add feature dimension if needed: (num_days, seq_len) -> (num_days, seq_len, 1)
        if self.sequences.ndim == 2:
            self.sequences = self.sequences.unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]


class TransactionPreprocessor:
    """
    Preprocessor for synthetic transaction data.

    Handles the complete preprocessing pipeline for each combo:
    1. Load CSV and aggregate to hourly counts per combo
    2. Segment into daily windows (24 hours each)
    3. Create train/val/threshold/test splits
    4. Normalize using StandardScaler (fit on train only)
    5. Generate PyTorch DataLoaders

    Each combo (network_type, transaction_type) gets its own:
    - Hourly count time series
    - Scaler (fitted on that combo's training data)
    - Set of DataLoaders
    """

    def __init__(self, config: Optional[TransactionPreprocessorConfig] = None):
        self.config = config or TransactionPreprocessorConfig()

        # Per-combo storage
        self.scalers: Dict[Tuple[str, str], StandardScaler] = {}
        self.combo_hourly: Dict[Tuple[str, str], pd.DataFrame] = {}
        self.combo_windows: Dict[Tuple[str, str], np.ndarray] = {}
        self.combo_window_info: Dict[Tuple[str, str], List[Dict]] = {}
        self.combo_splits: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

        # Raw data
        self.raw_df: Optional[pd.DataFrame] = None
        self.data_start: Optional[pd.Timestamp] = None
        self.data_end: Optional[pd.Timestamp] = None

        logger.info(f"Initialized TransactionPreprocessor with config: {self.config}")

    def load_and_aggregate(self, filepath: str) -> Dict[Tuple[str, str], pd.DataFrame]:
        """
        Load transaction CSV and aggregate to hourly counts per combo.

        The raw CSV has columns: timestamp, network_type, transaction_type, amount, is_anomaly
        This aggregates to hourly transaction counts for each (network_type, transaction_type) combo.

        Args:
            filepath: Path to synthetic_transactions.csv

        Returns:
            Dict mapping combo tuple to DataFrame with hourly counts
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        logger.info(f"Loading transaction data from {filepath}...")

        # Load with timestamp parsing
        df = pd.read_csv(filepath, parse_dates=["timestamp"])
        self.raw_df = df

        logger.info(f"Loaded {len(df):,} transactions")

        # Store date range
        self.data_start = df["timestamp"].min()
        self.data_end = df["timestamp"].max()
        logger.info(f"Date range: {self.data_start} to {self.data_end}")

        # Bin timestamps to hour
        df["hour_bucket"] = df["timestamp"].dt.floor("h")

        # Aggregate: count transactions per hour per combo
        counts = df.groupby(
            ["hour_bucket", "network_type", "transaction_type"]
        ).size().reset_index(name="count")

        # Create complete hourly index for the full date range
        full_hours = pd.date_range(
            start=self.data_start.floor("h"),
            end=self.data_end.floor("h"),
            freq="h"
        )
        expected_hours = len(full_hours)
        logger.info(f"Expected hours: {expected_hours} ({expected_hours // 24} days)")

        # Process each combo
        for combo in COMBO_KEYS:
            network_type, transaction_type = combo

            # Filter to this combo
            combo_counts = counts[
                (counts["network_type"] == network_type) &
                (counts["transaction_type"] == transaction_type)
            ][["hour_bucket", "count"]].copy()

            # Reindex to full hour range, fill missing with 0
            combo_df = pd.DataFrame({"hour_bucket": full_hours})
            combo_df = combo_df.merge(combo_counts, on="hour_bucket", how="left")
            combo_df["count"] = combo_df["count"].fillna(0).astype(int)

            self.combo_hourly[combo] = combo_df

            # Log stats
            total_txns = combo_df["count"].sum()
            mean_hourly = combo_df["count"].mean()
            logger.info(
                f"  {combo}: {len(combo_df)} hours, "
                f"{total_txns:,} total transactions, "
                f"{mean_hourly:.1f} mean/hour"
            )

        return self.combo_hourly

    def segment_into_days(
        self,
        combo: Tuple[str, str],
        hourly_df: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Segment hourly time series into daily windows (24 hours each) with DoW features.

        Each window is tagged with metadata including day-of-week for analysis.
        DoW cyclical encoding (sin/cos) is added as additional features.

        Args:
            combo: The (network_type, transaction_type) tuple
            hourly_df: DataFrame with hour_bucket and count columns

        Returns:
            Tuple of:
                - Array of shape (num_days, 24, 3) containing:
                    - Channel 0: transaction counts (to be normalized later)
                    - Channel 1: sin(2π * day_of_week / 7)
                    - Channel 2: cos(2π * day_of_week / 7)
                - List of dicts with window metadata
        """
        if hourly_df is None:
            hourly_df = self.combo_hourly.get(combo)
        if hourly_df is None:
            raise ValueError(f"No hourly data for combo {combo}. Call load_and_aggregate() first.")

        values = hourly_df["count"].values.astype(np.float32)
        hours = hourly_df["hour_bucket"].values

        # Calculate number of complete days
        num_complete_days = len(values) // SAMPLES_PER_DAY

        windows = []
        window_info = []

        for d in range(num_complete_days):
            start_idx = d * SAMPLES_PER_DAY
            end_idx = start_idx + SAMPLES_PER_DAY

            day_values = values[start_idx:end_idx]
            day_start = pd.Timestamp(hours[start_idx])
            day_end = pd.Timestamp(hours[end_idx - 1])

            # Compute DoW cyclical encoding (constant for all 24 hours in the window)
            dow = day_start.dayofweek  # 0=Mon, 6=Sun
            dow_sin = np.sin(2 * np.pi * dow / 7)
            dow_cos = np.cos(2 * np.pi * dow / 7)

            # Create 3-channel window: (24, 3)
            # Channel 0: transaction count, Channel 1: DoW sin, Channel 2: DoW cos
            window_3d = np.zeros((SAMPLES_PER_DAY, 3), dtype=np.float32)
            window_3d[:, 0] = day_values
            window_3d[:, 1] = dow_sin  # Broadcast to all 24 hours
            window_3d[:, 2] = dow_cos  # Broadcast to all 24 hours

            windows.append(window_3d)
            window_info.append({
                "day_index": d,
                "start_time": day_start,
                "end_time": day_end,
                "day_of_week": dow,  # 0=Mon, 6=Sun
                "day_name": day_start.strftime("%A"),
                "date": day_start.strftime("%Y-%m-%d"),
                "mean_count": float(day_values.mean()),
                "total_count": int(day_values.sum()),
                "dow_sin": float(dow_sin),
                "dow_cos": float(dow_cos),
                "is_anomaly": False,  # Will be set during test evaluation if needed
            })

        windows_array = np.array(windows)  # Shape: (num_days, 24, 3)
        self.combo_windows[combo] = windows_array
        self.combo_window_info[combo] = window_info

        logger.debug(
            f"  {combo}: Segmented into {len(windows)} daily windows, "
            f"shape {windows_array.shape}"
        )

        return windows_array, window_info

    def create_splits(
        self,
        combo: Tuple[str, str],
        daily_windows: Optional[np.ndarray] = None,
        window_info: Optional[List[Dict]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Create train/val/threshold/test splits for a single combo.

        Split strategy (30 days starting Monday):
        - Train: Days 0-17 (18 days)
        - Val: Days 18-21 (4 days, Thu-Sun)
        - Threshold: Days 22-24 (3 days, Mon-Wed)
        - Test: Days 25-29 (5 days, Thu-Mon)

        All days are normal in the base dataset. Anomalies are injected at evaluation time.

        Args:
            combo: The (network_type, transaction_type) tuple
            daily_windows: Array of shape (num_days, 24)
            window_info: List of window metadata dicts

        Returns:
            Dict with keys: train, val, threshold_val, test
            Each value is an array of shape (num_days, 24)
        """
        if daily_windows is None:
            daily_windows = self.combo_windows.get(combo)
        if window_info is None:
            window_info = self.combo_window_info.get(combo)

        if daily_windows is None:
            raise ValueError(f"No segmented data for combo {combo}. Call segment_into_days() first.")

        num_days = len(daily_windows)

        # Calculate split boundaries
        train_end = self.config.train_days
        val_end = train_end + self.config.val_days
        threshold_end = val_end + self.config.threshold_days

        # Create index ranges
        train_indices = list(range(0, min(train_end, num_days)))
        val_indices = list(range(train_end, min(val_end, num_days)))
        threshold_indices = list(range(val_end, min(threshold_end, num_days)))
        test_indices = list(range(threshold_end, num_days))

        splits = {
            "train": daily_windows[train_indices] if train_indices else np.array([]),
            "val": daily_windows[val_indices] if val_indices else np.array([]),
            "threshold_val": daily_windows[threshold_indices] if threshold_indices else np.array([]),
            "test": daily_windows[test_indices] if test_indices else np.array([]),
        }

        # Store split indices for later reference
        if not hasattr(self, "combo_split_indices"):
            self.combo_split_indices = {}
        self.combo_split_indices[combo] = {
            "train": train_indices,
            "val": val_indices,
            "threshold_val": threshold_indices,
            "test": test_indices,
        }

        self.combo_splits[combo] = splits

        logger.debug(
            f"  {combo}: train={len(train_indices)}, val={len(val_indices)}, "
            f"threshold={len(threshold_indices)}, test={len(test_indices)}"
        )

        return splits

    def normalize(
        self,
        combo: Tuple[str, str],
        splits: Optional[Dict[str, np.ndarray]] = None,
        fit_on: str = "train"
    ) -> Dict[str, np.ndarray]:
        """
        Normalize data using StandardScaler for a single combo.

        IMPORTANT: Scaler is fit only on training data to prevent data leakage.

        With DoW conditioning:
        - Only channel 0 (transaction count) is normalized
        - Channels 1-2 (DoW sin/cos) are left unchanged (already bounded [-1, 1])

        Args:
            combo: The (network_type, transaction_type) tuple
            splits: Dict of data splits, each with shape (num_days, 24, 3)
            fit_on: Which split to fit the scaler on (default: "train")

        Returns:
            Dict with same keys, but with channel 0 normalized
        """
        if splits is None:
            splits = self.combo_splits.get(combo)
        if splits is None:
            raise ValueError(f"No splits for combo {combo}. Call create_splits() first.")

        if fit_on not in splits or len(splits[fit_on]) == 0:
            raise ValueError(f"Cannot fit scaler: '{fit_on}' split is empty for {combo}")

        # Fit scaler on channel 0 (transaction count) of training data only
        train_data = splits[fit_on]
        if train_data.ndim == 3:
            # Shape: (num_days, 24, 3) - extract channel 0
            train_counts = train_data[:, :, 0].flatten().reshape(-1, 1)
        else:
            # Legacy 2D shape: (num_days, 24)
            train_counts = train_data.flatten().reshape(-1, 1)

        scaler = StandardScaler()
        scaler.fit(train_counts)
        self.scalers[combo] = scaler

        logger.debug(
            f"  {combo}: Scaler (channel 0) mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f}"
        )

        # Transform all splits - only normalize channel 0
        normalized_splits = {}
        for name, data in splits.items():
            if len(data) == 0:
                normalized_splits[name] = data
                continue

            if data.ndim == 3:
                # Shape: (num_days, 24, 3)
                normalized = data.copy()
                # Normalize channel 0 only
                counts = data[:, :, 0].flatten().reshape(-1, 1)
                counts_norm = scaler.transform(counts)
                normalized[:, :, 0] = counts_norm.reshape(data.shape[0], data.shape[1])
                # Channels 1-2 (DoW sin/cos) are left unchanged
                normalized_splits[name] = normalized
            else:
                # Legacy 2D shape: (num_days, 24)
                original_shape = data.shape
                flat = data.flatten().reshape(-1, 1)
                normalized = scaler.transform(flat)
                normalized_splits[name] = normalized.reshape(original_shape)

        return normalized_splits

    def create_dataloaders(
        self,
        normalized_splits: Dict[str, np.ndarray],
        batch_size: int = 4
    ) -> Dict[str, Optional[DataLoader]]:
        """
        Create PyTorch DataLoaders for each split.

        Note: Data is NOT shuffled to preserve temporal ordering.

        Args:
            normalized_splits: Dict of normalized data arrays
            batch_size: Batch size for training

        Returns:
            Dict with DataLoader for each split (None for empty splits)
        """
        dataloaders = {}

        for name, data in normalized_splits.items():
            if len(data) == 0:
                dataloaders[name] = None
                continue

            dataset = TimeSeriesDataset(data)

            # No shuffling - preserve temporal ordering
            loader = DataLoader(
                dataset,
                batch_size=min(batch_size, len(data)),
                shuffle=False,
                drop_last=False,
            )

            dataloaders[name] = loader

        return dataloaders

    def preprocess(
        self,
        filepath: str,
        batch_size: int = 4
    ) -> Dict[Tuple[str, str], Dict[str, DataLoader]]:
        """
        Run the complete preprocessing pipeline for all combos.

        This is the main entry point that:
        1. Loads and aggregates transaction data
        2. For each combo: segment, split, normalize, create DataLoaders

        Args:
            filepath: Path to synthetic_transactions.csv
            batch_size: Batch size for DataLoaders

        Returns:
            Dict mapping combo tuple to dict of DataLoaders
            Example: {("Accel", "CMP"): {"train": DataLoader, "val": DataLoader, ...}}
        """
        logger.info("=" * 60)
        logger.info("Starting transaction preprocessing pipeline")
        logger.info("=" * 60)

        # Step 1: Load and aggregate all combos
        self.load_and_aggregate(filepath)

        # Step 2-5: Process each combo
        combo_dataloaders: Dict[Tuple[str, str], Dict[str, DataLoader]] = {}

        for combo in COMBO_KEYS:
            logger.info(f"\nProcessing combo: {combo}")

            # Segment into days
            self.segment_into_days(combo)

            # Create splits
            splits = self.create_splits(combo)

            # Normalize
            normalized_splits = self.normalize(combo, splits)

            # Store normalized splits (overwrite raw splits)
            self.combo_splits[combo] = normalized_splits

            # Create DataLoaders
            dataloaders = self.create_dataloaders(normalized_splits, batch_size)

            combo_dataloaders[combo] = dataloaders

        # Log summary
        logger.info("\n" + "=" * 60)
        logger.info("Preprocessing complete")
        logger.info("=" * 60)

        for combo in COMBO_KEYS:
            loaders = combo_dataloaders[combo]
            sizes = {k: len(v.dataset) if v else 0 for k, v in loaders.items()}
            logger.info(f"  {combo}: {sizes}")

        return combo_dataloaders

    def get_combo_window_info(self, combo: Tuple[str, str]) -> List[Dict]:
        """Get metadata for all windows of a combo."""
        return self.combo_window_info.get(combo, [])

    def get_test_window_info(self, combo: Tuple[str, str]) -> List[Dict]:
        """Get metadata for test windows of a combo."""
        if combo not in self.combo_window_info:
            return []
        if not hasattr(self, "combo_split_indices") or combo not in self.combo_split_indices:
            return []

        test_indices = self.combo_split_indices[combo].get("test", [])
        return [self.combo_window_info[combo][i] for i in test_indices]

    def inverse_transform(
        self,
        combo: Tuple[str, str],
        data: np.ndarray,
        channel_only: bool = True
    ) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale.

        With DoW conditioning, only channel 0 is inverse transformed.

        Args:
            combo: The (network_type, transaction_type) tuple
            data: Normalized data array, shape (num_days, 24, 3) or (num_days, 24)
            channel_only: If True and data is 3D, return only channel 0 (counts)

        Returns:
            Data in original scale. If channel_only and 3D input, returns (num_days, 24).
            Otherwise returns same shape as input with channel 0 inverse transformed.
        """
        if combo not in self.scalers:
            raise ValueError(f"Scaler not fitted for {combo}. Call normalize() first.")

        scaler = self.scalers[combo]

        if data.ndim == 3:
            # Shape: (num_days, 24, 3) - inverse transform channel 0 only
            counts_norm = data[:, :, 0].flatten().reshape(-1, 1)
            counts_orig = scaler.inverse_transform(counts_norm)
            counts_orig = counts_orig.reshape(data.shape[0], data.shape[1])

            if channel_only:
                return counts_orig
            else:
                result = data.copy()
                result[:, :, 0] = counts_orig
                return result
        else:
            # Legacy 2D shape: (num_days, 24)
            original_shape = data.shape
            flat = data.flatten().reshape(-1, 1)
            original = scaler.inverse_transform(flat)
            return original.reshape(original_shape)

    def get_stats(self) -> Dict:
        """Get preprocessing statistics for all combos."""
        stats = {
            "config": {
                "sequence_length": self.config.sequence_length,
                "train_days": self.config.train_days,
                "val_days": self.config.val_days,
                "threshold_days": self.config.threshold_days,
            },
            "data_loaded": self.raw_df is not None,
            "combos_processed": list(self.combo_hourly.keys()),
        }

        if self.raw_df is not None:
            stats["total_transactions"] = len(self.raw_df)
            stats["date_range"] = {
                "start": str(self.data_start),
                "end": str(self.data_end),
            }

        # Per-combo stats
        combo_stats = {}
        for combo in COMBO_KEYS:
            c_stats = {}

            if combo in self.combo_hourly:
                c_stats["total_hours"] = len(self.combo_hourly[combo])
                c_stats["total_count"] = int(self.combo_hourly[combo]["count"].sum())

            if combo in self.combo_windows:
                c_stats["total_days"] = len(self.combo_windows[combo])

            if combo in self.scalers:
                c_stats["scaler"] = {
                    "mean": float(self.scalers[combo].mean_[0]),
                    "std": float(self.scalers[combo].scale_[0]),
                }

            if hasattr(self, "combo_split_indices") and combo in self.combo_split_indices:
                c_stats["splits"] = {
                    k: len(v) for k, v in self.combo_split_indices[combo].items()
                }

            combo_stats[f"{combo[0]}_{combo[1]}"] = c_stats

        stats["combos"] = combo_stats
        return stats


def main():
    """Test the preprocessor with synthetic transaction data."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess transaction data")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/synthetic_transactions.csv",
        help="Path to synthetic transactions CSV file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for DataLoaders"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Run preprocessing
    preprocessor = TransactionPreprocessor()
    combo_dataloaders = preprocessor.preprocess(
        args.data_path,
        batch_size=args.batch_size
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    stats = preprocessor.get_stats()
    print(f"\nTotal transactions: {stats.get('total_transactions', 'N/A'):,}")
    print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")

    print("\nPer-combo statistics:")
    for combo_key, combo_stats in stats.get("combos", {}).items():
        print(f"\n  {combo_key}:")
        print(f"    Total days: {combo_stats.get('total_days', 'N/A')}")
        print(f"    Total transactions: {combo_stats.get('total_count', 'N/A'):,}")
        if "splits" in combo_stats:
            print(f"    Splits: {combo_stats['splits']}")
        if "scaler" in combo_stats:
            print(f"    Scaler: mean={combo_stats['scaler']['mean']:.2f}, std={combo_stats['scaler']['std']:.2f}")

    # Verify tensor shapes
    print("\nTensor shapes (first combo):")
    first_combo = COMBO_KEYS[0]
    for name, loader in combo_dataloaders[first_combo].items():
        if loader is not None:
            sample = next(iter(loader))
            print(f"  {name}: {sample.shape} (batch, seq_len=24, features=1)")

    # Print day-of-week coverage for training split
    print("\nDay-of-week coverage in training split (first combo):")
    window_info = preprocessor.get_combo_window_info(first_combo)
    train_indices = preprocessor.combo_split_indices[first_combo]["train"]
    dow_counts = {}
    for idx in train_indices:
        dow = window_info[idx]["day_name"]
        dow_counts[dow] = dow_counts.get(dow, 0) + 1
    for dow in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        print(f"  {dow}: {dow_counts.get(dow, 0)}")


if __name__ == "__main__":
    main()
