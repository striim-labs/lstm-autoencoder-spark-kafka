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


class SlidingWindowDataset(Dataset):
    """
    PyTorch Dataset for FCVAE sliding windows.

    Each sample is a sliding window of hourly transaction counts,
    shaped as (1, window_size) for FCVAE input.

    Unlike TimeSeriesDataset, this does NOT include DoW features
    (FCVAE uses frequency conditioning instead).
    """

    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        """
        Args:
            windows: Array of shape (num_windows, window_size)
            labels: Array of shape (num_windows, window_size) - anomaly labels
        """
        # FCVAE expects (B, 1, W) shape
        self.windows = torch.FloatTensor(windows).unsqueeze(1)  # (N, 1, W)
        self.labels = torch.FloatTensor(labels)  # (N, W)
        # Missing mask (all zeros for normal data)
        self.missing = torch.zeros_like(self.labels)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (window, labels, missing_mask) matching FCVAE dataset format
        """
        return self.windows[idx], self.labels[idx], self.missing[idx]


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

        # Split info from CSV (if available)
        self.has_csv_splits: bool = False
        self.combo_hour_splits: Dict[Tuple[str, str], pd.Series] = {}
        self.combo_hour_anomalies: Dict[Tuple[str, str], pd.Series] = {}

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

        # Ensure is_anomaly column exists
        if "is_anomaly" not in df.columns:
            df["is_anomaly"] = 0

        # Check for split column (from 60-day dataset)
        has_split_col = "split" in df.columns
        if has_split_col and self.config.use_csv_splits:
            self.has_csv_splits = True
            logger.info("Found 'split' column in CSV - using CSV-based splits")
        else:
            self.has_csv_splits = False
            logger.info("No 'split' column - using day-based splits")

        # Aggregate: count transactions per hour per combo
        # Also aggregate is_anomaly (max = 1 if any transaction was anomalous)
        # And split (first = all transactions in an hour should have same split)
        agg_dict = {"timestamp": "count", "is_anomaly": "max"}
        if has_split_col:
            agg_dict["split"] = "first"

        counts = df.groupby(
            ["hour_bucket", "network_type", "transaction_type"]
        ).agg(agg_dict).reset_index()

        if has_split_col:
            counts.columns = ["hour_bucket", "network_type", "transaction_type", "count", "is_anomaly", "split"]
        else:
            counts.columns = ["hour_bucket", "network_type", "transaction_type", "count", "is_anomaly"]

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
            cols_to_keep = ["hour_bucket", "count", "is_anomaly"]
            if has_split_col:
                cols_to_keep.append("split")

            combo_counts = counts[
                (counts["network_type"] == network_type) &
                (counts["transaction_type"] == transaction_type)
            ][cols_to_keep].copy()

            # Reindex to full hour range, fill missing with 0
            combo_df = pd.DataFrame({"hour_bucket": full_hours})
            combo_df = combo_df.merge(combo_counts, on="hour_bucket", how="left")
            combo_df["count"] = combo_df["count"].fillna(0).astype(int)
            combo_df["is_anomaly"] = combo_df["is_anomaly"].fillna(0).astype(int)

            # Handle split column
            if has_split_col:
                # Forward-fill then backward-fill to handle any gaps
                combo_df["split"] = combo_df["split"].ffill().bfill()
                self.combo_hour_splits[combo] = combo_df["split"]
                self.combo_hour_anomalies[combo] = combo_df["is_anomaly"]

            self.combo_hourly[combo] = combo_df

            # Log stats
            total_txns = combo_df["count"].sum()
            mean_hourly = combo_df["count"].mean()
            anomaly_hours = combo_df["is_anomaly"].sum()
            log_msg = (
                f"  {combo}: {len(combo_df)} hours, "
                f"{total_txns:,} total transactions, "
                f"{mean_hourly:.1f} mean/hour"
            )
            if anomaly_hours > 0:
                log_msg += f", {anomaly_hours} anomaly hours"
            if has_split_col:
                split_counts = combo_df["split"].value_counts().to_dict()
                log_msg += f", splits: {split_counts}"
            logger.info(log_msg)

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

    def create_sliding_windows(
        self,
        combo: Tuple[str, str],
        window_size: int = 24,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create overlapping sliding windows from hourly count series for FCVAE.

        Unlike segment_into_days() which creates non-overlapping daily windows with
        DoW features, this creates overlapping sliding windows without DoW features
        (FCVAE uses frequency conditioning instead).

        Args:
            combo: (network_type, transaction_type) tuple
            window_size: Window size in hours (default 24)
            stride: Step size between windows (default 1 hour for maximum samples)

        Returns:
            Tuple of:
                - values: (num_windows, window_size) hourly counts (float32)
                - labels: (num_windows, window_size) binary anomaly labels
                - timestamps: (num_windows,) starting hour index for each window
        """
        hourly_df = self.combo_hourly.get(combo)
        if hourly_df is None:
            raise ValueError(f"No hourly data for combo {combo}. Call load_and_aggregate() first.")

        values = hourly_df["count"].values.astype(np.float32)
        anomalies = hourly_df["is_anomaly"].values.astype(np.float32)
        total_hours = len(values)

        # Calculate number of windows
        num_windows = (total_hours - window_size) // stride + 1

        if num_windows <= 0:
            logger.warning(f"  {combo}: Not enough data for sliding windows "
                          f"(need {window_size} hours, have {total_hours})")
            return np.array([]), np.array([]), np.array([])

        # Create windows
        windows = np.zeros((num_windows, window_size), dtype=np.float32)
        labels = np.zeros((num_windows, window_size), dtype=np.float32)
        timestamps = np.zeros(num_windows, dtype=np.int32)

        for i in range(num_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            windows[i] = values[start_idx:end_idx]
            labels[i] = anomalies[start_idx:end_idx]  # Use actual anomaly labels
            timestamps[i] = start_idx

        # Store for later use
        if not hasattr(self, "combo_sliding_windows"):
            self.combo_sliding_windows = {}
        self.combo_sliding_windows[combo] = {
            "windows": windows,
            "labels": labels,
            "timestamps": timestamps,
            "window_size": window_size,
            "stride": stride,
        }

        anomaly_windows = (labels.sum(axis=1) > 0).sum()
        logger.debug(
            f"  {combo}: Created {num_windows} sliding windows "
            f"(W={window_size}, stride={stride}, {anomaly_windows} with anomalies)"
        )

        return windows, labels, timestamps

    def create_sliding_splits(
        self,
        combo: Tuple[str, str],
        window_size: int = 24,
        stride: int = 1,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Create train/val/test splits for sliding windows.

        If CSV has 'split' column, uses that directly. Otherwise falls back to
        day-based splits:
        - Train: Windows starting in days 0-17
        - Val: Windows starting in days 18-21
        - Threshold: Windows starting in days 22-24
        - Test: Windows starting in days 25+

        Args:
            combo: (network_type, transaction_type) tuple
            window_size: Window size in hours (default 24)
            stride: Step size between windows (default 1 hour)

        Returns:
            Dict with keys: train, val, threshold_val, test
            Each value is tuple of (windows, labels) where:
                - windows: (num_windows, window_size) array
                - labels: (num_windows, window_size) array
        """
        # Create sliding windows if not already done
        if not hasattr(self, "combo_sliding_windows") or combo not in self.combo_sliding_windows:
            self.create_sliding_windows(combo, window_size, stride)

        sw_data = self.combo_sliding_windows[combo]
        windows = sw_data["windows"]
        labels = sw_data["labels"]
        timestamps = sw_data["timestamps"]

        if len(windows) == 0:
            empty = (np.array([]), np.array([]))
            return {"train": empty, "val": empty, "threshold_val": empty, "test": empty}

        # Use CSV-based splits if available
        if self.has_csv_splits and combo in self.combo_hour_splits:
            hourly_splits = self.combo_hour_splits[combo].values
            hourly_anomalies = self.combo_hour_anomalies[combo].values

            # Assign windows based on the split of their starting hour
            window_splits = np.array([hourly_splits[ts] for ts in timestamps])

            train_mask = window_splits == "train"
            val_mask = window_splits == "val"
            test_mask = window_splits == "test"

            # For CSV-based splits, val doubles as threshold_val (no separate threshold period)
            splits = {
                "train": (windows[train_mask], labels[train_mask]),
                "val": (windows[val_mask], labels[val_mask]),
                "threshold_val": (windows[val_mask], labels[val_mask]),  # Same as val
                "test": (windows[test_mask], labels[test_mask]),
            }

            # Log anomaly counts
            val_anomaly_windows = (labels[val_mask].sum(axis=1) > 0).sum() if val_mask.sum() > 0 else 0
            test_anomaly_windows = (labels[test_mask].sum(axis=1) > 0).sum() if test_mask.sum() > 0 else 0

            logger.info(
                f"  {combo}: CSV-based splits - "
                f"train={train_mask.sum()}, "
                f"val={val_mask.sum()} ({val_anomaly_windows} anomaly windows), "
                f"test={test_mask.sum()} ({test_anomaly_windows} anomaly windows)"
            )
        else:
            # Fall back to day-based splits
            hours_per_day = SAMPLES_PER_DAY
            train_end_hour = self.config.train_days * hours_per_day
            val_end_hour = (self.config.train_days + self.config.val_days) * hours_per_day
            threshold_end_hour = (self.config.train_days + self.config.val_days +
                                  self.config.threshold_days) * hours_per_day

            # Assign windows to splits based on starting timestamp
            train_mask = timestamps < train_end_hour
            val_mask = (timestamps >= train_end_hour) & (timestamps < val_end_hour)
            threshold_mask = (timestamps >= val_end_hour) & (timestamps < threshold_end_hour)
            test_mask = timestamps >= threshold_end_hour

            splits = {
                "train": (windows[train_mask], labels[train_mask]),
                "val": (windows[val_mask], labels[val_mask]),
                "threshold_val": (windows[threshold_mask], labels[threshold_mask]),
                "test": (windows[test_mask], labels[test_mask]),
            }

            logger.debug(
                f"  {combo}: Day-based splits - "
                f"train={len(splits['train'][0])}, "
                f"val={len(splits['val'][0])}, "
                f"threshold={len(splits['threshold_val'][0])}, "
                f"test={len(splits['test'][0])}"
            )

        # Store split info
        if not hasattr(self, "combo_sliding_split_counts"):
            self.combo_sliding_split_counts = {}
        self.combo_sliding_split_counts[combo] = {
            name: len(data[0]) for name, data in splits.items()
        }

        return splits

    def normalize_sliding_windows(
        self,
        combo: Tuple[str, str],
        splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
        fit_on: str = "train"
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Normalize sliding windows using StandardScaler.

        Creates a separate scaler for FCVAE sliding windows (stored with '_sliding' suffix).

        Args:
            combo: (network_type, transaction_type) tuple
            splits: Dict from create_sliding_splits()
            fit_on: Which split to fit the scaler on (default: "train")

        Returns:
            Dict with same structure, but with normalized window values
        """
        if fit_on not in splits or len(splits[fit_on][0]) == 0:
            raise ValueError(f"Cannot fit scaler: '{fit_on}' split is empty for {combo}")

        train_windows = splits[fit_on][0]
        train_flat = train_windows.flatten().reshape(-1, 1)

        scaler = StandardScaler()
        scaler.fit(train_flat)

        # Store with suffix to distinguish from daily window scaler
        scaler_key = (combo[0], combo[1] + "_sliding")
        self.scalers[scaler_key] = scaler

        logger.debug(
            f"  {combo}: Sliding scaler mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f}"
        )

        # Transform all splits
        normalized_splits = {}
        for name, (windows, labels) in splits.items():
            if len(windows) == 0:
                normalized_splits[name] = (windows, labels)
                continue

            flat = windows.flatten().reshape(-1, 1)
            normalized = scaler.transform(flat)
            normalized_splits[name] = (
                normalized.reshape(windows.shape),
                labels
            )

        return normalized_splits

    def create_sliding_dataloaders(
        self,
        normalized_splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
        batch_size: int = 64,
        shuffle_train: bool = True
    ) -> Dict[str, Optional[DataLoader]]:
        """
        Create PyTorch DataLoaders for FCVAE sliding windows.

        Args:
            normalized_splits: Dict from normalize_sliding_windows()
            batch_size: Batch size (larger than daily windows due to more samples)
            shuffle_train: Whether to shuffle training data (default True for FCVAE)

        Returns:
            Dict with DataLoader for each split (None for empty splits)
        """
        dataloaders = {}

        for name, (windows, labels) in normalized_splits.items():
            if len(windows) == 0:
                dataloaders[name] = None
                continue

            # Create dataset for FCVAE: (B, 1, W) format
            dataset = SlidingWindowDataset(windows, labels)

            loader = DataLoader(
                dataset,
                batch_size=min(batch_size, len(windows)),
                shuffle=(shuffle_train and name == "train"),
                drop_last=False,
            )

            dataloaders[name] = loader

        return dataloaders

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
