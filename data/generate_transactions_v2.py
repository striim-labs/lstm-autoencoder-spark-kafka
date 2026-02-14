"""
Synthetic Transaction Data Generator v2 - With Realistic Variability
=====================================================================
Generates time-series transaction records over 30 days with realistic
day-to-day variability that prevents the model from perfectly fitting
the data.

Key differences from v1:
  - Day-to-day amplitude jitter (±15%)
  - Phase shift variation (±1 hour per day)
  - Baseline drift (cumulative over days)
  - Micro-anomalies in normal data (~5% of hours)
  - Larger hourly multiplicative noise (8% std)

This creates data the LSTM-AE cannot perfectly reconstruct, resulting in:
  - Naturally wider error distribution
  - Reasonable anomaly scores without artificial variance floor
  - Better generalization to real-world data

Usage:
    python data/generate_transactions_v2.py --output data/synthetic_transactions_v2.csv
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, List

# ─── Base Configuration (same as v1) ─────────────────────────────────────────

SEED = 42
NUM_DAYS = 60
START_DATE = datetime(2025, 1, 6)  # a Monday

# Train/Val/Test split configuration
TRAIN_DAYS = 42
VAL_DAYS = 8
TEST_DAYS = 10

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Combo keys for iteration
COMBO_KEYS = [
    ("Accel", "CMP"),
    ("Accel", "no-pin"),
    ("Star", "CMP"),
    ("Star", "no-pin"),
]

# Intraday frequency parameters per combo
COMBO_FREQ_PARAMS = {
    ("Accel", "CMP"): {
        "base_rate": 1765,
        "amplitude": 1230,
        "period_h": 24,
        "phase": 0.0,           # peaks ~hour 6
    },
    ("Accel", "no-pin"): {
        "base_rate": 1370,
        "amplitude": 878,
        "period_h": 12,         # two cycles per day
        "phase": np.pi / 3,
    },
    ("Star", "CMP"): {
        "base_rate": 2000,
        "amplitude": 1500,
        "period_h": 24,
        "phase": np.pi,         # peaks ~hour 18
    },
    ("Star", "no-pin"): {
        "base_rate": 1252,
        "amplitude": 878,
        "period_h": 8,          # three cycles per day
        "phase": np.pi / 2,
    },
}

# Per-combo day-of-week multipliers applied to base_rate
#                               Mon   Tue   Wed   Thu   Fri   Sat   Sun
DOW_MULTIPLIERS = {
    ("Accel", "CMP"):    [1.00, 1.05, 1.15, 1.10, 0.95, 0.60, 0.55],
    ("Accel", "no-pin"): [0.90, 1.00, 1.05, 1.10, 1.15, 0.70, 0.65],
    ("Star",  "CMP"):    [1.05, 1.00, 0.95, 1.00, 1.10, 0.75, 0.70],
    ("Star",  "no-pin"): [0.85, 0.95, 1.00, 1.05, 1.00, 0.80, 0.90],
}

# Amount distribution (same for all combos)
AMOUNT_PARAMS = {
    "log_mean": 3.2,
    "log_sigma": 0.9,
}

# ─── NEW: Variability Configuration ──────────────────────────────────────────

VARIABILITY_CONFIG = {
    # Day-to-day amplitude variation
    "amplitude_jitter_pct": 0.15,       # ±15% amplitude variation per day

    # Phase shift variation
    "phase_jitter_hours": 1.0,          # ±1 hour phase shift per day

    # Baseline drift (cumulative over days)
    "baseline_drift_pct_per_day": 0.005,  # ±0.5% drift per day (cumulative)

    # Micro-anomalies in normal data
    "micro_anomaly_prob": 0.05,         # 5% of hours have small perturbations
    "micro_anomaly_magnitude": 0.15,    # ±15% magnitude of micro-anomalies

    # Hourly count noise (multiplicative)
    "hourly_noise_std": 0.08,           # 8% std multiplicative noise
}

# ─── Anomaly Injection Configuration ─────────────────────────────────────────

# Validation anomalies: mixed types spread across 8 days (days 43-50)
VAL_ANOMALIES = [
    {"day_offset": 0, "type": "spike", "hours": [10, 11, 12], "multiplier": 3.0},       # Day 43
    {"day_offset": 2, "type": "dip", "hours": [14], "multiplier": 0.1},                 # Day 45
    {"day_offset": 4, "type": "ramp", "hours": [8, 9, 10, 11],
     "start_mult": 1.5, "end_mult": 2.5},                                               # Day 47
    {"day_offset": 7, "type": "duration", "peak_shift_hours": 3},                       # Day 50
]

# Test anomalies: evenly spaced across 10 days (days 51-60)
TEST_ANOMALIES = [
    {"day_offset": 1, "type": "spike", "hours": [15, 16, 17], "multiplier": 3.0},       # Day 52
    {"day_offset": 5, "type": "dip", "hours": [11], "multiplier": 0.1},                 # Day 56
    {"day_offset": 9, "type": "spike", "hours": [9, 10, 11], "multiplier": 2.5},        # Day 60
]


# ─── Variability Functions ───────────────────────────────────────────────────

def generate_day_params(
    day_index: int,
    combo: Tuple[str, str],
    rng: np.random.Generator
) -> Dict:
    """
    Generate per-day variability parameters.

    Pre-computing ensures consistency within a day while varying across days.

    Args:
        day_index: Which day in the dataset (0 to NUM_DAYS-1)
        combo: (network_type, transaction_type) tuple
        rng: Random number generator

    Returns:
        Dict with variability parameters for this day
    """
    cfg = VARIABILITY_CONFIG

    # Amplitude jitter: different each day, same throughout the day
    amplitude_mult = 1.0 + rng.uniform(
        -cfg["amplitude_jitter_pct"],
        cfg["amplitude_jitter_pct"]
    )

    # Phase shift: different each day, converted to radians
    phase_shift = rng.uniform(
        -cfg["phase_jitter_hours"],
        cfg["phase_jitter_hours"]
    ) * (2 * np.pi / 24)

    # Baseline drift: cumulative random walk effect
    # Each day gets a random drift step, and we accumulate over days
    drift_step = rng.uniform(
        -cfg["baseline_drift_pct_per_day"],
        cfg["baseline_drift_pct_per_day"]
    )
    baseline_drift = drift_step * day_index  # Cumulative effect

    # Hourly multiplicative noise: independent per hour
    hourly_noise = 1.0 + rng.normal(0, cfg["hourly_noise_std"], size=24)
    hourly_noise = np.clip(hourly_noise, 0.7, 1.3)  # Prevent extreme values

    # Micro-anomaly mask: which hours have small perturbations
    micro_anomaly_mask = rng.random(24) < cfg["micro_anomaly_prob"]
    micro_anomaly_mults = 1.0 + rng.uniform(
        -cfg["micro_anomaly_magnitude"],
        cfg["micro_anomaly_magnitude"],
        size=24
    )

    return {
        "amplitude_mult": amplitude_mult,
        "phase_shift": phase_shift,
        "baseline_drift": baseline_drift,
        "hourly_noise": hourly_noise,
        "micro_anomaly_mask": micro_anomaly_mask,
        "micro_anomaly_mults": micro_anomaly_mults,
    }


def rate_function_v2(
    t: np.ndarray,
    params: dict,
    dow_multiplier: float,
    day_params: Dict
) -> np.ndarray:
    """
    Compute the instantaneous transaction rate at fractional hour(s) t,
    with day-of-week scaling AND day-specific variability.

    Args:
        t: Array of fractional hours [0, 24)
        params: Combo's base frequency parameters
        dow_multiplier: Day-of-week scaling factor
        day_params: Day-specific variability parameters

    Returns:
        Array of rates at each time t
    """
    # Apply day-specific jitter to parameters
    amplitude_jittered = params["amplitude"] * day_params["amplitude_mult"]
    phase_jittered = params["phase"] + day_params["phase_shift"]
    base_jittered = params["base_rate"] * (1 + day_params["baseline_drift"])

    # Compute base rate with sinusoidal pattern
    scaled_base = base_jittered * dow_multiplier
    rate = scaled_base + amplitude_jittered * np.sin(
        2 * np.pi * t / params["period_h"] + phase_jittered
    )

    # Apply hourly multiplicative noise
    # Map each time to its hour and apply that hour's noise
    hours = np.floor(t).astype(int) % 24
    rate = rate * day_params["hourly_noise"][hours]

    # Apply micro-anomalies where flagged
    for h in range(24):
        if day_params["micro_anomaly_mask"][h]:
            mask = (hours == h)
            rate[mask] = rate[mask] * day_params["micro_anomaly_mults"][h]

    return np.maximum(rate, 0.0)


def generate_combo_timestamps_v2(
    day_offset: int,
    params: dict,
    dow_multiplier: float,
    day_params: Dict,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate transaction timestamps for one combo for one day using
    inhomogeneous Poisson process thinning with variability.

    Args:
        day_offset: Day number (0 to NUM_DAYS-1)
        params: Combo's base frequency parameters
        dow_multiplier: Day-of-week scaling factor
        day_params: Day-specific variability parameters
        rng: Random number generator

    Returns:
        Array of timestamps in seconds from epoch
    """
    # Compute lambda_max with jitter effects
    # Account for possible increase from amplitude jitter and noise
    base_max = params["base_rate"] * (1 + abs(day_params["baseline_drift"]))
    amp_max = params["amplitude"] * day_params["amplitude_mult"]
    noise_max = max(day_params["hourly_noise"])
    micro_max = max(day_params["micro_anomaly_mults"])

    lambda_max = (base_max * dow_multiplier + amp_max) * noise_max * micro_max

    if lambda_max <= 0:
        return np.array([])

    # Generate candidates from homogeneous Poisson process
    expected_candidates = int(lambda_max * 24 * 1.2)
    inter_arrivals = rng.exponential(1.0 / lambda_max, size=expected_candidates)
    candidate_times_hours = np.cumsum(inter_arrivals)
    candidate_times_hours = candidate_times_hours[candidate_times_hours < 24.0]

    if len(candidate_times_hours) == 0:
        return np.array([])

    # Accept each candidate with probability lambda(t) / lambda_max
    acceptance_prob = rate_function_v2(
        candidate_times_hours, params, dow_multiplier, day_params
    ) / lambda_max
    uniform_draws = rng.random(len(candidate_times_hours))
    accepted = candidate_times_hours[uniform_draws < acceptance_prob]

    # Convert to seconds from start
    day_start_seconds = day_offset * 24 * 3600
    timestamps_seconds = day_start_seconds + accepted * 3600

    return timestamps_seconds


def generate_amounts(
    n: int,
    combo: Tuple[str, str],
    rng: np.random.Generator
) -> np.ndarray:
    """Generate random transaction amounts using log-normal distribution."""
    return np.round(
        rng.lognormal(
            mean=AMOUNT_PARAMS["log_mean"],
            sigma=AMOUNT_PARAMS["log_sigma"],
            size=n,
        ),
        2,
    )


# ─── Anomaly Injection Functions ─────────────────────────────────────────────

def inject_spike_anomaly(
    day_params: Dict,
    hours: List[int],
    multiplier: float
) -> List[int]:
    """
    Inject a spike anomaly by multiplying hourly noise for specified hours.

    Args:
        day_params: Day-specific variability parameters (modified in place)
        hours: Hours to inject spike
        multiplier: Rate multiplier for spike hours

    Returns:
        List of anomaly hours
    """
    for h in hours:
        day_params["hourly_noise"][h] *= multiplier
    return hours


def inject_dip_anomaly(
    day_params: Dict,
    hours: List[int],
    multiplier: float
) -> List[int]:
    """
    Inject a dip/outage anomaly by reducing rate for specified hours.

    Args:
        day_params: Day-specific variability parameters (modified in place)
        hours: Hours to inject dip
        multiplier: Rate multiplier for dip hours (e.g., 0.1 = 90% reduction)

    Returns:
        List of anomaly hours
    """
    for h in hours:
        day_params["hourly_noise"][h] *= multiplier
    return hours


def inject_gradual_ramp(
    day_params: Dict,
    hours: List[int],
    start_mult: float,
    end_mult: float
) -> List[int]:
    """
    Inject a gradual ramp anomaly with linearly increasing rate.

    Args:
        day_params: Day-specific variability parameters (modified in place)
        hours: Hours for the ramp (in order)
        start_mult: Starting multiplier
        end_mult: Ending multiplier

    Returns:
        List of anomaly hours
    """
    n_hours = len(hours)
    for i, h in enumerate(hours):
        # Linear interpolation from start_mult to end_mult
        mult = start_mult + (end_mult - start_mult) * (i / max(n_hours - 1, 1))
        day_params["hourly_noise"][h] *= mult
    return hours


def inject_duration_anomaly(
    day_params: Dict,
    peak_shift_hours: int
) -> List[int]:
    """
    Inject a duration anomaly by shifting peak timing.

    This creates an anomaly where the pattern is shifted, causing
    unexpected high/low values at certain hours.

    Args:
        day_params: Day-specific variability parameters (modified in place)
        peak_shift_hours: Hours to shift the peak (positive = later)

    Returns:
        List of anomaly hours (approximate affected hours)
    """
    # Shift the phase to create unexpected timing
    day_params["phase_shift"] += peak_shift_hours * (2 * np.pi / 24)

    # Return approximate affected hours (around typical peak times)
    # Since this affects the whole day's pattern, mark multiple hours
    return list(range(6, 18))  # Daytime hours most affected


def inject_anomaly(
    day_params: Dict,
    anomaly_config: Dict
) -> List[int]:
    """
    Dispatcher function to inject anomaly based on config type.

    Args:
        day_params: Day-specific variability parameters (modified in place)
        anomaly_config: Anomaly configuration dict with 'type' and parameters

    Returns:
        List of anomaly hours
    """
    anomaly_type = anomaly_config["type"]

    if anomaly_type == "spike":
        return inject_spike_anomaly(
            day_params,
            anomaly_config["hours"],
            anomaly_config["multiplier"]
        )
    elif anomaly_type == "dip":
        return inject_dip_anomaly(
            day_params,
            anomaly_config["hours"],
            anomaly_config["multiplier"]
        )
    elif anomaly_type == "ramp":
        return inject_gradual_ramp(
            day_params,
            anomaly_config["hours"],
            anomaly_config["start_mult"],
            anomaly_config["end_mult"]
        )
    elif anomaly_type == "duration":
        return inject_duration_anomaly(
            day_params,
            anomaly_config["peak_shift_hours"]
        )
    else:
        raise ValueError(f"Unknown anomaly type: {anomaly_type}")


# ─── Main Generation Pipeline ────────────────────────────────────────────────

def generate_dataset_v2(
    num_days: int = NUM_DAYS,
    start_date: datetime = START_DATE,
    seed: int = SEED
) -> pd.DataFrame:
    """
    Generate synthetic transaction dataset with realistic variability.

    Args:
        num_days: Number of days to generate
        start_date: Start date (should be a Monday)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: timestamp, network_type, transaction_type, amount, is_anomaly
    """
    rng = np.random.default_rng(seed)

    all_records = []

    for combo in COMBO_KEYS:
        params = COMBO_FREQ_PARAMS[combo]
        multipliers = DOW_MULTIPLIERS[combo]
        network, txn_type = combo
        combo_timestamps = []

        for day in range(num_days):
            dow = day % 7  # START_DATE is Monday, so day 0 = Mon
            dow_mult = multipliers[dow]

            # Generate day-specific variability parameters
            day_params = generate_day_params(day, combo, rng)

            # Generate timestamps with variability
            ts = generate_combo_timestamps_v2(
                day, params, dow_mult, day_params, rng
            )
            combo_timestamps.append(ts)

        combo_timestamps = np.concatenate(combo_timestamps)
        n = len(combo_timestamps)

        # Generate amounts
        amounts = generate_amounts(n, combo, rng)

        combo_df = pd.DataFrame({
            "timestamp_seconds": combo_timestamps,
            "network_type": network,
            "transaction_type": txn_type,
            "amount": amounts,
        })
        all_records.append(combo_df)

    df = pd.concat(all_records, ignore_index=True)

    # Convert to timestamps
    base_ts = pd.Timestamp(start_date)
    df["timestamp"] = base_ts + pd.to_timedelta(df["timestamp_seconds"], unit="s")
    df.drop(columns="timestamp_seconds", inplace=True)

    # Sort and finalize
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = df[["timestamp", "network_type", "transaction_type", "amount"]]
    df["is_anomaly"] = 0

    return df


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate realistic synthetic transaction data (v2) with day-to-day variability"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/synthetic_transactions_v2.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--num-days",
        type=int,
        default=NUM_DAYS,
        help="Number of days to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--amplitude-jitter",
        type=float,
        default=VARIABILITY_CONFIG["amplitude_jitter_pct"],
        help="Day-to-day amplitude variation (0.15 = ±15%%)"
    )
    parser.add_argument(
        "--phase-jitter",
        type=float,
        default=VARIABILITY_CONFIG["phase_jitter_hours"],
        help="Phase shift variation in hours"
    )
    parser.add_argument(
        "--hourly-noise",
        type=float,
        default=VARIABILITY_CONFIG["hourly_noise_std"],
        help="Hourly multiplicative noise std"
    )
    parser.add_argument(
        "--micro-anomaly-prob",
        type=float,
        default=VARIABILITY_CONFIG["micro_anomaly_prob"],
        help="Probability of micro-anomaly per hour"
    )
    args = parser.parse_args()

    # Update config from CLI args
    VARIABILITY_CONFIG["amplitude_jitter_pct"] = args.amplitude_jitter
    VARIABILITY_CONFIG["phase_jitter_hours"] = args.phase_jitter
    VARIABILITY_CONFIG["hourly_noise_std"] = args.hourly_noise
    VARIABILITY_CONFIG["micro_anomaly_prob"] = args.micro_anomaly_prob

    print("=" * 60)
    print("SYNTHETIC TRANSACTION GENERATOR V2")
    print("With Realistic Day-to-Day Variability")
    print("=" * 60)
    print(f"\nVariability settings:")
    print(f"  Amplitude jitter:    ±{args.amplitude_jitter*100:.0f}%")
    print(f"  Phase jitter:        ±{args.phase_jitter:.1f} hours")
    print(f"  Hourly noise std:    {args.hourly_noise*100:.0f}%")
    print(f"  Micro-anomaly prob:  {args.micro_anomaly_prob*100:.0f}%")
    print(f"  Baseline drift:      ±{VARIABILITY_CONFIG['baseline_drift_pct_per_day']*100:.1f}%/day")

    print(f"\nGenerating {args.num_days} days of data...")
    df = generate_dataset_v2(
        num_days=args.num_days,
        seed=args.seed
    )

    # Print statistics
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal records:  {len(df):,}")
    print(f"Date range:     {df['timestamp'].min()} to {df['timestamp'].max()}")

    print(f"\nRecords per combo:")
    combo_counts = df.groupby(["network_type", "transaction_type"]).size()
    for (net, txn), count in combo_counts.items():
        print(f"  {net}/{txn}: {count:,}")

    # Show daily variability to demonstrate the effect
    df["date"] = df["timestamp"].dt.date
    df["dow"] = df["timestamp"].dt.day_name()
    daily_counts = df.groupby(["network_type", "transaction_type", "date"]).size()

    print(f"\nDaily count variability (showing std across same-DoW days):")
    for combo in COMBO_KEYS:
        combo_daily = daily_counts.loc[combo[0], combo[1]]
        # Get day-of-week from date
        dow_groups = {}
        for date, count in combo_daily.items():
            dow = pd.Timestamp(date).day_name()
            if dow not in dow_groups:
                dow_groups[dow] = []
            dow_groups[dow].append(count)

        # Calculate std for each day of week
        std_by_dow = {dow: np.std(counts) for dow, counts in dow_groups.items() if len(counts) > 1}
        mean_std = np.mean(list(std_by_dow.values())) if std_by_dow else 0
        print(f"  {combo[0]}/{combo[1]}: avg within-DoW std = {mean_std:.1f} txns")

    df.drop(columns=["date", "dow"], inplace=True)

    # Save
    df.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")
    print(f"File size: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB (in memory)")


# ─── Dataset with Train/Val/Test Splits ──────────────────────────────────────

def generate_dataset_with_splits(
    train_days: int = TRAIN_DAYS,
    val_days: int = VAL_DAYS,
    test_days: int = TEST_DAYS,
    val_anomalies: List[Dict] = None,
    test_anomalies: List[Dict] = None,
    start_date: datetime = START_DATE,
    seed: int = SEED
) -> pd.DataFrame:
    """
    Generate synthetic transaction dataset with train/val/test splits.

    Args:
        train_days: Number of training days (normal data only)
        val_days: Number of validation days (with anomalies for threshold tuning)
        test_days: Number of test days (with anomalies for evaluation)
        val_anomalies: List of anomaly configs for validation split
        test_anomalies: List of anomaly configs for test split
        start_date: Start date (should be a Monday)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: timestamp, network_type, transaction_type,
                               amount, is_anomaly, split
    """
    if val_anomalies is None:
        val_anomalies = VAL_ANOMALIES
    if test_anomalies is None:
        test_anomalies = TEST_ANOMALIES

    rng = np.random.default_rng(seed)
    total_days = train_days + val_days + test_days

    # Build anomaly lookup: day_offset -> anomaly_config
    # Day offsets are relative to split start
    val_start_day = train_days
    test_start_day = train_days + val_days

    anomaly_by_day = {}
    for config in val_anomalies:
        absolute_day = val_start_day + config["day_offset"]
        anomaly_by_day[absolute_day] = config
    for config in test_anomalies:
        absolute_day = test_start_day + config["day_offset"]
        anomaly_by_day[absolute_day] = config

    all_records = []

    for combo in COMBO_KEYS:
        params = COMBO_FREQ_PARAMS[combo]
        multipliers = DOW_MULTIPLIERS[combo]
        network, txn_type = combo

        for day in range(total_days):
            dow = day % 7
            dow_mult = multipliers[dow]

            # Generate day-specific variability parameters
            day_params = generate_day_params(day, combo, rng)

            # Check if this day has an anomaly injection
            anomaly_hours = []
            if day in anomaly_by_day:
                anomaly_config = anomaly_by_day[day]
                anomaly_hours = inject_anomaly(day_params, anomaly_config)

            # Generate timestamps with variability
            ts = generate_combo_timestamps_v2(day, params, dow_mult, day_params, rng)

            if len(ts) == 0:
                continue

            # Generate amounts
            amounts = generate_amounts(len(ts), combo, rng)

            # Determine split for this day
            if day < train_days:
                split = "train"
            elif day < train_days + val_days:
                split = "val"
            else:
                split = "test"

            # Create records
            day_start_seconds = day * 24 * 3600
            for i, t in enumerate(ts):
                hour = int((t - day_start_seconds) / 3600)

                # Determine if this transaction is anomalous
                is_anomaly = 1 if hour in anomaly_hours else 0

                all_records.append({
                    "timestamp_seconds": t,
                    "network_type": network,
                    "transaction_type": txn_type,
                    "amount": amounts[i],
                    "is_anomaly": is_anomaly,
                    "split": split,
                })

    df = pd.DataFrame(all_records)

    # Convert to timestamps
    base_ts = pd.Timestamp(start_date)
    df["timestamp"] = base_ts + pd.to_timedelta(df["timestamp_seconds"], unit="s")
    df.drop(columns="timestamp_seconds", inplace=True)

    # Sort and finalize
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = df[["timestamp", "network_type", "transaction_type", "amount", "is_anomaly", "split"]]

    return df


# ─── Append Anomalous Days (Legacy) ──────────────────────────────────────────

def generate_anomalous_days(
    existing_df: pd.DataFrame,
    start_date: datetime,
    num_existing_days: int,
    spike_day_offset: int = 0,
    spike_hours: List[int] = [10, 11, 12],
    spike_multiplier: float = 3.0,
    outage_day_offset: int = 1,
    outage_hour: int = 14,
    outage_fraction: float = 0.1,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Generate 2 additional days with injected anomalies.

    Args:
        existing_df: Existing transaction DataFrame
        start_date: Original start date of the dataset
        num_existing_days: Number of days in existing data
        spike_day_offset: Which new day gets the spike (0 = first new day)
        spike_hours: Hours for the 3-hour spike
        spike_multiplier: How much to multiply transaction rate during spike
        outage_day_offset: Which new day gets the outage (1 = second new day)
        outage_hour: Hour for the 1-hour outage
        outage_fraction: Fraction of normal transactions during outage
        seed: Random seed

    Returns:
        DataFrame with 2 new days of data (with is_anomaly labels)
    """
    rng = np.random.default_rng(seed + 1000)  # Different seed for new days

    all_records = []
    new_days = [num_existing_days, num_existing_days + 1]

    for combo in COMBO_KEYS:
        params = COMBO_FREQ_PARAMS[combo]
        multipliers = DOW_MULTIPLIERS[combo]
        network, txn_type = combo

        for day_idx, day in enumerate(new_days):
            dow = day % 7
            dow_mult = multipliers[dow]

            # Generate day-specific variability
            day_params = generate_day_params(day, combo, rng)

            # Check if this day/hour needs anomaly injection
            is_spike_day = (day_idx == spike_day_offset)
            is_outage_day = (day_idx == outage_day_offset)

            # Modify day_params for anomaly injection
            if is_spike_day:
                # Increase hourly noise for spike hours
                for h in spike_hours:
                    day_params["hourly_noise"][h] *= spike_multiplier

            if is_outage_day:
                # Decrease rate dramatically for outage hour
                day_params["hourly_noise"][outage_hour] *= outage_fraction

            # Generate timestamps
            ts = generate_combo_timestamps_v2(day, params, dow_mult, day_params, rng)

            if len(ts) == 0:
                continue

            # Generate amounts
            amounts = generate_amounts(len(ts), combo, rng)

            # Create records
            day_start_seconds = day * 24 * 3600
            for i, t in enumerate(ts):
                hour = int((t - day_start_seconds) / 3600)

                # Determine if this transaction is anomalous
                is_anomaly = 0
                if is_spike_day and hour in spike_hours:
                    is_anomaly = 1
                if is_outage_day and hour == outage_hour:
                    is_anomaly = 1

                all_records.append({
                    "timestamp_seconds": t,
                    "network_type": network,
                    "transaction_type": txn_type,
                    "amount": amounts[i],
                    "is_anomaly": is_anomaly,
                })

    df = pd.DataFrame(all_records)

    # Convert to timestamps
    base_ts = pd.Timestamp(start_date)
    df["timestamp"] = base_ts + pd.to_timedelta(df["timestamp_seconds"], unit="s")
    df.drop(columns="timestamp_seconds", inplace=True)

    # Sort and finalize
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = df[["timestamp", "network_type", "transaction_type", "amount", "is_anomaly"]]

    return df


def append_anomalous_days():
    """CLI to append anomalous days to existing data."""
    parser = argparse.ArgumentParser(
        description="Append 2 anomalous test days to existing synthetic data"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/synthetic_transactions_v2.csv",
        help="Input CSV path (existing 30-day data)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/synthetic_transactions_v2_with_anomalies.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--spike-hours",
        type=str,
        default="10,11,12",
        help="Comma-separated hours for 3-hour spike (default: 10,11,12)"
    )
    parser.add_argument(
        "--spike-multiplier",
        type=float,
        default=3.0,
        help="Transaction rate multiplier during spike (default: 3.0)"
    )
    parser.add_argument(
        "--outage-hour",
        type=int,
        default=14,
        help="Hour for 1-hour outage (default: 14)"
    )
    parser.add_argument(
        "--outage-fraction",
        type=float,
        default=0.1,
        help="Fraction of normal transactions during outage (default: 0.1)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("APPENDING ANOMALOUS TEST DAYS")
    print("=" * 60)

    # Load existing data
    print(f"\nLoading existing data from: {args.input}")
    existing_df = pd.read_csv(args.input, parse_dates=["timestamp"])

    # Ensure is_anomaly column exists
    if "is_anomaly" not in existing_df.columns:
        existing_df["is_anomaly"] = 0

    # Parse spike hours
    spike_hours = [int(h.strip()) for h in args.spike_hours.split(",")]

    print(f"\nAnomaly configuration:")
    print(f"  Day 30 (6th test day): 3-hour spike at hours {spike_hours}")
    print(f"    - Transaction rate multiplier: {args.spike_multiplier}x")
    print(f"  Day 31 (7th test day): 1-hour outage at hour {args.outage_hour}")
    print(f"    - Transaction rate: {args.outage_fraction*100:.0f}% of normal")

    # Generate new days
    print("\nGenerating 2 anomalous days...")
    new_days_df = generate_anomalous_days(
        existing_df=existing_df,
        start_date=START_DATE,
        num_existing_days=NUM_DAYS,
        spike_day_offset=0,
        spike_hours=spike_hours,
        spike_multiplier=args.spike_multiplier,
        outage_day_offset=1,
        outage_hour=args.outage_hour,
        outage_fraction=args.outage_fraction,
    )

    # Combine
    combined_df = pd.concat([existing_df, new_days_df], ignore_index=True)
    combined_df.sort_values("timestamp", inplace=True)
    combined_df.reset_index(drop=True, inplace=True)

    # Print statistics
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print("=" * 60)

    print(f"\nExisting records: {len(existing_df):,}")
    print(f"New records: {len(new_days_df):,}")
    print(f"Total records: {len(combined_df):,}")

    print(f"\nDate range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")

    # Show anomaly counts
    anomaly_counts = new_days_df.groupby(["network_type", "transaction_type", "is_anomaly"]).size()
    print(f"\nNew days anomaly breakdown:")
    for (net, txn, is_anom), count in anomaly_counts.items():
        label = "ANOMALY" if is_anom else "normal"
        print(f"  {net}/{txn} - {label}: {count:,}")

    # Save
    combined_df.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")


# ─── Generate Split Dataset CLI ──────────────────────────────────────────────

def generate_split():
    """CLI to generate 60-day dataset with train/val/test splits."""
    parser = argparse.ArgumentParser(
        description="Generate 60-day synthetic data with train/val/test splits and anomalies"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/synthetic_transactions_v2_split.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=TRAIN_DAYS,
        help=f"Number of training days (default: {TRAIN_DAYS})"
    )
    parser.add_argument(
        "--val-days",
        type=int,
        default=VAL_DAYS,
        help=f"Number of validation days (default: {VAL_DAYS})"
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=TEST_DAYS,
        help=f"Number of test days (default: {TEST_DAYS})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    total_days = args.train_days + args.val_days + args.test_days

    print("=" * 60)
    print("SYNTHETIC TRANSACTION GENERATOR V2 - SPLIT MODE")
    print("With Train/Val/Test Splits and Mixed Anomalies")
    print("=" * 60)

    print(f"\nSplit configuration:")
    print(f"  Train:      {args.train_days} days (days 1-{args.train_days})")
    print(f"  Validation: {args.val_days} days (days {args.train_days+1}-{args.train_days+args.val_days})")
    print(f"  Test:       {args.test_days} days (days {args.train_days+args.val_days+1}-{total_days})")
    print(f"  Total:      {total_days} days")

    print(f"\nValidation anomalies (for F1 threshold tuning):")
    for config in VAL_ANOMALIES:
        day = args.train_days + config["day_offset"] + 1
        print(f"  Day {day}: {config['type']} - {config}")

    print(f"\nTest anomalies (for evaluation):")
    for config in TEST_ANOMALIES:
        day = args.train_days + args.val_days + config["day_offset"] + 1
        print(f"  Day {day}: {config['type']} - {config}")

    print(f"\nGenerating {total_days} days of data...")
    df = generate_dataset_with_splits(
        train_days=args.train_days,
        val_days=args.val_days,
        test_days=args.test_days,
        seed=args.seed
    )

    # Print statistics
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal records:  {len(df):,}")
    print(f"Date range:     {df['timestamp'].min()} to {df['timestamp'].max()}")

    print(f"\nRecords per split:")
    split_counts = df.groupby("split").size()
    for split, count in split_counts.items():
        print(f"  {split}: {count:,}")

    print(f"\nAnomaly breakdown by split:")
    anomaly_by_split = df.groupby(["split", "is_anomaly"]).size().unstack(fill_value=0)
    for split in ["train", "val", "test"]:
        if split in anomaly_by_split.index:
            normal = anomaly_by_split.loc[split, 0] if 0 in anomaly_by_split.columns else 0
            anomaly = anomaly_by_split.loc[split, 1] if 1 in anomaly_by_split.columns else 0
            print(f"  {split}: {normal:,} normal, {anomaly:,} anomaly")

    print(f"\nRecords per combo:")
    combo_counts = df.groupby(["network_type", "transaction_type"]).size()
    for (net, txn), count in combo_counts.items():
        print(f"  {net}/{txn}: {count:,}")

    # Save
    df.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")
    print(f"File size: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB (in memory)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "append":
        sys.argv.pop(1)  # Remove "append" from args
        append_anomalous_days()
    elif len(sys.argv) > 1 and sys.argv[1] == "split":
        sys.argv.pop(1)  # Remove "split" from args
        generate_split()
    else:
        main()
