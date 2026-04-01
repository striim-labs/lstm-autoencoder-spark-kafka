"""
Synthetic Anomaly Generation for Threshold Calibration

Functions to inject synthetic anomalies into normal time series data
for improved threshold calibration in anomaly detection.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SyntheticAnomalyConfig:
    """Configuration for synthetic anomaly generation."""
    anomaly_types: List[str] = field(
        default_factory=lambda: ["point", "level_shift", "temporal"]
    )
    num_synthetic_per_normal: int = 1
    point_n_points: int = 5
    point_magnitude: float = 3.0
    level_shift_fraction: float = 0.3
    level_shift_magnitude: float = 2.0
    noise_fraction: float = 0.3
    noise_scale: float = 0
    temporal_shift_hours: int = 6
    random_seed: Optional[int] = 42


def inject_point_anomalies(
    week: np.ndarray,
    rng: np.random.Generator,
    config: SyntheticAnomalyConfig,
) -> np.ndarray:
    """Inject random spikes/drops at individual points."""
    was_2d = week.ndim == 2
    if was_2d:
        week = week.squeeze(-1)

    corrupted = week.copy()
    seq_len = len(week)
    std = week.std()

    n_points = min(config.point_n_points, seq_len)
    indices = rng.choice(seq_len, size=n_points, replace=False)

    for idx in indices:
        sign = rng.choice([-1, 1])
        corrupted[idx] += sign * config.point_magnitude * std

    if was_2d:
        corrupted = corrupted.reshape(-1, 1)
    return corrupted


def inject_level_shift(
    week: np.ndarray,
    rng: np.random.Generator,
    config: SyntheticAnomalyConfig,
) -> np.ndarray:
    """Inject a sustained level shift over a portion of the week."""
    was_2d = week.ndim == 2
    if was_2d:
        week = week.squeeze(-1)

    corrupted = week.copy()
    seq_len = len(week)
    std = week.std()

    shift_len = int(seq_len * config.level_shift_fraction)
    max_start = seq_len - shift_len
    start = rng.integers(0, max_start + 1)

    sign = rng.choice([-1, 1])
    corrupted[start:start + shift_len] += sign * config.level_shift_magnitude * std

    if was_2d:
        corrupted = corrupted.reshape(-1, 1)
    return corrupted


def inject_noise(
    week: np.ndarray,
    rng: np.random.Generator,
    config: SyntheticAnomalyConfig,
) -> np.ndarray:
    """Inject Gaussian noise over a portion of the week."""
    was_2d = week.ndim == 2
    if was_2d:
        week = week.squeeze(-1)

    corrupted = week.copy()
    seq_len = len(week)
    std = week.std()

    noise_len = int(seq_len * config.noise_fraction)
    max_start = seq_len - noise_len
    start = rng.integers(0, max_start + 1)

    noise = rng.standard_normal(noise_len) * config.noise_scale * std
    corrupted[start:start + noise_len] += noise

    if was_2d:
        corrupted = corrupted.reshape(-1, 1)
    return corrupted


def inject_temporal_shift(
    week: np.ndarray,
    config: SyntheticAnomalyConfig,
) -> np.ndarray:
    """Shift the time pattern by a few hours (breaks daily cycle)."""
    was_2d = week.ndim == 2
    if was_2d:
        week = week.squeeze(-1)

    shift_samples = config.temporal_shift_hours * 2
    corrupted = np.roll(week, shift_samples)

    if was_2d:
        corrupted = corrupted.reshape(-1, 1)
    return corrupted


def generate_synthetic_anomaly(
    week: np.ndarray,
    rng: np.random.Generator,
    config: SyntheticAnomalyConfig,
    anomaly_type: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    """
    Generate a synthetic anomaly from a normal week.

    Returns:
        Tuple of (corrupted_week, anomaly_type_used)
    """
    if anomaly_type is None:
        anomaly_type = rng.choice(config.anomaly_types)

    if anomaly_type == "point":
        corrupted = inject_point_anomalies(week, rng, config)
    elif anomaly_type == "level_shift":
        corrupted = inject_level_shift(week, rng, config)
    elif anomaly_type == "noise":
        corrupted = inject_noise(week, rng, config)
    elif anomaly_type == "temporal":
        corrupted = inject_temporal_shift(week, config)
    else:
        raise ValueError(f"Unknown anomaly type: {anomaly_type}")

    return corrupted, anomaly_type


def generate_synthetic_dataset(
    normal_weeks: np.ndarray,
    config: Optional[SyntheticAnomalyConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate synthetic anomaly dataset from normal weeks.

    Returns:
        Tuple of (synthetic_weeks, labels, anomaly_types)
    """
    config = config or SyntheticAnomalyConfig()
    rng = np.random.default_rng(config.random_seed)

    was_3d = normal_weeks.ndim == 3
    if was_3d:
        normal_weeks_2d = normal_weeks.squeeze(-1)
    else:
        normal_weeks_2d = normal_weeks

    n_normal = len(normal_weeks_2d)
    n_synthetic = n_normal * config.num_synthetic_per_normal

    synthetic_list = []
    types_list = []

    for i in range(n_synthetic):
        source_idx = i % n_normal
        source_week = normal_weeks_2d[source_idx]

        corrupted, anom_type = generate_synthetic_anomaly(source_week, rng, config)
        synthetic_list.append(corrupted)
        types_list.append(anom_type)

    synthetic_weeks = np.array(synthetic_list)
    labels = np.ones(n_synthetic, dtype=np.int32)

    if was_3d:
        synthetic_weeks = synthetic_weeks.reshape(n_synthetic, -1, 1)

    logger.info(f"Generated {n_synthetic} synthetic anomalies from {n_normal} normal weeks")
    logger.info(f"  Anomaly types used: {dict(zip(*np.unique(types_list, return_counts=True)))}")

    return synthetic_weeks, labels, types_list
