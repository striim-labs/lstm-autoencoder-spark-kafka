"""
Data Augmentation for FCVAE Training

Adapted from FCVAE/data_augment.py (Wang et al. WWW 2024).

Changes from original:
1. Device-agnostic: removed hardcoded .to("cuda")
2. Adjusted seg_ano start range for W=24 (kernel_size=8 minimum)
3. Added AugmentConfig dataclass for cleaner API
4. Added batch_augment wrapper matching model.py training pattern
"""
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class AugmentConfig:
    """Configuration for FCVAE data augmentation.

    Default values from FCVAE model.py:214-216.
    """
    missing_data_rate: float = 0.01
    point_ano_rate: float = 0.05
    seg_ano_rate: float = 0.1


def missing_data_injection(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    rate: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomly zero out points to simulate missing data.

    Zeros out approximately (rate * B * 1 * W) random points across the batch.
    Updates the missing mask z to indicate which points were zeroed.

    Args:
        x: Input tensor (B, 1, W) - time series windows
        y: Labels tensor (B, W) - anomaly labels
        z: Missing mask tensor (B, W) - 1 where data is missing
        rate: Probability of zeroing each point

    Returns:
        Modified (x, y, z) tuple - modifies in-place but also returns
    """
    miss_size = int(rate * x.shape[0] * x.shape[1] * x.shape[2])
    if miss_size == 0:
        return x, y, z

    row = torch.randint(low=0, high=x.shape[0], size=(miss_size,), device=x.device)
    col = torch.randint(low=0, high=x.shape[2], size=(miss_size,), device=x.device)
    x[row, :, col] = 0
    z[row, col] = 1
    return x, y, z


def point_ano(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    rate: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create augmented samples with point anomalies at the last timestep.

    Samples (rate * B) windows from the batch, clones them, and adds
    random noise in range [-10, 10] (excluding 0) to the last point.

    Args:
        x: Input tensor (B, 1, W)
        y: Labels tensor (B, W)
        z: Missing mask tensor (B, W)
        rate: Fraction of batch to augment

    Returns:
        Augmented (x_aug, y_aug, z_aug) - NEW tensors to concatenate with original
    """
    aug_size = int(rate * x.shape[0])
    if aug_size == 0:
        return x[:0], y[:0], z[:0]  # Return empty tensors with correct shape

    id_x = torch.randint(low=0, high=x.shape[0], size=(aug_size,), device=x.device)
    x_aug = x[id_x].clone()
    y_aug = y[id_x].clone()
    z_aug = z[id_x].clone()

    if x_aug.shape[1] == 1:
        # Create noise: half positive [0.5, 10], half negative [-10, -0.5]
        half = aug_size // 2
        ano_noise1 = torch.randint(low=1, high=20, size=(half,), device=x.device)
        ano_noise2 = torch.randint(low=-20, high=-1, size=(aug_size - half,), device=x.device)
        ano_noise = torch.cat((ano_noise1, ano_noise2), dim=0).float() / 2

        # Add noise to last point only
        x_aug[:, 0, -1] += ano_noise

        # Mark last point as anomalous
        y_aug[:, -1] = torch.logical_or(
            y_aug[:, -1].bool(),
            torch.ones_like(y_aug[:, -1], dtype=torch.bool)
        ).to(y_aug.dtype)

    return x_aug, y_aug, z_aug


def seg_ano(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    rate: float,
    method: str = "swap"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create augmented samples by swapping tail segments between windows.

    Samples (rate * B) pairs of different windows and swaps the tail segment
    from one to the other, starting at a random position.

    For W=24, start positions range from 6 to 18, ensuring swapped segments
    are 6-18 hours long (meaningful for daily patterns).

    Args:
        x: Input tensor (B, 1, W)
        y: Labels tensor (B, W)
        z: Missing mask tensor (B, W)
        rate: Fraction of batch to augment
        method: Swap method ("swap" is the only supported method)

    Returns:
        Augmented (x_aug, y_aug, z_aug) - NEW tensors to concatenate with original
    """
    aug_size = int(rate * x.shape[0])
    if aug_size == 0 or x.shape[0] < 2:
        return x[:0], y[:0], z[:0]

    # Find pairs of different windows
    idx_1 = torch.arange(aug_size, device=x.device)
    idx_2 = torch.arange(aug_size, device=x.device)
    max_attempts = 100
    attempts = 0
    while torch.any(idx_1 == idx_2) and attempts < max_attempts:
        idx_1 = torch.randint(low=0, high=x.shape[0], size=(aug_size,), device=x.device)
        idx_2 = torch.randint(low=0, high=x.shape[0], size=(aug_size,), device=x.device)
        attempts += 1

    x_aug = x[idx_1].clone()
    y_aug = y[idx_1].clone()
    z_aug = z[idx_1].clone()

    # For W=24: start between 6 and W-6=18 to ensure meaningful segment swap
    # This means swapped segments will be 6-18 hours long
    window_size = x.shape[2]
    min_start = min(6, window_size // 4)  # At least 25% of window
    max_start = max(min_start + 1, window_size - 6)  # Leave at least 6 points

    time_start = torch.randint(
        low=min_start,
        high=max_start,
        size=(aug_size,),
        device=x.device
    )

    for i in range(len(idx_2)):
        if method == "swap":
            start = time_start[i].item()
            x_aug[i, :, start:] = x[idx_2[i], :, start:]
            # Mark swapped region as anomalous
            y_aug[i, start:] = torch.logical_or(
                y_aug[i, start:].bool(),
                torch.ones_like(y_aug[i, start:], dtype=torch.bool)
            ).to(y_aug.dtype)

    return x_aug, y_aug, z_aug


def batch_augment(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    config: AugmentConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply all augmentations to a batch, concatenating augmented samples.

    This matches the pattern from FCVAE model.py:batch_data_augmentation().
    Augmentations are applied in order:
    1. Point anomalies (adds new samples)
    2. Segment anomalies (adds new samples)
    3. Missing data injection (modifies existing + augmented samples)

    Args:
        x: Input tensor (B, 1, W)
        y: Labels tensor (B, W)
        z: Missing mask tensor (B, W)
        config: Augmentation configuration

    Returns:
        Augmented (x, y, z) with potentially larger batch size
    """
    # Point anomaly augmentation
    if config.point_ano_rate > 0:
        x_a, y_a, z_a = point_ano(x, y, z, config.point_ano_rate)
        if x_a.shape[0] > 0:
            x = torch.cat((x, x_a), dim=0)
            y = torch.cat((y, y_a), dim=0)
            z = torch.cat((z, z_a), dim=0)

    # Segment swap augmentation
    if config.seg_ano_rate > 0:
        x_a, y_a, z_a = seg_ano(x, y, z, config.seg_ano_rate, method="swap")
        if x_a.shape[0] > 0:
            x = torch.cat((x, x_a), dim=0)
            y = torch.cat((y, y_a), dim=0)
            z = torch.cat((z, z_a), dim=0)

    # Missing data injection (modifies in-place)
    x, y, z = missing_data_injection(x, y, z, config.missing_data_rate)

    return x, y, z
