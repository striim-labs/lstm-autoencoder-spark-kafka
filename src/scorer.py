"""
Anomaly Scorer for LSTM Encoder-Decoder

Implements the scoring methodology from Malhotra et al. (2016):
1. Fit error distribution (mu, Sigma) on normal training data
2. Compute Mahalanobis distance as anomaly score
3. Set threshold using validation data
4. Predict anomalies based on threshold
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class ScorerConfig:
    """Configuration for the anomaly scorer."""
    threshold_method: str = "percentile"  # "percentile" or "sigma"
    threshold_percentile: float = 95.0    # For percentile method
    threshold_sigma: float = 3.0          # For sigma method (k * std)
    regularization: float = 1e-6          # Regularization for covariance inversion
    scoring_mode: str = "point"           # "point" (paper) or "window" (legacy)
    hard_criterion_k: int = 5             # Points exceeding threshold to flag window


class AnomalyScorer:
    """
    Computes anomaly scores using Mahalanobis distance.

    The scorer learns the distribution of reconstruction errors on
    normal (training) data, fitting a multivariate Gaussian with
    mean mu and covariance Sigma.

    Anomaly Score:
        a(i) = (e(i) - mu)^T Sigma^-1 (e(i) - mu)
    """

    def __init__(self, config: Optional[ScorerConfig] = None):
        self.config = config or ScorerConfig()
        # Window-level attributes (legacy)
        self.mu: Optional[np.ndarray] = None
        self.cov: Optional[np.ndarray] = None
        self.cov_inv: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None
        # Point-level attributes (Malhotra et al. 2016)
        self.mu_point: Optional[np.ndarray] = None
        self.sigma_point: Optional[np.ndarray] = None
        self.sigma_inv_point: Optional[np.ndarray] = None
        self.cov_point: Optional[np.ndarray] = None
        self.cov_inv_point: Optional[np.ndarray] = None
        self.point_threshold: Optional[float] = None
        self.is_fitted: bool = False

        logger.info(f"Initialized AnomalyScorer with config: {self.config}")

    def fit(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        device: torch.device
    ) -> None:
        """
        Fit the error distribution on training (normal) data.

        Supports two modes:
        - "point": Pool all points across windows, compute mu and sigma per feature
        - "window": Compute mu and Sigma across time axis
        """
        model.eval()
        all_errors = []

        logger.info(f"Fitting error distribution on training data (mode={self.config.scoring_mode})...")

        with torch.no_grad():
            for batch in train_loader:
                x = batch.to(device)
                x_reconstructed = model(x)
                errors = torch.abs(x - x_reconstructed)
                all_errors.append(errors.cpu().numpy())

        all_errors = np.concatenate(all_errors, axis=0)
        num_sequences, seq_len, num_features = all_errors.shape

        if self.config.scoring_mode == "point":
            all_errors_pooled = all_errors.reshape(-1, num_features)
            self.mu_point = np.mean(all_errors_pooled, axis=0)

            if num_features == 1:
                self.sigma_point = np.var(all_errors_pooled, axis=0) + self.config.regularization
                self.sigma_inv_point = 1.0 / self.sigma_point
                logger.info(f"Fitted point-level scorer (univariate):")
                logger.info(f"  mu_point: {self.mu_point[0]:.6f}")
                logger.info(f"  sigma_point: {self.sigma_point[0]:.6f}")
            else:
                errors_centered = all_errors_pooled - self.mu_point
                self.cov_point = np.cov(errors_centered, rowvar=False)
                self.cov_point += self.config.regularization * np.eye(num_features)
                try:
                    self.cov_inv_point = np.linalg.inv(self.cov_point)
                except np.linalg.LinAlgError:
                    logger.warning("Point-level covariance singular, using pseudo-inverse")
                    self.cov_inv_point = np.linalg.pinv(self.cov_point)
                logger.info(f"Fitted point-level scorer (multivariate, m={num_features})")

            # Also fit window-level for backward compatibility
            all_errors_squeezed = all_errors.squeeze(-1)
            self.mu = np.mean(all_errors_squeezed, axis=0)
            errors_centered = all_errors_squeezed - self.mu
            self.cov = np.cov(errors_centered, rowvar=False)
            self.cov += self.config.regularization * np.eye(seq_len)
            try:
                self.cov_inv = np.linalg.inv(self.cov)
            except np.linalg.LinAlgError:
                self.cov_inv = np.linalg.pinv(self.cov)

        else:
            all_errors = all_errors.squeeze(-1)
            self.mu = np.mean(all_errors, axis=0)
            errors_centered = all_errors - self.mu
            self.cov = np.cov(errors_centered, rowvar=False)
            self.cov += self.config.regularization * np.eye(seq_len)
            try:
                self.cov_inv = np.linalg.inv(self.cov)
            except np.linalg.LinAlgError:
                logger.warning("Covariance matrix singular, using pseudo-inverse")
                self.cov_inv = np.linalg.pinv(self.cov)

            logger.info(f"Fitted window-level scorer:")
            logger.info(f"  Mean error range: [{self.mu.min():.4f}, {self.mu.max():.4f}]")
            logger.info(f"  Covariance matrix shape: {self.cov.shape}")

        self.is_fitted = True
        logger.info(f"Fitted on {num_sequences} sequences of length {seq_len}")

    def _mahalanobis_distance(self, error: np.ndarray) -> float:
        """Compute Mahalanobis distance for a single error vector."""
        diff = error - self.mu
        return diff @ self.cov_inv @ diff

    def compute_scores(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute window-level anomaly scores using Mahalanobis distance.

        Returns:
            Tuple of (sequence_scores, reconstruction_errors)
        """
        if not self.is_fitted:
            raise ValueError("Scorer not fitted. Call fit() first.")

        model.eval()
        all_scores = []
        all_errors = []

        with torch.no_grad():
            for batch in data_loader:
                x = batch.to(device)
                x_reconstructed = model(x)
                errors = torch.abs(x - x_reconstructed).cpu().numpy().squeeze(-1)
                for error in errors:
                    score = self._mahalanobis_distance(error)
                    all_scores.append(score)
                all_errors.append(errors)

        return np.array(all_scores), np.concatenate(all_errors, axis=0)

    def compute_point_scores(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute per-point anomaly scores (Malhotra et al. 2016).

        For univariate: a^(i) = ((e^(i) - mu) / sigma)^2

        Returns:
            Tuple of (point_scores, window_scores, reconstruction_errors)
        """
        if not self.is_fitted:
            raise ValueError("Scorer not fitted. Call fit() first.")
        if self.mu_point is None:
            raise ValueError("Point-level parameters not fitted. Ensure scoring_mode='point' during fit().")

        model.eval()
        all_point_scores = []
        all_errors = []

        with torch.no_grad():
            for batch in data_loader:
                x = batch.to(device)
                x_reconstructed = model(x)
                errors = torch.abs(x - x_reconstructed).cpu().numpy()

                if errors.shape[-1] == 1:
                    errors_squeezed = errors.squeeze(-1)
                    point_scores = ((errors_squeezed - self.mu_point[0]) ** 2) / self.sigma_point[0]
                else:
                    diff = errors - self.mu_point
                    point_scores = np.einsum('ijk,kl,ijl->ij', diff, self.cov_inv_point, diff)

                all_point_scores.append(point_scores)
                all_errors.append(errors)

        point_scores = np.concatenate(all_point_scores, axis=0)
        errors = np.concatenate(all_errors, axis=0)
        window_scores = np.max(point_scores, axis=1)

        return point_scores, window_scores, errors

    def set_threshold(
        self,
        val_scores: np.ndarray,
        method: Optional[str] = None,
        percentile: Optional[float] = None,
        sigma_k: Optional[float] = None
    ) -> float:
        """Set window-level anomaly threshold from validation scores."""
        method = method or self.config.threshold_method
        percentile = percentile or self.config.threshold_percentile
        sigma_k = sigma_k or self.config.threshold_sigma

        if method == "percentile":
            self.threshold = np.percentile(val_scores, percentile)
            logger.info(f"Set threshold using {percentile}th percentile: {self.threshold:.4f}")
        elif method == "sigma":
            mean_score = np.mean(val_scores)
            std_score = np.std(val_scores)
            self.threshold = mean_score + sigma_k * std_score
            logger.info(f"Set threshold using mu + {sigma_k}sigma: {self.threshold:.4f}")
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        return self.threshold

    def set_point_threshold(
        self,
        val_point_scores: np.ndarray,
        method: Optional[str] = None,
        percentile: Optional[float] = None,
        sigma_k: Optional[float] = None
    ) -> float:
        """Set point-level anomaly threshold from validation scores."""
        method = method or self.config.threshold_method
        percentile = percentile or self.config.threshold_percentile
        sigma_k = sigma_k or self.config.threshold_sigma

        all_scores = val_point_scores.flatten()

        if method == "percentile":
            self.point_threshold = float(np.percentile(all_scores, percentile))
            logger.info(f"Set point threshold using {percentile}th percentile: {self.point_threshold:.4f}")
        elif method == "sigma":
            mean_score = np.mean(all_scores)
            std_score = np.std(all_scores)
            self.point_threshold = float(mean_score + sigma_k * std_score)
            logger.info(f"Set point threshold using mu + {sigma_k}sigma: {self.point_threshold:.4f}")
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        return self.point_threshold

    @staticmethod
    def find_optimal_threshold(
        normal_scores: np.ndarray,
        anomaly_scores: np.ndarray,
        method: str = "midpoint",
        beta: float = 1.0
    ) -> Tuple[float, Dict]:
        """
        Find optimal threshold using labeled normal and anomaly scores.

        Methods:
            - "midpoint": Midpoint between max(normal) and min(anomaly)
            - "f1_max": Search for threshold that maximizes F-beta score
            - "youden": Maximize Youden's J statistic
        """
        max_normal = np.max(normal_scores)
        min_anomaly = np.min(anomaly_scores)
        gap = min_anomaly - max_normal

        metrics = {
            "method": method,
            "max_normal": float(max_normal),
            "min_anomaly": float(min_anomaly),
            "gap": float(gap),
            "separable": gap > 0,
        }

        if method == "midpoint":
            if gap > 0:
                threshold = (max_normal + min_anomaly) / 2
                metrics["threshold_source"] = "midpoint"
            else:
                threshold = float(np.percentile(normal_scores, 99))
                metrics["threshold_source"] = "fallback_percentile_99"
                logger.warning(
                    f"Distributions overlap (gap={gap:.2f}), "
                    f"falling back to 99th percentile: {threshold:.4f}"
                )
            return threshold, metrics

        elif method == "f1_max":
            all_scores = np.concatenate([normal_scores, anomaly_scores])
            labels = np.concatenate([
                np.zeros(len(normal_scores)),
                np.ones(len(anomaly_scores))
            ])
            candidates = np.percentile(all_scores, np.arange(1, 100, 0.5))

            best_f_beta = 0
            best_threshold = float(np.median(all_scores))

            for candidate in candidates:
                predictions = all_scores > candidate
                tp = np.sum(predictions & labels.astype(bool))
                fp = np.sum(predictions & ~labels.astype(bool))
                fn = np.sum(~predictions & labels.astype(bool))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                if precision + recall > 0:
                    f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
                else:
                    f_beta = 0

                if f_beta > best_f_beta:
                    best_f_beta = f_beta
                    best_threshold = float(candidate)

            metrics["best_f_beta"] = float(best_f_beta)
            metrics["beta"] = beta
            return best_threshold, metrics

        elif method == "youden":
            all_scores = np.concatenate([normal_scores, anomaly_scores])
            labels = np.concatenate([
                np.zeros(len(normal_scores)),
                np.ones(len(anomaly_scores))
            ])
            candidates = np.percentile(all_scores, np.arange(1, 100, 0.5))

            best_j = -1
            best_threshold = float(np.median(all_scores))

            for candidate in candidates:
                predictions = all_scores > candidate
                tp = np.sum(predictions & labels.astype(bool))
                tn = np.sum(~predictions & ~labels.astype(bool))
                fp = np.sum(predictions & ~labels.astype(bool))
                fn = np.sum(~predictions & labels.astype(bool))

                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                j = sensitivity + specificity - 1

                if j > best_j:
                    best_j = j
                    best_threshold = float(candidate)

            metrics["best_youden_j"] = float(best_j)
            return best_threshold, metrics

        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Predict anomalies based on window-level threshold."""
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")
        return scores > self.threshold

    def predict_points(self, point_scores: np.ndarray) -> np.ndarray:
        """Predict point-level anomalies based on point threshold."""
        if self.point_threshold is None:
            raise ValueError("Point threshold not set. Call set_point_threshold() first.")
        return point_scores > self.point_threshold

    def predict_windows_from_points(
        self,
        point_predictions: np.ndarray,
        k: Optional[int] = None
    ) -> np.ndarray:
        """
        Derive window-level labels using HardCriterion.

        A window is flagged as anomalous if k or more points exceed the threshold.
        """
        k = k or self.config.hard_criterion_k
        anomalous_point_counts = np.sum(point_predictions, axis=1)
        window_predictions = anomalous_point_counts >= k

        logger.debug(
            f"HardCriterion (k={k}): {window_predictions.sum()}/{len(window_predictions)} "
            f"windows flagged as anomalous"
        )
        return window_predictions

    def localize_anomaly(
        self,
        point_scores: np.ndarray,
        timestamps: List[str],
        scales_hours: List[int] = None
    ) -> Dict[str, Any]:
        """
        Localize anomaly within a flagged window using sliding window analysis.

        After a window is flagged, identifies the sub-window where anomalous
        behavior is most concentrated using a fixed 6-hour window.
        """
        if scales_hours is None:
            scales_hours = [1, 2, 3, 6, 12, 24, 48]

        seq_len = len(point_scores)
        if seq_len != len(timestamps):
            raise ValueError(f"point_scores length ({seq_len}) != timestamps length ({len(timestamps)})")

        prefix_sum = np.zeros(seq_len + 1)
        prefix_sum[1:] = np.cumsum(point_scores)

        all_scales_results = {}

        for h in scales_hours:
            w = h * 2  # hours to samples (30-min intervals)
            if w > seq_len:
                continue

            num_positions = seq_len - w + 1
            sliding_sums = prefix_sum[w:seq_len + 1] - prefix_sum[:num_positions]

            t_star = int(np.argmax(sliding_sums))
            peak_score = float(sliding_sums[t_star])
            mean_sum = float(np.mean(sliding_sums))
            rho = peak_score / mean_sum if mean_sum > 0 else 0.0

            all_scales_results[h] = {
                "t_star": t_star,
                "t_end": t_star + w - 1,
                "peak_score": peak_score,
                "mean_score": mean_sum,
                "contrast_ratio": rho,
                "window_samples": w,
            }

        # Fixed 6h window selection
        best_scale_h = 6
        best_w = 12

        if best_scale_h not in all_scales_results:
            best_scale_h = min(all_scales_results.keys()) if all_scales_results else 6
            best_w = best_scale_h * 2

        error_threshold = 0.2
        spike_multiplier = 8.0

        num_positions = seq_len - best_w + 1
        best_t_star = 0
        best_count = -1
        best_has_spike = False

        for t in range(num_positions):
            window_scores = point_scores[t:t + best_w]
            window_mean = np.mean(window_scores)

            count_above = np.sum(window_scores > error_threshold)
            has_spike = np.any(window_scores >= spike_multiplier * window_mean) if window_mean > 0 else False

            if has_spike and not best_has_spike:
                best_t_star = t
                best_count = count_above
                best_has_spike = True
            elif has_spike == best_has_spike and count_above > best_count:
                best_t_star = t
                best_count = count_above
                best_has_spike = has_spike

        best_rho = all_scales_results.get(best_scale_h, {}).get("contrast_ratio", 0.0)

        anomaly_start_idx = best_t_star
        anomaly_end_idx = min(best_t_star + best_w - 1, seq_len - 1)

        return {
            "anomaly_start": timestamps[anomaly_start_idx],
            "anomaly_end": timestamps[anomaly_end_idx],
            "anomaly_start_idx": anomaly_start_idx,
            "anomaly_end_idx": anomaly_end_idx,
            "scale_hours": best_scale_h,
            "scale_samples": best_w,
            "contrast_ratio": best_rho,
            "peak_score": all_scales_results.get(best_scale_h, {}).get("peak_score", 0.0),
            "all_scales": all_scales_results,
        }

    def score_and_predict(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device
    ) -> Dict[str, np.ndarray]:
        """Compute scores and predictions in one call."""
        scores, errors = self.compute_scores(model, data_loader, device)
        predictions = self.predict(scores)
        return {
            "scores": scores,
            "errors": errors,
            "predictions": predictions,
        }

    def get_stats(self) -> Dict:
        """Get scorer statistics."""
        stats = {
            "is_fitted": self.is_fitted,
            "threshold": self.threshold,
            "scoring_mode": self.config.scoring_mode,
            "config": {
                "threshold_method": self.config.threshold_method,
                "threshold_percentile": self.config.threshold_percentile,
                "threshold_sigma": self.config.threshold_sigma,
                "hard_criterion_k": self.config.hard_criterion_k,
            }
        }

        if self.mu is not None:
            stats["mu_range"] = (float(self.mu.min()), float(self.mu.max()))
            stats["cov_shape"] = self.cov.shape
            stats["cov_condition_number"] = float(np.linalg.cond(self.cov))

        if self.mu_point is not None:
            stats["mu_point"] = float(self.mu_point[0]) if len(self.mu_point) == 1 else self.mu_point.tolist()
            stats["sigma_point"] = float(self.sigma_point[0]) if self.sigma_point is not None and len(self.sigma_point) == 1 else None
            stats["point_threshold"] = self.point_threshold

        return stats

    def save(self, filepath: str) -> None:
        """Save scorer state to file."""
        state = {
            "mu": self.mu,
            "cov": self.cov,
            "cov_inv": self.cov_inv,
            "threshold": self.threshold,
            "config": self.config,
            "is_fitted": self.is_fitted,
            "mu_point": self.mu_point,
            "sigma_point": self.sigma_point,
            "sigma_inv_point": self.sigma_inv_point,
            "cov_point": self.cov_point,
            "cov_inv_point": self.cov_inv_point,
            "point_threshold": self.point_threshold,
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Saved scorer to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "AnomalyScorer":
        """Load scorer state from file."""
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        scorer = cls(config=state["config"])
        scorer.mu = state["mu"]
        scorer.cov = state["cov"]
        scorer.cov_inv = state["cov_inv"]
        scorer.threshold = state["threshold"]
        scorer.is_fitted = state["is_fitted"]

        scorer.mu_point = state.get("mu_point")
        scorer.sigma_point = state.get("sigma_point")
        scorer.sigma_inv_point = state.get("sigma_inv_point")
        scorer.cov_point = state.get("cov_point")
        scorer.cov_inv_point = state.get("cov_inv_point")
        scorer.point_threshold = state.get("point_threshold")

        logger.info(f"Loaded scorer from {filepath}")
        if scorer.mu_point is not None:
            logger.info(f"  Point-level scoring enabled (mu_point={scorer.mu_point[0]:.6f})")
        return scorer
