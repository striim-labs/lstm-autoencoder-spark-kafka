"""
FCVAE Anomaly Scorer

Implements NLL-based scoring for FCVAE models.

Key differences from AnomalyScorer (Mahalanobis-based):
- Score type: Negative log-likelihood (not Mahalanobis distance)
- Score semantics: Lower (more negative) = anomalous (inverted from LSTM-AE)
- Threshold direction: score < threshold => anomaly
- No error distribution fitting needed (NLL computed directly by model)
"""
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class FCVAEScorerConfig:
    """Configuration for FCVAE anomaly scorer."""
    threshold_method: str = "percentile"    # "percentile" or "f1_max"
    threshold_percentile: float = 5.0       # LOW percentile (anomalies are low scores)
    hard_criterion_k: int = 3               # Points below threshold to flag window
    score_mode: str = "single_pass"         # "single_pass" (fast) or "mcmc" (accurate)
    n_samples: int = 16                     # Latent samples for single-pass scoring

    # Decision logic configuration for window-level predictions
    # Modes: "count_only" (original), "k1", "severity", "zscore", "hybrid"
    decision_mode: str = "count_only"
    severity_margin: float = 0.5            # For "severity"/"hybrid": additional margin below threshold
    outlier_z_threshold: float = 3.0        # For "zscore"/"hybrid": z-score threshold within window


class FCVAEScorer:
    """
    Computes anomaly scores using FCVAE negative log-likelihood.

    The FCVAE decoder outputs a distribution p(x|z) with parameters (μ_x, σ²_x).
    The anomaly score for each point is the negative log-likelihood:

        score = -0.5 * (log(σ²_x) + (x - μ_x)² / σ²_x)

    Averaging over multiple latent samples provides more stable estimates.

    IMPORTANT: Lower scores indicate anomalies (inverted from Mahalanobis).
    - Normal points have high (less negative) NLL
    - Anomalous points have low (more negative) NLL

    Threshold semantics: score < threshold => anomaly
    """

    def __init__(self, config: Optional[FCVAEScorerConfig] = None):
        self.config = config or FCVAEScorerConfig()

        # Threshold attributes
        self.point_threshold: Optional[float] = None  # Per-point threshold
        self.window_threshold: Optional[float] = None  # Per-window threshold

        # Score statistics from normal validation data
        self.normal_score_mean: Optional[float] = None
        self.normal_score_std: Optional[float] = None

        self.is_fitted: bool = False

        logger.info(f"Initialized FCVAEScorer with config: {self.config}")

    def fit(
        self,
        model: torch.nn.Module,
        val_loader: DataLoader,
        device: torch.device
    ) -> None:
        """
        Fit scorer by computing score statistics on normal validation data.

        Args:
            model: Trained FCVAE model
            val_loader: DataLoader with normal validation windows
            device: Device for inference
        """
        logger.info("Fitting FCVAEScorer on validation data...")

        # Compute scores on validation data
        point_scores, window_scores = self.score_batch(model, val_loader, device)

        # Store statistics for threshold setting
        all_point_scores = point_scores.flatten()
        self.normal_score_mean = float(np.mean(all_point_scores))
        self.normal_score_std = float(np.std(all_point_scores))

        self.is_fitted = True

        logger.info(
            f"FCVAEScorer fitted: mean={self.normal_score_mean:.4f}, "
            f"std={self.normal_score_std:.4f}"
        )

    def score_batch(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score all windows in a DataLoader.

        Args:
            model: Trained FCVAE model
            data_loader: DataLoader with windows (x, y, z format from SlidingWindowDataset)
            device: Device for inference

        Returns:
            Tuple of:
                - point_scores: (num_windows, window_size) per-point NLL scores
                - window_scores: (num_windows,) mean NLL per window
        """
        model.eval()
        all_point_scores = []

        with torch.no_grad():
            for batch in data_loader:
                # Handle both (x, y, z) tuple and tensor-only formats
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                else:
                    x = batch.to(device)

                # Score using configured mode
                if self.config.score_mode == "mcmc":
                    _, scores = model.score_mcmc(x)
                    scores = scores.squeeze(1).cpu().numpy()  # (B, W)
                else:
                    scores = model.score_single_pass(x, self.config.n_samples)
                    scores = scores.cpu().numpy()  # (B, W)

                all_point_scores.append(scores)

        point_scores = np.concatenate(all_point_scores, axis=0)
        window_scores = np.mean(point_scores, axis=1)

        return point_scores, window_scores

    def score_window(
        self,
        model: torch.nn.Module,
        window: np.ndarray,
        device: torch.device
    ) -> Tuple[float, np.ndarray]:
        """
        Score a single window.

        Args:
            model: Trained FCVAE model
            window: Window array (window_size,) or (1, window_size)
            device: Device for inference

        Returns:
            Tuple of (window_score, point_scores)
        """
        model.eval()

        # Prepare tensor: (1, 1, W)
        if window.ndim == 1:
            x = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0)
        elif window.ndim == 2:
            x = torch.FloatTensor(window).unsqueeze(0)
        else:
            x = torch.FloatTensor(window)

        x = x.to(device)

        with torch.no_grad():
            if self.config.score_mode == "mcmc":
                _, scores = model.score_mcmc(x)
                point_scores = scores.squeeze().cpu().numpy()
            else:
                scores = model.score_single_pass(x, self.config.n_samples)
                point_scores = scores.squeeze().cpu().numpy()

        window_score = float(np.mean(point_scores))
        return window_score, point_scores

    def set_threshold(
        self,
        normal_scores: np.ndarray,
        method: Optional[str] = None,
        percentile: Optional[float] = None
    ) -> float:
        """
        Set point threshold from normal validation scores.

        IMPORTANT: For FCVAE, anomalies have LOWER scores, so we use
        a LOW percentile (e.g., 5th percentile) as the threshold.
        Points scoring BELOW this threshold are flagged as anomalous.

        Args:
            normal_scores: Point scores from normal validation, shape (N, T) or (N*T,)
            method: "percentile" (default from config)
            percentile: Percentile value (default from config)

        Returns:
            Point threshold τ
        """
        method = method or self.config.threshold_method
        percentile = percentile if percentile is not None else self.config.threshold_percentile

        # Flatten all point scores
        all_scores = normal_scores.flatten()

        if method == "percentile":
            # Use LOW percentile because anomalies have low scores
            self.point_threshold = float(np.percentile(all_scores, percentile))
            logger.info(
                f"Set point threshold using {percentile}th percentile: "
                f"{self.point_threshold:.4f}"
            )
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        return self.point_threshold

    def set_window_threshold(
        self,
        normal_window_scores: np.ndarray,
        method: Optional[str] = None,
        percentile: Optional[float] = None
    ) -> float:
        """
        Set window-level threshold from normal validation scores.

        Args:
            normal_window_scores: Window scores from normal validation, shape (N,)
            method: "percentile" (default)
            percentile: Percentile value

        Returns:
            Window threshold
        """
        method = method or self.config.threshold_method
        percentile = percentile if percentile is not None else self.config.threshold_percentile

        if method == "percentile":
            self.window_threshold = float(np.percentile(normal_window_scores, percentile))
            logger.info(
                f"Set window threshold using {percentile}th percentile: "
                f"{self.window_threshold:.4f}"
            )
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        return self.window_threshold

    @staticmethod
    def find_optimal_threshold(
        normal_scores: np.ndarray,
        anomaly_scores: np.ndarray,
        method: str = "f1_max",
        beta: float = 1.0
    ) -> Tuple[float, Dict]:
        """
        Find optimal threshold using labeled normal and anomaly scores.

        IMPORTANT: For FCVAE, the comparison is INVERTED:
        - Anomalies have LOWER scores
        - Prediction: score < threshold => anomaly

        Methods:
            - "midpoint": Midpoint between min(normal) and max(anomaly)
            - "f1_max": Search for threshold that maximizes F1 score
            - "youden": Maximize Youden's J statistic

        Args:
            normal_scores: Scores from normal validation windows
            anomaly_scores: Scores from synthetic/real anomaly windows
            method: Calibration method
            beta: Beta for F-beta score

        Returns:
            Tuple of (optimal_threshold, calibration_metrics)
        """
        min_normal = np.min(normal_scores)
        max_anomaly = np.max(anomaly_scores)
        gap = min_normal - max_anomaly  # Gap is inverted for FCVAE

        metrics = {
            "method": method,
            "min_normal": float(min_normal),
            "max_anomaly": float(max_anomaly),
            "gap": float(gap),
            "separable": gap > 0,  # Normal scores should be > anomaly scores
        }

        if method == "midpoint":
            if gap > 0:
                # Perfect separation: normal scores > anomaly scores
                threshold = (min_normal + max_anomaly) / 2
                metrics["threshold_source"] = "midpoint"
            else:
                # Overlap exists - use low percentile of normal
                threshold = float(np.percentile(normal_scores, 5))
                metrics["threshold_source"] = "fallback_percentile_5"
                logger.warning(
                    f"Distributions overlap (gap={gap:.2f}), "
                    f"falling back to 5th percentile: {threshold:.4f}"
                )

            logger.info(f"Optimal threshold (midpoint): {threshold:.4f}")
            return threshold, metrics

        elif method == "f1_max":
            # Search over candidate thresholds
            all_scores = np.concatenate([normal_scores, anomaly_scores])
            labels = np.concatenate([
                np.zeros(len(normal_scores)),   # 0 = normal
                np.ones(len(anomaly_scores))    # 1 = anomaly
            ])

            # Generate candidate thresholds
            candidates = np.percentile(all_scores, np.arange(1, 100, 0.5))

            best_f_beta = 0
            best_threshold = float(np.median(all_scores))
            best_metrics = {}

            for candidate in candidates:
                # INVERTED comparison: score < threshold => predicted anomaly
                predictions = all_scores < candidate

                tp = np.sum(predictions & labels.astype(bool))
                fp = np.sum(predictions & ~labels.astype(bool))
                fn = np.sum(~predictions & labels.astype(bool))
                tn = np.sum(~predictions & ~labels.astype(bool))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                if precision + recall > 0:
                    f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
                else:
                    f_beta = 0

                if f_beta > best_f_beta:
                    best_f_beta = f_beta
                    best_threshold = float(candidate)
                    best_metrics = {
                        "precision": float(precision),
                        "recall": float(recall),
                        "tp": int(tp),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tn": int(tn),
                    }

            metrics["f1"] = float(best_f_beta)
            metrics["beta"] = beta
            metrics.update(best_metrics)

            logger.info(
                f"Optimal threshold (F{beta:.1f} max): {best_threshold:.4f} "
                f"(F1={best_f_beta:.4f}, P={best_metrics.get('precision', 0):.3f}, "
                f"R={best_metrics.get('recall', 0):.3f})"
            )
            return best_threshold, metrics

        elif method == "youden":
            # Maximize Youden's J = sensitivity + specificity - 1
            all_scores = np.concatenate([normal_scores, anomaly_scores])
            labels = np.concatenate([
                np.zeros(len(normal_scores)),
                np.ones(len(anomaly_scores))
            ])

            candidates = np.percentile(all_scores, np.arange(1, 100, 0.5))

            best_j = -1
            best_threshold = float(np.median(all_scores))

            for candidate in candidates:
                # INVERTED comparison
                predictions = all_scores < candidate

                tp = np.sum(predictions & labels.astype(bool))
                fp = np.sum(predictions & ~labels.astype(bool))
                fn = np.sum(~predictions & labels.astype(bool))
                tn = np.sum(~predictions & ~labels.astype(bool))

                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                j = sensitivity + specificity - 1

                if j > best_j:
                    best_j = j
                    best_threshold = float(candidate)

            metrics["youden_j"] = float(best_j)
            logger.info(f"Optimal threshold (Youden): {best_threshold:.4f} (J={best_j:.4f})")
            return best_threshold, metrics

        else:
            raise ValueError(f"Unknown method: {method}")

    def predict_points(self, point_scores: np.ndarray) -> np.ndarray:
        """
        Predict which points are anomalous based on threshold.

        INVERTED: score < threshold => anomaly (True)

        Args:
            point_scores: Per-point NLL scores, shape (N, T)

        Returns:
            Binary predictions, shape (N, T), True = anomaly
        """
        if self.point_threshold is None:
            raise ValueError("Point threshold not set. Call set_threshold() first.")

        # INVERTED comparison for FCVAE
        return point_scores < self.point_threshold

    def predict_windows_from_points(
        self,
        point_predictions: np.ndarray,
        point_scores: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict which windows are anomalous using configurable decision logic.

        Decision modes:
        - "count_only": Original k-count criterion (>= k points below threshold)
        - "k1": Same as count_only but with k=1 (any single point triggers)
        - "severity": k-count OR any point below (threshold - severity_margin)
        - "zscore": k-count OR any point with z-score < -outlier_z_threshold within window
        - "hybrid": k-count OR severity OR zscore

        Args:
            point_predictions: Binary per-point predictions, shape (N, T)
            point_scores: Per-point NLL scores, shape (N, T). Required for severity/zscore/hybrid modes.

        Returns:
            Binary window predictions, shape (N,), True = anomaly
        """
        mode = self.config.decision_mode
        k = self.config.hard_criterion_k

        # Count-based criterion (always computed)
        anomalous_point_counts = np.sum(point_predictions, axis=1)

        if mode == "k1":
            # Simple: any single point below threshold
            return anomalous_point_counts >= 1

        elif mode == "count_only":
            # Original behavior
            return anomalous_point_counts >= k

        elif mode == "severity":
            if point_scores is None:
                logger.warning("severity mode requires point_scores, falling back to count_only")
                return anomalous_point_counts >= k

            count_criterion = anomalous_point_counts >= k
            severe_threshold = self.point_threshold - self.config.severity_margin
            severity_criterion = np.any(point_scores < severe_threshold, axis=1)
            return count_criterion | severity_criterion

        elif mode == "zscore":
            if point_scores is None:
                logger.warning("zscore mode requires point_scores, falling back to count_only")
                return anomalous_point_counts >= k

            count_criterion = anomalous_point_counts >= k
            # Compute within-window z-scores
            window_means = np.mean(point_scores, axis=1, keepdims=True)
            window_stds = np.std(point_scores, axis=1, keepdims=True)
            z_scores = (point_scores - window_means) / (window_stds + 1e-8)
            # Negative z-score = lower than window mean = more anomalous
            zscore_criterion = np.any(z_scores < -self.config.outlier_z_threshold, axis=1)
            return count_criterion | zscore_criterion

        elif mode == "hybrid":
            if point_scores is None:
                logger.warning("hybrid mode requires point_scores, falling back to count_only")
                return anomalous_point_counts >= k

            count_criterion = anomalous_point_counts >= k

            # Severity criterion
            severe_threshold = self.point_threshold - self.config.severity_margin
            severity_criterion = np.any(point_scores < severe_threshold, axis=1)

            # Z-score criterion
            window_means = np.mean(point_scores, axis=1, keepdims=True)
            window_stds = np.std(point_scores, axis=1, keepdims=True)
            z_scores = (point_scores - window_means) / (window_stds + 1e-8)
            zscore_criterion = np.any(z_scores < -self.config.outlier_z_threshold, axis=1)

            return count_criterion | severity_criterion | zscore_criterion

        else:
            logger.warning(f"Unknown decision mode: {mode}, using count_only")
            return anomalous_point_counts >= k

    def predict_windows(self, window_scores: np.ndarray) -> np.ndarray:
        """
        Predict which windows are anomalous based on window threshold.

        INVERTED: score < threshold => anomaly

        Args:
            window_scores: Window-level scores, shape (N,)

        Returns:
            Binary predictions, shape (N,), True = anomaly
        """
        if self.window_threshold is None:
            raise ValueError("Window threshold not set. Call set_window_threshold() first.")

        return window_scores < self.window_threshold

    def save(self, path: str) -> None:
        """Save scorer state to file."""
        path = Path(path)
        state = {
            "config": self.config,
            "point_threshold": self.point_threshold,
            "window_threshold": self.window_threshold,
            "normal_score_mean": self.normal_score_mean,
            "normal_score_std": self.normal_score_std,
            "is_fitted": self.is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.debug(f"Saved FCVAEScorer to {path}")

    @classmethod
    def load(cls, path: str) -> "FCVAEScorer":
        """Load scorer from file."""
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)

        scorer = cls(config=state["config"])
        scorer.point_threshold = state["point_threshold"]
        scorer.window_threshold = state.get("window_threshold")
        scorer.normal_score_mean = state.get("normal_score_mean")
        scorer.normal_score_std = state.get("normal_score_std")
        scorer.is_fitted = state["is_fitted"]

        logger.debug(f"Loaded FCVAEScorer from {path}")
        return scorer

    def get_stats(self) -> Dict:
        """Get scorer statistics."""
        return {
            "config": {
                "threshold_method": self.config.threshold_method,
                "threshold_percentile": self.config.threshold_percentile,
                "hard_criterion_k": self.config.hard_criterion_k,
                "score_mode": self.config.score_mode,
                "n_samples": self.config.n_samples,
                "decision_mode": self.config.decision_mode,
                "severity_margin": self.config.severity_margin,
                "outlier_z_threshold": self.config.outlier_z_threshold,
            },
            "is_fitted": self.is_fitted,
            "point_threshold": self.point_threshold,
            "window_threshold": self.window_threshold,
            "normal_score_mean": self.normal_score_mean,
            "normal_score_std": self.normal_score_std,
        }
