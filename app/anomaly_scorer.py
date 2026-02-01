"""
Anomaly Scorer for LSTM Encoder-Decoder

Implements the scoring methodology from Malhotra et al. (2016):
1. Fit error distribution (μ, Σ) on normal training data
2. Compute Mahalanobis distance as anomaly score
3. Set threshold using validation data
4. Predict anomalies based on threshold

Uses the full covariance matrix to capture dependencies
between reconstruction errors at different timesteps.
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

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


class AnomalyScorer:
    """
    Computes anomaly scores using Mahalanobis distance.

    The scorer learns the distribution of reconstruction errors on
    normal (training) data, fitting a multivariate Gaussian with
    mean μ and covariance Σ.

    Anomaly Score (Mahalanobis distance):
        a(i) = (e(i) - μ)ᵀ Σ⁻¹ (e(i) - μ)

    Where:
        - e(i) is the reconstruction error vector for sequence i
        - μ is the mean error vector (learned from training)
        - Σ is the covariance matrix of errors (learned from training)

    Attributes:
        mu: Mean reconstruction error vector, shape (seq_len,)
        cov: Covariance matrix of errors, shape (seq_len, seq_len)
        cov_inv: Inverse covariance matrix (precision matrix)
        threshold: Anomaly score threshold for classification
    """

    def __init__(self, config: Optional[ScorerConfig] = None):
        self.config = config or ScorerConfig()
        self.mu: Optional[np.ndarray] = None
        self.cov: Optional[np.ndarray] = None
        self.cov_inv: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None
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

        Learns the multivariate Gaussian parameters:
        - μ: Mean error vector at each timestep
        - Σ: Full covariance matrix capturing error dependencies

        Args:
            model: Trained LSTM encoder-decoder model
            train_loader: DataLoader with normal training sequences
            device: Device to run inference on (cpu/cuda)
        """
        model.eval()
        all_errors = []

        logger.info("Fitting error distribution on training data...")

        with torch.no_grad():
            for batch in train_loader:
                x = batch.to(device)
                x_reconstructed = model(x)

                # Point-wise absolute error: (batch, seq_len, 1)
                errors = torch.abs(x - x_reconstructed)
                all_errors.append(errors.cpu().numpy())

        # Concatenate all batches: (num_sequences, seq_len, 1)
        all_errors = np.concatenate(all_errors, axis=0)
        # Remove feature dimension: (num_sequences, seq_len)
        all_errors = all_errors.squeeze(-1)

        num_sequences, seq_len = all_errors.shape

        # Compute mean error vector
        self.mu = np.mean(all_errors, axis=0)  # (seq_len,)

        # Compute full covariance matrix
        # Center the errors
        errors_centered = all_errors - self.mu  # (num_sequences, seq_len)

        # Covariance: (seq_len, seq_len)
        self.cov = np.cov(errors_centered, rowvar=False)

        # Regularize covariance for numerical stability
        # Add small value to diagonal to ensure positive definiteness
        self.cov += self.config.regularization * np.eye(seq_len)

        # Compute inverse covariance (precision matrix)
        try:
            self.cov_inv = np.linalg.inv(self.cov)
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix singular, using pseudo-inverse")
            self.cov_inv = np.linalg.pinv(self.cov)

        self.is_fitted = True

        logger.info(f"Fitted on {num_sequences} sequences of length {seq_len}")
        logger.info(f"  Mean error range: [{self.mu.min():.4f}, {self.mu.max():.4f}]")
        logger.info(f"  Covariance matrix shape: {self.cov.shape}")
        logger.info(f"  Covariance condition number: {np.linalg.cond(self.cov):.2e}")

    def _mahalanobis_distance(self, error: np.ndarray) -> float:
        """
        Compute Mahalanobis distance for a single error vector.

        d² = (e - μ)ᵀ Σ⁻¹ (e - μ)

        Args:
            error: Error vector, shape (seq_len,)

        Returns:
            Squared Mahalanobis distance (anomaly score)
        """
        diff = error - self.mu  # (seq_len,)
        # d² = diff @ Σ⁻¹ @ diff
        return diff @ self.cov_inv @ diff

    def compute_scores(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute anomaly scores for sequences using Mahalanobis distance.

        Args:
            model: Trained LSTM encoder-decoder model
            data_loader: DataLoader with sequences to score
            device: Device to run inference on

        Returns:
            Tuple of:
                - sequence_scores: Mahalanobis distance per sequence, shape (num_sequences,)
                - reconstruction_errors: Raw errors, shape (num_sequences, seq_len)
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

                # Point-wise absolute error: (batch, seq_len)
                errors = torch.abs(x - x_reconstructed).cpu().numpy().squeeze(-1)

                # Compute Mahalanobis distance for each sequence in batch
                for error in errors:
                    score = self._mahalanobis_distance(error)
                    all_scores.append(score)

                all_errors.append(errors)

        return np.array(all_scores), np.concatenate(all_errors, axis=0)

    def set_threshold(
        self,
        val_scores: np.ndarray,
        method: Optional[str] = None,
        percentile: Optional[float] = None,
        sigma_k: Optional[float] = None
    ) -> float:
        """
        Determine anomaly threshold from validation scores.

        The validation set should contain only normal sequences.
        The threshold is set such that most normal sequences fall below it.

        Methods:
            - "percentile": Use nth percentile of validation scores
            - "sigma": Use mean + k*std of validation scores

        Args:
            val_scores: Anomaly scores from normal validation sequences
            method: Threshold method (default from config)
            percentile: Percentile value for percentile method
            sigma_k: Number of std deviations for sigma method

        Returns:
            The computed threshold value
        """
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
            logger.info(
                f"Set threshold using μ + {sigma_k}σ: {self.threshold:.4f} "
                f"(μ={mean_score:.4f}, σ={std_score:.4f})"
            )
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        return self.threshold

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """
        Predict anomalies based on threshold.

        Args:
            scores: Anomaly scores, shape (num_sequences,)

        Returns:
            Boolean array where True = anomaly, shape (num_sequences,)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")

        return scores > self.threshold

    def score_and_predict(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device
    ) -> Dict[str, np.ndarray]:
        """
        Convenience method to compute scores and predictions in one call.

        Args:
            model: Trained LSTM encoder-decoder model
            data_loader: DataLoader with sequences to evaluate
            device: Device to run inference on

        Returns:
            Dict with keys:
                - scores: Sequence-level anomaly scores (Mahalanobis distance)
                - errors: Raw reconstruction errors
                - predictions: Boolean anomaly predictions
        """
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
            "config": {
                "threshold_method": self.config.threshold_method,
                "threshold_percentile": self.config.threshold_percentile,
                "threshold_sigma": self.config.threshold_sigma,
            }
        }

        if self.mu is not None:
            stats["mu_range"] = (float(self.mu.min()), float(self.mu.max()))
            stats["cov_shape"] = self.cov.shape
            stats["cov_condition_number"] = float(np.linalg.cond(self.cov))

        return stats

    def save(self, filepath: str) -> None:
        """
        Save scorer state to file.

        Args:
            filepath: Path to save the scorer
        """
        state = {
            "mu": self.mu,
            "cov": self.cov,
            "cov_inv": self.cov_inv,
            "threshold": self.threshold,
            "config": self.config,
            "is_fitted": self.is_fitted,
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Saved scorer to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "AnomalyScorer":
        """
        Load scorer state from file.

        Args:
            filepath: Path to load the scorer from

        Returns:
            Loaded AnomalyScorer instance
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        scorer = cls(config=state["config"])
        scorer.mu = state["mu"]
        scorer.cov = state["cov"]
        scorer.cov_inv = state["cov_inv"]
        scorer.threshold = state["threshold"]
        scorer.is_fitted = state["is_fitted"]

        logger.info(f"Loaded scorer from {filepath}")
        return scorer


def main():
    """Test the anomaly scorer with dummy data."""
    import sys
    sys.path.insert(0, ".")

    from lstm_autoencoder import create_model
    from data_preprocessor import NYCTaxiPreprocessor

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("\n" + "=" * 60)
    print("ANOMALY SCORER TEST")
    print("=" * 60)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load and preprocess data
    print("\nLoading data...")
    preprocessor = NYCTaxiPreprocessor()
    dataloaders, _ = preprocessor.preprocess("data/nyc_taxi.csv", batch_size=4)

    # Create model (untrained - just for testing scorer mechanics)
    print("\nCreating model...")
    model = create_model(hidden_dim=64, num_layers=1)
    model.to(device)

    # Create scorer
    print("\nTesting AnomalyScorer...")
    scorer = AnomalyScorer()

    # Fit on training data
    print("\n1. Fitting error distribution on training data...")
    scorer.fit(model, dataloaders["train"], device)

    print(f"\nScorer stats after fit:")
    for key, value in scorer.get_stats().items():
        print(f"  {key}: {value}")

    # Compute scores on validation data (for threshold)
    print("\n2. Computing scores on validation data...")
    val_scores, val_errors = scorer.compute_scores(
        model, dataloaders["threshold_val"], device
    )
    print(f"  Validation scores shape: {val_scores.shape}")
    print(f"  Score range: [{val_scores.min():.4f}, {val_scores.max():.4f}]")

    # Set threshold
    print("\n3. Setting threshold...")
    threshold = scorer.set_threshold(val_scores, method="percentile", percentile=95)
    print(f"  Threshold (95th percentile): {threshold:.4f}")

    # Compute scores on test data
    print("\n4. Computing scores on test data...")
    test_scores, test_errors = scorer.compute_scores(
        model, dataloaders["test"], device
    )
    print(f"  Test scores shape: {test_scores.shape}")
    print(f"  Score range: [{test_scores.min():.4f}, {test_scores.max():.4f}]")

    # Predict anomalies
    print("\n5. Predicting anomalies...")
    predictions = scorer.predict(test_scores)
    num_anomalies = predictions.sum()
    print(f"  Predicted anomalies: {num_anomalies}/{len(predictions)}")

    # Test save/load
    print("\n6. Testing save/load...")
    scorer.save("/tmp/test_scorer.pkl")
    loaded_scorer = AnomalyScorer.load("/tmp/test_scorer.pkl")
    print(f"  Loaded threshold: {loaded_scorer.threshold:.4f}")
    assert loaded_scorer.threshold == scorer.threshold, "Load failed!"

    # Get test week info for context
    print("\n7. Test week anomaly status:")
    test_week_info = preprocessor.get_test_week_info()
    for i, (score, pred, week) in enumerate(zip(test_scores, predictions, test_week_info)):
        status = "ANOMALY" if pred else "normal"
        actual = "HAS_ANOMALY" if week["is_anomaly"] else "normal"
        match = "✓" if (pred == week["is_anomaly"]) else "✗"
        print(f"  Week {week['year_week']}: score={score:.2f}, pred={status:7s}, actual={actual:10s} {match}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
