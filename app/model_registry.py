"""
Model Registry for Multi-Combo LSTM-AE Anomaly Detection

Manages 4 independent LSTM Encoder-Decoder models, one per
network/transaction type combination. Provides unified interface
for training, inference, saving, and loading all models.

Each combo has:
- Its own EncDecAD model (trained on that combo's data)
- Its own AnomalyScorer (fitted on that combo's error distribution)
- Its own StandardScaler (from preprocessing)
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from app.lstm_autoencoder import EncDecAD, ModelConfig, create_model
from app.anomaly_scorer import AnomalyScorer, ScorerConfig
from app.transaction_config import COMBO_KEYS, TRANSACTION_SCORER_DEFAULTS

logger = logging.getLogger(__name__)


@dataclass
class TransactionModelConfig:
    """
    Model configuration optimized for transaction frequency data.

    Based on plan.md Section 4.1:
    - Shorter sequences (24 vs 336) allow smaller hidden dim
    - Single layer sufficient for daily patterns
    - Light dropout for regularization

    Day-of-week conditioning:
    - input_dim=3: transaction count + sin(DoW) + cos(DoW)
    - hidden_dim=12: restores bottleneck ratio h/L ≈ 0.5
    """
    input_dim: int = 3           # Transaction count + DoW sin/cos features
    hidden_dim: int = 18         # Bottleneck for DoW conditioning (was 32, trying 16)
    num_layers: int = 1          # Single layer sufficient
    dropout: float = 0.15        # Light regularization
    sequence_length: int = 24    # Daily windows


def combo_to_dirname(combo: Tuple[str, str]) -> str:
    """Convert combo tuple to valid directory name."""
    network, txn_type = combo
    return f"{network}_{txn_type.replace('-', '')}"


def dirname_to_combo(dirname: str) -> Tuple[str, str]:
    """Convert directory name back to combo tuple."""
    parts = dirname.split("_")
    network = parts[0]
    txn_type = "no-pin" if parts[1] == "nopin" else parts[1]
    return (network, txn_type)


class ModelRegistry:
    """
    Registry managing 4 independent LSTM-AE models for transaction anomaly detection.

    Provides a unified interface for:
    - Creating and accessing models per combo
    - Training individual or all models
    - Saving/loading complete model state
    - Updating scorers with new data (continual learning)
    """

    def __init__(
        self,
        model_config: Optional[TransactionModelConfig] = None,
        scorer_config: Optional[ScorerConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the registry.

        Args:
            model_config: Configuration for LSTM-AE models
            scorer_config: Configuration for anomaly scorers
            device: Torch device (defaults to CUDA if available)
        """
        self.model_config = model_config or TransactionModelConfig()
        self.scorer_config = scorer_config or ScorerConfig(
            scoring_mode="point",
            hard_criterion_k=TRANSACTION_SCORER_DEFAULTS["hard_criterion_k"],
            threshold_percentile=TRANSACTION_SCORER_DEFAULTS["threshold_percentile"],
        )
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Per-combo storage
        self.models: Dict[Tuple[str, str], EncDecAD] = {}
        self.scorers: Dict[Tuple[str, str], AnomalyScorer] = {}
        self.scalers: Dict[Tuple[str, str], StandardScaler] = {}
        self.training_histories: Dict[Tuple[str, str], Dict] = {}

        # Model versioning for continual learning
        self.model_versions: Dict[Tuple[str, str], int] = {
            combo: 0 for combo in COMBO_KEYS
        }

        logger.info(f"Initialized ModelRegistry with device={self.device}")
        logger.info(f"  Model config: hidden_dim={self.model_config.hidden_dim}, "
                    f"num_layers={self.model_config.num_layers}")
        logger.info(f"  Scorer config: mode={self.scorer_config.scoring_mode}, "
                    f"k={self.scorer_config.hard_criterion_k}")

    def _create_model_for_combo(self, combo: Tuple[str, str]) -> EncDecAD:
        """Create a fresh model for a combo."""
        # Convert our config to the base ModelConfig
        base_config = ModelConfig(
            input_dim=self.model_config.input_dim,
            hidden_dim=self.model_config.hidden_dim,
            num_layers=self.model_config.num_layers,
            dropout=self.model_config.dropout,
            sequence_length=self.model_config.sequence_length,
        )
        model = EncDecAD(config=base_config)
        model.to(self.device)
        return model

    def _create_scorer_for_combo(self, combo: Tuple[str, str]) -> AnomalyScorer:
        """Create a fresh scorer for a combo."""
        return AnomalyScorer(config=self.scorer_config)

    def get_model(self, combo: Tuple[str, str]) -> EncDecAD:
        """
        Get the model for a combo, creating if needed.

        Args:
            combo: (network_type, transaction_type) tuple

        Returns:
            EncDecAD model for the combo
        """
        if combo not in self.models:
            self.models[combo] = self._create_model_for_combo(combo)
            logger.debug(f"Created new model for {combo}")
        return self.models[combo]

    def get_scorer(self, combo: Tuple[str, str]) -> AnomalyScorer:
        """
        Get the scorer for a combo, creating if needed.

        Args:
            combo: (network_type, transaction_type) tuple

        Returns:
            AnomalyScorer for the combo
        """
        if combo not in self.scorers:
            self.scorers[combo] = self._create_scorer_for_combo(combo)
            logger.debug(f"Created new scorer for {combo}")
        return self.scorers[combo]

    def get_scaler(self, combo: Tuple[str, str]) -> Optional[StandardScaler]:
        """
        Get the scaler for a combo.

        Args:
            combo: (network_type, transaction_type) tuple

        Returns:
            StandardScaler if set, None otherwise
        """
        return self.scalers.get(combo)

    def set_scaler(self, combo: Tuple[str, str], scaler: StandardScaler) -> None:
        """Set the scaler for a combo (from preprocessor)."""
        self.scalers[combo] = scaler

    def get_model_version(self, combo: Tuple[str, str]) -> int:
        """Get the current model version for a combo."""
        return self.model_versions.get(combo, 0)

    def increment_model_version(self, combo: Tuple[str, str]) -> int:
        """Increment and return the new model version."""
        self.model_versions[combo] = self.model_versions.get(combo, 0) + 1
        return self.model_versions[combo]

    def train_combo(
        self,
        combo: Tuple[str, str],
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        patience: int = 10,
        grad_clip: float = 1.0,
    ) -> Dict:
        """
        Train the model for a single combo.

        Args:
            combo: (network_type, transaction_type) tuple
            train_loader: DataLoader with training data
            val_loader: DataLoader with validation data
            epochs: Maximum training epochs
            learning_rate: Learning rate
            patience: Early stopping patience
            grad_clip: Gradient clipping max norm

        Returns:
            Training history dict
        """
        model = self.get_model(combo)
        model.train()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        history = {
            "train_loss": [],
            "val_loss": [],
            "best_epoch": 0,
        }

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                x = batch.to(self.device)
                optimizer.zero_grad()
                x_reconstructed = model(x)
                loss = criterion(x_reconstructed, x)
                loss.backward()

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()
                train_loss += loss.item()
                num_batches += 1

            train_loss /= num_batches

            # Validation phase
            model.eval()
            val_loss = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    x = batch.to(self.device)
                    x_reconstructed = model(x)
                    loss = criterion(x_reconstructed, x)
                    val_loss += loss.item()
                    num_val_batches += 1

            val_loss /= num_val_batches

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # Check for best model
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                history["best_epoch"] = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            # Logging every 10 epochs or at end
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.debug(
                    f"  {combo} Epoch {epoch + 1:3d}: "
                    f"train={train_loss:.6f}, val={val_loss:.6f}"
                )

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"  {combo}: Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.to(self.device)

        self.training_histories[combo] = history
        logger.info(
            f"  {combo}: Training complete. Best val loss: {best_val_loss:.6f} "
            f"at epoch {history['best_epoch']}"
        )

        return history

    def fit_scorer(
        self,
        combo: Tuple[str, str],
        val_loader: DataLoader,
        threshold_loader: Optional[DataLoader] = None,
    ) -> None:
        """
        Fit the scorer for a combo on validation data.

        Args:
            combo: (network_type, transaction_type) tuple
            val_loader: DataLoader with validation data (for error distribution)
            threshold_loader: Optional DataLoader for threshold calibration
        """
        model = self.get_model(combo)
        scorer = self.get_scorer(combo)

        # Fit error distribution on validation data
        # Pass val_loader for both fitting and μ/σ computation to prevent overfitting
        scorer.fit(model, val_loader, self.device, val_loader=val_loader)

        # Set threshold using threshold_val or val data
        data_loader = threshold_loader or val_loader

        if self.scorer_config.scoring_mode == "point":
            point_scores, window_scores, _ = scorer.compute_point_scores(
                model, data_loader, self.device
            )
            scorer.set_point_threshold(point_scores)
            scorer.set_threshold(window_scores)
        else:
            scores, _ = scorer.compute_scores(model, data_loader, self.device)
            scorer.set_threshold(scores)

        logger.info(f"  {combo}: Scorer fitted. Point threshold: {scorer.point_threshold:.4f}")

    def fit_scorer_with_calibration(
        self,
        combo: Tuple[str, str],
        val_loader: DataLoader,
        preprocessor: "TransactionPreprocessor",
        normal_day_indices: List[int],
        spike_day_index: int,
        dip_day_index: int,
        spike_hours: Tuple[int, int] = (2, 5),
        dip_hours: Tuple[int, int] = (10, 14),
        magnitude_sigma: float = 2.0,
    ) -> Dict:
        """
        Fit scorer with balanced calibration using F1-max optimization.

        Two-step process:
        1. Fit error distribution (μ, σ) on validation data (normal only)
        2. Calibrate threshold (τ) using normal + synthetic anomaly windows with F1-max

        Args:
            combo: (network_type, transaction_type) tuple
            val_loader: DataLoader with validation data (for error distribution fitting)
            preprocessor: TransactionPreprocessor with window data
            normal_day_indices: Day indices for normal calibration windows (e.g., [20, 21, 22] for Sun-Tue)
            spike_day_index: Day index to inject spike anomaly (e.g., 23 for Wed)
            dip_day_index: Day index to inject dip anomaly (e.g., 19 for Sat)
            spike_hours: (start, end) hours for spike injection
            dip_hours: (start, end) hours for dip injection
            magnitude_sigma: Magnitude of anomalies in std units

        Returns:
            Dict with calibration metrics (thresholds, scores, F1)
        """
        from app.synthetic_anomaly import SyntheticAnomalyGenerator, SyntheticAnomalyConfig
        from app.transaction_preprocessor import TimeSeriesDataset

        model = self.get_model(combo)
        scorer = self.get_scorer(combo)

        # Step 1: Fit error distribution on validation data (normal only)
        # Pass val_loader for both fitting and μ/σ computation to prevent overfitting
        scorer.fit(model, val_loader, self.device, val_loader=val_loader)
        if scorer.sigma_point is not None:
            logger.info(f"  {combo}: Error distribution fitted (μ={scorer.mu_point[0]:.4f}, σ²={scorer.sigma_point[0]:.4f})")
        else:
            logger.info(f"  {combo}: Error distribution fitted (μ shape={scorer.mu_point.shape}, cov shape={scorer.cov_point.shape})")

        # Step 2: Extract calibration windows from preprocessor
        all_windows = preprocessor.combo_windows[combo]  # Shape: (30, 24, 3) with DoW conditioning
        window_info = preprocessor.combo_window_info[combo]

        # Check if windows have DoW features (3D)
        has_dow_features = all_windows.ndim == 3 and all_windows.shape[2] == 3

        # Get normal windows for calibration
        normal_windows = []
        for day_idx in normal_day_indices:
            if day_idx < len(all_windows):
                normal_windows.append(all_windows[day_idx].copy())
                logger.debug(f"    Normal: Day {day_idx} ({window_info[day_idx]['day_name']})")

        normal_windows = np.array(normal_windows)

        # Get windows for anomaly injection
        if spike_day_index >= len(all_windows) or dip_day_index >= len(all_windows):
            raise ValueError(f"Day indices out of range: spike={spike_day_index}, dip={dip_day_index}")

        spike_source = all_windows[spike_day_index].copy()
        dip_source = all_windows[dip_day_index].copy()

        logger.debug(f"    Spike source: Day {spike_day_index} ({window_info[spike_day_index]['day_name']})")
        logger.debug(f"    Dip source: Day {dip_day_index} ({window_info[dip_day_index]['day_name']})")

        # Create synthetic anomalies - inject into channel 0 only (transaction count)
        generator = SyntheticAnomalyGenerator(SyntheticAnomalyConfig())
        if has_dow_features:
            # Extract channel 0 for anomaly injection, preserve DoW features
            spike_counts = spike_source[:, 0].copy()
            dip_counts = dip_source[:, 0].copy()

            spike_counts = generator.inject_spike_at_hours(
                spike_counts, spike_hours[0], spike_hours[1], magnitude_sigma
            )
            dip_counts = generator.inject_dip_at_hours(
                dip_counts, dip_hours[0], dip_hours[1], magnitude_sigma
            )

            # Reconstruct 3D windows with injected anomalies
            spike_window = spike_source.copy()
            spike_window[:, 0] = spike_counts
            dip_window = dip_source.copy()
            dip_window[:, 0] = dip_counts
        else:
            spike_window = generator.inject_spike_at_hours(
                spike_source, spike_hours[0], spike_hours[1], magnitude_sigma
            )
            dip_window = generator.inject_dip_at_hours(
                dip_source, dip_hours[0], dip_hours[1], magnitude_sigma
            )

        anomaly_windows = np.array([spike_window, dip_window])

        # Normalize windows using the combo's scaler (channel 0 only for 3D)
        scaler = self.scalers[combo]
        if has_dow_features:
            # Normalize channel 0 only, preserve DoW features
            normal_norm = normal_windows.copy()
            normal_counts = normal_windows[:, :, 0].flatten().reshape(-1, 1)
            normal_counts_norm = scaler.transform(normal_counts)
            normal_norm[:, :, 0] = normal_counts_norm.reshape(normal_windows.shape[0], normal_windows.shape[1])

            anomaly_norm = anomaly_windows.copy()
            anomaly_counts = anomaly_windows[:, :, 0].flatten().reshape(-1, 1)
            anomaly_counts_norm = scaler.transform(anomaly_counts)
            anomaly_norm[:, :, 0] = anomaly_counts_norm.reshape(anomaly_windows.shape[0], anomaly_windows.shape[1])
        else:
            normal_norm = scaler.transform(normal_windows.reshape(-1, 1)).reshape(normal_windows.shape)
            anomaly_norm = scaler.transform(anomaly_windows.reshape(-1, 1)).reshape(anomaly_windows.shape)

        # Create DataLoaders
        normal_dataset = TimeSeriesDataset(normal_norm)
        anomaly_dataset = TimeSeriesDataset(anomaly_norm)
        normal_loader = DataLoader(normal_dataset, batch_size=len(normal_norm), shuffle=False)
        anomaly_loader = DataLoader(anomaly_dataset, batch_size=len(anomaly_norm), shuffle=False)

        # Compute scores
        model.eval()
        normal_point_scores, normal_window_scores, _ = scorer.compute_point_scores(
            model, normal_loader, self.device
        )
        anomaly_point_scores, anomaly_window_scores, _ = scorer.compute_point_scores(
            model, anomaly_loader, self.device
        )

        logger.info(f"  {combo}: Calibration scores:")
        logger.info(f"    Normal window scores: {normal_window_scores}")
        logger.info(f"    Anomaly window scores: {anomaly_window_scores}")

        # Step 3: Set thresholds
        # For point threshold: use percentile on normal scores (not F1-max)
        # This prevents the threshold from being too aggressive
        scorer.set_point_threshold(normal_point_scores)

        # For window threshold: use F1-max to find optimal separation
        optimal_window_threshold, window_metrics = scorer.find_optimal_threshold(
            normal_window_scores,
            anomaly_window_scores,
            method="f1_max"
        )
        scorer.threshold = optimal_window_threshold

        logger.info(f"  {combo}: Calibration complete:")
        logger.info(f"    Point threshold (97th pct): {scorer.point_threshold:.4f}")
        logger.info(f"    Window threshold (F1-max): {optimal_window_threshold:.4f}")
        logger.info(f"    Window F1 on calibration: {window_metrics.get('f1', 0):.3f}")

        return {
            "normal_window_scores": normal_window_scores.tolist(),
            "anomaly_window_scores": anomaly_window_scores.tolist(),
            "point_threshold": float(scorer.point_threshold),
            "window_threshold": optimal_window_threshold,
            "window_metrics": window_metrics,
        }

    def predict(
        self,
        combo: Tuple[str, str],
        data_loader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run prediction for a combo.

        Args:
            combo: (network_type, transaction_type) tuple
            data_loader: DataLoader with data to predict

        Returns:
            Tuple of (predictions, window_scores, point_scores)
        """
        model = self.get_model(combo)
        scorer = self.get_scorer(combo)

        model.eval()

        if self.scorer_config.scoring_mode == "point":
            point_scores, window_scores, _ = scorer.compute_point_scores(
                model, data_loader, self.device
            )
            point_predictions = scorer.predict_points(point_scores)
            predictions = scorer.predict_windows_from_points(point_predictions)
        else:
            window_scores, _ = scorer.compute_scores(model, data_loader, self.device)
            predictions = scorer.predict(window_scores)
            point_scores = window_scores  # No per-point scores in window mode

        return predictions, window_scores, point_scores

    def save_all(self, output_dir: str) -> None:
        """
        Save all models, scorers, and scalers.

        Directory structure:
        output_dir/
            Accel_CMP/
                model.pt
                scorer.pkl
                scaler.pkl
                history.pkl
            Accel_nopin/
                ...

        Args:
            output_dir: Base directory for saving
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for combo in COMBO_KEYS:
            combo_dir = output_dir / combo_to_dirname(combo)
            combo_dir.mkdir(parents=True, exist_ok=True)

            # Save model
            if combo in self.models:
                model = self.models[combo]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_config": model.config,
                    "model_version": self.model_versions.get(combo, 0),
                }, combo_dir / "model.pt")

            # Save scorer
            if combo in self.scorers:
                self.scorers[combo].save(combo_dir / "scorer.pkl")

            # Save scaler
            if combo in self.scalers:
                with open(combo_dir / "scaler.pkl", "wb") as f:
                    pickle.dump(self.scalers[combo], f)

            # Save training history
            if combo in self.training_histories:
                with open(combo_dir / "history.pkl", "wb") as f:
                    pickle.dump(self.training_histories[combo], f)

            logger.debug(f"Saved artifacts for {combo}")

        # Save registry config
        with open(output_dir / "registry_config.pkl", "wb") as f:
            pickle.dump({
                "model_config": self.model_config,
                "scorer_config": self.scorer_config,
                "model_versions": self.model_versions,
            }, f)

        logger.info(f"Saved all models to {output_dir}")

    def load_all(self, model_dir: str) -> None:
        """
        Load all models, scorers, and scalers.

        Args:
            model_dir: Base directory containing saved models
        """
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Load registry config if exists
        config_path = model_dir / "registry_config.pkl"
        if config_path.exists():
            with open(config_path, "rb") as f:
                config = pickle.load(f)
                self.model_config = config.get("model_config", self.model_config)
                self.scorer_config = config.get("scorer_config", self.scorer_config)
                self.model_versions = config.get("model_versions", self.model_versions)

        # Allow ModelConfig for safe loading
        torch.serialization.add_safe_globals([ModelConfig])

        for combo in COMBO_KEYS:
            combo_dir = model_dir / combo_to_dirname(combo)
            if not combo_dir.exists():
                logger.warning(f"No saved model for {combo}")
                continue

            # Load model
            model_path = combo_dir / "model.pt"
            if model_path.exists():
                checkpoint = torch.load(
                    model_path, map_location=self.device, weights_only=True
                )
                model = EncDecAD(config=checkpoint["model_config"])
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(self.device)
                model.eval()
                self.models[combo] = model

                if "model_version" in checkpoint:
                    self.model_versions[combo] = checkpoint["model_version"]

            # Load scorer
            scorer_path = combo_dir / "scorer.pkl"
            if scorer_path.exists():
                self.scorers[combo] = AnomalyScorer.load(scorer_path)

            # Load scaler
            scaler_path = combo_dir / "scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self.scalers[combo] = pickle.load(f)

            # Load history
            history_path = combo_dir / "history.pkl"
            if history_path.exists():
                with open(history_path, "rb") as f:
                    self.training_histories[combo] = pickle.load(f)

            logger.debug(f"Loaded artifacts for {combo}")

        logger.info(f"Loaded models from {model_dir}")

    def get_stats(self) -> Dict:
        """Get registry statistics."""
        stats = {
            "device": str(self.device),
            "model_config": {
                "hidden_dim": self.model_config.hidden_dim,
                "num_layers": self.model_config.num_layers,
                "sequence_length": self.model_config.sequence_length,
            },
            "scorer_config": {
                "scoring_mode": self.scorer_config.scoring_mode,
                "hard_criterion_k": self.scorer_config.hard_criterion_k,
            },
            "combos": {},
        }

        for combo in COMBO_KEYS:
            combo_stats = {
                "has_model": combo in self.models,
                "has_scorer": combo in self.scorers,
                "has_scaler": combo in self.scalers,
                "model_version": self.model_versions.get(combo, 0),
            }

            if combo in self.scorers and self.scorers[combo].is_fitted:
                scorer = self.scorers[combo]
                combo_stats["point_threshold"] = float(scorer.point_threshold) if scorer.point_threshold else None

            if combo in self.training_histories:
                history = self.training_histories[combo]
                combo_stats["best_epoch"] = history.get("best_epoch", 0)
                combo_stats["final_val_loss"] = history["val_loss"][-1] if history["val_loss"] else None

            stats["combos"][f"{combo[0]}_{combo[1]}"] = combo_stats

        return stats
