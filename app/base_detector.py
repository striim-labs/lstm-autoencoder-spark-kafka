"""
Base Detector Interface for Anomaly Detection

Provides an abstract interface that both IsolationForest and LSTM-based
detectors implement, allowing seamless switching between detection methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional

import pandas as pd


@dataclass
class DetectionResult:
    """
    Standard result format for anomaly detection.

    Attributes:
        anomalies: DataFrame with detected anomalies (timestamp, value, score)
        total_anomalies: Count of anomalies detected
        metadata: Additional detector-specific information
    """
    anomalies: pd.DataFrame
    total_anomalies: int
    metadata: Dict[str, Any]


class BaseDetector(ABC):
    """
    Abstract base class for anomaly detectors.

    All detector implementations must inherit from this class and
    implement the required methods to ensure consistent behavior
    across different detection algorithms.
    """

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run anomaly detection on the provided data.

        Args:
            df: DataFrame with at least 'timestamp' and 'value' columns

        Returns:
            DataFrame of detected anomalies with columns:
                - timestamp: When the anomaly occurred
                - value: The anomalous value
                - anomaly_score: Confidence/severity score
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get detector statistics and configuration.

        Returns:
            Dictionary with detector-specific statistics
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the detector name for display purposes.

        Returns:
            Human-readable detector name
        """
        pass

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """
        Check if the detector is ready to perform detection.

        For some detectors (like LSTM), this may require a minimum
        amount of data or a pre-trained model to be loaded.

        Returns:
            True if detector can perform detection
        """
        pass

    @property
    @abstractmethod
    def min_samples_required(self) -> int:
        """
        Minimum number of samples required for detection.

        Returns:
            Minimum sample count needed before detection can run
        """
        pass


def create_detector(
    detector_type: str,
    **kwargs
) -> BaseDetector:
    """
    Factory function to create a detector based on type.

    Args:
        detector_type: One of "isolation_forest" or "lstm"
        **kwargs: Detector-specific configuration

    Returns:
        Configured detector instance

    Raises:
        ValueError: If detector_type is not recognized
    """
    detector_type = detector_type.lower().strip()

    if detector_type == "isolation_forest":
        from isolation_forest_detector import IsolationForestDetector, DetectorConfig

        config = DetectorConfig(
            window_size=kwargs.get("window_size", 200),
            min_samples=kwargs.get("min_samples", 50),
            contamination=kwargs.get("contamination", 0.05),
            n_estimators=kwargs.get("n_estimators", 100),
        )
        return IsolationForestDetector(config=config)

    elif detector_type == "lstm":
        from streaming_detector import LSTMStreamingDetector

        return LSTMStreamingDetector(
            model_path=kwargs.get("model_path", "models/lstm_model.pt"),
            scaler_path=kwargs.get("scaler_path", "models/scaler.pkl"),
            scorer_path=kwargs.get("scorer_path", "models/scorer.pkl"),
            window_size=kwargs.get("window_size", 336),
            min_samples=kwargs.get("min_samples", 336),
        )

    elif detector_type == "fcvae":
        from fcvae_streaming_detector import FCVAEStreamingDetector

        return FCVAEStreamingDetector(
            model_path=kwargs.get("model_path", "models/transactions_fcvae"),
            scaler_path=kwargs.get("scaler_path"),
            scorer_path=kwargs.get("scorer_path"),
            combo=kwargs.get("combo"),
            window_size=kwargs.get("window_size", 24),
            min_samples=kwargs.get("min_samples", 24),
            n_samples=kwargs.get("n_samples", 16),
        )

    else:
        raise ValueError(
            f"Unknown detector type: {detector_type}. "
            f"Must be one of: 'isolation_forest', 'lstm', 'fcvae'"
        )
