"""
MLP Classifier for Credit Card Fraud Detection

Implements the MLP classifier that operates on autoencoder latent features.
Architecture: 15 → 13 → 7 → 1 (as specified in Misra et al. paper)
"""

import logging
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class MLPConfig:
    """Configuration for MLP classifier."""

    input_dim: int = 15       # Latent dimension from autoencoder
    hidden_dim1: int = 13     # First hidden layer
    hidden_dim2: int = 7      # Second hidden layer
    output_dim: int = 1       # Binary classification
    dropout: float = 0.0      # Dropout rate


class MLPClassifier(nn.Module):
    """
    MLP classifier for fraud detection on autoencoder latent features.

    Architecture: 15 → 13 → 7 → 1
    - Input: Latent representation from autoencoder (15 dimensions)
    - Output: Fraud probability (sigmoid activation)
    - Loss: Binary cross-entropy
    """

    def __init__(self, config: Optional[MLPConfig] = None):
        """
        Args:
            config: MLP configuration (uses defaults if None)
        """
        super().__init__()
        self.config = config or MLPConfig()

        # Layer 1: 15 → 13
        self.fc1 = nn.Linear(self.config.input_dim, self.config.hidden_dim1)
        self.dropout1 = nn.Dropout(self.config.dropout)

        # Layer 2: 13 → 7
        self.fc2 = nn.Linear(self.config.hidden_dim1, self.config.hidden_dim2)
        self.dropout2 = nn.Dropout(self.config.dropout)

        # Output layer: 7 → 1
        self.fc3 = nn.Linear(self.config.hidden_dim2, self.config.output_dim)

        logger.info(f"Created MLPClassifier:")
        logger.info(f"  Architecture: {self.config.input_dim} → {self.config.hidden_dim1} → "
                   f"{self.config.hidden_dim2} → {self.config.output_dim}")
        logger.info(f"  Parameters: {self.count_parameters():,}")
        logger.info(f"  Dropout: {self.config.dropout}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Latent features, shape (batch, input_dim)

        Returns:
            Fraud probability, shape (batch, 1)
        """
        # Hidden layer 1
        h1 = F.relu(self.fc1(x))
        h1 = self.dropout1(h1)

        # Hidden layer 2
        h2 = F.relu(self.fc2(h1))
        h2 = self.dropout2(h2)

        # Output layer with sigmoid
        logits = self.fc3(h2)
        probs = torch.sigmoid(logits)

        return probs

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary labels.

        Args:
            x: Latent features, shape (batch, input_dim)
            threshold: Classification threshold (default: 0.5)

        Returns:
            Binary predictions, shape (batch,)
        """
        probs = self.forward(x)
        predictions = (probs.squeeze() >= threshold).long()
        return predictions

    def count_parameters(self) -> int:
        """
        Count trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        """
        Get configuration as dictionary.

        Returns:
            Configuration dict
        """
        return asdict(self.config)


def create_mlp_classifier(
    input_dim: int = 15,
    hidden_dim1: int = 13,
    hidden_dim2: int = 7,
    dropout: float = 0.0
) -> MLPClassifier:
    """
    Factory function to create MLP classifier with custom configuration.

    Args:
        input_dim: Input dimension (default: 15, latent from autoencoder)
        hidden_dim1: First hidden layer dimension (default: 13)
        hidden_dim2: Second hidden layer dimension (default: 7)
        dropout: Dropout rate (default: 0.0)

    Returns:
        Configured MLPClassifier
    """
    config = MLPConfig(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        dropout=dropout
    )
    return MLPClassifier(config=config)


def main():
    """Test the MLP classifier architecture."""
    import argparse

    parser = argparse.ArgumentParser(description="Test MLP classifier")
    parser.add_argument("--input-dim", type=int, default=15, help="Input dimension")
    parser.add_argument("--hidden-dim1", type=int, default=13, help="First hidden dimension")
    parser.add_argument("--hidden-dim2", type=int, default=7, help="Second hidden dimension")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for testing")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create model
    model = create_mlp_classifier(
        input_dim=args.input_dim,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2
    )

    # Test forward pass
    logger.info(f"\nTesting forward pass with batch size {args.batch_size}")
    x = torch.randn(args.batch_size, args.input_dim)
    logger.info(f"  Input shape: {x.shape}")

    # Get probabilities
    probs = model(x)
    logger.info(f"  Output shape: {probs.shape}")
    logger.info(f"  Output range: [{probs.min():.3f}, {probs.max():.3f}]")

    # Get predictions
    predictions = model.predict(x, threshold=0.5)
    logger.info(f"  Predictions shape: {predictions.shape}")
    logger.info(f"  Unique predictions: {predictions.unique()}")

    # Verify shapes
    assert probs.shape == (args.batch_size, 1), "Output shape mismatch!"
    assert predictions.shape == (args.batch_size,), "Predictions shape mismatch!"

    logger.info("\n✓ All shape checks passed!")

    # Parameter breakdown
    logger.info("\nParameter breakdown:")
    for name, param in model.named_parameters():
        logger.info(f"  {name}: {param.shape} ({param.numel():,} params)")


if __name__ == "__main__":
    main()
