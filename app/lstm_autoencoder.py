"""
LSTM Encoder-Decoder for Anomaly Detection (EncDec-AD)

Implementation based on Malhotra et al. (2016):
"LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection"

Architecture:
- Encoder LSTM compresses input sequence into latent representation
- Decoder LSTM reconstructs sequence in REVERSE order
- Teacher forcing used during training
- Reconstruction error used for anomaly scoring
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the LSTM Encoder-Decoder model."""
    input_dim: int = 1          # Number of features (1 for univariate)
    hidden_dim: int = 64        # LSTM hidden state dimension
    num_layers: int = 1         # Number of stacked LSTM layers
    dropout: float = 0.2        # Dropout rate (applied if num_layers > 1)
    sequence_length: int = 336  # Expected sequence length (one week)


class LSTMEncoder(nn.Module):
    """
    Encoder LSTM that compresses a sequence into a latent representation.

    The encoder reads the input sequence step by step and produces
    a final hidden state h(L) that captures a compressed representation
    of the entire sequence.

    Args:
        input_dim: Number of input features per timestep
        hidden_dim: Dimension of LSTM hidden state
        num_layers: Number of stacked LSTM layers
        dropout: Dropout probability (only used if num_layers > 1)
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input sequence into latent representation.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Tuple of (h_n, c_n):
                h_n: Final hidden state, shape (num_layers, batch, hidden_dim)
                c_n: Final cell state, shape (num_layers, batch, hidden_dim)
        """
        # outputs: (batch, seq_len, hidden_dim)
        # h_n: (num_layers, batch, hidden_dim)
        # c_n: (num_layers, batch, hidden_dim)
        outputs, (h_n, c_n) = self.lstm(x)

        return h_n, c_n


class LSTMDecoder(nn.Module):
    """
    Decoder LSTM that reconstructs the sequence from latent representation.

    Key detail from the paper: The decoder reconstructs the sequence
    in REVERSE order (x'(L), x'(L-1), ..., x'(1)).

    During training, teacher forcing is used - the decoder receives
    the actual target values as input rather than its own predictions.

    Args:
        input_dim: Number of output features per timestep
        hidden_dim: Dimension of LSTM hidden state
        num_layers: Number of stacked LSTM layers
        dropout: Dropout probability (only used if num_layers > 1)
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Linear layer to project hidden state to output dimension
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Decode latent representation into reconstructed sequence.

        Args:
            x: Decoder input (teacher forcing with reversed sequence)
               Shape: (batch, seq_len, input_dim)
            hidden: Tuple of (h_n, c_n) from encoder

        Returns:
            Reconstructed sequence, shape (batch, seq_len, input_dim)
        """
        # outputs: (batch, seq_len, hidden_dim)
        outputs, _ = self.lstm(x, hidden)

        # Project to output dimension
        # reconstructed: (batch, seq_len, input_dim)
        reconstructed = self.output_layer(outputs)

        return reconstructed


class EncDecAD(nn.Module):
    """
    Complete LSTM Encoder-Decoder model for Anomaly Detection.

    This implements the EncDec-AD architecture from Malhotra et al. (2016).

    Training mode:
        - Encoder compresses input sequence to latent state
        - Decoder receives reversed input sequence (teacher forcing)
        - Decoder reconstructs sequence in reverse order
        - Output is reversed back to original order for loss computation

    Inference mode:
        - Same as training (we still use the input for teacher forcing)
        - For true autoregressive generation, use generate() method

    Args:
        config: ModelConfig with architecture hyperparameters
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()

        self.config = config or ModelConfig()

        self.encoder = LSTMEncoder(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        )

        self.decoder = LSTMDecoder(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        )

        logger.info(f"Initialized EncDecAD with config: {self.config}")
        logger.info(f"Total parameters: {self.count_parameters():,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with teacher forcing.

        The decoder reconstructs the sequence in reverse order as per
        the paper. Teacher forcing means we feed the actual (reversed)
        input to the decoder rather than its own predictions.

        Args:
            x: Input sequence, shape (batch, seq_len, input_dim)

        Returns:
            Reconstructed sequence, shape (batch, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Encode: compress sequence into latent state
        h_n, c_n = self.encoder(x)

        # Prepare decoder input: reverse the sequence for teacher forcing
        # The decoder learns to reconstruct x(L), x(L-1), ..., x(1)
        decoder_input = torch.flip(x, dims=[1])

        # Decode: reconstruct sequence (in reverse order)
        reconstructed_reversed = self.decoder(decoder_input, (h_n, c_n))

        # Reverse back to original order for loss computation
        reconstructed = torch.flip(reconstructed_reversed, dims=[1])

        return reconstructed

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sequence into latent representation.

        Useful for extracting learned representations.

        Args:
            x: Input sequence, shape (batch, seq_len, input_dim)

        Returns:
            Tuple of (h_n, c_n) latent states
        """
        return self.encoder(x)

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate reconstruction autoregressively (without teacher forcing).

        This is used during inference when we want the model to generate
        the reconstruction using only its own predictions as input.

        Note: For anomaly detection, we typically still use teacher forcing
        even during inference, as the reconstruction error is what matters.

        Args:
            x: Input sequence, shape (batch, seq_len, input_dim)

        Returns:
            Autoregressively generated sequence, shape (batch, seq_len, input_dim)
        """
        batch_size, seq_len, input_dim = x.shape
        device = x.device

        # Encode
        h_n, c_n = self.encoder(x)

        # Initialize decoder input with the last timestep of input
        # (decoder generates in reverse, so starts with x(L))
        decoder_input = x[:, -1:, :]  # Shape: (batch, 1, input_dim)

        # Generate sequence autoregressively
        generated = []
        hidden = (h_n, c_n)

        for t in range(seq_len):
            # Decode one step
            output, hidden = self._decode_step(decoder_input, hidden)
            generated.append(output)

            # Use output as next input
            decoder_input = output

        # Stack and reverse to get original order
        # generated is [x'(L), x'(L-1), ..., x'(1)]
        generated = torch.cat(generated, dim=1)  # (batch, seq_len, input_dim)
        generated = torch.flip(generated, dims=[1])  # Reverse to original order

        return generated

    def _decode_step(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single decoding step for autoregressive generation.

        Args:
            x: Input for this step, shape (batch, 1, input_dim)
            hidden: Tuple of (h, c) hidden states

        Returns:
            Tuple of (output, new_hidden)
        """
        output, new_hidden = self.decoder.lstm(x, hidden)
        output = self.decoder.output_layer(output)
        return output, new_hidden

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        """Get model configuration as dictionary."""
        return {
            "input_dim": self.config.input_dim,
            "hidden_dim": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "dropout": self.config.dropout,
            "sequence_length": self.config.sequence_length,
            "total_parameters": self.count_parameters(),
        }


def create_model(
    input_dim: int = 1,
    hidden_dim: int = 64,
    num_layers: int = 1,
    dropout: float = 0.2,
    sequence_length: int = 336
) -> EncDecAD:
    """
    Factory function to create an EncDecAD model.

    Args:
        input_dim: Number of input features
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        sequence_length: Expected sequence length

    Returns:
        Configured EncDecAD model
    """
    config = ModelConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        sequence_length=sequence_length
    )
    return EncDecAD(config)


def main():
    """Test the model with dummy data."""
    import argparse

    parser = argparse.ArgumentParser(description="Test LSTM Encoder-Decoder")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=336)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create model
    model = create_model(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        sequence_length=args.seq_len
    )

    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE TEST")
    print("=" * 60)

    # Print model summary
    print(f"\nModel Configuration:")
    for key, value in model.get_config().items():
        print(f"  {key}: {value}")

    # Create dummy input
    batch_size = args.batch_size
    seq_len = args.seq_len
    input_dim = 1

    x = torch.randn(batch_size, seq_len, input_dim)
    print(f"\nInput shape: {x.shape}")

    # Test forward pass (with teacher forcing)
    model.eval()
    with torch.no_grad():
        reconstructed = model(x)

    print(f"Output shape: {reconstructed.shape}")
    assert reconstructed.shape == x.shape, "Output shape mismatch!"

    # Test encoding
    h_n, c_n = model.encode(x)
    print(f"\nLatent state shapes:")
    print(f"  h_n: {h_n.shape}")
    print(f"  c_n: {c_n.shape}")

    # Test autoregressive generation
    print("\nTesting autoregressive generation...")
    with torch.no_grad():
        generated = model.generate(x)
    print(f"Generated shape: {generated.shape}")
    assert generated.shape == x.shape, "Generated shape mismatch!"

    # Compute reconstruction error
    mse = torch.mean((x - reconstructed) ** 2).item()
    print(f"\nReconstruction MSE (untrained): {mse:.4f}")

    # Test that reversed reconstruction mechanism works
    print("\nVerifying reverse reconstruction mechanism...")
    with torch.no_grad():
        # Manually trace through the forward pass
        h_n, c_n = model.encoder(x)
        decoder_input = torch.flip(x, dims=[1])
        reconstructed_reversed = model.decoder(decoder_input, (h_n, c_n))
        reconstructed_manual = torch.flip(reconstructed_reversed, dims=[1])

        # Should match forward pass
        diff = torch.abs(reconstructed - reconstructed_manual).max().item()
        print(f"  Max difference from manual trace: {diff:.2e}")
        assert diff < 1e-6, "Reverse reconstruction mismatch!"

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
