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
        outputs, (h_n, c_n) = self.lstm(x)
        return h_n, c_n


class LSTMDecoder(nn.Module):
    """
    Decoder LSTM that reconstructs the sequence from latent representation.

    Key detail from the paper: The decoder reconstructs the sequence
    in REVERSE order (x'(L), x'(L-1), ..., x'(1)).

    During training, teacher forcing is used - the decoder receives
    the actual target values as input rather than its own predictions.
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
        outputs, _ = self.lstm(x, hidden)
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
        decoder_input = torch.flip(x, dims=[1])

        # Decode: reconstruct sequence (in reverse order)
        reconstructed_reversed = self.decoder(decoder_input, (h_n, c_n))

        # Reverse back to original order for loss computation
        reconstructed = torch.flip(reconstructed_reversed, dims=[1])

        return reconstructed

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract latent representation from input sequence."""
        return self.encoder(x)

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate reconstruction autoregressively (without teacher forcing).

        Note: For anomaly detection, we typically still use teacher forcing
        even during inference, as the reconstruction error is what matters.

        Args:
            x: Input sequence, shape (batch, seq_len, input_dim)

        Returns:
            Autoregressively generated sequence, shape (batch, seq_len, input_dim)
        """
        batch_size, seq_len, input_dim = x.shape

        h_n, c_n = self.encoder(x)

        # Start with the last timestep (decoder generates in reverse)
        decoder_input = x[:, -1:, :]

        generated = []
        hidden = (h_n, c_n)

        for t in range(seq_len):
            output, hidden = self.decoder.lstm(decoder_input, hidden)
            output = self.decoder.output_layer(output)
            generated.append(output)
            decoder_input = output

        generated = torch.cat(generated, dim=1)
        generated = torch.flip(generated, dims=[1])

        return generated

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
    """Factory function to create an EncDecAD model."""
    config = ModelConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        sequence_length=sequence_length
    )
    return EncDecAD(config)
