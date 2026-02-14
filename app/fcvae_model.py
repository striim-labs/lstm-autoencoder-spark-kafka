"""
FCVAE Model for Transaction Anomaly Detection

Frequency-enhanced Conditional Variational Autoencoder adapted from Wang et al. (WWW 2024).
Source: FCVAE/CVAE.py

Key adaptations from original:
1. FCVAEConfig dataclass for clean parameterization
2. Device-agnostic code (no hardcoded .to("cuda"))
3. Simplified forward() for unsupervised training (no label weighting)
4. Simplified loss to type "C" only (reconstruction + KLD)
5. Added score_single_pass() for fast streaming inference
6. Parameterized hidden_dims via config
"""
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch import nn

from app.attention import EncoderLayer_selfattn


@dataclass
class FCVAEConfig:
    """Configuration for FCVAE model.

    Default values tuned for hourly transaction data with 24-hour windows.
    Scaled down from original FCVAE (designed for longer windows) to match
    our shorter sequences.
    """
    window: int = 24                      # Daily window (hourly counts)
    latent_dim: int = 8                   # Latent space dimension
    condition_emb_dim: int = 16           # Condition embedding dimension
    d_model: int = 64                     # Attention model dimension
    d_inner: int = 128                    # Attention FFN inner dimension
    n_head: int = 4                       # Attention heads
    kernel_size: int = 8                  # LFM sub-window size (8 hours)
    stride: int = 4                       # LFM stride (4 hours) -> 5 sub-windows
    dropout_rate: float = 0.05            # Dropout rate
    hidden_dims: Tuple[int, ...] = (64, 64)  # Encoder/decoder hidden layers
    mcmc_iterations: int = 10             # MCMC refinement steps (test time)
    mcmc_samples: int = 64                # Samples for score averaging
    mcmc_rate: float = 0.2                # MCMC threshold percentile
    mcmc_mode: int = 2                    # MCMC mode (2 = replace last point only)


class FCVAE(nn.Module):
    """
    Frequency-enhanced Conditional Variational Autoencoder.

    Uses two frequency conditioning modules:
    - Global Frequency Module (GFM): FFT of entire window -> captures overall periodicity
    - Local Frequency Module (LFM): FFT of sub-windows + self-attention -> captures local trends

    The frequency condition guides the CVAE to better reconstruct periodic patterns
    without needing explicit day-of-week features.
    """

    def __init__(self, config: FCVAEConfig):
        super().__init__()
        self.config = config

        # Store hidden dims as list (needs to be mutable for decoder construction)
        hidden_dims = list(config.hidden_dims)

        # Encoder: input (window + 2*condition_dim) -> hidden_dims -> latent
        encoder_modules = []
        in_channels = config.window + 2 * config.condition_emb_dim
        for h_dim in hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh(),
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*encoder_modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], config.latent_dim)
        self.fc_var = nn.Sequential(
            nn.Linear(hidden_dims[-1], config.latent_dim),
            nn.Softplus(),
        )

        # Decoder: latent + condition -> hidden_dims (reversed) -> window
        self.decoder_input = nn.Linear(
            config.latent_dim + 2 * config.condition_emb_dim,
            hidden_dims[-1]
        )

        decoder_hidden = list(reversed(hidden_dims))
        decoder_modules = []
        for i in range(len(decoder_hidden) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.Linear(decoder_hidden[i], decoder_hidden[i + 1]),
                    nn.Tanh(),
                )
            )
        decoder_modules.append(
            nn.Sequential(
                nn.Linear(decoder_hidden[-1], config.window),
                nn.Tanh(),
            )
        )
        self.decoder = nn.Sequential(*decoder_modules)

        # Output distribution parameters
        self.fc_mu_x = nn.Linear(config.window, config.window)
        self.fc_var_x = nn.Sequential(
            nn.Linear(config.window, config.window),
            nn.Softplus()
        )

        # Local Frequency Module: self-attention over sub-window FFT features
        self.atten = nn.ModuleList([
            EncoderLayer_selfattn(
                config.d_model,
                config.d_inner,
                config.n_head,
                config.d_inner // config.n_head,  # d_k
                config.d_inner // config.n_head,  # d_v
                dropout=0.1,
            )
            for _ in range(1)  # Single attention layer
        ])

        # Local FFT embedding: kernel_size -> d_model
        # Input: real + imag parts of FFT = kernel_size/2 + 1 complex values = kernel_size + 2 reals
        # Actually: rfft output size is kernel_size // 2 + 1, and we concat real+imag
        # For kernel_size=8: rfft gives 5 complex values -> 10 reals, but original uses 2 + kernel_size
        # Following original: 2 + kernel_size (this may be for a specific FFT normalization)
        self.emb_local = nn.Sequential(
            nn.Linear(2 + config.kernel_size, config.d_model),
            nn.Tanh(),
        )
        self.out_linear = nn.Sequential(
            nn.Linear(config.d_model, config.condition_emb_dim),
            nn.Tanh(),
        )

        # Global FFT embedding: window -> condition_emb_dim
        self.emb_global = nn.Sequential(
            nn.Linear(config.window, config.condition_emb_dim),
            nn.Tanh(),
        )

        self.dropout = nn.Dropout(config.dropout_rate)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Concatenated [input, condition] tensor (B, 1, W + 2*cond_dim)

        Returns:
            mu: Latent mean (B, latent_dim)
            var: Latent variance (B, latent_dim)
        """
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        var = self.fc_var(result)
        return mu, var

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent sample to output distribution parameters.

        Args:
            z: Latent sample concatenated with condition (B, latent_dim + 2*cond_dim)

        Returns:
            mu_x: Output mean (B, 1, window)
            var_x: Output variance (B, 1, window)
        """
        result = self.decoder_input(z)
        result = result.view(-1, 1, list(self.config.hidden_dims)[-1])
        result = self.decoder(result)
        mu_x = self.fc_mu_x(result)
        var_x = self.fc_var_x(result)
        return mu_x, var_x

    def reparameterize(self, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution using reparameterization trick."""
        std = torch.sqrt(1e-7 + var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_condition(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency-based condition from input.

        Combines:
        1. Global Frequency Module (GFM): FFT of entire window
        2. Local Frequency Module (LFM): FFT of sub-windows + self-attention

        Args:
            x: Input tensor (B, 1, W)

        Returns:
            condition: (B, 1, 2 * condition_emb_dim)
        """
        # Global Frequency Module: FFT of window (excluding last point for causal)
        x_g = x
        f_global = torch.fft.rfft(x_g[:, :, :-1], dim=-1)
        f_global = torch.cat((f_global.real, f_global.imag), dim=-1)
        f_global = self.emb_global(f_global)  # (B, 1, cond_dim)

        # Local Frequency Module: sub-window FFTs with self-attention
        x_g = x_g.view(x.shape[0], 1, 1, -1)  # (B, 1, 1, W)
        x_l = x_g.clone()
        x_l[:, :, :, -1] = 0  # Zero out last point (causal masking)

        # Unfold into sub-windows
        unfold = nn.Unfold(
            kernel_size=(1, self.config.kernel_size),
            dilation=1,
            padding=0,
            stride=(1, self.config.stride),
        )
        unfold_x = unfold(x_l)  # (B, kernel_size, num_windows)
        unfold_x = unfold_x.transpose(1, 2)  # (B, num_windows, kernel_size)

        # FFT of each sub-window
        f_local = torch.fft.rfft(unfold_x, dim=-1)
        f_local = torch.cat((f_local.real, f_local.imag), dim=-1)

        # Embed and apply self-attention
        f_local = self.emb_local(f_local)  # (B, num_windows, d_model)
        for enc_layer in self.atten:
            f_local, _ = enc_layer(f_local)

        # Project to condition dimension, take last sub-window
        f_local = self.out_linear(f_local)  # (B, num_windows, cond_dim)
        f_local = f_local[:, -1, :].unsqueeze(1)  # (B, 1, cond_dim)

        # Concatenate global and local conditions
        condition = torch.cat((f_global, f_local), dim=-1)  # (B, 1, 2*cond_dim)
        return condition

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "train",
        kl_weight: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training/validation.

        Args:
            x: Input tensor (B, 1, W)
            mode: "train" or "valid"
            kl_weight: Weight for KLD loss term (for KL annealing, ramps from 0 to 1)

        Returns:
            mu_x: Reconstruction mean (B, 1, W)
            var_x: Reconstruction variance (B, 1, W)
            rec_x: Sampled reconstruction (B, 1, W)
            mu: Latent mean (B, latent_dim)
            var: Latent variance (B, latent_dim)
            loss: ELBO loss (scalar)
        """
        condition = self.get_condition(x)
        if mode == "train":
            condition = self.dropout(condition)

        # Encode
        mu, var = self.encode(torch.cat((x, condition), dim=2))

        # Sample latent
        z = self.reparameterize(mu, var)

        # Decode
        mu_x, var_x = self.decode(torch.cat((z, condition.squeeze(1)), dim=1))

        # Sample reconstruction
        rec_x = self.reparameterize(mu_x, var_x)

        # Compute loss with KL weight
        loss = self.loss_func(mu_x, var_x, x, mu, var, kl_weight)

        return mu_x, var_x, rec_x, mu, var, loss

    def loss_func(
        self,
        mu_x: torch.Tensor,
        var_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        var: torch.Tensor,
        kl_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Compute ELBO loss (reconstruction + KLD) with optional KL annealing.

        KL annealing starts with kl_weight=0 and linearly increases to 1 over
        the first N epochs. This lets the decoder learn tight variance before
        KLD pushes the latent distribution toward the prior.

        Loss = recon_loss + kl_weight * kld_loss

        Args:
            mu_x: Decoder output mean (B, 1, W)
            var_x: Decoder output variance (B, 1, W)
            x: Input (B, 1, W)
            mu: Latent mean (B, latent_dim)
            var: Latent variance (B, latent_dim)
            kl_weight: Weight for KLD term (0 to 1, for annealing)

        Returns:
            loss: Scalar ELBO loss
        """
        mu_x = mu_x.squeeze(1)  # (B, W)
        var_x = var_x.squeeze(1)
        x = x.squeeze(1)

        # Reconstruction loss: negative log-likelihood
        recon_loss = torch.mean(
            0.5 * torch.mean(torch.log(var_x) + (x - mu_x) ** 2 / var_x, dim=1)
        )

        # KLD loss: KL(N(mu, var) || N(0, 1))
        kld_loss = torch.mean(
            0.5 * torch.mean(mu ** 2 + var - torch.log(var) - 1, dim=1)
        )

        return recon_loss + kl_weight * kld_loss

    def score_mcmc(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Score using MCMC refinement (original MCMC2 method).

        Iteratively refines the reconstruction by replacing suspected
        anomalous points, then averages log-likelihood over multiple samples.

        Args:
            x: Input tensor (B, 1, W)

        Returns:
            x_refined: Refined input after MCMC iterations
            scores: Per-point negative log-likelihood (B, 1, W)
        """
        device = x.device
        condition = self.get_condition(x)
        origin_x = x.clone()

        # MCMC refinement iterations
        for _ in range(self.config.mcmc_iterations):
            mu, var = self.encode(torch.cat((x, condition), dim=2))
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(torch.cat((z, condition.squeeze(1)), dim=1))

            # Log-likelihood per point
            recon = -0.5 * (torch.log(var_x) + (origin_x - mu_x) ** 2 / var_x)

            if self.config.mcmc_mode == 0:
                # Replace points below percentile threshold
                temp = torch.from_numpy(
                    np.percentile(recon.cpu().numpy(), self.config.mcmc_rate * 100, axis=-1)
                ).unsqueeze(2).repeat(1, 1, self.config.window).to(device)
                mask = (temp < recon).int()
                x = mu_x * (1 - mask) + origin_x * mask
            elif self.config.mcmc_mode == 1:
                # Replace points below fixed value
                mcmc_value = getattr(self.config, 'mcmc_value', -5)
                mask = (mcmc_value < recon).int()
                x = origin_x * mask + mu_x * (1 - mask)
            else:  # mode 2 (default)
                # Replace only the last point
                mask = torch.ones_like(origin_x)
                mask[:, :, -1] = 0
                x = origin_x * mask + (1 - mask) * mu_x

        # Average log-likelihood over multiple samples
        prob_all = torch.zeros_like(origin_x)
        mu, var = self.encode(torch.cat((x, condition), dim=2))
        for _ in range(self.config.mcmc_samples):
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(torch.cat((z, condition.squeeze(1)), dim=1))
            prob_all += -0.5 * (torch.log(var_x) + (origin_x - mu_x) ** 2 / var_x)

        scores = prob_all / self.config.mcmc_samples
        return x, scores

    def score_single_pass(
        self,
        x: torch.Tensor,
        n_samples: int = 16
    ) -> torch.Tensor:
        """
        Fast scoring for streaming (no MCMC refinement).

        Computes average negative log-likelihood over n_samples latent draws.
        Much faster than MCMC but potentially less accurate.

        Args:
            x: Input tensor (B, 1, W)
            n_samples: Number of latent samples for averaging

        Returns:
            scores: Per-point negative log-likelihood (B, W)
                   Higher (less negative) = more normal
                   Lower (more negative) = more anomalous
        """
        condition = self.get_condition(x)
        mu, var = self.encode(torch.cat((x, condition), dim=2))

        prob_all = torch.zeros(x.shape[0], self.config.window, device=x.device)

        for _ in range(n_samples):
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(torch.cat((z, condition.squeeze(1)), dim=1))
            # Negative log-likelihood per point
            nll = -0.5 * (torch.log(var_x) + (x - mu_x) ** 2 / var_x)
            prob_all += nll.squeeze(1)  # (B, W)

        return prob_all / n_samples

    def reconstruct(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct input (for visualization).

        Args:
            x: Input tensor (B, 1, W)

        Returns:
            mu_x: Reconstruction mean (B, 1, W)
            var_x: Reconstruction variance (B, 1, W)
        """
        condition = self.get_condition(x)
        mu, var = self.encode(torch.cat((x, condition), dim=2))
        z = self.reparameterize(mu, var)
        mu_x, var_x = self.decode(torch.cat((z, condition.squeeze(1)), dim=1))
        return mu_x, var_x
