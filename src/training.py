"""
Shared training loop for the LSTM Encoder-Decoder.

Lives here so that `code/1_train_model.py` and `code/4_grid_sweep.py`
both call the same code path. Without this, the grid sweep can drift
from the canonical training pipeline and produce results that don't
match a fresh `1_train_model.py` run.
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import EncDecAD

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 100
    learning_rate: float = 1e-3
    patience: int = 10
    min_delta: float = 1e-6
    weight_decay: float = 0.0
    grad_clip: float = 1.0


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_model(
    model: EncDecAD,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Optional[TrainingConfig] = None,
) -> Tuple[EncDecAD, Dict]:
    """
    Train the LSTM Encoder-Decoder model on reconstruction loss.

    Returns:
        Tuple of (trained_model, training_history).
    """
    config = config or TrainingConfig()

    logger.info("Starting training...")
    logger.info(
        f"  Epochs: {config.epochs}, LR: {config.learning_rate}, Patience: {config.patience}"
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)

    best_model_state = None
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "best_epoch": 0}

    for epoch in range(config.epochs):
        # Train
        model.train()
        train_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            x = batch.to(device)
            optimizer.zero_grad()
            x_reconstructed = model(x)
            loss = criterion(x_reconstructed, x)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
        train_loss /= num_batches

        # Validate
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                x_reconstructed = model(x)
                loss = criterion(x_reconstructed, x)
                val_loss += loss.item()
                num_val_batches += 1
        val_loss /= num_val_batches

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch + 1

        improved = "<-" if val_loss <= best_val_loss else ""
        logger.info(
            f"Epoch {epoch + 1:3d}/{config.epochs} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} {improved}"
        )

        if early_stopping.step(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
        logger.info(f"Restored best model from epoch {history['best_epoch']}")

    return model, history


def save_training_artifacts(
    output_dir: str,
    model: EncDecAD,
    scaler: Any,
    scorer: Any,
    history: Dict,
    preprocess_config: Any,
) -> None:
    """
    Persist a complete training run (model + scaler + scorer + history + split config)
    to a directory. Used by `code/1_train_model.py` and `code/4_grid_sweep.py`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "lstm_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config,
        },
        model_path,
    )
    logger.info(f"Saved model to {model_path}")

    scaler_path = output_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved scaler to {scaler_path}")

    scorer.save(str(output_dir / "scorer.pkl"))

    with open(output_dir / "training_history.pkl", "wb") as f:
        pickle.dump(history, f)
    logger.info("Saved training history")

    with open(output_dir / "preprocessor_config.pkl", "wb") as f:
        pickle.dump(preprocess_config, f)
    logger.info("Saved preprocessor config")
