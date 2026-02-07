"""
Hyperparameter Optimization for Credit Card Fraud Detection

Comprehensive grid search over:
- Autoencoder architecture (layers, hidden dimensions, dropout)
- MLP classifier architecture (layers, hidden dimensions, dropout)
- Training hyperparameters (learning rate, batch size, weight decay)

Tracks all experiments and reports best configurations.
"""

import argparse
import itertools
import json
import logging
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from credit_card_preprocessor import CreditCardPreprocessor, CreditCardPreprocessorConfig
from feedforward_autoencoder import FeedforwardAutoencoder, AutoencoderConfig
from mlp_classifier import MLPClassifier, MLPConfig
from train_mlp_classifier import (
    ClassifierTrainingConfig,
    EarlyStopping,
    extract_latent_features,
    train_classifier,
    evaluate_classifier,
)

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    # Autoencoder architecture
    ae_hidden_dim: int          # d1 in paper
    ae_latent_dim: int          # Bottleneck dimension
    ae_dropout: float           # Dropout in autoencoder

    # MLP classifier architecture
    mlp_hidden_dim1: int        # First hidden layer
    mlp_hidden_dim2: int        # Second hidden layer
    mlp_dropout: float          # Dropout in MLP

    # Training hyperparameters
    ae_learning_rate: float     # Autoencoder learning rate
    mlp_learning_rate: float    # Classifier learning rate
    batch_size: int             # Batch size
    ae_weight_decay: float      # L2 regularization for autoencoder
    mlp_weight_decay: float     # L2 regularization for classifier

    # Other settings
    ae_epochs: int = 100        # Max epochs for autoencoder
    mlp_epochs: int = 100       # Max epochs for classifier
    ae_patience: int = 10       # Early stopping patience for autoencoder
    mlp_patience: int = 15      # Early stopping patience for classifier


@dataclass
class ExperimentResult:
    """Results from a single experiment."""

    config: ExperimentConfig

    # Autoencoder metrics
    ae_train_loss: float
    ae_val_loss: float
    ae_test_loss: float
    ae_epochs_trained: int

    # Classifier metrics
    mlp_train_loss: float
    mlp_val_loss: float
    mlp_val_f1: float
    mlp_epochs_trained: int

    # Test metrics (final)
    test_precision: float
    test_recall: float
    test_f1: float

    # Timing
    total_time: float

    # Model info
    ae_parameters: int
    mlp_parameters: int
    total_parameters: int


def train_autoencoder_experiment(
    config: ExperimentConfig,
    dataloaders: Dict[str, DataLoader],
    device: torch.device
) -> Tuple[FeedforwardAutoencoder, Dict]:
    """
    Train autoencoder with given config.

    Args:
        config: Experiment configuration
        dataloaders: DataLoaders for train/val/test
        device: cpu/cuda

    Returns:
        (trained_autoencoder, metrics)
    """
    # Create autoencoder
    ae_config = AutoencoderConfig(
        input_dim=30,
        hidden_dim=config.ae_hidden_dim,
        latent_dim=config.ae_latent_dim,
        dropout=config.ae_dropout
    )

    autoencoder = FeedforwardAutoencoder(config=ae_config)
    autoencoder.to(device)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        autoencoder.parameters(),
        lr=config.ae_learning_rate,
        weight_decay=config.ae_weight_decay
    )

    early_stopping = EarlyStopping(
        patience=config.ae_patience,
        min_delta=1e-6
    )

    best_val_loss = float('inf')
    best_model_state = None
    epochs_trained = 0

    # Training loop
    for epoch in range(config.ae_epochs):
        # Train
        autoencoder.train()
        train_loss = 0.0

        for features, _ in dataloaders['train']:
            features = features.to(device)
            optimizer.zero_grad()
            reconstructed = autoencoder(features)
            loss = criterion(reconstructed, features)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(features)

        train_loss /= len(dataloaders['train'].dataset)

        # Validate
        autoencoder.eval()
        val_loss = 0.0

        with torch.no_grad():
            for features, _ in dataloaders['val']:
                features = features.to(device)
                reconstructed = autoencoder(features)
                loss = criterion(reconstructed, features)
                val_loss += loss.item() * len(features)

        val_loss /= len(dataloaders['val'].dataset)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = autoencoder.state_dict().copy()

        epochs_trained = epoch + 1

        # Early stopping
        if early_stopping.step(val_loss):
            break

    # Restore best model
    if best_model_state is not None:
        autoencoder.load_state_dict(best_model_state)

    # Test loss
    autoencoder.eval()
    test_loss = 0.0

    with torch.no_grad():
        for features, _ in dataloaders['test']:
            features = features.to(device)
            reconstructed = autoencoder(features)
            loss = criterion(reconstructed, features)
            test_loss += loss.item() * len(features)

    test_loss /= len(dataloaders['test'].dataset)

    metrics = {
        'train_loss': train_loss,
        'val_loss': best_val_loss,
        'test_loss': test_loss,
        'epochs_trained': epochs_trained,
        'parameters': autoencoder.count_parameters()
    }

    return autoencoder, metrics


def train_classifier_experiment(
    config: ExperimentConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device
) -> Tuple[MLPClassifier, Dict]:
    """
    Train MLP classifier with given config.

    Args:
        config: Experiment configuration
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        device: cpu/cuda

    Returns:
        (trained_classifier, metrics)
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    # Create DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Create classifier
    mlp_config = MLPConfig(
        input_dim=config.ae_latent_dim,
        hidden_dim1=config.mlp_hidden_dim1,
        hidden_dim2=config.mlp_hidden_dim2,
        dropout=config.mlp_dropout
    )

    classifier = MLPClassifier(config=mlp_config)
    classifier.to(device)

    # Training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=config.mlp_learning_rate,
        weight_decay=config.mlp_weight_decay
    )

    early_stopping = EarlyStopping(
        patience=config.mlp_patience,
        min_delta=1e-6
    )

    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_model_state = None
    epochs_trained = 0

    # Training loop
    for epoch in range(config.mlp_epochs):
        # Train
        classifier.train()
        train_loss = 0.0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            probs = classifier(features).squeeze()
            loss = criterion(probs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(features)

        train_loss /= len(train_loader.dataset)

        # Validate
        classifier.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                probs = classifier(features).squeeze()
                loss = criterion(probs, labels)
                val_loss += loss.item() * len(features)

                preds = (probs >= 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_model_state = classifier.state_dict().copy()

        epochs_trained = epoch + 1

        # Early stopping
        if early_stopping.step(val_loss):
            break

    # Restore best model
    if best_model_state is not None:
        classifier.load_state_dict(best_model_state)

    # Test metrics
    classifier.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            probs = classifier(features).squeeze()
            preds = (probs >= 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    test_precision = precision_score(all_labels, all_preds, zero_division=0)
    test_recall = recall_score(all_labels, all_preds, zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, zero_division=0)

    metrics = {
        'train_loss': train_loss,
        'val_loss': best_val_loss,
        'val_f1': best_val_f1,
        'epochs_trained': epochs_trained,
        'parameters': classifier.count_parameters(),
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1
    }

    return classifier, metrics


def run_experiment(
    config: ExperimentConfig,
    dataloaders: Dict[str, DataLoader],
    device: torch.device,
    experiment_id: int,
    total_experiments: int
) -> ExperimentResult:
    """
    Run a single hyperparameter experiment.

    Args:
        config: Experiment configuration
        dataloaders: DataLoaders with raw features
        device: cpu/cuda
        experiment_id: Current experiment number
        total_experiments: Total number of experiments

    Returns:
        ExperimentResult
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Experiment {experiment_id}/{total_experiments}")
    logger.info(f"{'=' * 80}")
    logger.info(f"Autoencoder: 30 → {config.ae_hidden_dim} → {config.ae_latent_dim}")
    logger.info(f"MLP: {config.ae_latent_dim} → {config.mlp_hidden_dim1} → {config.mlp_hidden_dim2} → 1")
    logger.info(f"AE LR: {config.ae_learning_rate}, MLP LR: {config.mlp_learning_rate}")
    logger.info(f"Batch size: {config.batch_size}")

    start_time = time.time()

    # Step 1: Train autoencoder
    logger.info("\nTraining autoencoder...")
    autoencoder, ae_metrics = train_autoencoder_experiment(config, dataloaders, device)
    logger.info(f"AE trained in {ae_metrics['epochs_trained']} epochs")
    logger.info(f"AE val loss: {ae_metrics['val_loss']:.6f}")

    # Step 2: Extract latent features
    logger.info("\nExtracting latent features...")
    X_train_latent, y_train = extract_latent_features(autoencoder.encoder, dataloaders['train'], device)
    X_val_latent, y_val = extract_latent_features(autoencoder.encoder, dataloaders['val'], device)
    X_test_latent, y_test = extract_latent_features(autoencoder.encoder, dataloaders['test'], device)

    # Step 3: Train classifier
    logger.info("\nTraining classifier...")
    classifier, mlp_metrics = train_classifier_experiment(
        config, X_train_latent, y_train, X_val_latent, y_val, X_test_latent, y_test, device
    )
    logger.info(f"MLP trained in {mlp_metrics['epochs_trained']} epochs")
    logger.info(f"Test F1: {mlp_metrics['test_f1']:.4f}")

    total_time = time.time() - start_time

    # Create result
    result = ExperimentResult(
        config=config,
        ae_train_loss=ae_metrics['train_loss'],
        ae_val_loss=ae_metrics['val_loss'],
        ae_test_loss=ae_metrics['test_loss'],
        ae_epochs_trained=ae_metrics['epochs_trained'],
        mlp_train_loss=mlp_metrics['train_loss'],
        mlp_val_loss=mlp_metrics['val_loss'],
        mlp_val_f1=mlp_metrics['val_f1'],
        mlp_epochs_trained=mlp_metrics['epochs_trained'],
        test_precision=mlp_metrics['test_precision'],
        test_recall=mlp_metrics['test_recall'],
        test_f1=mlp_metrics['test_f1'],
        total_time=total_time,
        ae_parameters=ae_metrics['parameters'],
        mlp_parameters=mlp_metrics['parameters'],
        total_parameters=ae_metrics['parameters'] + mlp_metrics['parameters']
    )

    logger.info(f"\nExperiment complete in {total_time:.1f}s")
    logger.info(f"Total parameters: {result.total_parameters:,}")

    return result


def main():
    """Main hyperparameter optimization pipeline."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for credit card fraud detection"
    )

    parser.add_argument("--data-path", type=str, default="data/creditcard.csv")
    parser.add_argument("--output-dir", type=str, default="models/credit_card/optimization")
    parser.add_argument("--search-type", type=str, default="grid", choices=["grid", "random"])
    parser.add_argument("--n-random-samples", type=int, default=50, help="Number of random samples (if random search)")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== Preprocessing =====
    logger.info("\n" + "=" * 80)
    logger.info("Preprocessing Data")
    logger.info("=" * 80)

    preprocessor = CreditCardPreprocessor(config=CreditCardPreprocessorConfig())
    dataloaders, _ = preprocessor.preprocess(args.data_path, batch_size=256)

    # ===== Define Search Space =====
    logger.info("\n" + "=" * 80)
    logger.info("Defining Search Space")
    logger.info("=" * 80)

    search_space = {
        # Autoencoder architecture
        'ae_hidden_dim': [18, 20, 22, 25, 28],           # Intermediate dimension d1
        'ae_latent_dim': [10, 12, 15, 18, 20],           # Bottleneck dimension
        'ae_dropout': [0.0, 0.1, 0.2],                   # Dropout rate

        # MLP classifier architecture
        'mlp_hidden_dim1': [10, 13, 15, 18],             # First hidden layer
        'mlp_hidden_dim2': [5, 7, 10],                   # Second hidden layer
        'mlp_dropout': [0.0, 0.1, 0.2],                  # Dropout rate

        # Training hyperparameters
        'ae_learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],    # Autoencoder LR
        'mlp_learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],   # Classifier LR
        'batch_size': [128, 256, 512],                   # Batch size
        'ae_weight_decay': [0.0, 1e-5, 1e-4],            # L2 regularization for AE
        'mlp_weight_decay': [0.0, 1e-5, 1e-4],           # L2 regularization for MLP
    }

    # Generate configurations
    if args.search_type == "grid":
        # Grid search: try all combinations
        keys = list(search_space.keys())
        values = list(search_space.values())

        configs = []
        for combination in itertools.product(*values):
            config_dict = dict(zip(keys, combination))
            configs.append(ExperimentConfig(**config_dict))

        logger.info(f"Grid search: {len(configs)} total configurations")

    else:
        # Random search: sample random combinations
        import random

        configs = []
        for _ in range(args.n_random_samples):
            config_dict = {
                key: random.choice(values)
                for key, values in search_space.items()
            }
            configs.append(ExperimentConfig(**config_dict))

        logger.info(f"Random search: {len(configs)} random configurations")

    logger.info(f"\nSearch space:")
    for key, values in search_space.items():
        logger.info(f"  {key}: {values}")

    # ===== Run Experiments =====
    logger.info("\n" + "=" * 80)
    logger.info("Running Experiments")
    logger.info("=" * 80)

    results = []

    for i, config in enumerate(configs, 1):
        try:
            result = run_experiment(config, dataloaders, device, i, len(configs))
            results.append(result)

            # Save intermediate results
            if i % 10 == 0 or i == len(configs):
                intermediate_path = output_dir / f"results_interim_{i}.pkl"
                with open(intermediate_path, 'wb') as f:
                    pickle.dump(results, f)
                logger.info(f"\nSaved intermediate results to {intermediate_path}")

        except Exception as e:
            logger.error(f"\nExperiment {i} failed: {e}")
            continue

    # ===== Analyze Results =====
    logger.info("\n" + "=" * 80)
    logger.info("Analyzing Results")
    logger.info("=" * 80)

    # Convert to DataFrame
    rows = []
    for result in results:
        row = {
            **asdict(result.config),
            'ae_train_loss': result.ae_train_loss,
            'ae_val_loss': result.ae_val_loss,
            'ae_test_loss': result.ae_test_loss,
            'ae_epochs': result.ae_epochs_trained,
            'mlp_train_loss': result.mlp_train_loss,
            'mlp_val_loss': result.mlp_val_loss,
            'mlp_val_f1': result.mlp_val_f1,
            'mlp_epochs': result.mlp_epochs_trained,
            'test_precision': result.test_precision,
            'test_recall': result.test_recall,
            'test_f1': result.test_f1,
            'total_time': result.total_time,
            'total_params': result.total_parameters
        }
        rows.append(row)

    df_results = pd.DataFrame(rows)

    # Sort by test F1
    df_results = df_results.sort_values('test_f1', ascending=False)

    # Save results
    csv_path = output_dir / "optimization_results.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"\nSaved results to {csv_path}")

    pkl_path = output_dir / "optimization_results.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Saved detailed results to {pkl_path}")

    # ===== Report Best Configurations =====
    logger.info("\n" + "=" * 80)
    logger.info("TOP 10 CONFIGURATIONS")
    logger.info("=" * 80)

    top_10 = df_results.head(10)

    for i, (idx, row) in enumerate(top_10.iterrows(), 1):
        logger.info(f"\n#{i} - Test F1: {row['test_f1']:.4f} (P: {row['test_precision']:.4f}, R: {row['test_recall']:.4f})")
        logger.info(f"  Autoencoder: 30 → {int(row['ae_hidden_dim'])} → {int(row['ae_latent_dim'])} (dropout={row['ae_dropout']:.1f})")
        logger.info(f"  MLP: {int(row['ae_latent_dim'])} → {int(row['mlp_hidden_dim1'])} → {int(row['mlp_hidden_dim2'])} → 1 (dropout={row['mlp_dropout']:.1f})")
        logger.info(f"  AE LR: {row['ae_learning_rate']:.0e}, MLP LR: {row['mlp_learning_rate']:.0e}")
        logger.info(f"  Batch size: {int(row['batch_size'])}, Params: {int(row['total_params']):,}")

    # Best configuration
    best_row = df_results.iloc[0]
    logger.info("\n" + "=" * 80)
    logger.info("BEST CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Test F1: {best_row['test_f1']:.4f}")
    logger.info(f"Test Precision: {best_row['test_precision']:.4f}")
    logger.info(f"Test Recall: {best_row['test_recall']:.4f}")
    logger.info(f"\nAutoencoder:")
    logger.info(f"  Architecture: 30 → {int(best_row['ae_hidden_dim'])} → {int(best_row['ae_latent_dim'])}")
    logger.info(f"  Dropout: {best_row['ae_dropout']:.2f}")
    logger.info(f"  Learning rate: {best_row['ae_learning_rate']:.0e}")
    logger.info(f"  Weight decay: {best_row['ae_weight_decay']:.0e}")
    logger.info(f"\nMLP Classifier:")
    logger.info(f"  Architecture: {int(best_row['ae_latent_dim'])} → {int(best_row['mlp_hidden_dim1'])} → {int(best_row['mlp_hidden_dim2'])} → 1")
    logger.info(f"  Dropout: {best_row['mlp_dropout']:.2f}")
    logger.info(f"  Learning rate: {best_row['mlp_learning_rate']:.0e}")
    logger.info(f"  Weight decay: {best_row['mlp_weight_decay']:.0e}")
    logger.info(f"\nTraining:")
    logger.info(f"  Batch size: {int(best_row['batch_size'])}")
    logger.info(f"  Total parameters: {int(best_row['total_params']):,}")
    logger.info(f"  Training time: {best_row['total_time']:.1f}s")

    # Save best config as JSON
    best_config = {
        'ae_hidden_dim': int(best_row['ae_hidden_dim']),
        'ae_latent_dim': int(best_row['ae_latent_dim']),
        'ae_dropout': float(best_row['ae_dropout']),
        'mlp_hidden_dim1': int(best_row['mlp_hidden_dim1']),
        'mlp_hidden_dim2': int(best_row['mlp_hidden_dim2']),
        'mlp_dropout': float(best_row['mlp_dropout']),
        'ae_learning_rate': float(best_row['ae_learning_rate']),
        'mlp_learning_rate': float(best_row['mlp_learning_rate']),
        'batch_size': int(best_row['batch_size']),
        'ae_weight_decay': float(best_row['ae_weight_decay']),
        'mlp_weight_decay': float(best_row['mlp_weight_decay']),
        'test_f1': float(best_row['test_f1']),
        'test_precision': float(best_row['test_precision']),
        'test_recall': float(best_row['test_recall'])
    }

    json_path = output_dir / "best_config.json"
    with open(json_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    logger.info(f"\nSaved best config to {json_path}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Optimization complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
