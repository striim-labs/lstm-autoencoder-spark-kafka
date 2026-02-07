"""
Comparison script for Phase 1, Step 5: Baseline Models

Implements and compares:
- M1: Autoencoder (AE) features + MLP (already trained)
- M2: Raw features + MLP (no autoencoder)
- M3: AE features + MLP with undersampled negative class
- M4: Raw features + MLP with undersampled negative class

Each model configuration tested with:
- MLP classifier
- KNN (k=3)
- Logistic Regression (L2, LBFGS)

Reproduces Tables 2-4 from Misra et al. paper.
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset

from credit_card_preprocessor import CreditCardPreprocessor, CreditCardPreprocessorConfig
from feedforward_autoencoder import AutoencoderConfig, FeedforwardAutoencoder
from mlp_classifier import MLPClassifier, MLPConfig
from train_mlp_classifier import (
    ClassifierTrainingConfig,
    EarlyStopping,
    extract_latent_features,
    train_classifier,
    evaluate_classifier,
)

logger = logging.getLogger(__name__)


def undersample_majority_class(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Undersample the majority class to balance the dataset.

    Args:
        X: Features
        y: Labels
        random_state: Random seed

    Returns:
        (X_resampled, y_resampled)
    """
    # Separate majority and minority classes
    X_majority = X[y == 0]
    X_minority = X[y == 1]
    y_majority = y[y == 0]
    y_minority = y[y == 1]

    # Undersample majority class to match minority class size
    X_majority_undersampled, y_majority_undersampled = resample(
        X_majority,
        y_majority,
        n_samples=len(X_minority),
        replace=False,
        random_state=random_state
    )

    # Combine minority class with undersampled majority class
    X_resampled = np.vstack([X_majority_undersampled, X_minority])
    y_resampled = np.concatenate([y_majority_undersampled, y_minority])

    # Shuffle
    indices = np.random.RandomState(random_state).permutation(len(y_resampled))
    X_resampled = X_resampled[indices]
    y_resampled = y_resampled[indices]

    logger.info(f"Undersampling: {len(y)} → {len(y_resampled)} samples")
    logger.info(f"  Original fraud ratio: {y.mean():.4f}")
    logger.info(f"  Resampled fraud ratio: {y_resampled.mean():.4f}")

    return X_resampled, y_resampled


def train_mlp_pytorch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
    device: torch.device,
    model_name: str = "MLP"
) -> Dict:
    """
    Train PyTorch MLP classifier.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        input_dim: Input feature dimension
        device: cpu/cuda
        model_name: Name for logging

    Returns:
        Dict with test metrics
    """
    logger.info(f"\nTraining {model_name} (PyTorch MLP)...")

    # Create DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Create model
    config = MLPConfig(
        input_dim=input_dim,
        hidden_dim1=13,
        hidden_dim2=7
    )
    model = MLPClassifier(config=config)
    model.to(device)

    # Training config
    training_config = ClassifierTrainingConfig(
        epochs=100,
        learning_rate=1e-4,
        patience=15
    )

    # Train
    model, history = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=training_config
    )

    # Evaluate on test set
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            probs = model(features).squeeze()
            preds = (probs >= 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    logger.info(f"{model_name} Results: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'model': model,
        'history': history
    }


def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k: int = 3,
    model_name: str = "KNN"
) -> Dict:
    """
    Train KNN classifier.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        k: Number of neighbors
        model_name: Name for logging

    Returns:
        Dict with test metrics
    """
    logger.info(f"\nTraining {model_name} (k={k})...")

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict on test set
    y_pred = knn.predict(X_test)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logger.info(f"{model_name} Results: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'model': knn
    }


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "LR"
) -> Dict:
    """
    Train Logistic Regression classifier.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_name: Name for logging

    Returns:
        Dict with test metrics
    """
    logger.info(f"\nTraining {model_name} (L2, LBFGS)...")

    # Train Logistic Regression with L2 regularization
    lr = LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    lr.fit(X_train, y_train)

    # Predict on test set
    y_pred = lr.predict(X_test)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logger.info(f"{model_name} Results: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'model': lr
    }


def load_m1_results(results_path: str) -> Dict:
    """
    Load M1 results from saved test_results.pkl.

    Args:
        results_path: Path to test_results.pkl

    Returns:
        Dict with M1 metrics
    """
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    logger.info(f"\nLoaded M1 (AE + MLP) results:")
    logger.info(f"  Precision: {results['precision']:.4f}")
    logger.info(f"  Recall: {results['recall']:.4f}")
    logger.info(f"  F1: {results['f1']:.4f}")

    return {
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1']
    }


def main():
    """Main comparison pipeline."""
    parser = argparse.ArgumentParser(
        description="Compare baseline models for credit card fraud detection"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/creditcard.csv",
        help="Path to credit card CSV"
    )
    parser.add_argument(
        "--autoencoder-path",
        type=str,
        default="models/credit_card/autoencoder.pt",
        help="Path to trained autoencoder"
    )
    parser.add_argument(
        "--m1-results-path",
        type=str,
        default="models/credit_card/test_results.pkl",
        help="Path to M1 test results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/credit_card/comparisons",
        help="Directory to save comparison results"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== Step 1: Preprocess Data =====
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Preprocessing Data")
    logger.info("=" * 80)

    preprocessor_config = CreditCardPreprocessorConfig()
    preprocessor = CreditCardPreprocessor(config=preprocessor_config)

    dataloaders, normalized_splits = preprocessor.preprocess(
        filepath=args.data_path,
        batch_size=256
    )

    # Extract raw features and labels
    X_train_raw, y_train = normalized_splits['train']
    X_val_raw, y_val = normalized_splits['val']
    X_test_raw, y_test = normalized_splits['test']

    # ===== Step 2: Load Autoencoder and Extract Latent Features =====
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Loading Autoencoder and Extracting Features")
    logger.info("=" * 80)

    torch.serialization.add_safe_globals([AutoencoderConfig])
    checkpoint = torch.load(args.autoencoder_path, map_location=device, weights_only=True)
    autoencoder = FeedforwardAutoencoder(config=checkpoint['model_config'])
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder.to(device)
    autoencoder.eval()

    logger.info(f"Loaded autoencoder from {args.autoencoder_path}")

    # Extract latent features for all splits
    X_train_latent, _ = extract_latent_features(autoencoder.encoder, dataloaders['train'], device)
    X_val_latent, _ = extract_latent_features(autoencoder.encoder, dataloaders['val'], device)
    X_test_latent, _ = extract_latent_features(autoencoder.encoder, dataloaders['test'], device)

    logger.info(f"Extracted latent features:")
    logger.info(f"  Train: {X_train_latent.shape}")
    logger.info(f"  Val: {X_val_latent.shape}")
    logger.info(f"  Test: {X_test_latent.shape}")

    # ===== Step 3: Load M1 Results =====
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Loading M1 Results")
    logger.info("=" * 80)

    m1_mlp = load_m1_results(args.m1_results_path)

    # ===== Step 4: Train M1 with KNN and LR =====
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Training M1 (AE features) with KNN and LR")
    logger.info("=" * 80)

    m1_knn = train_knn(X_train_latent, y_train, X_test_latent, y_test, k=3, model_name="M1 KNN")
    m1_lr = train_logistic_regression(X_train_latent, y_train, X_test_latent, y_test, model_name="M1 LR")

    # ===== Step 5: Train M2 (Raw Features) =====
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Training M2 (Raw Features)")
    logger.info("=" * 80)

    m2_mlp = train_mlp_pytorch(
        X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test,
        input_dim=30, device=device, model_name="M2 MLP"
    )
    m2_knn = train_knn(X_train_raw, y_train, X_test_raw, y_test, k=3, model_name="M2 KNN")
    m2_lr = train_logistic_regression(X_train_raw, y_train, X_test_raw, y_test, model_name="M2 LR")

    # ===== Step 6: Create Undersampled Datasets =====
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Creating Undersampled Datasets")
    logger.info("=" * 80)

    # Undersample training data (combine train + val for undersampling)
    X_train_full_raw = np.vstack([X_train_raw, X_val_raw])
    y_train_full = np.concatenate([y_train, y_val])
    X_train_full_latent = np.vstack([X_train_latent, X_val_latent])

    logger.info("\nUndersampling raw features...")
    X_train_raw_under, y_train_under = undersample_majority_class(X_train_full_raw, y_train_full)

    logger.info("\nUndersampling latent features...")
    X_train_latent_under, y_train_latent_under = undersample_majority_class(X_train_full_latent, y_train_full)

    # ===== Step 7: Train M3 (AE Features + Undersampling) =====
    logger.info("\n" + "=" * 80)
    logger.info("Step 7: Training M3 (AE Features + Undersampling)")
    logger.info("=" * 80)

    # For M3, we don't have separate val set after undersampling
    # Use a portion of undersampled data for validation
    split_idx = int(0.8 * len(X_train_latent_under))
    X_train_m3 = X_train_latent_under[:split_idx]
    y_train_m3 = y_train_latent_under[:split_idx]
    X_val_m3 = X_train_latent_under[split_idx:]
    y_val_m3 = y_train_latent_under[split_idx:]

    m3_mlp = train_mlp_pytorch(
        X_train_m3, y_train_m3, X_val_m3, y_val_m3, X_test_latent, y_test,
        input_dim=15, device=device, model_name="M3 MLP"
    )
    m3_knn = train_knn(X_train_latent_under, y_train_latent_under, X_test_latent, y_test, k=3, model_name="M3 KNN")
    m3_lr = train_logistic_regression(X_train_latent_under, y_train_latent_under, X_test_latent, y_test, model_name="M3 LR")

    # ===== Step 8: Train M4 (Raw Features + Undersampling) =====
    logger.info("\n" + "=" * 80)
    logger.info("Step 8: Training M4 (Raw Features + Undersampling)")
    logger.info("=" * 80)

    split_idx = int(0.8 * len(X_train_raw_under))
    X_train_m4 = X_train_raw_under[:split_idx]
    y_train_m4 = y_train_under[:split_idx]
    X_val_m4 = X_train_raw_under[split_idx:]
    y_val_m4 = y_train_under[split_idx:]

    m4_mlp = train_mlp_pytorch(
        X_train_m4, y_train_m4, X_val_m4, y_val_m4, X_test_raw, y_test,
        input_dim=30, device=device, model_name="M4 MLP"
    )
    m4_knn = train_knn(X_train_raw_under, y_train_under, X_test_raw, y_test, k=3, model_name="M4 KNN")
    m4_lr = train_logistic_regression(X_train_raw_under, y_train_under, X_test_raw, y_test, model_name="M4 LR")

    # ===== Step 9: Create Comparison Tables =====
    logger.info("\n" + "=" * 80)
    logger.info("Step 9: Generating Comparison Tables")
    logger.info("=" * 80)

    # Collect all results
    results = {
        'M1 (AE + MLP)': {'MLP': m1_mlp, 'KNN': m1_knn, 'LR': m1_lr},
        'M2 (Raw + MLP)': {'MLP': m2_mlp, 'KNN': m2_knn, 'LR': m2_lr},
        'M3 (AE + Undersample)': {'MLP': m3_mlp, 'KNN': m3_knn, 'LR': m3_lr},
        'M4 (Raw + Undersample)': {'MLP': m4_mlp, 'KNN': m4_knn, 'LR': m4_lr},
    }

    # Create comparison DataFrame
    rows = []
    for model_name, classifiers in results.items():
        for clf_name, metrics in classifiers.items():
            rows.append({
                'Model': model_name,
                'Classifier': clf_name,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1']
            })

    df_results = pd.DataFrame(rows)

    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON TABLE")
    logger.info("=" * 80)
    logger.info("\n" + df_results.to_string(index=False))

    # Find best model
    best_idx = df_results['F1'].idxmax()
    best_model = df_results.iloc[best_idx]
    logger.info("\n" + "=" * 80)
    logger.info("BEST MODEL")
    logger.info("=" * 80)
    logger.info(f"Model: {best_model['Model']}")
    logger.info(f"Classifier: {best_model['Classifier']}")
    logger.info(f"Precision: {best_model['Precision']:.4f}")
    logger.info(f"Recall: {best_model['Recall']:.4f}")
    logger.info(f"F1: {best_model['F1']:.4f}")

    # Save results
    results_csv = output_dir / "comparison_results.csv"
    df_results.to_csv(results_csv, index=False)
    logger.info(f"\nSaved comparison results to {results_csv}")

    # Save detailed results
    detailed_results_path = output_dir / "detailed_results.pkl"
    with open(detailed_results_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Saved detailed results to {detailed_results_path}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Comparison complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
