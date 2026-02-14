"""
Training Pipeline for FCVAE Transaction Anomaly Detection

Trains 4 independent FCVAE (Frequency-enhanced Conditional VAE) models,
one per network/transaction type combination. Each model learns normal
hourly transaction frequency patterns using frequency-domain conditioning
and VAE-based generative modeling.

Key differences from LSTM-AE training (train_transactions.py):
- Uses sliding windows with configurable stride (default: stride=1)
- ELBO loss (reconstruction NLL + KLD) instead of MSE
- Data augmentation during training (point anomalies, segment swaps, missing data)
- NLL-based scoring (lower = anomalous, inverted from Mahalanobis)
- CosineAnnealingLR scheduler
- Larger batch size (64 vs 4) due to more training windows

Usage:
    python -m app.train_fcvae --data-path data/synthetic_transactions.csv
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from app.transaction_config import TransactionPreprocessorConfig, COMBO_KEYS
from app.transaction_preprocessor import TransactionPreprocessor
from app.fcvae_registry import FCVAERegistry
from app.fcvae_model import FCVAEConfig
from app.fcvae_scorer import FCVAEScorerConfig
from app.fcvae_augment import AugmentConfig

logger = logging.getLogger(__name__)


def optimize_threshold_f1(
    registry: FCVAERegistry,
    combo: Tuple[str, str],
    val_loader: torch.utils.data.DataLoader,
    beta: float = 1.0,
) -> Dict:
    """
    Optimize threshold using F1 score on validation set with real anomaly labels.

    This is used when the validation set contains real injected anomalies
    (e.g., from the 60-day split dataset).

    Args:
        registry: FCVAERegistry with trained model
        combo: (network_type, transaction_type) tuple
        val_loader: Validation DataLoader with anomaly labels
        beta: F-beta score parameter (1.0 = F1)

    Returns:
        Dict with calibration metrics
    """
    model = registry.get_model(combo)
    scorer = registry.get_scorer(combo)

    if model is None or scorer is None:
        raise ValueError(f"Model or scorer not found for {combo}")

    # Collect all validation labels first
    all_labels = []
    for batch in val_loader:
        _, labels, _ = batch
        all_labels.append(labels.numpy())
    all_labels = np.concatenate(all_labels)  # (N, W)

    # Score all windows using scorer.score_batch
    all_point_scores, _ = scorer.score_batch(model, val_loader, registry.device)

    # Flatten for point-level threshold optimization
    flat_scores = all_point_scores.flatten()
    flat_labels = all_labels.flatten().astype(int)

    # Separate normal and anomaly scores
    normal_scores = flat_scores[flat_labels == 0]
    anomaly_scores = flat_scores[flat_labels == 1]

    logger.info(f"  Validation set: {len(normal_scores)} normal points, {len(anomaly_scores)} anomaly points")

    if len(anomaly_scores) == 0:
        logger.warning(f"  No anomaly points in validation set - falling back to percentile threshold")
        scorer.set_threshold(flat_scores, method="percentile", percentile=5.0)
        return {"method": "percentile_fallback", "reason": "no_anomalies"}

    # Use F1-max to find optimal threshold
    optimal_threshold, metrics = scorer.find_optimal_threshold(
        normal_scores=normal_scores,
        anomaly_scores=anomaly_scores,
        method="f1_max",
        beta=beta,
    )

    # Set the threshold
    scorer.point_threshold = optimal_threshold

    # Also set window threshold using the same percentile of window scores
    # that corresponds to the point threshold
    window_scores = all_point_scores.mean(axis=1)  # Mean NLL per window
    window_labels = (all_labels.sum(axis=1) > 0).astype(int)  # Window is anomaly if any point is

    normal_window_scores = window_scores[window_labels == 0]
    anomaly_window_scores = window_scores[window_labels == 1]

    if len(anomaly_window_scores) > 0:
        window_threshold, window_metrics = scorer.find_optimal_threshold(
            normal_scores=normal_window_scores,
            anomaly_scores=anomaly_window_scores,
            method="f1_max",
            beta=beta,
        )
        scorer.window_threshold = window_threshold
        metrics["window_threshold"] = window_threshold
        metrics["window_f1"] = window_metrics.get("best_f1", 0)
    else:
        # Fallback: use percentile
        scorer.set_window_threshold(normal_window_scores, method="percentile", percentile=5.0)
        metrics["window_threshold"] = scorer.window_threshold
        metrics["window_method"] = "percentile_fallback"

    logger.info(f"  F1-optimized thresholds: point={scorer.point_threshold:.4f}, window={scorer.window_threshold:.4f}")
    logger.info(f"  Point F1={metrics.get('best_f1', 0):.4f}, Window F1={metrics.get('window_f1', 0):.4f}")

    return metrics


def train_all_combos(
    registry: FCVAERegistry,
    preprocessor: TransactionPreprocessor,
    combo_dataloaders: Dict[Tuple[str, str], Dict[str, torch.utils.data.DataLoader]],
    epochs: int = 30,
    learning_rate: float = 5e-4,
    patience: int = 5,
    grad_clip: float = 2.0,
    use_balanced_calibration: bool = True,
    calibration_magnitude: float = 1.5,
    use_f1_optimization: bool = True,
    kl_warmup_epochs: int = 10,
) -> Dict[Tuple[str, str], Dict]:
    """
    Train all 4 combo models using FCVAE.

    Args:
        registry: FCVAERegistry to store models
        preprocessor: TransactionPreprocessor with fitted scalers
        combo_dataloaders: Dict of DataLoaders per combo
        epochs: Max training epochs
        learning_rate: Learning rate
        patience: Early stopping patience
        grad_clip: Gradient clipping value
        use_balanced_calibration: If True, use normal windows + synthetic anomalies for calibration
        calibration_magnitude: Magnitude of synthetic anomalies in std units
        use_f1_optimization: If True and validation has real anomalies, use F1 optimization
        kl_warmup_epochs: Epochs to linearly ramp KLD weight from 0 to 1

    Returns:
        Dict of calibration results per combo
    """
    print("\n" + "=" * 60)
    print("TRAINING ALL COMBOS (FCVAE)")
    print("=" * 60)

    calibration_results = {}

    for combo in COMBO_KEYS:
        print(f"\n--- Training {combo[0]}/{combo[1]} ---")

        loaders = combo_dataloaders[combo]
        train_loader = loaders.get("train")
        val_loader = loaders.get("val")

        if train_loader is None or val_loader is None:
            logger.warning(f"Skipping {combo}: missing train or val loader")
            continue

        # Store scaler from preprocessor (use sliding scaler key)
        scaler_key = (combo[0], combo[1] + "_sliding")
        if scaler_key in preprocessor.scalers:
            registry.set_scaler(scaler_key, preprocessor.scalers[scaler_key])
        elif combo in preprocessor.scalers:
            registry.set_scaler(combo, preprocessor.scalers[combo])

        # Train model
        history = registry.train_combo(
            combo=combo,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            patience=patience,
            grad_clip=grad_clip,
            kl_warmup_epochs=kl_warmup_epochs,
        )

        # Fit scorer with threshold calibration
        # Check if validation set has real anomaly labels
        val_has_anomalies = False
        if val_loader is not None:
            for batch in val_loader:
                _, labels, _ = batch
                if labels.sum() > 0:
                    val_has_anomalies = True
                    break

        if use_f1_optimization and val_has_anomalies:
            # Use F1 optimization with real validation anomalies
            print(f"    Using F1-based threshold optimization (validation has real anomalies)")
            try:
                cal_result = optimize_threshold_f1(
                    registry=registry,
                    combo=combo,
                    val_loader=val_loader,
                    beta=1.0,
                )
                cal_result["method"] = "f1_optimization"
                calibration_results[combo] = cal_result
            except Exception as e:
                logger.warning(f"F1 optimization failed for {combo}: {e}")
                logger.info("Falling back to percentile-based threshold")
                registry.fit_scorer(combo=combo, val_loader=val_loader)
                calibration_results[combo] = {"method": "percentile_fallback"}
        elif use_balanced_calibration:
            # Use balanced calibration with normal windows + synthetic anomalies
            # For sliding windows, use window indices rather than day indices
            # Estimate: ~17 windows per day with stride=1, so:
            # Days 20-22 (Sun-Tue) start at roughly windows 17*2=34 to 17*5=85
            print(f"    Using balanced calibration (synthetic anomalies)")
            try:
                # Note: spike_day_index and dip_day_index are RELATIVE to the threshold period
                # Threshold period: 3 days starting at day 22
                # Index 0 = day 22, Index 1 = day 23, Index 2 = day 24
                cal_result = registry.fit_scorer_with_calibration(
                    combo=combo,
                    val_loader=val_loader,
                    preprocessor=preprocessor,
                    normal_day_indices=[0, 1, 2],     # All 3 days of threshold period for normal
                    spike_day_index=1,                # Day 1 of threshold (day 23) - spike at 2-6 AM
                    dip_day_index=2,                  # Day 2 of threshold (day 24) - dip at 10-14
                    spike_hours=(2, 6),               # 4 hours: 2, 3, 4, 5 AM
                    dip_hours=(10, 14),               # 4 hours: 10, 11, 12, 13
                    magnitude_sigma=calibration_magnitude,
                )
                calibration_results[combo] = cal_result
            except Exception as e:
                logger.warning(f"Calibration failed for {combo}: {e}")
                logger.info("Falling back to percentile-based threshold")
                registry.fit_scorer(combo=combo, val_loader=val_loader)
                calibration_results[combo] = {"method": "percentile_fallback"}
        else:
            # Use simple percentile-based threshold
            print(f"    Using percentile-based threshold")
            registry.fit_scorer(combo=combo, val_loader=val_loader)
            calibration_results[combo] = {"method": "percentile"}

    return calibration_results


def evaluate_all_combos(
    registry: FCVAERegistry,
    preprocessor: TransactionPreprocessor,
    combo_dataloaders: Dict[Tuple[str, str], Dict[str, torch.utils.data.DataLoader]],
) -> Dict[Tuple[str, str], Dict]:
    """
    Evaluate all combo models on test data.

    Args:
        registry: Trained FCVAERegistry
        preprocessor: TransactionPreprocessor with window info
        combo_dataloaders: Dict of DataLoaders per combo

    Returns:
        Dict of evaluation results per combo
    """
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET (FCVAE)")
    print("=" * 60)

    all_results = {}

    for combo in COMBO_KEYS:
        print(f"\n--- {combo[0]}/{combo[1]} ---")

        loaders = combo_dataloaders[combo]
        test_loader = loaders.get("test")

        if test_loader is None:
            logger.warning(f"Skipping {combo}: no test loader")
            continue

        # Collect test labels
        all_labels = []
        for batch in test_loader:
            _, labels, _ = batch
            all_labels.append(labels.numpy())
        all_labels = np.concatenate(all_labels)  # (N, W)

        # Get predictions
        predictions, window_scores, point_scores = registry.predict(combo, test_loader)

        # Compute window-level labels (window is anomaly if any point is anomaly)
        window_labels = (all_labels.sum(axis=1) > 0).astype(int)

        # Compute P/R/F1 at window level
        tp = int(((predictions == 1) & (window_labels == 1)).sum())
        fp = int(((predictions == 1) & (window_labels == 0)).sum())
        fn = int(((predictions == 0) & (window_labels == 1)).sum())
        tn = int(((predictions == 0) & (window_labels == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Summarize results
        results = {
            "num_windows": len(predictions),
            "num_predicted_anomaly": int(predictions.sum()),
            "num_actual_anomaly": int(window_labels.sum()),
            "mean_window_score": float(window_scores.mean()),
            "min_window_score": float(window_scores.min()),  # Lower = more anomalous
            "max_window_score": float(window_scores.max()),
            "predictions": predictions,
            "window_scores": window_scores,
            "point_scores": point_scores,
            "window_labels": window_labels,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        # Print summary
        scorer = registry.get_scorer(combo)
        print(f"  Test windows: {results['num_windows']}")
        print(f"  Predicted anomalies: {results['num_predicted_anomaly']}")
        print(f"  Window score range: [{results['min_window_score']:.4f}, {results['max_window_score']:.4f}]")
        print(f"  Mean window score: {results['mean_window_score']:.4f}")
        print(f"  Window threshold: {scorer.window_threshold:.4f}")
        print(f"  Point threshold: {scorer.point_threshold:.4f}")

        # Print per-window scores (first 10 and last 10 if many)
        print(f"\n  {'Window':<8} {'Score':>12} {'Prediction':>12}")
        print("  " + "-" * 35)

        display_indices = list(range(min(5, len(window_scores))))
        if len(window_scores) > 10:
            display_indices.extend(list(range(len(window_scores) - 5, len(window_scores))))

        for i in sorted(set(display_indices)):
            score = window_scores[i]
            pred = "ANOMALY" if predictions[i] else "normal"
            print(f"  {i:<8} {score:>12.4f} {pred:>12}")

        if len(window_scores) > 10:
            print(f"  ... ({len(window_scores) - 10} more windows)")

        print("  " + "-" * 35)

        all_results[combo] = results

    return all_results


def print_summary(
    registry: FCVAERegistry,
    eval_results: Dict[Tuple[str, str], Dict],
) -> None:
    """Print training and evaluation summary."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    stats = registry.get_stats()

    print("\nFCVAE Configuration:")
    print(f"  Window size: {stats['model_config']['window']}")
    print(f"  Latent dim: {stats['model_config']['latent_dim']}")
    print(f"  Condition embedding dim: {stats['model_config']['condition_emb_dim']}")

    print("\nScorer Configuration:")
    print(f"  Score mode: {stats['scorer_config']['score_mode']}")
    print(f"  Hard criterion k: {stats['scorer_config']['hard_criterion_k']}")

    print("\nAugmentation Configuration:")
    print(f"  Point anomaly rate: {stats['augment_config']['point_ano_rate']}")
    print(f"  Segment anomaly rate: {stats['augment_config']['seg_ano_rate']}")
    print(f"  Missing data rate: {stats['augment_config']['missing_data_rate']}")

    print("\nPer-Combo Training Results:")
    print(f"{'Combo':<20} {'Best Ep':>10} {'Val Loss':>12} {'Test Anom':>12} {'W-Thresh':>12}")
    print("-" * 70)

    for combo in COMBO_KEYS:
        combo_key = f"{combo[0]}_{combo[1]}"
        combo_stats = stats["combos"].get(combo_key, {})

        best_epoch = combo_stats.get("best_epoch", "N/A")
        val_loss = combo_stats.get("final_val_loss", None)
        val_loss_str = f"{val_loss:.6f}" if val_loss else "N/A"

        eval_res = eval_results.get(combo, {})
        test_anom = eval_res.get("num_predicted_anomaly", None)
        num_windows = eval_res.get("num_windows", 1)
        test_anom_str = f"{test_anom}/{num_windows}" if test_anom is not None else "N/A"

        threshold = combo_stats.get("window_threshold", None)
        threshold_str = f"{threshold:.4f}" if threshold else "N/A"

        combo_name = f"{combo[0]}/{combo[1]}"
        print(f"{combo_name:<20} {best_epoch:>10} {val_loss_str:>12} {test_anom_str:>12} {threshold_str:>12}")

    # P/R/F1 Table
    print("\nTest Set Performance (Window-Level):")
    print(f"{'Combo':<20} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 86)

    for combo in COMBO_KEYS:
        eval_res = eval_results.get(combo, {})
        combo_name = f"{combo[0]}/{combo[1]}"

        tp = eval_res.get("tp", 0)
        fp = eval_res.get("fp", 0)
        fn = eval_res.get("fn", 0)
        tn = eval_res.get("tn", 0)
        precision = eval_res.get("precision", 0.0)
        recall = eval_res.get("recall", 0.0)
        f1 = eval_res.get("f1", 0.0)

        print(f"{combo_name:<20} {tp:>6} {fp:>6} {fn:>6} {tn:>6} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")


def main():
    """Main training script for FCVAE transaction anomaly detection."""
    parser = argparse.ArgumentParser(
        description="Train FCVAE models for transaction frequency anomaly detection"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/synthetic_transactions_v2_split60.csv",
        help="Path to synthetic transactions CSV (with train/val/test splits)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/transactions_fcvae_60",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=24,
        help="Sliding window size in hours"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding window stride in hours"
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=8,
        help="FCVAE latent space dimension"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (larger than LSTM-AE due to more training windows)"
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=2.0,
        help="Gradient clipping max value"
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=5.0,
        help="Percentile for anomaly threshold (LOW for FCVAE - anomalies have low scores)"
    )
    parser.add_argument(
        "--hard-criterion-k",
        type=int,
        default=3,
        help="Number of anomalous points to flag window (3/24 = 12.5%%)"
    )
    parser.add_argument(
        "--skip-save",
        action="store_true",
        help="Skip saving models (for quick testing)"
    )
    parser.add_argument(
        "--no-balanced-calibration",
        action="store_true",
        help="Disable balanced calibration (use simple percentile threshold)"
    )
    parser.add_argument(
        "--no-f1-optimization",
        action="store_true",
        help="Disable F1-based threshold optimization even if validation has real anomalies"
    )
    parser.add_argument(
        "--calibration-magnitude",
        type=float,
        default=1.5,
        help="Magnitude of synthetic anomalies in std units for calibration"
    )
    parser.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=10,
        help="Epochs to linearly ramp KL divergence weight from 0 to 1 (slower = more stable)"
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable data augmentation during training"
    )
    parser.add_argument(
        "--score-mode",
        type=str,
        choices=["single_pass", "mcmc"],
        default="single_pass",
        help="Scoring mode: single_pass (fast) or mcmc (accurate)"
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("\n" + "=" * 60)
    print("FCVAE TRANSACTION FREQUENCY ANOMALY DETECTION")
    print("Frequency-enhanced Conditional VAE Training Pipeline")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Step 1: Preprocess data with sliding windows
    print("\n" + "-" * 40)
    print("Step 1: Preprocessing data (sliding windows)")
    print("-" * 40)

    preprocess_config = TransactionPreprocessorConfig()
    preprocessor = TransactionPreprocessor(config=preprocess_config)

    # Load and aggregate data (this populates combo_hourly)
    preprocessor.load_and_aggregate(args.data_path)

    print(f"\nSliding window config: window_size={args.window_size}, stride={args.stride}")

    # Create sliding window DataLoaders for each combo
    combo_dataloaders = {}

    for combo in COMBO_KEYS:
        print(f"\n  Creating sliding windows for {combo[0]}/{combo[1]}...")

        # Create sliding windows and splits
        splits = preprocessor.create_sliding_splits(
            combo=combo,
            window_size=args.window_size,
            stride=args.stride
        )

        # Normalize
        normalized = preprocessor.normalize_sliding_windows(
            combo=combo,
            splits=splits,
            fit_on="train"
        )

        # Create DataLoaders
        loaders = preprocessor.create_sliding_dataloaders(
            normalized_splits=normalized,
            batch_size=args.batch_size,
            shuffle_train=True
        )

        combo_dataloaders[combo] = loaders

        # Print stats
        for split_name in ["train", "val", "test"]:
            if split_name in loaders and loaders[split_name] is not None:
                num_windows = len(loaders[split_name].dataset)
                print(f"    {split_name}: {num_windows} windows")

    # Step 2: Initialize FCVAE registry
    print("\n" + "-" * 40)
    print("Step 2: Initializing FCVAE registry")
    print("-" * 40)

    model_config = FCVAEConfig(
        window=args.window_size,
        latent_dim=args.latent_dim,
    )
    scorer_config = FCVAEScorerConfig(
        threshold_percentile=args.threshold_percentile,
        hard_criterion_k=args.hard_criterion_k,
        score_mode=args.score_mode,
    )
    augment_config = AugmentConfig() if not args.no_augmentation else AugmentConfig(
        missing_data_rate=0.0,
        point_ano_rate=0.0,
        seg_ano_rate=0.0,
    )

    registry = FCVAERegistry(
        model_config=model_config,
        scorer_config=scorer_config,
        augment_config=augment_config,
        device=device,
    )

    # Step 3: Train all combos
    print("\n" + "-" * 40)
    print("Step 3: Training FCVAE models")
    print("-" * 40)

    use_balanced = not args.no_balanced_calibration
    use_f1 = not args.no_f1_optimization

    if use_f1:
        print("\nF1-based threshold optimization enabled")
        print("  Will use real anomaly labels from validation set if available")
    elif use_balanced:
        print("\nUsing balanced calibration:")
        print("  Normal days: Sun (20), Mon (21), Tue (22)")
        print("  Spike injection: Wed (23) at 2-5 AM")
        print("  Dip injection: Sat (19) at 10 AM-2 PM")
        print(f"  Magnitude: {args.calibration_magnitude}sigma")
        print("  Threshold method: F1-max")
    else:
        print("\nUsing simple percentile-based threshold")

    if not args.no_augmentation:
        print("\nData augmentation enabled:")
        print(f"  Point anomaly rate: {augment_config.point_ano_rate}")
        print(f"  Segment anomaly rate: {augment_config.seg_ano_rate}")
        print(f"  Missing data rate: {augment_config.missing_data_rate}")
    else:
        print("\nData augmentation disabled")

    calibration_results = train_all_combos(
        registry=registry,
        preprocessor=preprocessor,
        combo_dataloaders=combo_dataloaders,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        grad_clip=args.grad_clip,
        use_balanced_calibration=use_balanced,
        calibration_magnitude=args.calibration_magnitude,
        use_f1_optimization=use_f1,
        kl_warmup_epochs=args.kl_warmup_epochs,
    )

    # Step 4: Evaluate
    print("\n" + "-" * 40)
    print("Step 4: Evaluating on test set")
    print("-" * 40)

    eval_results = evaluate_all_combos(
        registry=registry,
        preprocessor=preprocessor,
        combo_dataloaders=combo_dataloaders,
    )

    # Print summary
    print_summary(registry, eval_results)

    # Step 5: Save models
    if not args.skip_save:
        print("\n" + "-" * 40)
        print("Step 5: Saving FCVAE models")
        print("-" * 40)

        registry.save_all(args.output_dir)

        print(f"\nArtifacts saved to: {args.output_dir}/")
        for combo in COMBO_KEYS:
            dirname = f"{combo[0]}_{combo[1].replace('-', '')}"
            print(f"  {dirname}/")
            print(f"    - model.pt")
            print(f"    - scorer.pkl")
            print(f"    - scaler.pkl")
            print(f"    - history.pkl")

    print("\n" + "=" * 60)
    print("FCVAE TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
