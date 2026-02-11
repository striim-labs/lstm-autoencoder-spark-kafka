"""
Training Pipeline for Transaction Frequency Anomaly Detection

Trains 4 independent LSTM Encoder-Decoder models, one per
network/transaction type combination. Each model learns the
normal hourly transaction frequency pattern for its combo
and flags deviations as anomalies.

Usage:
    python -m app.train_transactions --data-path data/synthetic_transactions.csv
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from app.transaction_config import TransactionPreprocessorConfig, COMBO_KEYS
from app.transaction_preprocessor import TransactionPreprocessor
from app.model_registry import ModelRegistry, TransactionModelConfig
from app.anomaly_scorer import ScorerConfig

logger = logging.getLogger(__name__)


def train_all_combos(
    registry: ModelRegistry,
    preprocessor: TransactionPreprocessor,
    combo_dataloaders: Dict[Tuple[str, str], Dict[str, torch.utils.data.DataLoader]],
    epochs: int = 100,
    learning_rate: float = 1e-3,
    patience: int = 10,
    use_balanced_calibration: bool = True,
    calibration_magnitude: float = 2.0,
) -> Dict[Tuple[str, str], Dict]:
    """
    Train all 4 combo models.

    Args:
        registry: ModelRegistry to store models
        preprocessor: TransactionPreprocessor with fitted scalers
        combo_dataloaders: Dict of DataLoaders per combo
        epochs: Max training epochs
        learning_rate: Learning rate
        patience: Early stopping patience
        use_balanced_calibration: If True, use Sun-Tue + synthetic anomalies for calibration
        calibration_magnitude: Magnitude of synthetic anomalies in std units

    Returns:
        Dict of calibration results per combo
    """
    print("\n" + "=" * 60)
    print("TRAINING ALL COMBOS")
    print("=" * 60)

    calibration_results = {}

    for combo in COMBO_KEYS:
        print(f"\n--- Training {combo[0]}/{combo[1]} ---")

        loaders = combo_dataloaders[combo]
        train_loader = loaders["train"]
        val_loader = loaders["val"]

        if train_loader is None or val_loader is None:
            logger.warning(f"Skipping {combo}: missing train or val loader")
            continue

        # Store scaler from preprocessor
        registry.set_scaler(combo, preprocessor.scalers[combo])

        # Train model
        history = registry.train_combo(
            combo=combo,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            patience=patience,
        )

        # Fit scorer with calibration
        if use_balanced_calibration:
            # Use balanced calibration with Sun-Tue normal + synthetic anomalies
            # Day mapping: 19=Sat, 20=Sun, 21=Mon, 22=Tue, 23=Wed
            cal_result = registry.fit_scorer_with_calibration(
                combo=combo,
                val_loader=val_loader,
                preprocessor=preprocessor,
                normal_day_indices=[20, 21, 22],  # Sun, Mon, Tue
                spike_day_index=23,                # Wed - spike at 2-5 AM
                dip_day_index=19,                  # Sat - dip at 10 AM-2 PM
                spike_hours=(2, 5),
                dip_hours=(10, 14),
                magnitude_sigma=calibration_magnitude,
            )
            calibration_results[combo] = cal_result
        else:
            # Use simple percentile-based threshold
            threshold_loader = loaders.get("threshold_val") or val_loader
            registry.fit_scorer(
                combo=combo,
                val_loader=val_loader,
                threshold_loader=threshold_loader,
            )
            calibration_results[combo] = {"method": "percentile"}

    return calibration_results


def evaluate_all_combos(
    registry: ModelRegistry,
    preprocessor: TransactionPreprocessor,
    combo_dataloaders: Dict[Tuple[str, str], Dict[str, torch.utils.data.DataLoader]],
) -> Dict[Tuple[str, str], Dict]:
    """
    Evaluate all combo models on test data.

    Args:
        registry: Trained ModelRegistry
        preprocessor: TransactionPreprocessor with window info
        combo_dataloaders: Dict of DataLoaders per combo

    Returns:
        Dict of evaluation results per combo
    """
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)

    all_results = {}

    for combo in COMBO_KEYS:
        print(f"\n--- {combo[0]}/{combo[1]} ---")

        loaders = combo_dataloaders[combo]
        test_loader = loaders.get("test")

        if test_loader is None:
            logger.warning(f"Skipping {combo}: no test loader")
            continue

        # Get predictions
        predictions, window_scores, point_scores = registry.predict(combo, test_loader)

        # Get test window metadata
        test_info = preprocessor.get_test_window_info(combo)

        # For now, all test windows are labeled as normal (anomalies injected later)
        # We can still report reconstruction quality
        results = {
            "num_windows": len(predictions),
            "num_predicted_anomaly": int(predictions.sum()),
            "mean_window_score": float(window_scores.mean()),
            "max_window_score": float(window_scores.max()),
            "predictions": predictions,
            "window_scores": window_scores,
            "point_scores": point_scores,
        }

        # Print per-window results
        print(f"{'Day':<12} {'Date':<12} {'DoW':<10} {'Score':>10} {'Pred':>10}")
        print("-" * 60)

        for i, (score, pred, info) in enumerate(zip(window_scores, predictions, test_info)):
            pred_str = "ANOMALY" if pred else "normal"
            print(
                f"{info['day_index']:<12} "
                f"{info['date']:<12} "
                f"{info['day_name']:<10} "
                f"{score:>10.4f} "
                f"{pred_str:>10}"
            )

        print("-" * 60)
        print(f"Predicted anomalies: {results['num_predicted_anomaly']}/{results['num_windows']}")

        # Compute reconstruction MSE stats
        scorer = registry.get_scorer(combo)
        model = registry.get_model(combo)
        model.eval()

        all_mse = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch.to(registry.device)
                x_recon = model(x)
                mse = ((x - x_recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
                all_mse.extend(mse)

        results["mean_mse"] = float(np.mean(all_mse))
        results["std_mse"] = float(np.std(all_mse))
        print(f"Reconstruction MSE: {results['mean_mse']:.6f} ± {results['std_mse']:.6f}")

        all_results[combo] = results

    return all_results


def print_summary(
    registry: ModelRegistry,
    preprocessor: TransactionPreprocessor,
    eval_results: Dict[Tuple[str, str], Dict],
) -> None:
    """Print training and evaluation summary."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    stats = registry.get_stats()

    print("\nModel Configuration:")
    print(f"  Hidden dim: {stats['model_config']['hidden_dim']}")
    print(f"  Num layers: {stats['model_config']['num_layers']}")
    print(f"  Sequence length: {stats['model_config']['sequence_length']}")

    print("\nScorer Configuration:")
    print(f"  Scoring mode: {stats['scorer_config']['scoring_mode']}")
    print(f"  Hard criterion k: {stats['scorer_config']['hard_criterion_k']}")

    print("\nPer-Combo Results:")
    print(f"{'Combo':<20} {'Best Epoch':>12} {'Val Loss':>12} {'Test MSE':>12} {'Threshold':>12}")
    print("-" * 70)

    for combo in COMBO_KEYS:
        combo_key = f"{combo[0]}_{combo[1]}"
        combo_stats = stats["combos"].get(combo_key, {})

        best_epoch = combo_stats.get("best_epoch", "N/A")
        val_loss = combo_stats.get("final_val_loss", None)
        val_loss_str = f"{val_loss:.6f}" if val_loss else "N/A"

        eval_res = eval_results.get(combo, {})
        test_mse = eval_res.get("mean_mse", None)
        test_mse_str = f"{test_mse:.6f}" if test_mse else "N/A"

        threshold = combo_stats.get("point_threshold", None)
        threshold_str = f"{threshold:.4f}" if threshold else "N/A"

        combo_name = f"{combo[0]}/{combo[1]}"
        print(f"{combo_name:<20} {best_epoch:>12} {val_loss_str:>12} {test_mse_str:>12} {threshold_str:>12}")


def main():
    """Main training script for transaction anomaly detection."""
    parser = argparse.ArgumentParser(
        description="Train LSTM-AE models for transaction frequency anomaly detection"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/synthetic_transactions.csv",
        help="Path to synthetic transactions CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/transactions",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=18,
        help="LSTM hidden dimension (16 for DoW conditioning bottleneck)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=97.0,
        help="Percentile for anomaly threshold"
    )
    parser.add_argument(
        "--hard-criterion-k",
        type=int,
        default=3,
        help="Number of anomalous points to flag window (3/24 = 12.5%)"
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
        "--calibration-magnitude",
        type=float,
        default=2.0,
        help="Magnitude of synthetic anomalies in std units for calibration"
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("\n" + "=" * 60)
    print("TRANSACTION FREQUENCY ANOMALY DETECTION")
    print("LSTM Encoder-Decoder Training Pipeline")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Step 1: Preprocess data
    print("\n" + "-" * 40)
    print("Step 1: Preprocessing data")
    print("-" * 40)

    preprocess_config = TransactionPreprocessorConfig()
    preprocessor = TransactionPreprocessor(config=preprocess_config)
    combo_dataloaders = preprocessor.preprocess(
        args.data_path,
        batch_size=args.batch_size
    )

    # Print data summary
    stats = preprocessor.get_stats()
    print(f"\nTotal transactions: {stats['total_transactions']:,}")
    print(f"Date range: {stats['date_range']['start'][:10]} to {stats['date_range']['end'][:10]}")
    print(f"Combos: {len(COMBO_KEYS)}")

    for combo in COMBO_KEYS:
        combo_key = f"{combo[0]}_{combo[1]}"
        combo_stats = stats["combos"].get(combo_key, {})
        splits = combo_stats.get("splits", {})
        print(f"  {combo[0]}/{combo[1]}: train={splits.get('train', 0)}, "
              f"val={splits.get('val', 0)}, threshold={splits.get('threshold_val', 0)}, "
              f"test={splits.get('test', 0)}")

    # Step 2: Initialize registry
    print("\n" + "-" * 40)
    print("Step 2: Initializing model registry")
    print("-" * 40)

    model_config = TransactionModelConfig(
        hidden_dim=args.hidden_dim,
    )
    scorer_config = ScorerConfig(
        scoring_mode="point",
        threshold_percentile=args.threshold_percentile,
        hard_criterion_k=args.hard_criterion_k,
    )
    registry = ModelRegistry(
        model_config=model_config,
        scorer_config=scorer_config,
        device=device,
    )

    # Step 3: Train all combos
    print("\n" + "-" * 40)
    print("Step 3: Training models")
    print("-" * 40)

    use_balanced = not args.no_balanced_calibration
    if use_balanced:
        print("\nUsing balanced calibration:")
        print("  Normal days: Sun (20), Mon (21), Tue (22)")
        print("  Spike injection: Wed (23) at 2-5 AM")
        print("  Dip injection: Sat (19) at 10 AM-2 PM")
        print(f"  Magnitude: {args.calibration_magnitude}σ")
        print("  Threshold method: F1-max")
    else:
        print("\nUsing simple percentile-based threshold")

    calibration_results = train_all_combos(
        registry=registry,
        preprocessor=preprocessor,
        combo_dataloaders=combo_dataloaders,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        use_balanced_calibration=use_balanced,
        calibration_magnitude=args.calibration_magnitude,
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
    print_summary(registry, preprocessor, eval_results)

    # Step 5: Save models
    if not args.skip_save:
        print("\n" + "-" * 40)
        print("Step 5: Saving models")
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
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
