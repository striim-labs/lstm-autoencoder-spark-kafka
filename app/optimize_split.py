"""
Train/Validation Split Optimizer for LSTM Encoder-Decoder

Tests different train/val/threshold split configurations on the first N normal weeks
and evaluates on the held-out test set to find the optimal split.

Based on the paper's approach:
- sN (train): Training sequences for encoder-decoder
- vN1 (val): Validation for early stopping AND fitting error distribution (μ, Σ)
- vN2 (threshold_val): Validation for threshold τ selection
- Test: Remaining weeks (normal + anomalous)
"""

import argparse
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import torch

from data_preprocessor import NYCTaxiPreprocessor, PreprocessorConfig
from lstm_autoencoder import create_model
from anomaly_scorer import AnomalyScorer, ScorerConfig
from train import train_model, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for a single split experiment."""
    train_weeks: int
    val_weeks: int
    threshold_weeks: int

    @property
    def total_dev_weeks(self) -> int:
        return self.train_weeks + self.val_weeks + self.threshold_weeks


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    config: SplitConfig
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    train_loss: float
    val_loss: float
    threshold: float
    predictions: List[bool]
    actuals: List[bool]
    scores: List[float]

    def to_dict(self) -> Dict:
        return {
            "train_weeks": self.config.train_weeks,
            "val_weeks": self.config.val_weeks,
            "threshold_weeks": self.config.threshold_weeks,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "threshold": self.threshold,
        }


def compute_metrics(predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    tp = np.sum(predictions & actuals)
    fp = np.sum(predictions & ~actuals)
    fn = np.sum(~predictions & actuals)
    tn = np.sum(~predictions & ~actuals)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def run_experiment(
    split_config: SplitConfig,
    data_path: str,
    device: torch.device,
    hidden_dim: int = 64,
    num_layers: int = 1,
    dropout: float = 0.2,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    patience: int = 10,
    batch_size: int = 4,
    threshold_percentile: float = 95.0,
    verbose: bool = False
) -> Optional[ExperimentResult]:
    """
    Run a single experiment with the given split configuration.

    Returns:
        ExperimentResult or None if experiment failed
    """
    try:
        # Setup preprocessor with this split config
        preprocess_config = PreprocessorConfig(
            train_weeks=split_config.train_weeks,
            val_weeks=split_config.val_weeks,
            threshold_weeks=split_config.threshold_weeks
        )
        preprocessor = NYCTaxiPreprocessor(config=preprocess_config)
        dataloaders, _ = preprocessor.preprocess(data_path, batch_size=batch_size)

        # Check we have enough data in all splits
        if dataloaders["train"] is None or len(dataloaders["train"].dataset) == 0:
            logger.warning(f"Skipping config {split_config}: empty train set")
            return None
        if dataloaders["val"] is None or len(dataloaders["val"].dataset) == 0:
            logger.warning(f"Skipping config {split_config}: empty val set")
            return None
        if dataloaders["threshold_val"] is None or len(dataloaders["threshold_val"].dataset) == 0:
            logger.warning(f"Skipping config {split_config}: empty threshold_val set")
            return None
        if dataloaders["test"] is None or len(dataloaders["test"].dataset) == 0:
            logger.warning(f"Skipping config {split_config}: empty test set")
            return None

        # Create model
        model = create_model(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        model.to(device)

        # Train model
        training_config = TrainingConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            patience=patience
        )

        # Suppress verbose logging during optimization
        if not verbose:
            logging.getLogger("train").setLevel(logging.WARNING)

        model, history = train_model(
            model=model,
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            device=device,
            config=training_config
        )

        # Fit anomaly scorer on validation set (vN1) per paper
        scorer = AnomalyScorer(
            config=ScorerConfig(threshold_percentile=threshold_percentile)
        )
        scorer.fit(model, dataloaders["val"], device)

        # Set threshold using threshold_val set (vN2)
        val_scores, _ = scorer.compute_scores(model, dataloaders["threshold_val"], device)
        scorer.set_threshold(val_scores)

        # Evaluate on test set
        test_scores, _ = scorer.compute_scores(model, dataloaders["test"], device)
        predictions = scorer.predict(test_scores)

        # Get actuals
        test_week_info = preprocessor.get_test_week_info()
        actuals = np.array([w["is_anomaly"] for w in test_week_info])

        # Compute metrics
        metrics = compute_metrics(predictions, actuals)

        result = ExperimentResult(
            config=split_config,
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            train_loss=history["train_loss"][-1],
            val_loss=history["val_loss"][-1] if history["val_loss"] else float('inf'),
            threshold=scorer.threshold,
            predictions=predictions.tolist(),
            actuals=actuals.tolist(),
            scores=test_scores.tolist(),
        )

        return result

    except Exception as e:
        logger.error(f"Experiment failed for {split_config}: {e}")
        return None


def generate_split_configs(
    total_dev_weeks: int,
    min_train: int = 4,
    min_val: int = 2,
    min_threshold: int = 1
) -> List[SplitConfig]:
    """
    Generate all valid split configurations.

    Args:
        total_dev_weeks: Total weeks to allocate to train + val + threshold
        min_train: Minimum training weeks
        min_val: Minimum validation weeks (for early stopping + error distribution)
        min_threshold: Minimum threshold calibration weeks

    Returns:
        List of valid SplitConfig objects
    """
    configs = []

    for train_weeks in range(min_train, total_dev_weeks - min_val - min_threshold + 1):
        remaining = total_dev_weeks - train_weeks
        for val_weeks in range(min_val, remaining - min_threshold + 1):
            threshold_weeks = remaining - val_weeks
            if threshold_weeks >= min_threshold:
                configs.append(SplitConfig(
                    train_weeks=train_weeks,
                    val_weeks=val_weeks,
                    threshold_weeks=threshold_weeks
                ))

    return configs


def print_results_table(results: List[ExperimentResult], baseline_f1: Optional[float] = None) -> None:
    """Print results in a formatted table."""
    print("\n" + "=" * 90)
    print("SPLIT OPTIMIZATION RESULTS")
    print("=" * 90)

    # Sort by F1 score descending
    sorted_results = sorted(results, key=lambda r: r.f1_score, reverse=True)

    print(f"\n{'Train':>6} {'Val':>5} {'Thresh':>6} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Δ vs Base':>10}")
    print("-" * 90)

    for r in sorted_results:
        delta = ""
        if baseline_f1 is not None:
            diff = r.f1_score - baseline_f1
            delta = f"{diff:+.2%}"

        print(f"{r.config.train_weeks:>6} {r.config.val_weeks:>5} {r.config.threshold_weeks:>6} "
              f"{r.accuracy:>8.2%} {r.precision:>8.2%} {r.recall:>8.2%} {r.f1_score:>8.2%} {delta:>10}")

    # Best result
    best = sorted_results[0]
    print("\n" + "-" * 90)
    print(f"BEST CONFIG: train={best.config.train_weeks}, val={best.config.val_weeks}, "
          f"threshold={best.config.threshold_weeks}")
    print(f"  F1-Score: {best.f1_score:.2%}")
    print(f"  Precision: {best.precision:.2%}, Recall: {best.recall:.2%}")
    print("=" * 90)


def main():
    """Main optimization script."""
    parser = argparse.ArgumentParser(
        description="Optimize train/validation split for LSTM anomaly detection"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/nyc_taxi.csv",
        help="Path to NYC taxi CSV file"
    )
    parser.add_argument(
        "--total-dev-weeks",
        type=int,
        default=14,
        help="Total weeks to use for train + val + threshold (remaining are test)"
    )
    parser.add_argument(
        "--min-train",
        type=int,
        default=4,
        help="Minimum training weeks"
    )
    parser.add_argument(
        "--min-val",
        type=int,
        default=2,
        help="Minimum validation weeks"
    )
    parser.add_argument(
        "--min-threshold",
        type=int,
        default=1,
        help="Minimum threshold calibration weeks"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="LSTM hidden dimension"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--baseline-f1",
        type=float,
        default=None,
        help="Baseline F1 score to compare against"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output during training"
    )
    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("\n" + "=" * 60)
    print("SPLIT OPTIMIZATION")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Generate split configurations
    configs = generate_split_configs(
        total_dev_weeks=args.total_dev_weeks,
        min_train=args.min_train,
        min_val=args.min_val,
        min_threshold=args.min_threshold
    )

    print(f"\nTesting {len(configs)} split configurations...")
    print(f"Total development weeks: {args.total_dev_weeks}")
    print(f"Constraints: min_train={args.min_train}, min_val={args.min_val}, min_threshold={args.min_threshold}")

    # Run experiments
    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testing: train={config.train_weeks}, "
              f"val={config.val_weeks}, threshold={config.threshold_weeks}...")

        result = run_experiment(
            split_config=config,
            data_path=args.data_path,
            device=device,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            patience=args.patience,
            verbose=args.verbose
        )

        if result is not None:
            results.append(result)
            print(f"    -> F1: {result.f1_score:.2%}, Precision: {result.precision:.2%}, "
                  f"Recall: {result.recall:.2%}")

    # Print results
    if results:
        print_results_table(results, baseline_f1=args.baseline_f1)

        # Save results if output specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            output_data = {
                "configs_tested": len(configs),
                "successful_runs": len(results),
                "results": [r.to_dict() for r in results],
                "best_config": {
                    "train_weeks": results[0].config.train_weeks,
                    "val_weeks": results[0].config.val_weeks,
                    "threshold_weeks": results[0].config.threshold_weeks,
                    "f1_score": results[0].f1_score
                }
            }

            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to {output_path}")
    else:
        print("\nNo successful experiments!")

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
