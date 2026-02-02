"""
Hyperparameter Optimizer for LSTM Encoder-Decoder

Tests different LSTM configurations (hidden_dim, num_layers, dropout, etc.)
using the optimal data split configuration.

Based on Malhotra et al. (2016):
- Paper uses c=40 hidden units for power demand
- Hyperparameters are chosen to maximize Fβ on validation set
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import random
import time

import numpy as np
import torch

from data_preprocessor import NYCTaxiPreprocessor, PreprocessorConfig
from lstm_autoencoder import create_model
from anomaly_scorer import AnomalyScorer, ScorerConfig
from train import train_model, TrainingConfig

logger = logging.getLogger(__name__)


def setup_logging(log_file: Optional[str] = None, verbose: bool = False) -> None:
    """Setup logging to both console and file."""
    log_level = logging.INFO if verbose else logging.WARNING

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (always INFO level for detailed logs)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")


def save_incremental_results(
    results: List["ExperimentResult"],
    output_path: Path,
    total_configs: int,
    sort_by: str,
    beta: float
) -> None:
    """Save results incrementally after each experiment."""
    sorted_results = sorted(results, key=lambda r: getattr(r, sort_by), reverse=True)
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "configs_tested": total_configs,
        "completed_runs": len(results),
        "progress_pct": round(100 * len(results) / total_configs, 1),
        "sort_by": sort_by,
        "beta": beta,
        "results": [r.to_dict() for r in sorted_results],
        "best_config": sorted_results[0].to_dict() if sorted_results else None
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


@dataclass
class HyperparamConfig:
    """Configuration for a hyperparameter experiment."""
    hidden_dim: int
    num_layers: int
    dropout: float
    learning_rate: float
    threshold_percentile: float
    batch_size: int = 4

    def to_dict(self) -> Dict:
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "threshold_percentile": self.threshold_percentile,
            "batch_size": self.batch_size,
        }


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    config: HyperparamConfig
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    f_beta_score: float  # Fβ with β=0.1 as per paper
    train_loss: float
    val_loss: float
    threshold: float
    best_epoch: int
    total_params: int

    def to_dict(self) -> Dict:
        return {
            **self.config.to_dict(),
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "f_beta_score": self.f_beta_score,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "threshold": self.threshold,
            "best_epoch": self.best_epoch,
            "total_params": self.total_params,
        }


def compute_f_beta(precision: float, recall: float, beta: float = 0.1) -> float:
    """
    Compute Fβ score.

    β < 1 weights precision higher (paper uses β=0.1 for power demand).
    """
    if precision + recall == 0:
        return 0
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def compute_metrics(predictions: np.ndarray, actuals: np.ndarray, beta: float = 0.1) -> Dict[str, float]:
    """Compute classification metrics including Fβ."""
    tp = np.sum(predictions & actuals)
    fp = np.sum(predictions & ~actuals)
    fn = np.sum(~predictions & actuals)
    tn = np.sum(~predictions & ~actuals)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f_beta = compute_f_beta(precision, recall, beta)
    accuracy = (tp + tn) / len(predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "f_beta_score": f_beta,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(
    hyperparam_config: HyperparamConfig,
    data_path: str,
    device: torch.device,
    train_weeks: int = 8,
    val_weeks: int = 2,
    threshold_weeks: int = 4,
    epochs: int = 100,
    patience: int = 10,
    n_trials: int = 1,
    beta: float = 0.1,
    verbose: bool = False
) -> Optional[ExperimentResult]:
    """
    Run experiment(s) with the given hyperparameter configuration.

    Args:
        hyperparam_config: Hyperparameter configuration
        data_path: Path to data file
        device: Compute device
        train_weeks: Training weeks
        val_weeks: Validation weeks
        threshold_weeks: Threshold calibration weeks
        epochs: Max training epochs
        patience: Early stopping patience
        n_trials: Number of trials to average (for stability)
        beta: Beta value for Fβ score
        verbose: Enable verbose output

    Returns:
        ExperimentResult with averaged metrics or None if failed
    """
    all_metrics = []
    all_train_losses = []
    all_val_losses = []
    all_thresholds = []
    all_best_epochs = []
    total_params = 0

    try:
        # Setup preprocessor
        preprocess_config = PreprocessorConfig(
            train_weeks=train_weeks,
            val_weeks=val_weeks,
            threshold_weeks=threshold_weeks
        )

        for trial in range(n_trials):
            set_seed(42 + trial)  # Different seed each trial

            preprocessor = NYCTaxiPreprocessor(config=preprocess_config)
            dataloaders, _ = preprocessor.preprocess(
                data_path,
                batch_size=hyperparam_config.batch_size
            )

            # Check data availability
            for split in ["train", "val", "threshold_val", "test"]:
                if dataloaders[split] is None or len(dataloaders[split].dataset) == 0:
                    logger.warning(f"Empty {split} set, skipping")
                    return None

            # Create model
            model = create_model(
                hidden_dim=hyperparam_config.hidden_dim,
                num_layers=hyperparam_config.num_layers,
                dropout=hyperparam_config.dropout
            )
            model.to(device)
            total_params = sum(p.numel() for p in model.parameters())

            # Train
            training_config = TrainingConfig(
                epochs=epochs,
                learning_rate=hyperparam_config.learning_rate,
                patience=patience
            )

            if not verbose:
                logging.getLogger().setLevel(logging.WARNING)

            model, history = train_model(
                model=model,
                train_loader=dataloaders["train"],
                val_loader=dataloaders["val"],
                device=device,
                config=training_config
            )

            # Fit scorer
            scorer = AnomalyScorer(
                config=ScorerConfig(
                    threshold_percentile=hyperparam_config.threshold_percentile
                )
            )
            scorer.fit(model, dataloaders["val"], device)

            # Set threshold
            val_scores, _ = scorer.compute_scores(model, dataloaders["threshold_val"], device)
            scorer.set_threshold(val_scores)

            # Evaluate
            test_scores, _ = scorer.compute_scores(model, dataloaders["test"], device)
            predictions = scorer.predict(test_scores)

            test_week_info = preprocessor.get_test_week_info()
            actuals = np.array([w["is_anomaly"] for w in test_week_info])

            metrics = compute_metrics(predictions, actuals, beta)
            all_metrics.append(metrics)
            all_train_losses.append(history["train_loss"][-1])
            all_val_losses.append(history["val_loss"][-1] if history["val_loss"] else float('inf'))
            all_thresholds.append(scorer.threshold)
            all_best_epochs.append(history.get("best_epoch", len(history["train_loss"])))

        # Average results
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in ["accuracy", "precision", "recall", "f1_score", "f_beta_score"]
        }

        return ExperimentResult(
            config=hyperparam_config,
            accuracy=avg_metrics["accuracy"],
            precision=avg_metrics["precision"],
            recall=avg_metrics["recall"],
            f1_score=avg_metrics["f1_score"],
            f_beta_score=avg_metrics["f_beta_score"],
            train_loss=np.mean(all_train_losses),
            val_loss=np.mean(all_val_losses),
            threshold=np.mean(all_thresholds),
            best_epoch=int(np.mean(all_best_epochs)),
            total_params=total_params,
        )

    except Exception as e:
        logger.error(f"Experiment failed for {hyperparam_config}: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_grid_configs(
    hidden_dims: List[int] = [32, 40, 64, 128],
    num_layers_list: List[int] = [1, 2],
    dropouts: List[float] = [0.0, 0.1, 0.2, 0.3],
    learning_rates: List[float] = [1e-4, 5e-4, 1e-3],
    threshold_percentiles: List[float] = [90.0, 95.0, 99.0]
) -> List[HyperparamConfig]:
    """Generate grid of hyperparameter configurations."""
    configs = []
    for hidden_dim, num_layers, dropout, lr, thresh_pct in product(
        hidden_dims, num_layers_list, dropouts, learning_rates, threshold_percentiles
    ):
        configs.append(HyperparamConfig(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=lr,
            threshold_percentile=thresh_pct,
        ))
    return configs


def generate_random_configs(
    n_configs: int = 50,
    hidden_dim_range: Tuple[int, int] = (16, 128),
    num_layers_range: Tuple[int, int] = (1, 3),
    dropout_range: Tuple[float, float] = (0.0, 0.4),
    lr_range: Tuple[float, float] = (1e-4, 1e-2),
    threshold_pct_range: Tuple[float, float] = (85.0, 99.0)
) -> List[HyperparamConfig]:
    """Generate random hyperparameter configurations."""
    configs = []
    for _ in range(n_configs):
        configs.append(HyperparamConfig(
            hidden_dim=random.randint(hidden_dim_range[0], hidden_dim_range[1]),
            num_layers=random.randint(num_layers_range[0], num_layers_range[1]),
            dropout=round(random.uniform(dropout_range[0], dropout_range[1]), 2),
            learning_rate=10 ** random.uniform(np.log10(lr_range[0]), np.log10(lr_range[1])),
            threshold_percentile=round(random.uniform(threshold_pct_range[0], threshold_pct_range[1]), 1),
        ))
    return configs


def print_results_table(results: List[ExperimentResult], sort_by: str = "f1_score") -> None:
    """Print results in a formatted table."""
    print("\n" + "=" * 120)
    print("HYPERPARAMETER OPTIMIZATION RESULTS")
    print("=" * 120)

    # Sort by specified metric descending
    sorted_results = sorted(results, key=lambda r: getattr(r, sort_by), reverse=True)

    print(f"\n{'Hidden':>7} {'Layers':>7} {'Drop':>6} {'LR':>10} {'Thresh%':>8} "
          f"{'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Fβ':>7} {'Params':>8}")
    print("-" * 120)

    for r in sorted_results[:20]:  # Top 20
        print(f"{r.config.hidden_dim:>7} {r.config.num_layers:>7} {r.config.dropout:>6.2f} "
              f"{r.config.learning_rate:>10.6f} {r.config.threshold_percentile:>8.1f} "
              f"{r.accuracy:>7.2%} {r.precision:>7.2%} {r.recall:>7.2%} "
              f"{r.f1_score:>7.2%} {r.f_beta_score:>7.2%} {r.total_params:>8}")

    # Best result
    best = sorted_results[0]
    print("\n" + "-" * 120)
    print(f"BEST CONFIG (by {sort_by}):")
    print(f"  hidden_dim={best.config.hidden_dim}, num_layers={best.config.num_layers}, "
          f"dropout={best.config.dropout}, lr={best.config.learning_rate:.6f}, "
          f"threshold_pct={best.config.threshold_percentile}")
    print(f"  F1: {best.f1_score:.2%}, Fβ: {best.f_beta_score:.2%}, "
          f"Precision: {best.precision:.2%}, Recall: {best.recall:.2%}")
    print("=" * 120)


def main():
    """Main hyperparameter optimization script."""
    parser = argparse.ArgumentParser(
        description="Optimize LSTM hyperparameters for anomaly detection"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/nyc_taxi.csv",
        help="Path to NYC taxi CSV file"
    )
    parser.add_argument(
        "--train-weeks",
        type=int,
        default=8,
        help="Training weeks (from split optimization)"
    )
    parser.add_argument(
        "--val-weeks",
        type=int,
        default=2,
        help="Validation weeks"
    )
    parser.add_argument(
        "--threshold-weeks",
        type=int,
        default=4,
        help="Threshold calibration weeks"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["grid", "random", "focused"],
        default="focused",
        help="Search mode: grid (exhaustive), random (sampled), focused (targeted)"
    )
    parser.add_argument(
        "--n-configs",
        type=int,
        default=30,
        help="Number of configurations to try (for random mode)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=1,
        help="Number of trials per config for stability"
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
        "--beta",
        type=float,
        default=0.1,
        help="Beta value for Fβ score (paper uses 0.1)"
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        choices=["f1_score", "f_beta_score", "precision", "recall", "accuracy"],
        default="f1_score",
        help="Metric to sort results by"
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
        help="Enable verbose output"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (logs to file in addition to console)"
    )
    args = parser.parse_args()

    # Setup logging with file output
    log_file = args.log_file
    if log_file is None and args.output:
        # Default log file next to output file
        log_file = str(Path(args.output).with_suffix('.log'))

    setup_logging(log_file=log_file, verbose=args.verbose)

    logger.info("=" * 60)
    logger.info("HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 60)
    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Data split: train={args.train_weeks}, val={args.val_weeks}, threshold={args.threshold_weeks}")
    print(f"Mode: {args.mode}")

    # Generate configurations
    if args.mode == "grid":
        # Full grid (288 combinations)
        configs = generate_grid_configs(
            hidden_dims=[32, 40, 64, 128],
            num_layers_list=[1, 2],
            dropouts=[0.0, 0.1, 0.2, 0.3],
            learning_rates=[1e-4, 5e-4, 1e-3],
            threshold_percentiles=[90.0, 95.0, 99.0]
        )
    elif args.mode == "random":
        configs = generate_random_configs(n_configs=args.n_configs)
    else:  # focused
        # Focused search around paper's recommendations and current config
        configs = [
            # Paper's recommendation: c=40
            HyperparamConfig(hidden_dim=40, num_layers=1, dropout=0.0, learning_rate=1e-3, threshold_percentile=95.0),
            HyperparamConfig(hidden_dim=40, num_layers=1, dropout=0.1, learning_rate=1e-3, threshold_percentile=95.0),
            HyperparamConfig(hidden_dim=40, num_layers=1, dropout=0.2, learning_rate=1e-3, threshold_percentile=95.0),
            HyperparamConfig(hidden_dim=40, num_layers=2, dropout=0.1, learning_rate=1e-3, threshold_percentile=95.0),

            # Current config variations
            HyperparamConfig(hidden_dim=64, num_layers=1, dropout=0.0, learning_rate=1e-3, threshold_percentile=95.0),
            HyperparamConfig(hidden_dim=64, num_layers=1, dropout=0.1, learning_rate=1e-3, threshold_percentile=95.0),
            HyperparamConfig(hidden_dim=64, num_layers=1, dropout=0.2, learning_rate=1e-3, threshold_percentile=95.0),
            HyperparamConfig(hidden_dim=64, num_layers=2, dropout=0.1, learning_rate=1e-3, threshold_percentile=95.0),
            HyperparamConfig(hidden_dim=64, num_layers=2, dropout=0.2, learning_rate=1e-3, threshold_percentile=95.0),

            # Smaller models
            HyperparamConfig(hidden_dim=32, num_layers=1, dropout=0.1, learning_rate=1e-3, threshold_percentile=95.0),
            HyperparamConfig(hidden_dim=32, num_layers=2, dropout=0.1, learning_rate=1e-3, threshold_percentile=95.0),

            # Larger models
            HyperparamConfig(hidden_dim=128, num_layers=1, dropout=0.2, learning_rate=1e-3, threshold_percentile=95.0),
            HyperparamConfig(hidden_dim=128, num_layers=1, dropout=0.3, learning_rate=1e-3, threshold_percentile=95.0),

            # Different learning rates
            HyperparamConfig(hidden_dim=64, num_layers=1, dropout=0.2, learning_rate=5e-4, threshold_percentile=95.0),
            HyperparamConfig(hidden_dim=64, num_layers=1, dropout=0.2, learning_rate=2e-3, threshold_percentile=95.0),

            # Different thresholds
            HyperparamConfig(hidden_dim=64, num_layers=1, dropout=0.2, learning_rate=1e-3, threshold_percentile=90.0),
            HyperparamConfig(hidden_dim=64, num_layers=1, dropout=0.2, learning_rate=1e-3, threshold_percentile=99.0),

            # Combinations
            HyperparamConfig(hidden_dim=40, num_layers=1, dropout=0.1, learning_rate=5e-4, threshold_percentile=90.0),
            HyperparamConfig(hidden_dim=40, num_layers=1, dropout=0.2, learning_rate=1e-3, threshold_percentile=90.0),
            HyperparamConfig(hidden_dim=64, num_layers=1, dropout=0.1, learning_rate=5e-4, threshold_percentile=90.0),
        ]

    print(f"Testing {len(configs)} configurations...")
    logger.info(f"Testing {len(configs)} configurations...")

    # Prepare output path for incremental saving
    output_path = Path(args.output) if args.output else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run experiments
    results = []
    start_time = time.time()

    for i, config in enumerate(configs):
        elapsed = time.time() - start_time
        elapsed_str = f"{elapsed/60:.1f}min" if elapsed > 60 else f"{elapsed:.0f}s"

        progress_msg = (
            f"[{i+1}/{len(configs)}] ({elapsed_str}) Testing: hidden={config.hidden_dim}, "
            f"layers={config.num_layers}, dropout={config.dropout}, "
            f"lr={config.learning_rate:.6f}, thresh={config.threshold_percentile}%"
        )
        print(f"\n{progress_msg}...")
        logger.info(progress_msg)

        exp_start = time.time()
        result = run_experiment(
            hyperparam_config=config,
            data_path=args.data_path,
            device=device,
            train_weeks=args.train_weeks,
            val_weeks=args.val_weeks,
            threshold_weeks=args.threshold_weeks,
            epochs=args.epochs,
            patience=args.patience,
            n_trials=args.n_trials,
            beta=args.beta,
            verbose=args.verbose
        )
        exp_elapsed = time.time() - exp_start

        if result is not None:
            results.append(result)
            result_msg = (
                f"    -> F1: {result.f1_score:.2%}, Fβ: {result.f_beta_score:.2%}, "
                f"Prec: {result.precision:.2%}, Rec: {result.recall:.2%} ({exp_elapsed:.1f}s)"
            )
            print(result_msg)
            logger.info(result_msg)

            # Save incrementally
            if output_path:
                save_incremental_results(
                    results, output_path, len(configs), args.sort_by, args.beta
                )
                logger.info(f"Saved incremental results to {output_path}")
        else:
            logger.warning(f"Experiment {i+1} failed")

    # Print results
    total_elapsed = time.time() - start_time
    total_elapsed_str = f"{total_elapsed/60:.1f} minutes" if total_elapsed > 60 else f"{total_elapsed:.0f} seconds"

    if results:
        print_results_table(results, sort_by=args.sort_by)

        # Final save (incremental saves already done, but ensure final state)
        if output_path:
            save_incremental_results(
                results, output_path, len(configs), args.sort_by, args.beta
            )
            print(f"\nFinal results saved to {output_path}")
            logger.info(f"Final results saved to {output_path}")
    else:
        print("\nNo successful experiments!")
        logger.warning("No successful experiments!")

    completion_msg = f"OPTIMIZATION COMPLETE - {len(results)}/{len(configs)} successful in {total_elapsed_str}"
    print("\n" + "=" * 60)
    print(completion_msg)
    print("=" * 60)
    logger.info("=" * 60)
    logger.info(completion_msg)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
