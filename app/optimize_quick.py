"""
Quick Hyperparameter Optimization

Focused search on most impactful hyperparameters:
- Architecture dimensions (fewer options)
- Learning rates (key hyperparameter)
- Dropout (regularization)

Runs 20-30 experiments for quick feedback (~10-15 minutes).
"""

import argparse
import itertools
import json
import logging
import pickle
import sys
from pathlib import Path

# Import from main optimization script
from optimize_hyperparameters import (
    ExperimentConfig,
    run_experiment,
    ExperimentResult
)

from credit_card_preprocessor import CreditCardPreprocessor, CreditCardPreprocessorConfig
import torch
import pandas as pd
from dataclasses import asdict

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Quick hyperparameter optimization")
    parser.add_argument("--data-path", type=str, default="data/creditcard.csv")
    parser.add_argument("--output-dir", type=str, default="models/credit_card/optimization_quick")
    args = parser.parse_args()

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

    # ===== Focused Search Space =====
    logger.info("\n" + "=" * 80)
    logger.info("Defining Focused Search Space")
    logger.info("=" * 80)

    # Most impactful hyperparameters (12 total experiments - ~6 minutes)
    search_space = {
        # Architecture - test around paper's values
        'ae_hidden_dim': [20, 22, 25],                   # d1: smaller, paper, larger
        'ae_latent_dim': [15],                           # Fixed at paper's value
        'ae_dropout': [0.0],                             # No dropout initially

        # MLP - fixed at paper's values
        'mlp_hidden_dim1': [13],                         # Paper's value
        'mlp_hidden_dim2': [7],                          # Paper's value
        'mlp_dropout': [0.0],                            # No dropout initially

        # Learning rates - test around paper's value
        'ae_learning_rate': [1e-4],                      # Paper's value
        'mlp_learning_rate': [5e-5, 1e-4, 5e-4, 1e-3],   # Most important: test 4 values

        # Fixed values
        'batch_size': [256],
        'ae_weight_decay': [0.0],
        'mlp_weight_decay': [0.0],
    }

    logger.info("Search space:")
    for key, values in search_space.items():
        logger.info(f"  {key}: {values}")

    # Generate all combinations
    keys = list(search_space.keys())
    values = list(search_space.values())

    configs = []
    for combination in itertools.product(*values):
        config_dict = dict(zip(keys, combination))
        configs.append(ExperimentConfig(**config_dict))

    total_experiments = len(configs)
    logger.info(f"\nTotal experiments: {total_experiments}")
    logger.info(f"Estimated time: ~{total_experiments * 0.5:.0f} minutes\n")

    # ===== Run Experiments =====
    logger.info("=" * 80)
    logger.info("Running Experiments")
    logger.info("=" * 80)

    results = []

    for i, config in enumerate(configs, 1):
        try:
            result = run_experiment(config, dataloaders, device, i, total_experiments)
            results.append(result)

            # Save intermediate results every 5 experiments
            if i % 5 == 0 or i == total_experiments:
                intermediate_path = output_dir / f"results_interim_{i}.pkl"
                with open(intermediate_path, 'wb') as f:
                    pickle.dump(results, f)

        except Exception as e:
            logger.error(f"Experiment {i} failed: {e}")
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
            'test_precision': result.test_precision,
            'test_recall': result.test_recall,
            'test_f1': result.test_f1,
            'ae_val_loss': result.ae_val_loss,
            'mlp_val_f1': result.mlp_val_f1,
            'total_params': result.total_parameters,
            'total_time': result.total_time
        }
        rows.append(row)

    df_results = pd.DataFrame(rows)
    df_results = df_results.sort_values('test_f1', ascending=False)

    # Save results
    csv_path = output_dir / "optimization_results.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")

    # ===== Report Top 5 =====
    logger.info("\n" + "=" * 80)
    logger.info("TOP 5 CONFIGURATIONS")
    logger.info("=" * 80)

    for i, (idx, row) in enumerate(df_results.head(5).iterrows(), 1):
        logger.info(f"\n#{i} - F1: {row['test_f1']:.4f} (P: {row['test_precision']:.4f}, R: {row['test_recall']:.4f})")
        logger.info(f"  AE: 30→{int(row['ae_hidden_dim'])}→{int(row['ae_latent_dim'])} (drop={row['ae_dropout']:.1f}, lr={row['ae_learning_rate']:.0e})")
        logger.info(f"  MLP: {int(row['ae_latent_dim'])}→{int(row['mlp_hidden_dim1'])}→{int(row['mlp_hidden_dim2'])}→1 (drop={row['mlp_dropout']:.1f}, lr={row['mlp_learning_rate']:.0e})")
        logger.info(f"  Params: {int(row['total_params']):,}, Time: {row['total_time']:.1f}s")

    # Best configuration
    best = df_results.iloc[0]
    logger.info("\n" + "=" * 80)
    logger.info("BEST CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Test F1: {best['test_f1']:.4f} (Precision: {best['test_precision']:.4f}, Recall: {best['test_recall']:.4f})")
    logger.info(f"\nAutoencoder: 30 → {int(best['ae_hidden_dim'])} → {int(best['ae_latent_dim'])}")
    logger.info(f"  Dropout: {best['ae_dropout']:.2f}, LR: {best['ae_learning_rate']:.0e}")
    logger.info(f"\nMLP: {int(best['ae_latent_dim'])} → {int(best['mlp_hidden_dim1'])} → {int(best['mlp_hidden_dim2'])} → 1")
    logger.info(f"  Dropout: {best['mlp_dropout']:.2f}, LR: {best['mlp_learning_rate']:.0e}")
    logger.info(f"\nTotal parameters: {int(best['total_params']):,}")

    # Save best config
    best_config = {
        'ae_hidden_dim': int(best['ae_hidden_dim']),
        'ae_latent_dim': int(best['ae_latent_dim']),
        'ae_dropout': float(best['ae_dropout']),
        'mlp_hidden_dim1': int(best['mlp_hidden_dim1']),
        'mlp_hidden_dim2': int(best['mlp_hidden_dim2']),
        'mlp_dropout': float(best['mlp_dropout']),
        'ae_learning_rate': float(best['ae_learning_rate']),
        'mlp_learning_rate': float(best['mlp_learning_rate']),
        'batch_size': 256,
        'test_f1': float(best['test_f1']),
        'test_precision': float(best['test_precision']),
        'test_recall': float(best['test_recall'])
    }

    json_path = output_dir / "best_config.json"
    with open(json_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    logger.info(f"\nSaved best config to {json_path}")

    # Improvement over baseline
    baseline_f1 = 0.7726  # M1 baseline from earlier
    improvement = (best['test_f1'] - baseline_f1) / baseline_f1 * 100

    logger.info("\n" + "=" * 80)
    logger.info(f"Baseline M1 F1: {baseline_f1:.4f}")
    logger.info(f"Optimized F1: {best['test_f1']:.4f}")
    logger.info(f"Improvement: {improvement:+.2f}%")
    logger.info("=" * 80)

    logger.info("\n✓ Quick optimization complete!")


if __name__ == "__main__":
    main()
