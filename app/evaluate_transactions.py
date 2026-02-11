"""
Evaluation and Visualization for Transaction Frequency Anomaly Detection

Provides:
- Reconstruction visualizations for 24-hour test windows
- Normal sequence reconstruction grids
- Score distribution plots per combo
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from app.transaction_config import TransactionPreprocessorConfig, COMBO_KEYS
from app.transaction_preprocessor import TransactionPreprocessor
from app.model_registry import ModelRegistry, TransactionModelConfig, combo_to_dirname
from app.anomaly_scorer import ScorerConfig

logger = logging.getLogger(__name__)


def plot_daily_reconstruction(
    model: torch.nn.Module,
    sequence: np.ndarray,
    day_info: Dict,
    window_score: float,
    prediction: bool,
    device: torch.device,
    save_path: str
) -> None:
    """
    Plot original, reconstructed, and error for a 24-hour window.

    Args:
        model: Trained LSTM-AE model
        sequence: Input sequence, shape (24,) or (24, 1)
        day_info: Dict with day metadata (day_index, date, day_name)
        window_score: Window-level anomaly score
        prediction: True if predicted as anomaly
        device: Torch device
        save_path: Path to save figure
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plot")
        return

    model.eval()

    # Prepare input
    if sequence.ndim == 1:
        sequence = sequence.reshape(-1, 1)

    x = torch.FloatTensor(sequence).unsqueeze(0).to(device)

    with torch.no_grad():
        x_reconstructed = model(x)

    original = sequence.squeeze()
    reconstructed = x_reconstructed.cpu().numpy().squeeze()
    error = np.abs(original - reconstructed)

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    hours = np.arange(len(original))

    # Original
    axes[0].plot(hours, original, label="Original", color="blue", linewidth=1.5)
    axes[0].set_ylabel("Normalized Value")
    axes[0].set_title(
        f"Day {day_info['day_index']} - {day_info['day_name']} ({day_info['date']}) - Original"
    )
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 23)

    # Reconstructed
    axes[1].plot(hours, reconstructed, label="Reconstructed", color="orange", linewidth=1.5)
    axes[1].set_ylabel("Normalized Value")
    axes[1].set_title("Reconstructed Sequence")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 23)

    # Error
    axes[2].fill_between(hours, 0, error, alpha=0.5, color="red", label="Error")
    axes[2].plot(hours, error, color="red", linewidth=0.5)
    axes[2].set_xlabel("Hour of Day")
    axes[2].set_ylabel("Absolute Error")

    pred_str = "ANOMALY" if prediction else "Normal"
    axes[2].set_title(
        f"Reconstruction Error (Mean: {error.mean():.4f}) | "
        f"Score: {window_score:.4f} | Prediction: {pred_str}"
    )
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, 23)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"Saved reconstruction plot to {save_path}")


def plot_normal_reconstructions(
    model: torch.nn.Module,
    sequences: np.ndarray,
    window_info: List[Dict],
    predictions: np.ndarray,
    device: torch.device,
    save_path: str
) -> None:
    """
    Grid of correctly predicted normal sequences.

    Args:
        model: Trained LSTM-AE model
        sequences: All test sequences, shape (N, 24) or (N, 24, 1)
        window_info: List of window metadata dicts
        predictions: Boolean array of predictions (True=anomaly)
        device: Torch device
        save_path: Path to save figure
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plot")
        return

    model.eval()

    # Find correctly predicted normal windows
    normal_indices = [i for i, pred in enumerate(predictions) if not pred]

    if len(normal_indices) == 0:
        logger.warning("No normal predictions found for grid plot")
        return

    # Limit to 4 windows
    n_windows = min(len(normal_indices), 4)
    selected_indices = normal_indices[:n_windows]

    fig, axes = plt.subplots(n_windows, 3, figsize=(16, 4 * n_windows))

    if n_windows == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(selected_indices):
        seq = sequences[idx]
        info = window_info[idx]

        if seq.ndim == 1:
            seq = seq.reshape(-1, 1)

        x = torch.FloatTensor(seq).unsqueeze(0).to(device)

        with torch.no_grad():
            x_reconstructed = model(x)

        original = seq.squeeze()
        reconstructed = x_reconstructed.cpu().numpy().squeeze()
        error = np.abs(original - reconstructed)

        hours = np.arange(len(original))

        # Original
        axes[row, 0].plot(hours, original, color="blue", linewidth=1)
        axes[row, 0].set_title(f"Day {info['day_index']} ({info['day_name']}) - Original")
        axes[row, 0].set_ylabel("Normalized")
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 0].set_xlim(0, 23)

        # Reconstructed
        axes[row, 1].plot(hours, reconstructed, color="orange", linewidth=1)
        axes[row, 1].set_title("Reconstructed")
        axes[row, 1].grid(True, alpha=0.3)
        axes[row, 1].set_xlim(0, 23)

        # Error
        axes[row, 2].fill_between(hours, 0, error, alpha=0.5, color="red")
        axes[row, 2].set_title(f"Error (Mean: {error.mean():.4f})")
        axes[row, 2].grid(True, alpha=0.3)
        axes[row, 2].set_xlim(0, 23)

        if row == n_windows - 1:
            for col in range(3):
                axes[row, col].set_xlabel("Hour")

    plt.suptitle("Correctly Predicted Normal Sequences", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"Saved normal reconstructions to {save_path}")


def plot_combo_score_distribution(
    window_scores: np.ndarray,
    predictions: np.ndarray,
    day_info: List[Dict],
    threshold: float,
    combo: Tuple[str, str],
    save_path: str
) -> None:
    """
    Score distribution with histogram and bar chart.

    Args:
        window_scores: Array of window scores
        predictions: Boolean array of predictions (True=anomaly)
        day_info: List of window metadata dicts
        threshold: Window threshold
        combo: (network_type, transaction_type) tuple
        save_path: Path to save figure
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Histogram of scores
    ax = axes[0]

    normal_scores = window_scores[~predictions]
    anomaly_scores = window_scores[predictions]

    bins = np.linspace(window_scores.min(), window_scores.max(), 15)

    if len(normal_scores) > 0:
        ax.hist(normal_scores, bins=bins, alpha=0.6, label="Predicted Normal", color="blue")
    if len(anomaly_scores) > 0:
        ax.hist(anomaly_scores, bins=bins, alpha=0.6, label="Predicted Anomaly", color="red")

    ax.axvline(x=threshold, color="black", linestyle="--", linewidth=2,
               label=f"Threshold: {threshold:.4f}")

    ax.set_xlabel("Window Score")
    ax.set_ylabel("Count")
    ax.set_title(f"{combo[0]}/{combo[1]} - Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Bar chart of scores by day
    ax = axes[1]

    x = np.arange(len(window_scores))
    colors = ["red" if p else "blue" for p in predictions]

    ax.bar(x, window_scores, color=colors, alpha=0.7)
    ax.axhline(y=threshold, color="black", linestyle="--", linewidth=2,
               label=f"Threshold: {threshold:.4f}")

    # X-axis labels with day info
    labels = [f"D{info['day_index']}\n{info['day_name'][:3]}" for info in day_info]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)

    ax.set_xlabel("Test Window")
    ax.set_ylabel("Window Score")
    ax.set_title("Scores by Test Day (Blue=Normal, Red=Anomaly)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"Saved score distribution to {save_path}")


def evaluate_and_plot(
    registry: ModelRegistry,
    preprocessor: TransactionPreprocessor,
    combo_dataloaders: Dict[Tuple[str, str], Dict[str, torch.utils.data.DataLoader]],
    output_dir: str
) -> Dict[Tuple[str, str], Dict]:
    """
    Main evaluation function that generates all plots.

    Args:
        registry: Trained ModelRegistry with all combo models
        preprocessor: TransactionPreprocessor with window info
        combo_dataloaders: Dict of DataLoaders per combo
        output_dir: Base output directory

    Returns:
        Dict of evaluation results per combo
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for combo in COMBO_KEYS:
        combo_dirname = combo_to_dirname(combo)
        combo_dir = output_dir / combo_dirname
        combo_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Evaluating {combo[0]}/{combo[1]} ---")

        # Get model, scorer, and data
        model = registry.get_model(combo)
        scorer = registry.get_scorer(combo)
        test_loader = combo_dataloaders[combo].get("test")

        if test_loader is None:
            logger.warning(f"No test loader for {combo}, skipping")
            continue

        # Get predictions and scores
        predictions, window_scores, point_scores = registry.predict(combo, test_loader)

        # Get test window info
        test_info = preprocessor.get_test_window_info(combo)

        # Get test sequences (normalized)
        test_sequences = preprocessor.combo_splits[combo]["test"]

        # Print summary
        n_anomalies = predictions.sum()
        print(f"  Test windows: {len(predictions)}")
        print(f"  Predicted anomalies: {n_anomalies}")
        print(f"  Window threshold: {scorer.threshold:.4f}")

        # Store results
        results = {
            "predictions": predictions,
            "window_scores": window_scores,
            "point_scores": point_scores,
            "test_info": test_info,
        }
        all_results[combo] = results

        # 1. Plot individual daily reconstructions
        print(f"  Generating reconstruction plots...")
        for i, (seq, info, score, pred) in enumerate(
            zip(test_sequences, test_info, window_scores, predictions)
        ):
            save_path = combo_dir / f"day{info['day_index']}_reconstruction.png"
            plot_daily_reconstruction(
                model=model,
                sequence=seq,
                day_info=info,
                window_score=float(score),
                prediction=bool(pred),
                device=registry.device,
                save_path=str(save_path)
            )

        # 2. Plot normal reconstructions grid
        print(f"  Generating normal reconstructions grid...")
        plot_normal_reconstructions(
            model=model,
            sequences=test_sequences,
            window_info=test_info,
            predictions=predictions,
            device=registry.device,
            save_path=str(combo_dir / "normal_reconstructions.png")
        )

        # 3. Plot score distribution
        print(f"  Generating score distribution...")
        plot_combo_score_distribution(
            window_scores=window_scores,
            predictions=predictions,
            day_info=test_info,
            threshold=float(scorer.threshold),
            combo=combo,
            save_path=str(combo_dir / "score_distribution.png")
        )

        print(f"  Plots saved to {combo_dir}/")

    return all_results


def main():
    """Main evaluation script for transaction anomaly detection."""
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize transaction anomaly detection models"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/transactions",
        help="Directory containing trained model artifacts"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/synthetic_transactions.csv",
        help="Path to transaction CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluate/fiserv",
        help="Directory to save evaluation outputs"
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("\n" + "=" * 60)
    print("TRANSACTION ANOMALY DETECTION - EVALUATION")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Check matplotlib
    if not HAS_MATPLOTLIB:
        print("\nERROR: matplotlib not available. Please install with: pip install matplotlib")
        return

    # Load trained models
    print(f"\nLoading models from {args.model_dir}...")
    model_dir = Path(args.model_dir)

    if not model_dir.exists():
        print(f"\nERROR: Model directory not found: {model_dir}")
        print("Please run training first: python -m app.train_transactions")
        return

    registry = ModelRegistry(device=device)
    registry.load_all(args.model_dir)

    # Load and preprocess data
    print(f"\nLoading data from {args.data_path}...")
    preprocess_config = TransactionPreprocessorConfig()
    preprocessor = TransactionPreprocessor(config=preprocess_config)
    combo_dataloaders = preprocessor.preprocess(args.data_path, batch_size=1)

    # Run evaluation and generate plots
    print("\nGenerating evaluation plots...")
    results = evaluate_and_plot(
        registry=registry,
        preprocessor=preprocessor,
        combo_dataloaders=combo_dataloaders,
        output_dir=args.output_dir
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\n{'Combo':<20} {'Windows':>10} {'Anomalies':>12} {'Threshold':>12}")
    print("-" * 56)

    for combo in COMBO_KEYS:
        if combo in results:
            res = results[combo]
            n_windows = len(res["predictions"])
            n_anomalies = int(res["predictions"].sum())
            scorer = registry.get_scorer(combo)
            threshold = scorer.threshold

            combo_name = f"{combo[0]}/{combo[1]}"
            print(f"{combo_name:<20} {n_windows:>10} {n_anomalies:>12} {threshold:>12.4f}")

    print(f"\nPlots saved to: {args.output_dir}/")
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
