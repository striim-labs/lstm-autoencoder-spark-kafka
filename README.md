# LSTM Autoencoder Anomaly Detection

Real-time anomaly detection on NYC taxi demand data using an LSTM Encoder-Decoder model, Kafka streaming, Spark Structured Streaming, and a Dash visualization dashboard.

Based on [Malhotra et al. (2016)](https://arxiv.org/abs/1607.00148): "LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection"

---

## Project Structure

```
lstm-autoencoder-spark-kafka/
│
├── code/                                    # Numbered workflow (start here)
│   ├── 0_verify_setup.py                    # Optional environment / artifact check
│   ├── 1_data_exploration.ipynb             # EDA: patterns, seasonality, motivation
│   ├── 2_model_design.ipynb                 # Architecture walkthrough, train demo
│   ├── 3_train_model.py                     # Full training pipeline
│   ├── 4_evaluate_model.py                  # Evaluate model, generate plots
│   ├── 5_streaming_app.py                   # Real-time Dash + Spark + Kafka app
│   └── 6_optimize.py                        # Hyperparameter + split optimization
│
├── src/                                     # Reusable library code
│   ├── model.py                             # EncDecAD architecture (Malhotra et al.)
│   ├── scorer.py                            # Mahalanobis anomaly scoring
│   ├── preprocess.py                        # Data loading, segmentation, splits
│   └── synthetic.py                         # Synthetic anomaly generation
│
├── producer/                                # Kafka producer service
│   └── producer.py                          # Streams CSV data to Kafka
│
├── data/
│   └── nyc_taxi.csv                         # NYC taxi demand dataset (included)
│
├── models/                                  # Pre-trained model artifacts
│   ├── lstm_model.pt                        # LSTM Encoder-Decoder weights
│   ├── scaler.pkl                           # StandardScaler (train-fitted)
│   ├── scorer.pkl                           # Anomaly scorer with threshold
│   ├── training_history.pkl                 # Training loss curves
│   └── preprocessor_config.pkl              # Data split configuration
│
├── Dockerfile.app                           # App container definition
├── Dockerfile.producer                      # Producer container definition
├── docker-compose.yml                       # Full stack orchestration
├── pyproject.toml                           # Python dependencies
└── TECHNICAL.md                             # Detailed technical reference
```

## Prerequisites

- **Python 3.11+**
- **uv** (Python package manager) — install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Docker** and **Docker Compose** (for the streaming demo)

## Quick Start

### 1. Install dependencies

```bash
git clone <repo-url>
cd lstm-autoencoder-spark-kafka
uv sync
```

### 2. Open the notebooks

```bash
uv run jupyter notebook code/
```

Work through the notebooks in order:

| Notebook | What you'll learn |
|----------|-------------------|
| **`1_data_exploration.ipynb`** | Dataset overview, periodicity analysis, why simple thresholds fail, motivation for reconstruction-based detection |
| **`2_model_design.ipynb`** | LSTM Encoder-Decoder architecture, training demo, anomaly scoring methodology, localization |

Both notebooks run cell-by-cell with no external setup beyond `uv sync`. All cells are self-contained — just press Shift+Enter through them.

> **Note:** A pre-trained model is included in `models/`. The model design notebook trains a small demo model (10 epochs) for illustration, but the pre-trained model was trained with full hyperparameter optimization.
>
> `0_verify_setup.py` is optional troubleshooting. You do not need to run it before the notebooks.

### 3. Run the real-time streaming demo

```bash
MESSAGE_DELAY_SECONDS=0.005 START_OFFSET=4944 LOOP_DATA=false docker compose up --build -d
```

Open http://localhost:8050 to view the live dashboard.

On subsequent runs (after images are built):
```bash
MESSAGE_DELAY_SECONDS=0.005 START_OFFSET=4944 LOOP_DATA=false docker compose up -d
```

### View Logs

```bash
docker compose logs -f app
docker compose logs -f producer
```

---

## Workflow

The numbered files in `code/` tell the full story of the project:

| Step | File | Purpose |
|------|------|---------|
| 0 | `0_verify_setup.py` | Optional troubleshooting: verify environment, data, and model artifacts |
| 1 | `1_data_exploration.ipynb` | Explore data patterns, motivate the approach |
| 2 | `2_model_design.ipynb` | Walk through model architecture and scoring |
| 3 | `3_train_model.py` | Full training pipeline (optional — pre-trained model included) |
| 4 | `4_evaluate_model.py` | Evaluate model performance, generate plots |
| 5 | `5_streaming_app.py` | Real-time streaming application (Docker entrypoint) |
| 6 | `6_optimize.py` | Hyperparameter and split optimization (advanced) |

**Start with the notebooks** (steps 1-2) to understand the project. The Python scripts (steps 3-6) are for training, evaluation, and deployment. `0_verify_setup.py` is optional.

## Architecture

```
┌─────────────┐     ┌─────────┐     ┌───────────────┐     ┌──────────────┐
│  Producer   │────▶│  Kafka  │────▶│ Spark Stream  │────▶│     Dash     │
│ (NYC Taxi)  │     │         │     │  + Detector   │     │  Dashboard   │
└─────────────┘     └─────────┘     └───────────────┘     └──────────────┘
```

- **Producer**: Streams NYC taxi data CSV to Kafka topic
- **Kafka**: Message broker for real-time data streaming
- **Spark**: Structured Streaming consumes micro-batches from Kafka
- **LSTM Detector**: Pre-trained Encoder-Decoder flags anomalous weekly windows
- **Dash**: Real-time visualization with anomaly markers and 6-hour localization

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `START_OFFSET` | `0` | Record index to start streaming from (4944 for test data) |
| `LOOP_DATA` | `true` | Whether to loop through data continuously |
| `MESSAGE_DELAY_SECONDS` | `0.1` | Delay between messages (0.005 for fast demo) |

## Detection Performance

The LSTM Encoder-Decoder detects all 5 known anomalies in the NYC taxi dataset:

- **NYC Marathon** (Nov 1-3, 2014) — demand spike
- **Thanksgiving** (Nov 25-29, 2014) — demand drop
- **Christmas** (Dec 23-27, 2014) — demand drop
- **New Year's** (Dec 29 - Jan 3, 2015) — pattern disruption
- **January Blizzard** (Jan 24-29, 2015) — demand drop

**Precision: 100% | Recall: 100% | Inference: <5ms per weekly window**

## Services & Ports

| Service | Port | URL |
|---------|------|-----|
| Dash Dashboard | 8050 | http://localhost:8050 |
| Spark Master UI | 8080 | http://localhost:8080 |
| Kafka | 9092 | External access |
| Zookeeper | 2181 | Kafka coordination |
