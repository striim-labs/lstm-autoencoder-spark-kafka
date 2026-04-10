"""
LSTM-AE Scoring API -- FastAPI service for Striim Open Processor integration.

Loads the pre-trained LSTM Encoder-Decoder model, scaler, and scorer at startup.
Exposes a /v1/score endpoint that accepts a 336-point window of taxi demand counts
and returns an anomaly decision with optional localization.

Run:
    cd striim/api && uvicorn main:app --port 8000
"""

import logging
import os
import pickle
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, field_validator

# Add project root to path so we can import src.model and src.scorer
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import EncDecAD, ModelConfig
from src.scorer import AnomalyScorer

# Register old module paths so pickled artifacts can be loaded
import src.model, src.scorer
sys.modules["lstm_autoencoder"] = src.model
sys.modules["anomaly_scorer"] = src.scorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", str(PROJECT_ROOT / "models" / "lstm_model.pt"))
SCALER_PATH = os.environ.get("SCALER_PATH", str(PROJECT_ROOT / "models" / "scaler.pkl"))
SCORER_PATH = os.environ.get("SCORER_PATH", str(PROJECT_ROOT / "models" / "scorer.pkl"))
WINDOW_SIZE = 336

# ---------------------------------------------------------------------------
# Global model state (populated at startup)
# ---------------------------------------------------------------------------
model: Optional[EncDecAD] = None
scaler = None
scorer: Optional[AnomalyScorer] = None
device: Optional[torch.device] = None
startup_time: Optional[float] = None


def load_model_artifacts():
    """Load pre-trained model, scaler, and scorer from disk."""
    global model, scaler, scorer, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model from %s", MODEL_PATH)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = EncDecAD(config=checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info("Loading scaler from %s", SCALER_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    logger.info("Loading scorer from %s", SCORER_PATH)
    scorer = AnomalyScorer.load(SCORER_PATH)

    logger.info("All artifacts loaded (device=%s, threshold=%.4f)", device, scorer.threshold)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ScoreRequest(BaseModel):
    """Scoring request: a 336-point window of taxi demand counts."""
    values: List[float] = Field(
        ...,
        description="336 taxi demand counts (30-min intervals, one week)",
    )
    window_start: Optional[str] = Field(
        default=None,
        description="Timestamp of the first data point (for logging/response)",
    )
    window_end: Optional[str] = Field(
        default=None,
        description="Timestamp of the last data point (for logging/response)",
    )

    @field_validator("values")
    @classmethod
    def validate_values_length(cls, v: List[float]) -> List[float]:
        if len(v) != WINDOW_SIZE:
            raise ValueError(
                f"values must contain exactly {WINDOW_SIZE} elements, got {len(v)}"
            )
        return v


class LocalizationInfo(BaseModel):
    """Sub-window localization for detected anomalies."""
    anomaly_start: str
    anomaly_end: str
    scale_hours: float
    contrast_ratio: float


class ScoreResponse(BaseModel):
    """Response from a scoring request."""
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    localization: Optional[LocalizationInfo] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    window_size: int
    threshold: Optional[float] = None
    uptime_seconds: Optional[float] = None


# ---------------------------------------------------------------------------
# Scoring logic (mirrors detect_anomalies from code/3_streaming_app.py)
# ---------------------------------------------------------------------------
def score_window(values: List[float], window_start: Optional[str], window_end: Optional[str]) -> ScoreResponse:
    """Score a single 336-point window for anomalies."""
    values_array = np.array(values, dtype=float).reshape(-1, 1)
    values_normalized = scaler.transform(values_array).flatten()

    # Forward pass
    x = torch.FloatTensor(values_normalized).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        x_reconstructed = model(x)

    error = torch.abs(x - x_reconstructed).cpu().numpy().squeeze()

    # Check scoring mode
    use_point_level = scorer.mu_point is not None and scorer.point_threshold is not None

    if use_point_level:
        point_scores = ((error - scorer.mu_point[0]) ** 2) / scorer.sigma_point[0]
        k = scorer.config.hard_criterion_k
        is_anomaly = bool(np.sum(point_scores > scorer.point_threshold) >= k)
        window_score = float(np.max(point_scores))
    else:
        window_score = float(scorer._mahalanobis_distance(error))
        is_anomaly = window_score > scorer.threshold
        point_scores = error

    logger.info(
        "score_request score=%.4f is_anomaly=%s window=[%s, %s]",
        window_score, is_anomaly, window_start, window_end,
    )

    localization = None
    if is_anomaly:
        # Build timestamp list for localization
        # If window_start is provided, generate 30-min interval timestamps
        # Otherwise use index-based placeholders
        if window_start:
            import pandas as pd
            ts_start = pd.Timestamp(window_start)
            timestamps = [str(ts_start + pd.Timedelta(minutes=30 * i)) for i in range(WINDOW_SIZE)]
        else:
            timestamps = [str(i) for i in range(WINDOW_SIZE)]

        loc = scorer.localize_anomaly(error, timestamps)
        localization = LocalizationInfo(
            anomaly_start=loc["anomaly_start"],
            anomaly_end=loc["anomaly_end"],
            scale_hours=loc["scale_hours"],
            contrast_ratio=loc["contrast_ratio"],
        )
        logger.info(
            "  Localized to %dh: %s - %s (contrast=%.2f)",
            loc["scale_hours"], loc["anomaly_start"], loc["anomaly_end"], loc["contrast_ratio"],
        )

    return ScoreResponse(
        is_anomaly=is_anomaly,
        anomaly_score=window_score,
        threshold=float(scorer.threshold),
        window_start=window_start,
        window_end=window_end,
        localization=localization,
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global startup_time
    load_model_artifacts()
    startup_time = time.time()
    yield
    logger.info("Shutting down LSTM-AE Scoring API")


app = FastAPI(
    title="LSTM-AE Anomaly Detection API",
    version="1.0.0",
    description="Anomaly scoring on NYC taxi demand time series using an LSTM Encoder-Decoder.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    uptime = time.time() - startup_time if startup_time else None
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model="lstm-ae",
        window_size=WINDOW_SIZE,
        threshold=float(scorer.threshold) if scorer and scorer.threshold else None,
        uptime_seconds=round(uptime, 1) if uptime else None,
    )


@app.post("/v1/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    """Score a 336-point taxi demand window for anomalies.

    When weekly_only=true (default), only scores windows whose start
    timestamp falls on a Sunday at midnight. All other windows return
    a skip response (is_anomaly=false, anomaly_score=0). This produces
    one score per non-overlapping week, matching the model's training
    paradigm when used with a sliding window upstream.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Deduplicate sliding window output: only score Sunday-aligned weeks
    if req.window_start:
        import pandas as pd
        try:
            ts = pd.Timestamp(req.window_start)
            if ts.day_name() != "Sunday" or ts.hour != 0 or ts.minute != 0:
                return Response(status_code=204)
            # Skip training/validation era (model trained on data before Sep 2014)
            if ts < pd.Timestamp("2014-08-31"):
                return Response(status_code=204)
        except Exception:
            pass  # If timestamp can't be parsed, score anyway

    start = time.perf_counter()
    result = score_window(req.values, req.window_start, req.window_end)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info("  latency=%.1fms", latency_ms)
    return result
