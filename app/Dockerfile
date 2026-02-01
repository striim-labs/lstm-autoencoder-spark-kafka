FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (Java for Spark, librdkafka for confluent-kafka)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    librdkafka-dev \
    curl \
    openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME (detect architecture)
RUN JAVA_PATH=$(find /usr/lib/jvm -name "java-21-openjdk-*" -type d | head -1) && \
    echo "JAVA_HOME=$JAVA_PATH" >> /etc/environment
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy pyproject.toml and install dependencies
COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml

# Copy all application code
COPY main.py .
COPY base_detector.py .
COPY isolation_forest_detector.py .
COPY streaming_detector.py .
COPY lstm_autoencoder.py .
COPY anomaly_scorer.py .
COPY data_preprocessor.py .
COPY train.py .
COPY evaluate.py .

# Create models directory (will be mounted as volume)
RUN mkdir -p models

# Expose Dash port
EXPOSE 8050

# Default environment variables
ENV DETECTOR_TYPE=isolation_forest
ENV WINDOW_SIZE=200
ENV MIN_SAMPLES=50
ENV CONTAMINATION=0.05
ENV MODEL_PATH=models/lstm_model.pt
ENV SCALER_PATH=models/scaler.pkl
ENV SCORER_PATH=models/scorer.pkl

# Run the application
CMD ["python", "-u", "main.py"]
