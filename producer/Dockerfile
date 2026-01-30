FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for confluent-kafka and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    librdkafka-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy pyproject.toml and install dependencies
COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml

# Copy producer code
COPY producer.py .

# Run the producer
CMD ["python", "-u", "producer.py"]
