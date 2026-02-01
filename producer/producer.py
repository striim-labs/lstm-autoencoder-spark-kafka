"""
Kafka Producer for NYC Taxi Data Streaming

Reads NYC taxi data from CSV and streams it to a Kafka topic
for real-time anomaly detection processing.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime

import pandas as pd
from confluent_kafka import Producer, KafkaError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def delivery_callback(err, msg):
    """Callback for message delivery reports."""
    if err is not None:
        logger.error(f"Message delivery failed: {err}")
    else:
        logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}] @ {msg.offset()}")


def create_producer(bootstrap_servers: str) -> Producer:
    """Create and configure the Kafka producer."""
    config = {
        "bootstrap.servers": bootstrap_servers,
        "client.id": "nyc-taxi-producer",
        "acks": "all",
        "retries": 3,
        "retry.backoff.ms": 1000,
        "linger.ms": 5,
        "batch.size": 16384,
    }
    return Producer(config)


def load_taxi_data(file_path: str) -> pd.DataFrame:
    """Load NYC taxi data from CSV file."""
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info(f"Loaded {len(df)} records")
    return df


def stream_data(
    producer: Producer,
    topic: str,
    df: pd.DataFrame,
    delay_seconds: float = 0.1,
    loop: bool = True,
    start_offset: int = 0
):
    """
    Stream taxi data to Kafka topic.

    Args:
        producer: Kafka producer instance
        topic: Kafka topic name
        df: DataFrame containing taxi data
        delay_seconds: Delay between messages (simulates real-time)
        loop: Whether to continuously loop through data
        start_offset: Record index to start streaming from (useful for
                      skipping training data when using LSTM detector)
    """
    # Apply starting offset
    if start_offset > 0:
        if start_offset >= len(df):
            logger.error(f"Start offset {start_offset} >= data length {len(df)}")
            return
        df = df.iloc[start_offset:].reset_index(drop=True)
        logger.info(f"Starting from record {start_offset}, {len(df)} records remaining")

    logger.info(f"Starting to stream data to topic '{topic}'")
    logger.info(f"Delay between messages: {delay_seconds}s, Loop: {loop}")

    iteration = 0
    total_messages = 0

    while True:
        iteration += 1
        logger.info(f"Starting iteration {iteration}")

        for idx, row in df.iterrows():
            message = {
                "timestamp": row["timestamp"].isoformat(),
                "value": int(row["value"]),
                "produced_at": datetime.utcnow().isoformat(),
                "sequence_id": total_messages
            }

            try:
                producer.produce(
                    topic=topic,
                    key=str(total_messages).encode("utf-8"),
                    value=json.dumps(message).encode("utf-8"),
                    callback=delivery_callback
                )
                total_messages += 1

                if total_messages % 100 == 0:
                    logger.info(f"Produced {total_messages} messages")
                    producer.flush()

                producer.poll(0)
                time.sleep(delay_seconds)

            except BufferError:
                logger.warning("Producer queue full, waiting...")
                producer.flush()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error producing message: {e}")
                raise

        producer.flush()
        logger.info(f"Completed iteration {iteration}, total messages: {total_messages}")

        if not loop:
            break

    logger.info(f"Streaming complete. Total messages produced: {total_messages}")


def wait_for_kafka(bootstrap_servers: str, max_retries: int = 30, retry_interval: int = 2):
    """Wait for Kafka to be available before starting."""
    logger.info(f"Waiting for Kafka at {bootstrap_servers}...")

    for attempt in range(max_retries):
        try:
            producer = Producer({"bootstrap.servers": bootstrap_servers})
            metadata = producer.list_topics(timeout=5)
            logger.info(f"Connected to Kafka. Available topics: {list(metadata.topics.keys())}")
            return True
        except Exception as e:
            logger.warning(f"Kafka not ready (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(retry_interval)

    raise RuntimeError(f"Could not connect to Kafka after {max_retries} attempts")


def wait_for_app_ready(app_url: str, max_retries: int = 60, retry_interval: int = 2):
    """
    Wait for the app's Spark streaming to be ready before producing.

    Polls the app's /health endpoint until spark_ready is True.
    """
    import urllib.request
    import urllib.error

    health_url = f"{app_url}/health"
    logger.info(f"Waiting for app to be ready at {health_url}...")

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                data = json.loads(response.read().decode())
                if data.get("spark_ready", False):
                    logger.info(f"App is ready! Spark streaming connected.")
                    return True
                else:
                    logger.info(f"App starting... (attempt {attempt + 1}/{max_retries})")
        except urllib.error.URLError as e:
            logger.debug(f"App not reachable yet (attempt {attempt + 1}/{max_retries}): {e}")
        except Exception as e:
            logger.debug(f"Health check failed (attempt {attempt + 1}/{max_retries}): {e}")

        time.sleep(retry_interval)

    logger.warning(f"App did not become ready after {max_retries} attempts, proceeding anyway")
    return False


def main():
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("KAFKA_TOPIC", "anomaly_stream")
    data_path = os.getenv("DATA_PATH", "/app/data/nyc_taxi.csv")
    delay = float(os.getenv("MESSAGE_DELAY_SECONDS", "0.1"))
    loop_data = os.getenv("LOOP_DATA", "true").lower() == "true"
    start_offset = int(os.getenv("START_OFFSET", "0"))
    app_url = os.getenv("APP_URL", "http://app:8050")
    wait_for_app = os.getenv("WAIT_FOR_APP", "true").lower() == "true"

    logger.info("=" * 50)
    logger.info("NYC Taxi Data Kafka Producer")
    logger.info("=" * 50)
    logger.info(f"Bootstrap Servers: {bootstrap_servers}")
    logger.info(f"Topic: {topic}")
    logger.info(f"Data Path: {data_path}")
    logger.info(f"Message Delay: {delay}s")
    logger.info(f"Loop Data: {loop_data}")
    logger.info(f"Start Offset: {start_offset}")
    logger.info(f"Wait for App: {wait_for_app}")
    logger.info("=" * 50)

    wait_for_kafka(bootstrap_servers)

    # Wait for app's Spark streaming to be ready before producing
    if wait_for_app:
        wait_for_app_ready(app_url)

    producer = create_producer(bootstrap_servers)

    df = load_taxi_data(data_path)

    try:
        stream_data(
            producer, topic, df,
            delay_seconds=delay,
            loop=loop_data,
            start_offset=start_offset
        )
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        logger.info("Flushing remaining messages...")
        producer.flush(timeout=10)
        logger.info("Producer shutdown complete")


if __name__ == "__main__":
    main()
