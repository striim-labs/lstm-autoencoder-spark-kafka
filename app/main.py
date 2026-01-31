"""
NYC Taxi Data Stream Viewer

Simple dashboard to verify Kafka → Spark Streaming pipeline is working.
No ML/anomaly detection - just displays incoming data.
"""

import os
import threading
import time
import logging
from collections import deque

import pandas as pd

import dash
from dash.dependencies import Output, Input
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Thread-safe data storage
data_lock = threading.Lock()
WINDOW_SIZE = 200

data_store = {
    "data": deque(maxlen=WINDOW_SIZE),
    "total_received": 0,
    "last_batch_size": 0,
    "last_update": None
}


def wait_for_kafka(bootstrap_servers: str, max_retries: int = 30, retry_interval: int = 5):
    """Wait for Kafka to be available."""
    from confluent_kafka import Producer

    logger.info(f"Waiting for Kafka at {bootstrap_servers}...")
    for attempt in range(max_retries):
        try:
            producer = Producer({"bootstrap.servers": bootstrap_servers})
            metadata = producer.list_topics(timeout=10)
            logger.info(f"Connected to Kafka. Topics: {list(metadata.topics.keys())}")
            return True
        except Exception as e:
            logger.warning(f"Kafka not ready (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(retry_interval)

    raise RuntimeError(f"Could not connect to Kafka after {max_retries} attempts")


def start_spark_streaming():
    """Start Spark Structured Streaming to consume from Kafka."""
    kafka_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
    kafka_topic = os.environ.get("KAFKA_TOPIC", "anomaly_stream")
    spark_master = os.environ.get("SPARK_MASTER", "spark://spark-master:7077")

    wait_for_kafka(kafka_servers)

    logger.info(f"Starting Spark session connecting to {spark_master}")

    spark = SparkSession.builder \
        .appName("NYCTaxiStreamViewer") \
        .master(spark_master) \
        .config("spark.driver.host", "app") \
        .config("spark.driver.bindAddress", "0.0.0.0") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3") \
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # Schema matching producer output
    schema = StructType([
        StructField("timestamp", StringType()),
        StructField("value", IntegerType()),
        StructField("produced_at", StringType()),
        StructField("sequence_id", IntegerType())
    ])

    logger.info(f"Subscribing to Kafka topic: {kafka_topic}")

    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_servers) \
        .option("subscribe", kafka_topic) \
        .option("startingOffsets", "latest") \
        .option("maxOffsetsPerTrigger", 50) \
        .load()

    df = df.selectExpr("CAST(value AS STRING)")
    df_parsed = df.select(from_json(col("value"), schema).alias("data")).select("data.*")
    df_parsed = df_parsed.withColumn("timestamp", to_timestamp(col("timestamp")))

    def process_batch(batch_df, batch_id):
        """Process each micro-batch - just store the data."""
        try:
            pandas_df = batch_df.toPandas()

            if pandas_df.empty:
                logger.debug(f"Batch {batch_id}: empty")
                return

            logger.info(f"Batch {batch_id}: received {len(pandas_df)} records")

            with data_lock:
                for _, row in pandas_df.iterrows():
                    data_store["data"].append({
                        "timestamp": str(row["timestamp"]),
                        "value": row["value"],
                        "produced_at": row["produced_at"],
                        "sequence_id": row["sequence_id"]
                    })

                data_store["total_received"] += len(pandas_df)
                data_store["last_batch_size"] = len(pandas_df)
                data_store["last_update"] = pd.Timestamp.now().isoformat()

        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {e}")

    query = df_parsed.writeStream \
        .trigger(processingTime="2 seconds") \
        .foreachBatch(process_batch) \
        .start()

    logger.info("Spark Streaming started - waiting for data...")
    query.awaitTermination()


# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Kafka → Spark Stream Viewer"

app.layout = dbc.Container([
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand(
                "Kafka → Spark Streaming Test",
                style={"color": "white", "fontSize": "24px", "fontWeight": "bold"}
            ),
        ]),
        color="#2c3e50",
        dark=True,
        sticky="top",
        className="mb-4"
    ),

    # Status cards
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Total Received", className="text-center"),
                html.H2(id="total-received", className="text-center text-success")
            ])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Window Size", className="text-center"),
                html.H2(id="window-size", className="text-center text-info")
            ])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Last Batch", className="text-center"),
                html.H2(id="last-batch", className="text-center text-primary")
            ])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Last Update", className="text-center"),
                html.P(id="last-update", className="text-center text-muted", style={"fontSize": "14px"})
            ])
        ]), width=3),
    ], className="mb-4"),

    # Chart
    dbc.Card([
        dbc.CardHeader(html.H4("Taxi Demand - Live Stream", className="mb-0")),
        dbc.CardBody([
            dcc.Graph(id="stream-graph", style={"height": "350px"})
        ])
    ], className="mb-4"),

    # Recent data table
    dbc.Card([
        dbc.CardHeader(html.H4("Recent Records (Last 20)", className="mb-0")),
        dbc.CardBody([
            dash_table.DataTable(
                id="data-table",
                columns=[
                    {"name": "Timestamp", "id": "timestamp"},
                    {"name": "Value", "id": "value"},
                    {"name": "Sequence ID", "id": "sequence_id"},
                    {"name": "Produced At", "id": "produced_at"},
                ],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "8px"},
                style_header={"backgroundColor": "#2c3e50", "color": "white", "fontWeight": "bold"},
                page_size=20
            )
        ])
    ]),

    dcc.Interval(id="refresh", interval=2000, n_intervals=0),
], fluid=True, style={"backgroundColor": "#ecf0f1", "minHeight": "100vh", "paddingBottom": "20px"})


@app.callback(
    [
        Output("stream-graph", "figure"),
        Output("data-table", "data"),
        Output("total-received", "children"),
        Output("window-size", "children"),
        Output("last-batch", "children"),
        Output("last-update", "children"),
    ],
    [Input("refresh", "n_intervals")]
)
def update_dashboard(n):
    with data_lock:
        data_list = list(data_store["data"])
        total = data_store["total_received"]
        last_batch = data_store["last_batch_size"]
        last_update = data_store["last_update"] or "Waiting..."

    fig = go.Figure()

    if data_list:
        df = pd.DataFrame(data_list)
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["value"],
            mode="lines+markers",
            name="Taxi Demand",
            line=dict(color="#3498db", width=2),
            marker=dict(size=4)
        ))

    fig.update_layout(
        xaxis=dict(title="Timestamp"),
        yaxis=dict(title="Value"),
        plot_bgcolor="white",
        margin=dict(l=50, r=30, t=30, b=50),
        hovermode="x unified"
    )

    if not data_list:
        fig.add_annotation(
            text="Waiting for data from Kafka...",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )

    # Show last 20 records in table (reversed so newest first)
    table_data = list(reversed(data_list[-20:])) if data_list else []

    return fig, table_data, f"{total:,}", f"{len(data_list)}", f"{last_batch}", last_update


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Kafka -> Spark Streaming Test Viewer")
    logger.info("=" * 50)

    streaming_thread = threading.Thread(target=start_spark_streaming, daemon=True)
    streaming_thread.start()

    logger.info("Starting Dash server on http://0.0.0.0:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)
