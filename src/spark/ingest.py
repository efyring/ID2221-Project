"""Kafka → Spark Structured Streaming ingest with periodic GBT retrain."""

from __future__ import annotations

import logging
import os

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ml.features import OPENSKY_SCHEMA, add_proxy_label, compute_features
from ml.train import fit_and_save
from spark.aggregate import aggregate_for_training
from spark.config import SparkConfig
from spark.session import create_spark_session

LOGGER = logging.getLogger(__name__)

_batches_since_start = 0


def _with_date_partition(batch_df: DataFrame) -> DataFrame:
    """Add calendar-date partition column derived from ``fetched_at`` (UTC)."""
    return batch_df.withColumn(
        "fetched_at_date",
        F.from_unixtime(F.col("fetched_at"), "yyyy-MM-dd"),
    )


def _process_batch(batch_df: DataFrame, batch_id: int, config: SparkConfig) -> None:
    """Append a micro-batch to Parquet and periodically retrain the model."""
    global _batches_since_start

    if batch_df.isEmpty():
        LOGGER.debug("Skipping empty batch %s", batch_id)
        return

    partitioned = _with_date_partition(batch_df)
    row_count = partitioned.count()
    (
        partitioned.write.mode("append")
        .partitionBy("fetched_at_date")
        .parquet(config.train_data_path)
    )
    _batches_since_start += 1

    LOGGER.info(
        "Appended batch %s (%d rows) to %s [batches_since_start=%d] "
        "partitioned by fetched_at_date",
        batch_id,
        row_count,
        config.train_data_path,
        _batches_since_start,
    )

    if _batches_since_start % config.retrain_every_n_batches != 0:
        return

    spark = batch_df.sparkSession
    LOGGER.info(
        "Retrain trigger: every %d batches (at %d). Loading %s",
        config.retrain_every_n_batches,
        _batches_since_start,
        config.train_data_path,
    )
    accumulated = spark.read.parquet(config.train_data_path)
    LOGGER.info("Aggregating per-aircraft features for retrain")
    training_df = aggregate_for_training(accumulated)
    fit_and_save(training_df, config.model_output_path)


def run() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    config = SparkConfig.from_env()

    # Workers must reach the driver by a compose-resolvable hostname (not the
    # ephemeral container id). Override with SPARK_DRIVER_HOST when needed.
    spark = create_spark_session(
        "OpenSky-ML-KafkaIngest",
        enable_kafka=True,
        config=config,
    )

    LOGGER.info(
        "Starting ingest: master=%s kafka=%s topic=%s checkpoint=%s",
        config.master,
        config.kafka_bootstrap_servers,
        config.kafka_topic,
        config.checkpoint_path,
    )

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", config.kafka_bootstrap_servers)
        .option("subscribe", config.kafka_topic)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = (
        raw.select(
            F.from_json(F.col("value").cast("string"), OPENSKY_SCHEMA).alias("data")
        )
        .select("data.*")
        .filter(F.col("icao24").isNotNull())
    )

    featured = add_proxy_label(compute_features(parsed))

    query = (
        featured.writeStream.foreachBatch(
            lambda batch_df, batch_id: _process_batch(batch_df, batch_id, config)
        )
        .option("checkpointLocation", config.checkpoint_path)
        .outputMode("append")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    run()
