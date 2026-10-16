"""Kafka -> Spark Structured Streaming ingest: append raw OpenSky states to Parquet."""

from __future__ import annotations

import logging
import os

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ml.features import OPENSKY_SCHEMA
from spark.config import SparkConfig
from spark.session import create_spark_session

LOGGER = logging.getLogger(__name__)


def _process_batch(batch_df: DataFrame, batch_id: int, config: SparkConfig) -> None:
    """Append a micro-batch to Parquet, partitioned by UTC date of ``fetched_at``."""
    if batch_df.isEmpty():
        LOGGER.debug("Skipping empty batch %s", batch_id)
        return

    partitioned = batch_df.withColumn(
        "fetched_at_date", F.from_unixtime(F.col("fetched_at"), "yyyy-MM-dd")
    )
    (
        partitioned.write.mode("append")
        .partitionBy("fetched_at_date")
        .parquet(config.train_data_path)
    )
    LOGGER.info("Appended batch %s to %s", batch_id, config.train_data_path)


def run() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    config = SparkConfig.from_env()
    spark = create_spark_session("OpenSky-KafkaIngest", enable_kafka=True, config=config)

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
        raw.select(F.from_json(F.col("value").cast("string"), OPENSKY_SCHEMA).alias("data"))
        .select("data.*")
        .filter(F.col("icao24").isNotNull())
    )

    query = (
        parsed.writeStream.foreachBatch(
            lambda batch_df, batch_id: _process_batch(batch_df, batch_id, config)
        )
        .option("checkpointLocation", config.checkpoint_path)
        .outputMode("append")
        .start()
    )
    query.awaitTermination()


if __name__ == "__main__":
    run()