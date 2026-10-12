"""Shared SparkSession factory."""

from __future__ import annotations

from pyspark.sql import SparkSession

from spark.config import SparkConfig


def create_spark_session(
    app_name: str,
    *,
    enable_kafka: bool = False,
    config: SparkConfig | None = None,
) -> SparkSession:
    """Build a SparkSession from centralized env config."""
    cfg = config or SparkConfig.from_env()
    builder = SparkSession.builder.appName(app_name).master(cfg.master)
    for key, value in cfg.session_configs(enable_kafka=enable_kafka).items():
        builder = builder.config(key, value)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(cfg.log_level)
    return spark
