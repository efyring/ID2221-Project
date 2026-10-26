from __future__ import annotations

import os
from pyspark.sql import SparkSession
from spark.config import SparkConfig


def create_spark_session(
    app_name: str,
    *,
    enable_kafka: bool = False,
    enable_cassandra: bool = False,
    config: SparkConfig | None = None,
) -> SparkSession:
    """Build a SparkSession from centralized env config."""
    cfg = config or SparkConfig.from_env()
    builder = SparkSession.builder.appName(app_name).master(cfg.master)

    try:
        session_dict = cfg.session_configs(
            enable_kafka=enable_kafka, enable_cassandra=enable_cassandra
        )
    except TypeError:
        session_dict = cfg.session_configs(enable_kafka=enable_kafka)

    # Prefer explicit IVY_HOME (Docker sets this); fall back to a local cache on the host.
    ivy = os.environ.get("IVY_HOME") or os.path.abspath("./.ivy2")
    session_dict["spark.jars.ivy"] = ivy

    for key, value in session_dict.items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(cfg.log_level)
    return spark
