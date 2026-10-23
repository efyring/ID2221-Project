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
    builder = SparkSession.builder.appName(app_name)
    
    # Safely fetch session configs supporting both kafka and cassandra flags
    try:
        session_dict = cfg.session_configs(
            enable_kafka=enable_kafka, enable_cassandra=enable_cassandra
        )
    except TypeError:
        session_dict = cfg.session_configs(enable_kafka=enable_kafka)
        
    session_dict.pop("spark.jars.ivy", None)  # Remove Docker container path
    
    for key, value in session_dict.items():
        builder = builder.config(key, value)
        
    # Force Ivy cache to a local, writable folder on your Mac
    builder = builder.config("spark.jars.ivy", os.path.abspath("./.ivy2"))
    
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(cfg.log_level)
    return spark