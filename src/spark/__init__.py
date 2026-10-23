"""Spark runtime: config, session factory, and Kafka ingest."""

from spark.config import SparkConfig

__all__ = ["SparkConfig", "create_spark_session"]


def __getattr__(name: str):
    if name == "create_spark_session":
        from spark.session import create_spark_session

        return create_spark_session
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")