"""Spark runtime: config, session factory, aggregations, and Kafka ingest."""

from spark.config import SparkConfig

__all__ = ["SparkConfig", "aggregate_for_training", "create_spark_session"]


def __getattr__(name: str):
    if name == "create_spark_session":
        from spark.session import create_spark_session

        return create_spark_session
    if name == "aggregate_for_training":
        from spark.aggregate import aggregate_for_training

        return aggregate_for_training
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
