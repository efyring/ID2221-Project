"""Spark aggregations that turn ingested snapshots into ML training rows."""

from __future__ import annotations

from pyspark.sql import DataFrame

from ml.features import prepare_training_dataset


def aggregate_for_training(df: DataFrame) -> DataFrame:
    """Run the Spark groupBy aggregations required before GBT fitting."""
    return prepare_training_dataset(df)
