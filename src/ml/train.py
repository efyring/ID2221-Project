"""Train a GBT regressor that predicts minutes remaining until landing."""

import logging
import os
from dataclasses import replace

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ml.features import FEATURE_COLS, LABEL_COL, add_features, add_label, load_airports
from spark.config import SparkConfig
from spark.session import create_spark_session

LOGGER = logging.getLogger(__name__)

MIN_TRAIN_ROWS = 100


def offline_config() -> SparkConfig:
    config = SparkConfig.from_env()
    if os.getenv("SPARK_MASTER") is None:
        config = replace(config, master="local[*]", driver_host=None)
        if os.getenv("DEFAULT_FS") is None:
            config = replace(config, default_fs="file:///")
    return config


def fit_and_save(df: DataFrame, model_output_path: str) -> None:
    """Fit on a featurized + labelled DataFrame, report metrics, save the model."""
    df = df.cache()
    row_count = df.count()
    if row_count < MIN_TRAIN_ROWS:
        LOGGER.warning("Need at least %d rows, got %d. Skipping.", MIN_TRAIN_ROWS, row_count)
        return

    # Split by flight (not by row) so snapshots of one flight never leak into the test set.
    bucket = F.pmod(F.hash("flight_key"), F.lit(5))
    train_df = df.filter(bucket != 0)
    test_df = df.filter(bucket == 0)

    pipeline = Pipeline(stages=[
        VectorAssembler(inputCols=FEATURE_COLS, outputCol="features", handleInvalid="skip"),
        GBTRegressor(featuresCol="features", labelCol=LABEL_COL, maxDepth=6, maxIter=50, seed=42),
    ])

    LOGGER.info("Fitting on %d rows", row_count)
    model = pipeline.fit(train_df)

    preds = model.transform(test_df)
    for metric in ("rmse", "mae"):
        ev = RegressionEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName=metric)
        base = RegressionEvaluator(labelCol=LABEL_COL, predictionCol="naive_eta_min", metricName=metric)
        LOGGER.info(
            "Test %s: model %.2f min | baseline (distance/speed) %.2f min",
            metric.upper(), ev.evaluate(preds), base.evaluate(preds),
        )

    model.write().overwrite().save(model_output_path)
    LOGGER.info("Saved model to %s", model_output_path)


def train() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    config = offline_config()
    spark = create_spark_session("OpenSky-ETA-Training", config=config)

    airports = load_airports(spark, config.airports_path)
    raw = spark.read.parquet(config.train_data_path)
    data = add_label(add_features(raw, airports))
    fit_and_save(data, config.model_output_path)
    spark.stop()


if __name__ == "__main__":
    train()