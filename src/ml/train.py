import logging
import os
from dataclasses import replace

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import DataFrame

from ml.features import AGGREGATED_FEATURE_COLS
from spark.aggregate import aggregate_for_training
from spark.config import SparkConfig
from spark.session import create_spark_session

LOGGER = logging.getLogger(__name__)

FEATURE_COLS = AGGREGATED_FEATURE_COLS

MIN_TRAIN_ROWS = 10


def fit_and_save(df: DataFrame, model_output_path: str) -> None:
    """Fit the GBT pipeline on aggregated training rows and overwrite the model path."""
    row_count = df.count()
    if row_count < MIN_TRAIN_ROWS:
        LOGGER.warning(
            "Skipping retrain: need at least %d aggregated rows, got %d",
            MIN_TRAIN_ROWS,
            row_count,
        )
        return

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features")
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="actual_delay_min",
        maxDepth=5,
        maxIter=30,
        seed=42,
    )

    pipeline = Pipeline(stages=[assembler, scaler, gbt])

    LOGGER.info("Fitting GBT Regression model on %d aggregated rows...", row_count)
    model_pipeline = pipeline.fit(train_df)

    eval_df = test_df if not test_df.isEmpty() else train_df
    if test_df.isEmpty():
        LOGGER.warning("Test split empty; reporting metrics on train split instead")

    predictions = model_pipeline.transform(eval_df)
    evaluator_rmse = RegressionEvaluator(
        labelCol="actual_delay_min", predictionCol="prediction", metricName="rmse"
    )
    evaluator_mae = RegressionEvaluator(
        labelCol="actual_delay_min", predictionCol="prediction", metricName="mae"
    )

    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)

    LOGGER.info("Model Evaluation -> RMSE: %.3f min, MAE: %.3f min", rmse, mae)

    LOGGER.info("Saving fitted model pipeline to %s", model_output_path)
    model_pipeline.write().overwrite().save(model_output_path)


def train() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    # Offline / local default differs from the compose cluster default.
    config = SparkConfig.from_env()
    if os.getenv("SPARK_MASTER") is None:
        config = replace(config, master="local[*]", driver_host=None)

    spark = create_spark_session(
        "OpenSky-ML-BatchTraining",
        enable_kafka=False,
        config=config,
    )

    LOGGER.info("Loading training data from %s", config.train_data_path)
    raw_df = spark.read.parquet(config.train_data_path)

    LOGGER.info("Aggregating per-aircraft features for training")
    training_df = aggregate_for_training(raw_df)
    fit_and_save(training_df, config.model_output_path)
    spark.stop()


if __name__ == "__main__":
    train()
