import logging
import os
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import SparkSession

from features import compute_features

LOGGER = logging.getLogger(__name__)


def train() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    spark = (
        SparkSession.builder.appName("OpenSky-ML-BatchTraining")
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        .getOrCreate()
    )

    data_path = os.getenv("TRAIN_DATA_PATH", "./data/historical_opensky")
    model_output_path = os.getenv("MODEL_OUTPUT_PATH", "./models/delay_gbt_pipeline")

    LOGGER.info("Loading training data from %s", data_path)
    raw_df = spark.read.parquet(data_path)

    # Apply spatial feature extraction
    featured_df = compute_features(raw_df)

    # Train / Test split
    train_df, test_df = featured_df.randomSplit([0.8, 0.2], seed=42)

    # ML Pipeline Stages
    feature_cols = [
        "dist_to_airport_km",
        "clean_altitude",
        "clean_velocity",
        "clean_vertical_rate",
        "descent_gradient",
    ]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features")
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="actual_delay_min",
        maxDepth=5,
        maxIter=30,
        seed=42,
    )

    pipeline = Pipeline(stages=[assembler, scaler, gbt])

    LOGGER.info("Fitting GBT Regression model...")
    model_pipeline = pipeline.fit(train_df)

    # Evaluation
    predictions = model_pipeline.transform(test_df)
    evaluator_rmse = RegressionEvaluator(
        labelCol="actual_delay_min", predictionCol="prediction", metricName="rmse"
    )
    evaluator_mae = RegressionEvaluator(
        labelCol="actual_delay_min", predictionCol="prediction", metricName="mae"
    )

    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)

    LOGGER.info("Model Evaluation -> Test RMSE: %.3f min, Test MAE: %.3f min", rmse, mae)

    LOGGER.info("Saving fitted model pipeline to %s", model_output_path)
    model_pipeline.write().overwrite().save(model_output_path)
    spark.stop()


if __name__ == "__main__":
    train()