"""Predict minutes left and the landing time for flights in progress."""

import logging
import os

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ml.features import add_features, load_airports
from ml.train import offline_config
from spark.session import create_spark_session

LOGGER = logging.getLogger(__name__)


def predict(model: PipelineModel, df: DataFrame, airports: DataFrame) -> DataFrame:
    """Score flights. If ``ts`` is missing, ``fetched_at`` is used.
    """
    if "ts" not in df.columns:
        df = df.withColumn("ts", F.col("fetched_at"))

    scored = model.transform(add_features(df, airports))
    return (
        scored.withColumn("pred_remaining_min", F.greatest(F.col("prediction"), F.lit(0.0)))
        .withColumn("pred_landing_ts", (F.col("ts") + F.col("pred_remaining_min") * 60).cast("long"))
        .withColumn("pred_landing_time", F.timestamp_seconds("pred_landing_ts"))
        .select(
            "icao24", "arr_airport", "ts",
            "pred_remaining_min", "pred_landing_ts", "pred_landing_time",
        )
    )


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    config = offline_config()
    spark = create_spark_session("OpenSky-ETA-Predict", config=config)

    model = PipelineModel.load(config.model_output_path)
    airports = load_airports(spark, config.airports_path)
    flights = spark.read.parquet(os.environ["PREDICT_INPUT_PATH"])

    result = predict(model, flights, airports)
    output_path = os.getenv("PREDICT_OUTPUT_PATH")
    if output_path:
        result.write.mode("overwrite").parquet(output_path)
    else:
        result.show(20, truncate=False)
    spark.stop()


if __name__ == "__main__":
    main()