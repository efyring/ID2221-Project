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

# Time zone of the arrival airport (Frankfurt). Spark itself runs in UTC.
DISPLAY_TZ = os.getenv("DISPLAY_TZ", "Europe/Berlin")


def predict(model: PipelineModel, df: DataFrame, airports: DataFrame) -> DataFrame:
    """Score flights. Input needs the columns described in ml/features.py
    (``actual_arrival_ts`` is not needed). If ``ts`` is missing, ``fetched_at`` is used.
    """
    if "ts" not in df.columns:
        df = df.withColumn("ts", F.col("fetched_at"))

    scored = model.transform(add_features(df, airports))
    scored = (
        scored.withColumn("pred_remaining_min", F.greatest(F.col("prediction"), F.lit(0.0)))
        .withColumn("pred_landing_ts", (F.col("ts") + F.col("pred_remaining_min") * 60).cast("long"))
        .withColumn("pred_landing_time", F.timestamp_seconds("pred_landing_ts"))  # UTC instant
        # Wall-clock time at the arrival airport, for display only
        .withColumn("pred_landing_local", F.from_utc_timestamp("pred_landing_time", DISPLAY_TZ))
    )
    cols = [
        "icao24", "arr_airport", "ts",
        "latitude", "longitude",
        "dist_to_arr_km", "naive_eta_min", "altitude", "velocity", "vertical_rate",
        "pred_remaining_min", "pred_landing_ts", "pred_landing_time", "pred_landing_local",
    ]
    if "callsign" in scored.columns:  # live OpenSky states have it
        cols.insert(1, "callsign")
    if "true_track" in scored.columns:  # ground track heading, degrees clockwise from north
        cols.insert(cols.index("longitude") + 1, "true_track")
    if "true_arrival_ts" in scored.columns:  # only present in the synthetic demo data
        scored = scored.withColumn(
            "err_min", (F.col("pred_landing_ts") - F.col("true_arrival_ts")) / 60.0
        )
        cols += ["true_arrival_ts", "err_min"]
    return scored.select(*cols)


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
        if "err_min" in result.columns:
            result.agg(F.avg(F.abs("err_min")).alias("MAE_min")).show()
    spark.stop()


if __name__ == "__main__":
    main()