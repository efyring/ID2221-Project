"""Run batch ETA predictions and return a pandas DataFrame for the dashboard."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

# Spark is imported inside ``run_predictions`` so Cassandra-only Streamlit
# (``make viz-cassandra``) can load without pulling PySpark at import time.


def run_predictions(
    input_path: str | None = None,
    model_path: str | None = None,
    airports_path: str | None = None,
) -> pd.DataFrame:
    """Score ``PREDICT_INPUT_PATH`` (or ``input_path``) and return results as pandas.

    Adds ``delay_vs_naive_min``: model remaining time minus the naive distance/speed
    ETA (positive ⇒ model expects a longer trip than kinematics alone).
    """
    from pyspark.ml import PipelineModel

    from ml.features import load_airports
    from ml.predict import predict
    from ml.train import offline_config
    from spark.session import create_spark_session

    config = offline_config()
    input_path = input_path or os.environ.get(
        "PREDICT_INPUT_PATH", "./data/synthetic/current_flights"
    )
    model_path = model_path or config.model_output_path
    airports_path = airports_path or config.airports_path

    spark = create_spark_session("OpenSky-ETA-Viz", config=config)
    try:
        model = PipelineModel.load(model_path)
        airports = load_airports(spark, airports_path)
        flights = spark.read.parquet(input_path)
        pdf = predict(model, flights, airports).toPandas()
    finally:
        spark.stop()

    if {"naive_eta_min", "pred_remaining_min"}.issubset(pdf.columns):
        pdf["delay_vs_naive_min"] = pdf["pred_remaining_min"] - pdf["naive_eta_min"]
    if {"true_arrival_ts", "ts"}.issubset(pdf.columns):
        pdf["true_remaining_min"] = (pdf["true_arrival_ts"] - pdf["ts"]) / 60.0
    return pdf


def summary_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compact KPIs for the Streamlit header."""
    out: dict[str, Any] = {
        "n_flights": int(len(df)),
        "avg_remaining_min": float(df["pred_remaining_min"].mean()) if len(df) else 0.0,
        "avg_dist_km": float(df["dist_to_arr_km"].mean()) if len(df) else 0.0,
    }
    if "err_min" in df.columns and len(df):
        out["mae_min"] = float(df["err_min"].abs().mean())
        out["rmse_min"] = float((df["err_min"] ** 2).mean() ** 0.5)
    if "delay_vs_naive_min" in df.columns and len(df):
        out["avg_delay_vs_naive_min"] = float(df["delay_vs_naive_min"].mean())
    return out
