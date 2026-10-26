"""Fetch live OpenSky states, keep flights inbound to the hub, run ETA predictions.

Does not need Kafka. Writes:
  - ``LIVE_STATES_PATH``     (raw inbound snapshots as parquet)
  - ``LIVE_PREDICTIONS_PATH`` (model scores for the Streamlit dashboard)

Example::

    PYTHONPATH=src python src/ml/predict_opensky.py
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

from ml.features import load_airports
from ml.predict_stream import live_predictions
from ml.train import offline_config
from producer.producer import _state_to_dict
from spark.session import create_spark_session

LOGGER = logging.getLogger(__name__)

OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"
# Bounding box covering typical EDDF inbounds (Western / Central Europe).
DEFAULT_BBOX = {
    "lamin": 42.0,
    "lomin": -10.0,
    "lamax": 60.0,
    "lomax": 25.0,
}


def fetch_opensky_states(
    session: requests.Session | None = None,
    bbox: dict[str, float] | None = None,
    timeout: float = 45.0,
) -> pd.DataFrame:
    """Pull current ADS-B states from OpenSky (optionally bbox-limited)."""
    session = session or requests.Session()
    params = bbox or {
        "lamin": float(os.getenv("OPENSKY_LAMIN", DEFAULT_BBOX["lamin"])),
        "lomin": float(os.getenv("OPENSKY_LOMIN", DEFAULT_BBOX["lomin"])),
        "lamax": float(os.getenv("OPENSKY_LAMAX", DEFAULT_BBOX["lamax"])),
        "lomax": float(os.getenv("OPENSKY_LOMAX", DEFAULT_BBOX["lomax"])),
    }
    auth = None
    user = os.getenv("OPENSKY_USERNAME")
    password = os.getenv("OPENSKY_PASSWORD")
    if user and password:
        auth = (user, password)

    response = session.get(OPENSKY_STATES_URL, params=params, auth=auth, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    fetched_at = int(payload.get("time") or time.time())
    rows = [_state_to_dict(state, fetched_at) for state in (payload.get("states") or [])]
    df = pd.DataFrame(rows)
    LOGGER.info("OpenSky returned %d states (fetched_at=%s)", len(df), fetched_at)
    return df


def run(
    states_out: str | None = None,
    predictions_out: str | None = None,
) -> pd.DataFrame:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    config = offline_config()
    states_out = states_out or os.getenv("LIVE_STATES_PATH", "./data/live/current_states")
    predictions_out = predictions_out or os.getenv(
        "LIVE_PREDICTIONS_PATH", "./data/live/predictions.parquet"
    )
    max_km = float(os.getenv("LIVE_MAX_KM", "1500"))
    max_angle = float(os.getenv("LIVE_MAX_ANGLE", "60"))

    pdf = fetch_opensky_states()
    if pdf.empty:
        raise SystemExit("OpenSky returned no states (rate-limited or empty bbox).")

    states_path = Path(states_out)
    states_path.mkdir(parents=True, exist_ok=True)
    # Spark-friendly single parquet file / directory.
    raw_path = states_path / "part-00000.parquet"
    pdf.to_parquet(raw_path, index=False)

    spark = create_spark_session("OpenSky-ETA-Live", config=config)
    try:
        airports = load_airports(spark, config.airports_path)
        hub = airports.filter(F.col("icao") == config.arrival_airport).first()
        if hub is None:
            raise SystemExit(f"{config.arrival_airport} not in {config.airports_path}")

        model = PipelineModel.load(config.model_output_path)
        raw = spark.read.parquet(str(states_path))
        preds = live_predictions(
            raw,
            model,
            airports,
            float(hub["lat"]),
            float(hub["lon"]),
            config.arrival_airport,
            max_km,
            max_angle,
            hub_elev_m=float(hub["elev_m"] or 0.0),
            hub_dh_m=float(hub["decision_height_m"]),
        )
        # Also keep a count of how many states were inbound vs raw.
        n_raw = raw.count()
        result = preds.toPandas()
        n_in = len(result)
        LOGGER.info(
            "Inbound to %s: %d / %d states (max_km=%.0f, max_angle=%.0f)",
            config.arrival_airport,
            n_in,
            n_raw,
            max_km,
            max_angle,
        )
    finally:
        spark.stop()

    if {"naive_eta_min", "pred_remaining_min"}.issubset(result.columns):
        result["delay_vs_naive_min"] = result["pred_remaining_min"] - result["naive_eta_min"]

    out = Path(predictions_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    LOGGER.info("Wrote %d predictions → %s", len(result), out)

    if not result.empty:
        show = result.sort_values("pred_remaining_min")[
            [c for c in [
                "icao24", "callsign", "dist_to_arr_km", "altitude",
                "pred_remaining_min", "pred_landing_local", "delay_vs_naive_min",
            ] if c in result.columns]
        ].head(25)
        print(show.to_string(index=False))
        hit = result.loc[result["icao24"].astype(str).str.lower() == "3c6744"]
        if len(hit):
            print("\nFound 3c6744:")
            print(hit.to_string(index=False))
        else:
            print("\nNote: icao24 3c6744 is not in the inbound set this poll "
                  "(not seen, filtered out, or OpenSky bbox miss).")
    return result


if __name__ == "__main__":
    run()
