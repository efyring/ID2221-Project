"""Infer arrival airport + landing time from OpenSky track kinematics.

Used to turn raw ingested states into labelled training rows
(``arr_airport``, ``actual_arrival_ts``, ``ts``).
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from ml.features import _haversine_km

FLIGHT_GAP_SEC = 45 * 60
DESCENT_LOOKBACK = 8
MIN_DESCENT_SAMPLES = 3
MAX_DESCENT_RATE_MPS = 0.5
MIN_DESCENT_RATE_MPS = -0.05
MAX_APPROACH_DIST_KM = 40.0
APPROACH_AGL_M = 2000.0 * 0.3048  # ~2000 ft


def _with_ts(states: DataFrame) -> DataFrame:
    if "ts" in states.columns:
        return states
    if "fetched_at" in states.columns:
        return states.withColumn("ts", F.col("fetched_at"))
    raise ValueError("states need ``ts`` or ``fetched_at``")


def _with_descent_flag(states: DataFrame) -> DataFrame:
    """Mark rows that follow a gradual descent over recent samples."""
    w = (
        Window.partitionBy("icao24")
        .orderBy("ts")
        .rowsBetween(-DESCENT_LOOKBACK, 0)
    )
    hist = (
        states.filter(F.col("baro_altitude").isNotNull())
        .withColumn("prev_alt", F.lag("baro_altitude").over(Window.partitionBy("icao24").orderBy("ts")))
        .withColumn(
            "descending_step",
            (F.col("prev_alt").isNotNull())
            & (F.col("baro_altitude") <= F.col("prev_alt") + 5.0)
            & (F.coalesce(F.col("vertical_rate"), F.lit(0.0)) <= MAX_DESCENT_RATE_MPS),
        )
        .withColumn("avg_vr", F.avg(F.coalesce(F.col("vertical_rate"), F.lit(0.0))).over(w))
        .withColumn("descent_counts", F.sum(F.col("descending_step").cast("int")).over(w))
        .withColumn(
            "gradually_descending",
            (F.col("descent_counts") >= MIN_DESCENT_SAMPLES)
            & (F.col("avg_vr") <= MIN_DESCENT_RATE_MPS),
        )
        .drop("prev_alt", "descending_step", "avg_vr", "descent_counts")
    )
    return hist


def _approach_assignments(states: DataFrame, airports: DataFrame) -> DataFrame:
    """Rows in approach: AGL < 2000 ft vs airport elev, descending, nearest airport."""
    apt = airports.select(
        F.col("icao").alias("_apt_icao"),
        F.col("lat").alias("_apt_lat"),
        F.col("lon").alias("_apt_lon"),
        F.coalesce(F.col("elev_m"), F.lit(0.0)).alias("_apt_elev_m"),
    )
    paired = states.crossJoin(F.broadcast(apt)).withColumn(
        "_dist_km",
        _haversine_km(F.col("latitude"), F.col("longitude"), F.col("_apt_lat"), F.col("_apt_lon")),
    ).withColumn(
        "_agl_m", F.coalesce(F.col("baro_altitude"), F.lit(0.0)) - F.col("_apt_elev_m")
    )
    approach = paired.filter(
        (F.col("_dist_km") <= MAX_APPROACH_DIST_KM)
        & (F.col("_agl_m") <= APPROACH_AGL_M)
        & (F.col("_agl_m") >= -50.0)
        & F.coalesce(F.col("gradually_descending"), F.lit(False))
    )
    w = Window.partitionBy("icao24", "ts").orderBy(F.col("_dist_km").asc())
    nearest = (
        approach.withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .select(
            "icao24",
            "ts",
            F.col("_apt_icao").alias("approach_airport"),
            F.col("_dist_km").alias("approach_dist_km"),
            F.col("_agl_m").alias("approach_agl_m"),
        )
    )
    return nearest


def _flight_segments(states: DataFrame) -> DataFrame:
    """Split each icao24 track into flights when the time gap is large."""
    w = Window.partitionBy("icao24").orderBy("ts")
    ordered = states.withColumn("prev_ts", F.lag("ts").over(w))
    return (
        ordered.withColumn(
            "new_flight",
            F.col("prev_ts").isNull()
            | ((F.col("ts") - F.col("prev_ts")) > FLIGHT_GAP_SEC),
        )
        .withColumn(
            "flight_seg",
            F.sum(F.col("new_flight").cast("int")).over(
                w.rowsBetween(Window.unboundedPreceding, 0)
            ),
        )
        .drop("prev_ts", "new_flight")
    )


def enrich_arrivals(states: DataFrame, airports: DataFrame) -> DataFrame:
    """Return airborne snapshots labelled with ``arr_airport`` and ``actual_arrival_ts``.

    Landing time = last timestamp of a flight segment that entered approach
    to an airport (low AGL + descending near that airport).
    """
    base = _with_ts(states).filter(
        F.col("latitude").isNotNull() & F.col("longitude").isNotNull() & F.col("ts").isNotNull()
    )
    with_descent = _with_descent_flag(base)
    approach = _approach_assignments(with_descent, airports)
    segmented = _flight_segments(with_descent).join(approach, on=["icao24", "ts"], how="left")

    flight_w = Window.partitionBy("icao24", "flight_seg")
    labelled = (
        segmented.withColumn(
            "arr_airport",
            F.last("approach_airport", ignorenulls=True).over(
                flight_w.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
            ),
        )
        .withColumn("actual_arrival_ts", F.max("ts").over(flight_w))
        .filter(F.col("arr_airport").isNotNull())
        .filter(~F.coalesce(F.col("on_ground"), F.lit(False)))
    )
    return labelled
