import math
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

# Exact schema matching producer.py STATE_FIELDS + fetched_at
OPENSKY_SCHEMA = StructType([
    StructField("icao24", StringType(), True),
    StructField("callsign", StringType(), True),
    StructField("origin_country", StringType(), True),
    StructField("time_position", LongType(), True),
    StructField("last_contact", LongType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("baro_altitude", DoubleType(), True),
    StructField("on_ground", BooleanType(), True),
    StructField("velocity", DoubleType(), True),
    StructField("true_track", DoubleType(), True),
    StructField("vertical_rate", DoubleType(), True),
    # OpenSky sends an int array (or null), not a string.
    StructField("sensors", ArrayType(IntegerType()), True),
    StructField("geo_altitude", DoubleType(), True),
    StructField("squawk", StringType(), True),
    StructField("spi", BooleanType(), True),
    StructField("position_source", LongType(), True),
    StructField("category", LongType(), True),
    StructField("fetched_at", LongType(), True),
])

# Stockholm Arlanda Airport (ARN) Coordinates
ARN_LAT = 59.6519
ARN_LON = 17.9186

# Pre-compute static radian values in Python to avoid PySpark Column conversion errors
ARN_LAT_RAD = math.radians(ARN_LAT)
COS_ARN_LAT = math.cos(ARN_LAT_RAD)


def compute_features(df: DataFrame) -> DataFrame:
    """Filter airborne state vectors and compute spatial/kinematic features."""
    # Filter out grounded aircraft or records missing positional telemetry
    filtered_df = df.filter(
        (F.col("on_ground") == False)
        & F.col("latitude").isNotNull()
        & F.col("longitude").isNotNull()
    )

    # Haversine distance formula to target airport (in kilometers)
    dlat = F.radians(F.col("latitude") - ARN_LAT)
    dlon = F.radians(F.col("longitude") - ARN_LON)
    a = (
        F.sin(dlat / 2) ** 2
        + F.lit(COS_ARN_LAT)
        * F.cos(F.radians(F.col("latitude")))
        * F.sin(dlon / 2) ** 2
    )
    c = 2 * F.atan2(F.sqrt(a), F.sqrt(1 - a))
    dist_km = 6371.0 * c

    # Handle null kinematic values with defaults
    altitude = F.coalesce(F.col("baro_altitude"), F.lit(0.0))
    velocity = F.coalesce(F.col("velocity"), F.lit(0.0))
    v_rate = F.coalesce(F.col("vertical_rate"), F.lit(0.0))

    # Derived physical indicators
    descent_gradient = altitude / (dist_km + 1e-5)

    return (
        filtered_df.withColumn("dist_to_airport_km", dist_km)
        .withColumn("clean_altitude", altitude)
        .withColumn("clean_velocity", velocity)
        .withColumn("clean_vertical_rate", v_rate)
        .withColumn("descent_gradient", descent_gradient)
    )


def add_proxy_label(df: DataFrame) -> DataFrame:
    """Add a kinematics-based proxy delay label for continuous-learning demos.

    This is not schedule ground truth. It synthesizes ``actual_delay_min`` from
    distance, speed, altitude, and vertical rate so the GBT pipeline can train
    without joining flight schedules.
    """
    # Floor speed so ETA stays finite for near-hovering tracks
    speed_km_h = F.greatest(F.col("clean_velocity") * 3.6, F.lit(50.0))
    eta_min = (F.col("dist_to_airport_km") / speed_km_h) * 60.0
    altitude_term = F.col("clean_altitude") / 300.0
    climb_term = F.when(
        F.col("clean_vertical_rate") > 0,
        F.col("clean_vertical_rate") * 0.1,
    ).otherwise(F.lit(0.0))
    proxy = F.greatest(
        F.lit(0.0),
        eta_min * 0.15 + altitude_term * 0.05 + climb_term,
    )
    return df.withColumn("actual_delay_min", proxy)


# Columns produced by ``aggregate_training_features`` (excluding the label).
AGGREGATED_FEATURE_COLS = [
    "n_samples",
    "avg_dist_to_airport_km",
    "min_dist_to_airport_km",
    "max_dist_to_airport_km",
    "avg_altitude",
    "std_altitude",
    "avg_velocity",
    "std_velocity",
    "avg_vertical_rate",
    "avg_descent_gradient",
    "min_descent_gradient",
]


def aggregate_training_features(df: DataFrame) -> DataFrame:
    """Collapse per-snapshot rows into one aggregated training row per aircraft.

    Groups by ``icao24`` and computes mean/min/max/stddev track statistics that
    the GBT model uses as features. The label is the mean proxy delay over the
    track. Stddev columns are filled with 0 when a track has a single sample.
    """
    aggregated = (
        df.groupBy("icao24")
        .agg(
            F.count(F.lit(1)).alias("n_samples"),
            F.avg("dist_to_airport_km").alias("avg_dist_to_airport_km"),
            F.min("dist_to_airport_km").alias("min_dist_to_airport_km"),
            F.max("dist_to_airport_km").alias("max_dist_to_airport_km"),
            F.avg("clean_altitude").alias("avg_altitude"),
            F.stddev_samp("clean_altitude").alias("std_altitude"),
            F.avg("clean_velocity").alias("avg_velocity"),
            F.stddev_samp("clean_velocity").alias("std_velocity"),
            F.avg("clean_vertical_rate").alias("avg_vertical_rate"),
            F.avg("descent_gradient").alias("avg_descent_gradient"),
            F.min("descent_gradient").alias("min_descent_gradient"),
            F.avg("actual_delay_min").alias("actual_delay_min"),
        )
        .na.fill(0.0, ["std_altitude", "std_velocity"])
    )
    return aggregated.filter(F.col("n_samples") >= 1)


def prepare_training_dataset(df: DataFrame) -> DataFrame:
    """Featureize (if needed), label (if needed), then aggregate for training."""
    working = df
    if "dist_to_airport_km" not in working.columns:
        working = compute_features(working)
    if "actual_delay_min" not in working.columns:
        working = add_proxy_label(working)
    return aggregate_training_features(working)