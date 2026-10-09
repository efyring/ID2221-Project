import math
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
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
    StructField("sensors", StringType(), True),
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