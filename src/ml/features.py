from pyspark.sql import DataFrame, SparkSession
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
    StructField("sensors", ArrayType(IntegerType()), True),
    StructField("geo_altitude", DoubleType(), True),
    StructField("squawk", StringType(), True),
    StructField("spi", BooleanType(), True),
    StructField("position_source", LongType(), True),
    StructField("category", LongType(), True),
    StructField("fetched_at", LongType(), True),
])

LABEL_COL = "remaining_min"

FEATURE_COLS = [
    "dist_to_arr_km",
    "naive_eta_min",
    "altitude",
    "velocity",
    "vertical_rate",
    "hour_utc",
    "day_of_week",
]


def load_airports(spark: SparkSession, path: str) -> DataFrame:
    """Read the airport lookup CSV (columns: icao, lat, lon)."""
    return spark.read.csv(path, header=True, inferSchema=True).select("icao", "lat", "lon")


def _haversine_km(lat1, lon1, lat2, lon2):
    dlat = F.radians(lat2 - lat1)
    dlon = F.radians(lon2 - lon1)
    a = (
        F.sin(dlat / 2) ** 2
        + F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) * F.sin(dlon / 2) ** 2
    )
    return 2 * 6371.0 * F.asin(F.sqrt(F.least(F.lit(1.0), a)))


def add_features(df: DataFrame, airports: DataFrame) -> DataFrame:
    """Add model features. Rows with unknown arrival airport or no position are dropped."""
    out = df.filter(
        F.col("latitude").isNotNull()
        & F.col("longitude").isNotNull()
        & F.col("ts").isNotNull()
    )
    if "on_ground" in df.columns:
        out = out.filter(~F.coalesce(F.col("on_ground"), F.lit(False)))

    arr = airports.select(
        F.col("icao").alias("arr_airport"),
        F.col("lat").alias("arr_lat"),
        F.col("lon").alias("arr_lon"),
    )
    out = out.join(F.broadcast(arr), "arr_airport")

    velocity = F.coalesce(F.col("velocity"), F.lit(0.0))
    speed_kmh = F.greatest(velocity * 3.6, F.lit(50.0))  # floor keeps ETA finite
    dist_to_arr = _haversine_km(F.col("latitude"), F.col("longitude"), F.col("arr_lat"), F.col("arr_lon"))
    ts = F.timestamp_seconds("ts")

    return (
        out.withColumn("dist_to_arr_km", dist_to_arr)
        # Simple physics baseline: distance left / current ground speed.
        .withColumn("naive_eta_min", F.col("dist_to_arr_km") / speed_kmh * 60.0)
        .withColumn("altitude", F.coalesce(F.col("baro_altitude"), F.lit(0.0)))
        .withColumn("velocity", velocity)
        .withColumn("vertical_rate", F.coalesce(F.col("vertical_rate"), F.lit(0.0)))
        .withColumn("hour_utc", F.hour(ts).cast("double"))
        .withColumn("day_of_week", F.dayofweek(ts).cast("double"))
    )


def add_label(df: DataFrame) -> DataFrame:
    """Add ``remaining_min`` (training label) and ``flight_key`` (for leak-free splits)."""
    return (
        df.filter(F.col("actual_arrival_ts").isNotNull())
        .withColumn(LABEL_COL, (F.col("actual_arrival_ts") - F.col("ts")) / 60.0)
        .filter((F.col(LABEL_COL) >= 0) & (F.col(LABEL_COL) <= 24 * 60))
        .withColumn(
            "flight_key",
            F.concat_ws("_", F.col("icao24"), F.col("actual_arrival_ts").cast("string")),
        )
    )