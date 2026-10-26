"""Real-time ETA: Kafka -> inbound European hubs -> model -> Cassandra / parquet."""

import logging
import os

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from ml.features import OPENSKY_SCHEMA, _haversine_km, load_airports
from ml.predict import predict
from ml.train import offline_config
from spark.session import create_spark_session

LOGGER = logging.getLogger(__name__)


def _bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2, dl = F.radians(lat1), F.radians(lat2), F.radians(lon2 - lon1)
    x = F.sin(dl) * F.cos(p2)
    y = F.cos(p1) * F.sin(p2) - F.sin(p1) * F.cos(p2) * F.cos(dl)
    return F.pmod(F.degrees(F.atan2(x, y)) + 360.0, F.lit(360.0))


def inbound_condition(hub_lat, hub_lon, max_km: float, max_angle: float):
    """Column expression for "probably landing at this hub" (lat/lon may be literals or columns)."""
    dist = _haversine_km(F.col("latitude"), F.col("longitude"), hub_lat, hub_lon)
    brg = _bearing_deg(F.col("latitude"), F.col("longitude"), hub_lat, hub_lon)
    off = F.abs(F.pmod(F.col("true_track") - brg + 540.0, F.lit(360.0)) - 180.0)
    airborne = ~F.coalesce(F.col("on_ground"), F.lit(False))
    has_pos = F.col("latitude").isNotNull() & F.col("longitude").isNotNull()
    fast_enough = F.coalesce(F.col("velocity"), F.lit(0.0)) >= 55.0
    heading_in = (dist > 60.0) & (dist <= max_km) & (off <= max_angle)
    low_and_close = (
        (dist <= 60.0)
        & (F.coalesce(F.col("baro_altitude"), F.lit(0.0)) < 5000.0)
        & (F.coalesce(F.col("vertical_rate"), F.lit(0.0)) <= 0.5)
    )
    return airborne & has_pos & fast_enough & (heading_in | low_and_close), dist


CASSANDRA_COLUMNS = [
    "arr_airport", "icao24", "callsign", "ts",
    "latitude", "longitude", "true_track",
    "dist_to_arr_km", "altitude", "velocity", "vertical_rate",
    "pred_remaining_min", "pred_landing_ts", "pred_landing_time",
]


def write_cassandra(batch_df: DataFrame, keyspace: str) -> None:
    """Upsert into flights.eta_latest and append to eta_history."""
    for col in CASSANDRA_COLUMNS:
        if col not in batch_df.columns:
            batch_df = batch_df.withColumn(col, F.lit(None))
    out = batch_df.select(*CASSANDRA_COLUMNS)
    for table in ("eta_latest", "eta_history"):
        (
            out.write.format("org.apache.spark.sql.cassandra")
            .options(keyspace=keyspace, table=table)
            .mode("append")
            .save()
        )


def european_hubs(airports: DataFrame) -> DataFrame:
    """Hubs to score against. Override with ARRIVAL_AIRPORTS=EDDF,EHAM,LFPG,..."""
    raw = os.getenv("ARRIVAL_AIRPORTS", "").strip()
    hubs = airports.select(
        F.col("icao").alias("hub_icao"),
        F.col("lat").alias("hub_lat"),
        F.col("lon").alias("hub_lon"),
    )
    if raw:
        codes = [c.strip().upper() for c in raw.split(",") if c.strip()]
        return hubs.filter(F.col("hub_icao").isin(codes))
    # Default: European ICAO regions in our airports file (exclude US K***).
    return hubs.filter(~F.col("hub_icao").startswith("K"))


def live_predictions(
    parsed: DataFrame,
    model: PipelineModel,
    airports: DataFrame,
    hub_lat: float,
    hub_lon: float,
    arr_airport: str,
    max_km: float,
    max_angle: float,
) -> DataFrame:
    """Single-hub path (kept for predict_opensky / demos)."""
    cond, _ = inbound_condition(F.lit(hub_lat), F.lit(hub_lon), max_km, max_angle)
    inbound = (
        parsed.filter(cond)
        .withColumn("arr_airport", F.lit(arr_airport))
        .withColumn("ts", F.col("fetched_at"))
        .withColumn("callsign", F.trim(F.col("callsign")))
    )
    return predict(model, inbound, airports)


def live_predictions_multi(
    parsed: DataFrame,
    model: PipelineModel,
    airports: DataFrame,
    hubs: DataFrame,
    max_km: float,
    max_angle: float,
) -> DataFrame:
    """Assign each state to the closest European hub it is inbound toward."""
    paired = parsed.crossJoin(F.broadcast(hubs))
    cond, dist = inbound_condition(F.col("hub_lat"), F.col("hub_lon"), max_km, max_angle)
    inbound = (
        paired.filter(cond)
        .withColumn("_dist_hub", dist)
        .withColumn(
            "_rn",
            F.row_number().over(
                Window.partitionBy("icao24").orderBy(F.col("_dist_hub").asc())
            ),
        )
        .filter(F.col("_rn") == 1)
        .withColumn("arr_airport", F.col("hub_icao"))
        .withColumn("ts", F.col("fetched_at"))
        .withColumn("callsign", F.trim(F.col("callsign")))
        .drop("hub_icao", "hub_lat", "hub_lon", "_dist_hub", "_rn")
    )
    return predict(model, inbound, airports)


def check_cassandra(spark, keyspace: str) -> None:
    for table in ("eta_latest", "eta_history"):
        try:
            spark.read.format("org.apache.spark.sql.cassandra").options(
                keyspace=keyspace, table=table
            ).load()
        except Exception as exc:
            first_line = str(exc).strip().splitlines()[0]
            raise SystemExit(
                f"Cassandra table {keyspace}.{table} is not usable: {first_line}\n"
                "Run `make cassandra`."
            ) from exc


def _sync_cassandra_python(preds: DataFrame) -> int:
    """Upsert this micro-batch via cassandra-driver (no Spark Cassandra connector)."""
    from viz.sync_cassandra import sync

    pdf = preds.select(*[c for c in CASSANDRA_COLUMNS if c in preds.columns]).toPandas()
    return sync(pdf)


def make_batch_handler(model, airports, hubs, config, max_km, max_angle, out_path):
    # Prefer Python Cassandra sync when CASSANDRA_HOST is set (avoids Ivy jar downloads).
    use_cassandra_spark = (
        config.cassandra_host is not None
        and os.getenv("CASSANDRA_SPARK_SINK", "").lower() in {"1", "true", "yes"}
    )
    use_cassandra_python = (
        config.cassandra_host is not None and not use_cassandra_spark
    )
    multi = hubs.count() > 1

    def handle(raw_batch: DataFrame, batch_id: int) -> None:
        if raw_batch.isEmpty():
            return
        # One row per aircraft: keep the freshest OpenSky snapshot in this minute.
        w = Window.partitionBy("icao24").orderBy(F.col("fetched_at").desc())
        raw_batch = (
            raw_batch.withColumn("_rn", F.row_number().over(w))
            .filter(F.col("_rn") == 1)
            .drop("_rn")
            .persist()
        )
        if multi:
            preds = live_predictions_multi(
                raw_batch, model, airports, hubs, max_km, max_angle
            ).persist()
        else:
            hub = hubs.first()
            preds = live_predictions(
                raw_batch, model, airports,
                float(hub["hub_lat"]), float(hub["hub_lon"]), hub["hub_icao"],
                max_km, max_angle,
            ).persist()
        try:
            n_raw, n_inbound = raw_batch.count(), preds.count()
            if n_inbound == 0:
                LOGGER.info("Batch %s: %d states received, none inbound", batch_id, n_raw)
                return
            by_hub = {
                r["arr_airport"]: r["count"]
                for r in preds.groupBy("arr_airport").count().collect()
            }
            LOGGER.info(
                "Batch %s: %d unique aircraft → %d inbound %s",
                batch_id, n_raw, n_inbound, by_hub,
            )
            preds.orderBy("pred_remaining_min").show(15, truncate=False)
            if use_cassandra_spark:
                write_cassandra(preds, config.cassandra_keyspace)
                LOGGER.info("Batch %s → Cassandra.%s (Spark sink)", batch_id, config.cassandra_keyspace)
            elif use_cassandra_python:
                n = _sync_cassandra_python(preds)
                LOGGER.info("Batch %s → Cassandra (%d upserts, python driver)", batch_id, n)
            if out_path:
                # overwrite-style snapshot dir per batch is heavy; append keeps history
                preds.write.mode("append").parquet(out_path)
        finally:
            preds.unpersist()
            raw_batch.unpersist()

    return handle


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    config = offline_config()
    use_cassandra = config.cassandra_host is not None
    use_cassandra_spark = (
        use_cassandra
        and os.getenv("CASSANDRA_SPARK_SINK", "").lower() in {"1", "true", "yes"}
    )
    spark = create_spark_session(
        "OpenSky-ETA-Stream",
        enable_kafka=True,
        enable_cassandra=use_cassandra_spark,
        config=config,
    )

    airports = load_airports(spark, config.airports_path)
    hubs = european_hubs(airports).cache()
    n_hubs = hubs.count()
    if n_hubs == 0:
        raise SystemExit("No arrival hubs configured — check AIRPORTS_PATH / ARRIVAL_AIRPORTS")
    trigger_s = int(os.getenv("TRIGGER_SECONDS", "60"))
    LOGGER.info(
        "Scoring inbound traffic to %d hubs every %ds (Cassandra=%s)",
        n_hubs, trigger_s, "on" if use_cassandra else "off",
    )
    hubs.show(50, truncate=False)

    model = PipelineModel.load(config.model_output_path)
    if use_cassandra_spark:
        check_cassandra(spark, config.cassandra_keyspace)

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", config.kafka_bootstrap_servers)
        .option("subscribe", config.kafka_topic)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )
    parsed = (
        raw.select(F.from_json(F.col("value").cast("string"), OPENSKY_SCHEMA).alias("d"))
        .select("d.*")
        .filter(F.col("icao24").isNotNull())
    )
    handler = make_batch_handler(
        model, airports, hubs, config,
        max_km=float(os.getenv("LIVE_MAX_KM", "1500")),
        max_angle=float(os.getenv("LIVE_MAX_ANGLE", "60")),
        out_path=os.getenv("PREDICTIONS_PATH", "./data/live/stream_predictions"),
    )
    query = (
        parsed.writeStream.foreachBatch(handler)
        .option("checkpointLocation", os.getenv(
            "STREAM_CHECKPOINT_PATH", "./data/checkpoints/eta_stream"
        ))
        .trigger(processingTime=f"{trigger_s} seconds")
        .start()
    )
    query.awaitTermination()


if __name__ == "__main__":
    main()
