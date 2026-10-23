"""Real-time ETA: Kafka -> keep flights inbound to the hub -> model -> landing time."""

import logging
import os

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame
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


def inbound_condition(hub_lat: float, hub_lon: float, max_km: float, max_angle: float):
    """Column expression for "probably landing at the hub". Live states carry no
    destination, so this is a heuristic:

    * airborne, has a position, flying at airliner speed (>= 55 m/s), and either
    * 60 km to max_km away and heading toward the hub (within max_angle degrees), or
    * within 60 km, low (< 5000 m) and not climbing. Heading is ignored this close
      because it is too noisy; this drops departures and overflights.
    """
    dist = _haversine_km(F.col("latitude"), F.col("longitude"), F.lit(hub_lat), F.lit(hub_lon))
    brg = _bearing_deg(F.col("latitude"), F.col("longitude"), F.lit(hub_lat), F.lit(hub_lon))
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
    return airborne & has_pos & fast_enough & (heading_in | low_and_close)


CASSANDRA_COLUMNS = [
    "arr_airport", "icao24", "callsign", "ts",
    "dist_to_arr_km", "altitude", "velocity", "vertical_rate",
    "pred_remaining_min", "pred_landing_ts", "pred_landing_time",
]


def write_cassandra(batch_df: DataFrame, keyspace: str) -> None:
    """Upsert into flights.eta_latest (one row per aircraft) and append to eta_history."""
    out = batch_df.select(*CASSANDRA_COLUMNS)
    for table in ("eta_latest", "eta_history"):
        (
            out.write.format("org.apache.spark.sql.cassandra")
            .options(keyspace=keyspace, table=table)
            .mode("append")
            .save()
        )


def live_predictions(parsed: DataFrame, model: PipelineModel, airports: DataFrame,
                     hub_lat: float, hub_lon: float, arr_airport: str,
                     max_km: float, max_angle: float) -> DataFrame:
    """Works on batch and streaming DataFrames with OPENSKY_SCHEMA columns."""
    inbound = (
        parsed.filter(inbound_condition(hub_lat, hub_lon, max_km, max_angle))
        .withColumn("arr_airport", F.lit(arr_airport))
        .withColumn("ts", F.col("fetched_at"))
        .withColumn("callsign", F.trim(F.col("callsign")))  # OpenSky pads callsigns
    )
    return predict(model, inbound, airports)


def check_cassandra(spark, keyspace: str) -> None:
    """Fail fast with a readable message if the keyspace/tables are missing."""
    for table in ("eta_latest", "eta_history"):
        try:
            spark.read.format("org.apache.spark.sql.cassandra").options(
                keyspace=keyspace, table=table
            ).load()
        except Exception as exc:
            first_line = str(exc).strip().splitlines()[0]
            raise SystemExit(
                f"Cassandra table {keyspace}.{table} is not usable: {first_line}\n"
                "Run `make cassandra` and check `docker compose logs cassandra-init`."
            ) from exc


def make_batch_handler(model, airports, hub_lat, hub_lon, config, max_km, max_angle, out_path):
    use_cassandra = config.cassandra_host is not None

    def handle(raw_batch: DataFrame, batch_id: int) -> None:
        if raw_batch.isEmpty():
            return
        raw_batch = raw_batch.persist()
        preds = live_predictions(
            raw_batch, model, airports, hub_lat, hub_lon, config.arrival_airport, max_km, max_angle
        ).persist()  # reused below; avoid re-running the model
        try:
            n_raw, n_inbound = raw_batch.count(), preds.count()
            if n_inbound == 0:
                LOGGER.info("Batch %s: %d states received, none inbound", batch_id, n_raw)
                return
            farthest = preds.agg(F.max("dist_to_arr_km")).first()[0]
            LOGGER.info(
                "Batch %s: %d states received, %d inbound (farthest %.0f km from %s)",
                batch_id, n_raw, n_inbound, farthest, config.arrival_airport,
            )
            preds.orderBy("pred_remaining_min").show(25, truncate=False)
            if use_cassandra:
                write_cassandra(preds, config.cassandra_keyspace)
                LOGGER.info("Batch %s written to Cassandra keyspace %s", batch_id, config.cassandra_keyspace)
            if out_path:
                preds.write.mode("append").parquet(out_path)
        finally:
            preds.unpersist()
            raw_batch.unpersist()

    return handle


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    config = offline_config()
    use_cassandra = config.cassandra_host is not None
    spark = create_spark_session(
        "OpenSky-ETA-Stream", enable_kafka=True, enable_cassandra=use_cassandra, config=config
    )

    airports = load_airports(spark, config.airports_path)
    hub = airports.filter(F.col("icao") == config.arrival_airport).first()
    if hub is None:
        raise SystemExit(f"{config.arrival_airport} not found in {config.airports_path}")
    model = PipelineModel.load(config.model_output_path)
    if use_cassandra:
        check_cassandra(spark, config.cassandra_keyspace)

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", config.kafka_bootstrap_servers)
        .option("subscribe", config.kafka_topic)
        .option("startingOffsets", "latest")
        .load()
    )
    parsed = (
        raw.select(F.from_json(F.col("value").cast("string"), OPENSKY_SCHEMA).alias("d"))
        .select("d.*")
        .filter(F.col("icao24").isNotNull())
    )
    handler = make_batch_handler(
        model, airports, hub["lat"], hub["lon"], config,
        max_km=float(os.getenv("LIVE_MAX_KM", "1500")),
        max_angle=float(os.getenv("LIVE_MAX_ANGLE", "60")),
        out_path=os.getenv("PREDICTIONS_PATH"),
    )
    query = (
        parsed.writeStream.foreachBatch(handler)
        .trigger(processingTime=f"{os.getenv('TRIGGER_SECONDS', '15')} seconds")
        .start()
    )
    query.awaitTermination()


if __name__ == "__main__":
    main()