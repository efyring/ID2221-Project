"""Real-time ETA: Kafka -> inbound European hubs -> model -> Cassandra / parquet."""

from __future__ import annotations

import logging
import os
from typing import Optional

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from ml.airport_alt import DEFAULT_DECISION_HEIGHT_AGL_M, NEAR_HUB_AGL_M
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


def inbound_condition(
    hub_lat,
    hub_lon,
    max_km: float,
    max_angle: float,
    hub_elev_m=None,
    hub_dh_m=None,
):
    """Inbound / approaching this hub; landed once baro ≤ decision height MSL."""
    elev = hub_elev_m if hub_elev_m is not None else F.lit(0.0)
    dh = hub_dh_m if hub_dh_m is not None else (elev + F.lit(DEFAULT_DECISION_HEIGHT_AGL_M))
    dist = _haversine_km(F.col("latitude"), F.col("longitude"), hub_lat, hub_lon)
    brg = _bearing_deg(F.col("latitude"), F.col("longitude"), hub_lat, hub_lon)
    off = F.abs(F.pmod(F.col("true_track") - brg + 540.0, F.lit(360.0)) - 180.0)
    baro = F.coalesce(F.col("baro_altitude"), F.lit(0.0))
    agl = baro - elev
    airborne = ~F.coalesce(F.col("on_ground"), F.lit(False))
    has_pos = F.col("latitude").isNotNull() & F.col("longitude").isNotNull()
    # At/below decision height MSL near the field ⇒ considered landed.
    landed = (baro <= dh) & (dist <= 40.0)
    fast_enough = F.coalesce(F.col("velocity"), F.lit(0.0)) >= 55.0
    heading_in = (dist > 60.0) & (dist <= max_km) & (off <= max_angle)
    # Close to the hub, low and not climbing => inbound
    approaching = (
        (dist <= 60.0)
        & (agl <= NEAR_HUB_AGL_M)
        & (baro > dh)
        & (F.coalesce(F.col("vertical_rate"), F.lit(0.0)) <= 0.5)
    )
    return (
        airborne & has_pos & ~landed & (heading_in | approaching)
        & (fast_enough | approaching),
        dist,
    )


def sticky_condition(hub_lat, hub_lon, max_km: float, hub_dh_m):
    """Keep a flight's earlier hub while it is plausibly still going there: inside the
    terminal area, or pointing roughly at the hub. Released once it is flying away from it."""
    dist = _haversine_km(F.col("latitude"), F.col("longitude"), hub_lat, hub_lon)
    brg = _bearing_deg(F.col("latitude"), F.col("longitude"), hub_lat, hub_lon)
    off = F.abs(F.pmod(F.col("true_track") - brg + 540.0, F.lit(360.0)) - 180.0)
    airborne = ~F.coalesce(F.col("on_ground"), F.lit(False))
    has_pos = F.col("latitude").isNotNull() & F.col("longitude").isNotNull()
    landed = (F.coalesce(F.col("baro_altitude"), F.lit(0.0)) <= hub_dh_m) & (dist <= 40.0)
    return airborne & has_pos & ~landed & (dist <= max_km) & ((dist <= 150.0) | (off <= 90.0)), dist


class HubAssignments:
    """Remembers which hub each aircraft was last assigned to, so the target does not
    flip between airports from one batch to the next."""

    MAX_AGE_S = 30 * 60

    def __init__(self) -> None:
        self._by_icao: dict[str, tuple[str, int]] = {}

    def as_df(self, spark) -> DataFrame:
        return spark.createDataFrame(
            [(k, v[0]) for k, v in self._by_icao.items()],
            "icao24 string, sticky_airport string",
        )

    def update(self, preds: DataFrame) -> None:
        rows = preds.select("icao24", "arr_airport", "ts").collect()
        for r in rows:
            self._by_icao[r["icao24"]] = (r["arr_airport"], int(r["ts"]))
        if rows:
            newest = max(int(r["ts"]) for r in rows)
            self._by_icao = {
                k: v for k, v in self._by_icao.items() if newest - v[1] <= self.MAX_AGE_S
            }


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
        F.col("elev_m").alias("hub_elev_m"),
        F.col("decision_height_m").alias("hub_dh_m"),
    )
    if raw:
        codes = [c.strip().upper() for c in raw.split(",") if c.strip()]
        return hubs.filter(F.col("hub_icao").isin(codes))
    # Default: airports inside a Europe box (the file may also hold US/Gulf airports).
    return hubs.filter(
        F.col("hub_lat").between(34.0, 72.0) & F.col("hub_lon").between(-25.0, 45.0)
    )


def live_predictions(
    parsed: DataFrame,
    model: PipelineModel,
    airports: DataFrame,
    hub_lat: float,
    hub_lon: float,
    arr_airport: str,
    max_km: float,
    max_angle: float,
    hub_elev_m: float = 0.0,
    hub_dh_m: Optional[float] = None,
) -> DataFrame:
    """Single-hub path (kept for predict_opensky / demos)."""
    dh = hub_dh_m if hub_dh_m is not None else hub_elev_m + DEFAULT_DECISION_HEIGHT_AGL_M
    cond, _ = inbound_condition(
        F.lit(hub_lat), F.lit(hub_lon), max_km, max_angle,
        hub_elev_m=F.lit(hub_elev_m), hub_dh_m=F.lit(dh),
    )
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
    sticky: Optional[DataFrame] = None,
) -> DataFrame:
    """Assign each aircraft to one hub and score it.

    ``sticky`` (icao24, sticky_airport) holds earlier assignments. If an aircraft's
    earlier hub still looks plausible it keeps it; otherwise it goes to the closest
    hub it is inbound toward.
    """
    paired = parsed.crossJoin(F.broadcast(hubs))
    cond, dist = inbound_condition(
        F.col("hub_lat"), F.col("hub_lon"), max_km, max_angle,
        hub_elev_m=F.col("hub_elev_m"), hub_dh_m=F.col("hub_dh_m"),
    )
    candidates = paired.filter(cond).withColumn("_dist_hub", dist).withColumn("_prio", F.lit(1))
    if sticky is not None:
        kept = (
            parsed.join(F.broadcast(sticky), "icao24")
            .join(F.broadcast(hubs), F.col("sticky_airport") == F.col("hub_icao"))
            .drop("sticky_airport")
        )
        keep_cond, keep_dist = sticky_condition(
            F.col("hub_lat"), F.col("hub_lon"), max_km, F.col("hub_dh_m")
        )
        kept = kept.filter(keep_cond).withColumn("_dist_hub", keep_dist).withColumn("_prio", F.lit(0))
        candidates = candidates.unionByName(kept)
    inbound = (
        candidates.withColumn(
            "_rn",
            F.row_number().over(
                Window.partitionBy("icao24").orderBy(F.col("_prio").asc(), F.col("_dist_hub").asc())
            ),
        )
        .filter(F.col("_rn") == 1)
        .withColumn("arr_airport", F.col("hub_icao"))
        .withColumn("ts", F.col("fetched_at"))
        .withColumn("callsign", F.trim(F.col("callsign")))
        .drop("hub_icao", "hub_lat", "hub_lon", "hub_elev_m", "hub_dh_m", "_dist_hub", "_prio", "_rn")
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
    assignments = HubAssignments()

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
                raw_batch, model, airports, hubs, max_km, max_angle,
                sticky=assignments.as_df(raw_batch.sparkSession),
            ).persist()
        else:
            hub = hubs.first()
            preds = live_predictions(
                raw_batch, model, airports,
                float(hub["hub_lat"]), float(hub["hub_lon"]), hub["hub_icao"],
                max_km, max_angle,
                hub_elev_m=float(hub["hub_elev_m"] or 0.0),
                hub_dh_m=float(hub["hub_dh_m"]),
            ).persist()
        try:
            n_raw, n_inbound = raw_batch.count(), preds.count()
            if multi:
                assignments.update(preds)
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