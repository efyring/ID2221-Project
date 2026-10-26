"""Load the newest stream parquet predictions into Cassandra (no Spark connector)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from cassandra.cluster import Cluster
from cassandra.query import BatchStatement, ConsistencyLevel, SimpleStatement

PRED_PATH = Path(os.getenv("PREDICTIONS_PATH", "./data/live/stream_predictions"))
HOST = os.getenv("CASSANDRA_HOST", "localhost")
PORT = int(os.getenv("CASSANDRA_PORT", "9042"))
KEYSPACE = os.getenv("CASSANDRA_KEYSPACE", "flights")


def _latest_parquet(root: Path) -> pd.DataFrame:
    files = sorted(root.rglob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise SystemExit(f"No parquet under {root} — is stream-live writing?")
    # Merge recent micro-batches; keep newest row per (arr_airport, icao24).
    frames = [pd.read_parquet(f) for f in files[:20]]
    df = pd.concat(frames, ignore_index=True)
    if "ts" in df.columns:
        df = df.sort_values("ts").drop_duplicates(["arr_airport", "icao24"], keep="last")
    return df


def sync(df: pd.DataFrame | None = None) -> int:
    df = df if df is not None else _latest_parquet(PRED_PATH)
    cluster = Cluster([HOST], port=PORT)
    session = cluster.connect(KEYSPACE)
    insert_latest = session.prepare(
        "INSERT INTO eta_latest ("
        "arr_airport, icao24, callsign, ts, latitude, longitude, true_track, "
        "dist_to_arr_km, altitude, velocity, vertical_rate, "
        "pred_remaining_min, pred_landing_ts, pred_landing_time"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    insert_hist = session.prepare(
        "INSERT INTO eta_history ("
        "icao24, ts, arr_airport, callsign, latitude, longitude, true_track, "
        "dist_to_arr_km, altitude, velocity, vertical_rate, "
        "pred_remaining_min, pred_landing_ts, pred_landing_time"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    n = 0
    batch = BatchStatement(consistency_level=ConsistencyLevel.ONE)
    batch_size = 50

    def _flush() -> None:
        nonlocal batch
        if batch:
            session.execute(batch)
            batch = BatchStatement(consistency_level=ConsistencyLevel.ONE)

    for _, r in df.iterrows():
        callsign = None if pd.isna(r.get("callsign")) else str(r.get("callsign"))
        landing = None
        if pd.notna(r.get("pred_landing_time")):
            landing = pd.to_datetime(r["pred_landing_time"], utc=True).to_pydatetime()
        vals = (
            str(r["arr_airport"]),
            str(r["icao24"]),
            callsign,
            int(r["ts"]) if pd.notna(r.get("ts")) else None,
            float(r["latitude"]) if pd.notna(r.get("latitude")) else None,
            float(r["longitude"]) if pd.notna(r.get("longitude")) else None,
            float(r["true_track"]) if pd.notna(r.get("true_track")) else None,
            float(r["dist_to_arr_km"]) if pd.notna(r.get("dist_to_arr_km")) else None,
            float(r["altitude"]) if pd.notna(r.get("altitude")) else None,
            float(r["velocity"]) if pd.notna(r.get("velocity")) else None,
            float(r["vertical_rate"]) if pd.notna(r.get("vertical_rate")) else None,
            float(r["pred_remaining_min"]) if pd.notna(r.get("pred_remaining_min")) else None,
            int(r["pred_landing_ts"]) if pd.notna(r.get("pred_landing_ts")) else None,
            landing,
        )
        batch.add(insert_latest, vals)
        batch.add(insert_hist, (vals[1], vals[3], vals[0], vals[2], *vals[4:]))
        n += 1
        if n % batch_size == 0:
            _flush()
    _flush()
    cluster.shutdown()
    print(f"Synced {n} rows → Cassandra {KEYSPACE}.eta_latest / eta_history")
    return n


if __name__ == "__main__":
    sync()
