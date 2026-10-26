"""Read live ETA predictions from Cassandra for the Streamlit UI."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

DISPLAY_TZ = os.getenv("DISPLAY_TZ", "Europe/Berlin")


def cassandra_available(host: str | None = None, port: int = 9042) -> bool:
    host = host or os.getenv("CASSANDRA_HOST")
    if not host:
        return False
    try:
        from cassandra.cluster import Cluster

        cluster = Cluster([host], port=int(os.getenv("CASSANDRA_PORT", port)))
        session = cluster.connect()
        session.execute("SELECT keyspace_name FROM system_schema.keyspaces LIMIT 1")
        cluster.shutdown()
        return True
    except Exception:
        return False


def load_eta_latest(
    host: str | None = None,
    keyspace: str = "flights",
    arr_airport: str | None = None,
) -> pd.DataFrame:
    """Newest prediction per aircraft (optionally filtered to one arrival airport)."""
    from cassandra.cluster import Cluster

    host = host or os.getenv("CASSANDRA_HOST", "localhost")
    port = int(os.getenv("CASSANDRA_PORT", "9042"))
    keyspace = os.getenv("CASSANDRA_KEYSPACE", keyspace)
    cluster = Cluster([host], port=port)
    session = cluster.connect(keyspace)
    try:
        cols = (
            "arr_airport, icao24, callsign, ts, latitude, longitude, true_track, "
            "dist_to_arr_km, altitude, velocity, vertical_rate, "
            "pred_remaining_min, pred_landing_ts, pred_landing_time"
        )
        if arr_airport:
            rows = session.execute(
                f"SELECT {cols} FROM eta_latest WHERE arr_airport = %s",
                [arr_airport],
            )
        else:
            rows = session.execute(f"SELECT {cols} FROM eta_latest")
        data = [dict(r._asdict()) for r in rows]
    finally:
        cluster.shutdown()

    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "pred_landing_time" in df.columns:
        df["pred_landing_time"] = pd.to_datetime(df["pred_landing_time"], utc=True)
    if "pred_landing_ts" in df.columns and "pred_landing_time" not in df.columns:
        df["pred_landing_time"] = pd.to_datetime(df["pred_landing_ts"], unit="s", utc=True)
    if {"pred_remaining_min", "dist_to_arr_km"}.issubset(df.columns):
        # Approximate naive ETA for delay display when stream didn't persist it.
        speed = df.get("velocity", pd.Series(0.0, index=df.index)).fillna(0.0).clip(lower=50 / 3.6)
        df["naive_eta_min"] = df["dist_to_arr_km"] / (speed * 3.6) * 60.0
        df["delay_vs_naive_min"] = df["pred_remaining_min"] - df["naive_eta_min"]
    return df


def load_eta_history(icao24: str, host: str | None = None, keyspace: str = "flights",
                     limit: int = 50) -> pd.DataFrame:
    """Past predictions for one aircraft (newest first)."""
    from cassandra.cluster import Cluster

    host = host or os.getenv("CASSANDRA_HOST", "localhost")
    port = int(os.getenv("CASSANDRA_PORT", "9042"))
    keyspace = os.getenv("CASSANDRA_KEYSPACE", keyspace)
    cluster = Cluster([host], port=port)
    session = cluster.connect(keyspace)
    try:
        rows = session.execute(
            "SELECT icao24, ts, arr_airport, callsign, dist_to_arr_km, altitude, "
            "velocity, vertical_rate, pred_remaining_min, pred_landing_ts, pred_landing_time "
            "FROM eta_history WHERE icao24 = %s LIMIT %s",
            [icao24, limit],
        )
        data = [dict(r._asdict()) for r in rows]
    finally:
        cluster.shutdown()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "pred_landing_time" in df.columns:
        df["pred_landing_time"] = pd.to_datetime(df["pred_landing_time"], utc=True)
    df["pred_at_local"] = (
        pd.to_datetime(df["ts"], unit="s", utc=True)
        .dt.tz_convert(DISPLAY_TZ)
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    return df


def list_arrival_airports(host: str | None = None, keyspace: str = "flights") -> list[str]:
    df = load_eta_latest(host=host, keyspace=keyspace)
    if df.empty or "arr_airport" not in df.columns:
        return []
    return sorted(df["arr_airport"].dropna().unique().tolist())
