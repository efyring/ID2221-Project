"""Past landings / delays for an aircraft, from historical training parquet."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

HUB_LAT = 50.0379
HUB_LON = 8.5622
DISPLAY_TZ = os.getenv("DISPLAY_TZ", "Europe/Berlin")
DEFAULT_HISTORY = os.environ.get(
    "TRAIN_DATA_PATH", "./data/synthetic/historical_opensky"
)
LANDING_CACHE = Path(
    os.environ.get("LANDING_HISTORY_PATH", "./data/synthetic/landing_history.parquet")
)


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def _fmt_ts(ts: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(ts, unit="s", utc=True)
        .dt.tz_convert(DISPLAY_TZ)
        .dt.strftime("%Y-%m-%d %H:%M")
    )


def build_landing_history(history_path: str | Path = DEFAULT_HISTORY) -> pd.DataFrame:
    """One row per completed flight: landing time + mid-flight delay vs naive ETA."""
    cols = [
        "icao24",
        "callsign",
        "arr_airport",
        "ts",
        "latitude",
        "longitude",
        "baro_altitude",
        "velocity",
        "actual_arrival_ts",
        "on_ground",
    ]
    raw = pd.read_parquet(history_path, columns=cols)
    airborne = raw.loc[~raw["on_ground"].fillna(False)].copy()
    if airborne.empty:
        return pd.DataFrame()

    airborne["flight_key"] = (
        airborne["icao24"].astype(str)
        + "_"
        + airborne["actual_arrival_ts"].astype("int64").astype(str)
    )
    # Mid-track snapshot ≈ halfway through the airborne samples of that flight.
    airborne["ord"] = airborne.groupby("flight_key").cumcount()
    counts = airborne.groupby("flight_key")["ord"].transform("size")
    mid = airborne.loc[airborne["ord"] == (counts // 2)].copy()

    dist = _haversine_km(mid["latitude"], mid["longitude"], HUB_LAT, HUB_LON)
    speed_kmh = np.maximum(mid["velocity"].fillna(0.0).to_numpy() * 3.6, 50.0)
    naive = dist / speed_kmh * 60.0
    true_rem = (mid["actual_arrival_ts"] - mid["ts"]) / 60.0
    first_ts = airborne.groupby("flight_key")["ts"].min()
    duration = (mid.set_index("flight_key")["actual_arrival_ts"] - first_ts) / 60.0

    out = pd.DataFrame(
        {
            "icao24": mid["icao24"].to_numpy(),
            "callsign": mid["callsign"].astype(str).str.strip().to_numpy(),
            "arr_airport": mid["arr_airport"].to_numpy(),
            "landing_ts": mid["actual_arrival_ts"].astype("int64").to_numpy(),
            "landing_local": _fmt_ts(mid["actual_arrival_ts"]).to_numpy(),
            "flight_duration_min": duration.reindex(mid["flight_key"]).to_numpy(),
            "mid_true_remaining_min": true_rem.to_numpy(),
            "mid_naive_eta_min": naive.to_numpy(),
            # Positive ⇒ actual remaining exceeded distance/speed baseline at mid-flight.
            "delay_vs_naive_min": (true_rem - naive).to_numpy(),
            "mid_altitude_m": mid["baro_altitude"].fillna(0.0).to_numpy(),
            "mid_dist_km": dist.to_numpy(),
        }
    )
    return out.sort_values(["icao24", "landing_ts"]).reset_index(drop=True)


def ensure_landing_history(
    history_path: str | Path = DEFAULT_HISTORY,
    cache_path: Path = LANDING_CACHE,
    force: bool = False,
) -> pd.DataFrame:
    """Load cached landing summaries, building them once from historical parquet."""
    hist = Path(history_path)
    if (
        not force
        and cache_path.exists()
        and hist.exists()
        and cache_path.stat().st_mtime >= hist.stat().st_mtime
    ):
        return pd.read_parquet(cache_path)
    if not hist.exists():
        return pd.DataFrame()
    out = build_landing_history(hist)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    return out


def history_for_aircraft(landings: pd.DataFrame, icao24: str) -> pd.DataFrame:
    if landings.empty:
        return landings
    return landings.loc[landings["icao24"] == icao24].sort_values(
        "landing_ts", ascending=False
    )
