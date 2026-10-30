"""Generate synthetic historical flights that all land at one big European hub."""

from __future__ import annotations

import argparse
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HUB = ("EDDF", 50.0379, 8.5622)  # Frankfurt

# (icao, lat, lon, weight): mostly short-haul feeders, a few long-haul.
ORIGINS = [
    ("EGLL", 51.4706, -0.4619, 6), ("LFPG", 49.0097, 2.5479, 6),
    ("EHAM", 52.3086, 4.7639, 5), ("LEMD", 40.4719, -3.5626, 5),
    ("LIRF", 41.8003, 12.2389, 4), ("LSZH", 47.4647, 8.5492, 5),
    ("LOWW", 48.1103, 16.5697, 5), ("EKCH", 55.6180, 12.6508, 4),
    ("ESSA", 59.6519, 17.9186, 4), ("ENGM", 60.1939, 11.1004, 3),
    ("EFHK", 60.3172, 24.9633, 3), ("EPWA", 52.1657, 20.9671, 3),
    ("LKPR", 50.1008, 14.2600, 4), ("LHBP", 47.4369, 19.2556, 3),
    ("LGAV", 37.9364, 23.9445, 3), ("LTFM", 41.2753, 28.7519, 3),
    ("LPPT", 38.7813, -9.1359, 3), ("EIDW", 53.4213, -6.2701, 3),
    ("EBBR", 50.9014, 4.4844, 3), ("LEBL", 41.2974, 2.0833, 4),
    ("LFMN", 43.6584, 7.2159, 2), ("EGKK", 51.1537, -0.1821, 3),
    ("KJFK", 40.6398, -73.7789, 1), ("KORD", 41.9742, -87.9073, 1),
    ("OMDB", 25.2528, 55.3644, 1),
]

EARTH_KM = 6371.0
DT_S = 30  # snapshot spacing


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(min(1.0, a)))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def destination_point(lat, lon, bearing, dist_km):
    """Point reached from (lat, lon) travelling dist_km along bearing (degrees)."""
    d, b = dist_km / EARTH_KM, math.radians(bearing)
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(
        math.sin(b) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2)
    )
    return math.degrees(p2), (math.degrees(l2) + 540) % 360 - 180


def congestion(hour: float) -> float:
    """0..1, peaks around 07 and 17 UTC (rush hours at the hub)."""
    return min(1.0, math.exp(-(((hour - 7) / 2.0) ** 2)) + math.exp(-(((hour - 17) / 2.5) ** 2)))


def simulate_flight(rng, origin, start_ts: int):
    """Return (list of snapshot dicts, actual_arrival_ts) for one flight."""
    icao, o_lat, o_lon, _ = origin
    h_lat, h_lon = HUB[1], HUB[2]
    route_km = haversine_km(o_lat, o_lon, h_lat, h_lon)
    out_bearing = bearing_deg(h_lat, h_lon, o_lat, o_lon)  # hub -> origin

    v_cruise = float(np.clip(rng.normal(235, 12), 200, 262))  # m/s
    wind = float(np.clip(rng.normal(1.0, 0.07), 0.8, 1.2))  # ground speed factor
    cruise_alt = float(rng.choice([9500, 10000, 10700, 11300]))
    climb_s = 1200.0
    d_desc_km = cruise_alt / 1000.0 * 20.0  # start of descent
    hold_r_km = 45.0

    # Holding probability depends on the (approximate) landing hour.
    approx_dur_h = route_km / (v_cruise * 3.6 * 0.92) + 0.25
    land_hour = ((start_ts / 3600.0) + approx_dur_h) % 24
    hold_s = 0.0
    if rng.random() < 0.04 + 0.40 * congestion(land_hour):
        hold_s = float(rng.exponential(480.0)) + 120.0

    def alt_at(d_km):
        if d_km >= d_desc_km:
            return cruise_alt
        if d_km > 30.0:
            return 1500.0 + (cruise_alt - 1500.0) * (d_km - 30.0) / (d_desc_km - 30.0)
        return 1500.0 * max(d_km, 0.0) / 30.0

    def speed_at(d_km):
        if d_km >= d_desc_km:
            return v_cruise
        if d_km > 30.0:
            return 140.0 + (v_cruise - 140.0) * (d_km - 30.0) / (d_desc_km - 30.0)
        return 75.0 + 65.0 * max(d_km, 0.0) / 30.0

    d, t, hold_left, held = route_km, 0.0, hold_s, False
    hold_angle = out_bearing
    rows = []
    while True:
        ts = start_ts + t
        in_hold = (not held) and d <= hold_r_km and hold_s > 0
        if in_hold:
            speed = 110.0 * wind
            alt = alt_at(hold_r_km)
            lat, lon = destination_point(h_lat, h_lon, hold_angle, hold_r_km)
            track = (hold_angle + 90.0) % 360
            hold_angle = (hold_angle + math.degrees(speed * DT_S / (hold_r_km * 1000.0))) % 360
            hold_left -= DT_S
            if hold_left <= 0:
                held = True
            dist_now = hold_r_km
        else:
            speed = speed_at(d) * wind
            if t < climb_s:
                speed = min(speed, (120.0 + (v_cruise - 120.0) * t / climb_s) * wind)
            alt = min(alt_at(d), cruise_alt * min(1.0, t / climb_s)) if t < climb_s else alt_at(d)
            lat, lon = destination_point(h_lat, h_lon, out_bearing, d)
            track = bearing_deg(lat, lon, h_lat, h_lon)
            dist_now = d

        rows.append({
            "icao24": None, "dep_airport": icao, "arr_airport": HUB[0], "ts": int(ts),
            "latitude": lat + rng.normal(0, 0.0004), "longitude": lon + rng.normal(0, 0.0004),
            "baro_altitude": max(0.0, alt + rng.normal(0, 15)),
            "velocity": max(0.0, speed + rng.normal(0, 1.5)),
            "true_track": track, "on_ground": False, "_alt": alt,
        })

        step_km = speed * DT_S / 1000.0
        if not in_hold:
            if d - step_km <= 0:
                t += DT_S * d / step_km  # interpolate the touchdown moment
                break
            d -= step_km
        t += DT_S

    # vertical rate from altitude differences (m/s), plus noise
    alts = np.array([r["_alt"] for r in rows])
    vr = np.gradient(alts, DT_S) if len(alts) > 1 else np.zeros(1)
    for r, v in zip(rows, vr):
        r["vertical_rate"] = float(v + rng.normal(0, 0.3))
        del r["_alt"]
    return rows, int(round(start_ts + t))


def generate(n_flights: int, start_ts: int, days: int, seed: int):
    rng = np.random.default_rng(seed)
    weights = np.array([o[3] for o in ORIGINS], dtype=float)
    weights /= weights.sum()
    hours = np.arange(24)
    hour_w = 0.3 + np.array([congestion(h + 0.5) for h in hours])
    hour_w /= hour_w.sum()

    snaps = []
    for i in range(n_flights):
        origin = ORIGINS[int(rng.choice(len(ORIGINS), p=weights))]
        day = int(rng.integers(0, days))
        land_hour = int(rng.choice(hours, p=hour_w))
        # departure ~ landing minus a rough duration; exact landing comes from the sim
        route_km = haversine_km(origin[1], origin[2], HUB[1], HUB[2])
        rough_dur = route_km / 800.0 * 3600 + 900
        land_nominal = start_ts + day * 86400 + land_hour * 3600 + int(rng.integers(0, 3600))
        dep_ts = int(land_nominal - rough_dur)
        icao24 = f"{i:06x}"  # unique per flight
        rows, arrival_ts = simulate_flight(rng, origin, dep_ts)
        for r in rows:
            r["icao24"] = icao24
            r["callsign"] = f"SYN{i % 100000:05d}"
            r["actual_arrival_ts"] = arrival_ts
        snaps.extend(rows)
    df = pd.DataFrame(snaps)
    df["fetched_at"] = df["ts"]
    for c in ("ts", "actual_arrival_ts", "fetched_at"):
        df[c] = df[c].astype("int64")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", type=int, default=3000)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.getenv("DATA_DIR", "data/synthetic"))
    args = ap.parse_args()

    out = Path(args.out)
    # Own folder, wiped on every run, so old files can never mix schemas with this.
    for sub in ("historical_opensky", "current_flights"):
        shutil.rmtree(out / sub, ignore_errors=True)
        (out / sub).mkdir(parents=True, exist_ok=True)

    airports = pd.DataFrame(
        [(HUB[0], HUB[1], HUB[2])] + [(o[0], o[1], o[2]) for o in ORIGINS],
        columns=["icao", "lat", "lon"],
    )
    airports.to_csv(out / "airports.csv", index=False)

    start_ts = int(pd.Timestamp("2026-10-01", tz="UTC").timestamp())
    train = generate(args.flights, start_ts, args.days, args.seed)
    train.to_parquet(out / "historical_opensky" / "synthetic.parquet", index=False)

    # Fresh, separate flights (different seed) for the predict demo: keep one
    # mid-flight snapshot per flight and hide the label
    test = generate(max(200, args.flights // 15), start_ts, args.days, args.seed + 1)
    rng = np.random.default_rng(args.seed + 2)
    sampled = []
    for _, g in test.groupby("icao24"):
        if len(g) > 2:
            sampled.append(g.iloc[int(rng.integers(len(g) // 10, max(len(g) // 10 + 1, len(g) * 9 // 10)))])
    cur = pd.DataFrame(sampled).rename(columns={"actual_arrival_ts": "true_arrival_ts"})
    cur.to_parquet(out / "current_flights" / "synthetic.parquet", index=False)

    print(f"airports: {len(airports)} | train rows: {len(train):,} "
          f"({train['icao24'].nunique():,} flights) | current flights: {len(cur):,}")
    print(f"written under {out.resolve()}")


if __name__ == "__main__":
    main()