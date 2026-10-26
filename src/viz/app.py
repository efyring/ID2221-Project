"""Streamlit dashboard for the OpenSky ETA / delay prediction engine.

Primary UX: look up one flight → current ETA + past landings / delays.

Run (from repo root, after ``make demo``)::

    make viz
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from viz.history import ensure_landing_history, history_for_aircraft
from viz.predict_batch import run_predictions, summary_metrics

HUB_LAT = 50.0379
HUB_LON = 8.5622
HUB_ICAO = "EDDF"
EUROPE_CENTER = {"lat": 50.0, "lon": 8.5}
EUROPE_ZOOM = 5
DISPLAY_TZ = os.environ.get("DISPLAY_TZ", "Europe/Berlin")
# Prefer live OpenSky predictions when present (written by ``make predict-live``).
_LIVE_PRED = Path("./data/live/predictions.parquet")
_DEFAULT_CACHE = (
    str(_LIVE_PRED)
    if _LIVE_PRED.exists()
    else "./data/synthetic/predictions.parquet"
)
DEFAULT_INPUT = os.environ.get(
    "PREDICT_INPUT_PATH",
    "./data/live/current_states" if _LIVE_PRED.exists() else "./data/synthetic/current_flights",
)
DEFAULT_MODEL = os.environ.get("MODEL_OUTPUT_PATH", "./models/synthetic/eta_gbt_pipeline")
DEFAULT_AIRPORTS = os.environ.get(
    "AIRPORTS_PATH",
    "./data/airports.csv" if _LIVE_PRED.exists() else "./data/synthetic/airports.csv",
)
DEFAULT_HISTORY = os.environ.get(
    "TRAIN_DATA_PATH",
    "./data/historical_opensky" if _LIVE_PRED.exists() else "./data/synthetic/historical_opensky",
)
CACHE_PATH = Path(os.environ.get("VIZ_CACHE_PATH", _DEFAULT_CACHE))


@st.cache_data
def _airport_coords(airports_path: str) -> dict[str, tuple[float, float]]:
    path = Path(airports_path)
    if not path.exists():
        return {HUB_ICAO: (HUB_LAT, HUB_LON)}
    df = pd.read_csv(path)
    return {
        str(r["icao"]).upper(): (float(r["lat"]), float(r["lon"]))
        for _, r in df.iterrows()
        if pd.notna(r.get("icao"))
    }


def _hub_for_row(row: pd.Series, airports_path: str = DEFAULT_AIRPORTS) -> tuple[str, float, float]:
    """Arrival hub ICAO + lat/lon for this prediction (not a hardcoded Frankfurt)."""
    icao = str(row.get("arr_airport") or HUB_ICAO).strip().upper() or HUB_ICAO
    coords = _airport_coords(airports_path)
    lat, lon = coords.get(icao, (HUB_LAT, HUB_LON))
    return icao, lat, lon


def _alt_label(alt_m: float) -> str:
    """OpenSky baro altitude is metres; also show feet for readability."""
    if pd.isna(alt_m):
        return "—"
    return f"{alt_m:,.0f} m ({alt_m * 3.28084:,.0f} ft)"


st.set_page_config(
    page_title="Flight ETA lookup",
    page_icon="✈️",
    layout="wide",
)


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _destination_point(lat: float, lon: float, bearing_deg: float, dist_km: float) -> tuple[float, float]:
    r = 6371.0
    δ = dist_km / r
    θ = math.radians(bearing_deg)
    φ1, λο = math.radians(lat), math.radians(lon)
    φ2 = math.asin(math.sin(φ1) * math.cos(δ) + math.cos(φ1) * math.sin(δ) * math.cos(θ))
    λ2 = λο + math.atan2(
        math.sin(θ) * math.sin(δ) * math.cos(φ1),
        math.cos(δ) - math.sin(φ1) * math.sin(φ2),
    )
    return math.degrees(φ2), (math.degrees(λ2) + 540.0) % 360.0 - 180.0


def _load_cached() -> pd.DataFrame | None:
    if CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)
    return None


def _save_cache(df: pd.DataFrame) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)


@st.cache_data(show_spinner="Running Spark GBT predictions…")
def _cached_run(input_path: str, model_path: str, airports_path: str, bust: int) -> pd.DataFrame:
    del bust
    return run_predictions(input_path, model_path, airports_path)


@st.cache_data(show_spinner="Loading past landings…")
def _cached_landings(history_path: str, bust: int) -> pd.DataFrame:
    del bust
    return ensure_landing_history(history_path, force=True)


def _fmt_landing_value(ts) -> str:
    if pd.isna(ts):
        return "—"
    return (
        pd.to_datetime(ts, utc=True)
        .tz_convert(DISPLAY_TZ)
        .strftime(f"%Y-%m-%d %H:%M {DISPLAY_TZ}")
    )


def _flight_label(row: pd.Series) -> str:
    cs = str(row.get("callsign") or "").strip() or "—"
    return f"{cs}  ·  {row['icao24']}"


def load_predictions(input_path: str, model_path: str, airports_path: str) -> pd.DataFrame:
    with st.sidebar:
        st.header("Data")
        use_cache = st.checkbox("Use cached predictions", value=CACHE_PATH.exists())
        fetch_live = st.button("Fetch live from OpenSky", type="primary")
        run_btn = st.button("Re-score saved input parquet")
        if "bust" not in st.session_state:
            st.session_state.bust = 0

    df: pd.DataFrame | None = None
    if fetch_live:
        try:
            from ml.predict_opensky import run as run_live

            with st.spinner("Polling OpenSky and scoring inbound flights…"):
                # Ensure live paths use the sidebar model / airports.
                os.environ.setdefault("MODEL_OUTPUT_PATH", model_path)
                os.environ.setdefault("AIRPORTS_PATH", airports_path)
                os.environ["MODEL_OUTPUT_PATH"] = model_path
                os.environ["AIRPORTS_PATH"] = airports_path
                df = run_live(
                    states_out=os.environ.get("LIVE_STATES_PATH", "./data/live/current_states"),
                    predictions_out=str(CACHE_PATH)
                    if "live" in str(CACHE_PATH)
                    else "./data/live/predictions.parquet",
                )
                live_out = Path("./data/live/predictions.parquet")
                if live_out.exists():
                    _save_cache(pd.read_parquet(live_out) if CACHE_PATH != live_out else df)
                st.session_state.bust += 1
                st.success(f"Live poll scored {len(df)} inbound flights.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Live OpenSky prediction failed: {exc}")
            st.stop()
    elif use_cache and not run_btn:
        df = _load_cached()

    if df is None:
        if run_btn:
            st.session_state.bust += 1
        missing = [p for p in (input_path, model_path, airports_path) if not Path(p).exists()]
        if missing:
            st.error(
                "Missing: " + ", ".join(missing)
                + ". Run `make demo` or `make predict-live` first (needs Java 17+)."
            )
            st.stop()
        try:
            df = _cached_run(input_path, model_path, airports_path, st.session_state.bust)
            _save_cache(df)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction failed: {exc}")
            st.stop()
    if df is None or df.empty:
        st.warning("No predictions available. Try **Fetch live from OpenSky**.")
        st.stop()
    return df


def render_flight_eta(row: pd.Series) -> None:
    """Current ETA card for the selected airborne flight."""
    callsign = str(row.get("callsign") or "").strip() or "—"
    hub_icao, _, _ = _hub_for_row(row)
    st.subheader(f"{callsign}")
    as_of = ""
    if pd.notna(row.get("ts")):
        as_of = (
            pd.to_datetime(int(row["ts"]), unit="s", utc=True)
            .tz_convert(DISPLAY_TZ)
            .strftime(f"%H:%M:%S {DISPLAY_TZ}")
        )
    st.caption(
        f"icao24 `{row['icao24']}` → **{hub_icao}**"
        + (f" · OpenSky snapshot {as_of}" if as_of else "")
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ETA (remaining)", f"{row['pred_remaining_min']:.0f} min")
    c2.metric("Predicted landing", _fmt_landing_value(row["pred_landing_time"]))
    delay = row.get("delay_vs_naive_min")
    c3.metric(
        "Delay vs naive ETA",
        f"{delay:+.0f} min" if pd.notna(delay) else "n/a",
        help="Model remaining − distance/speed baseline. Not airline schedule delay.",
    )
    if "err_min" in row.index and pd.notna(row.get("err_min")):
        c4.metric("Error vs true landing", f"{row['err_min']:+.0f} min")
    elif "true_arrival_ts" in row.index and pd.notna(row.get("true_arrival_ts")):
        c4.metric("True landing (demo)", _fmt_landing_value(pd.to_datetime(row["true_arrival_ts"], unit="s", utc=True)))
    else:
        c4.metric("Distance to hub", f"{row.get('dist_to_arr_km', float('nan')):.0f} km")

    hub_icao, hub_lat, hub_lon = _hub_for_row(row)
    d1, d2, d3, d4 = st.columns(4)
    alt_m = row.get("altitude", float("nan"))
    d1.metric("Altitude", _alt_label(alt_m))
    hdg = row.get("true_track")
    if pd.isna(hdg) and pd.notna(row.get("latitude")):
        hdg = _bearing_deg(float(row["latitude"]), float(row["longitude"]), hub_lat, hub_lon)
    d2.metric("Heading", f"{hdg:.0f}°" if pd.notna(hdg) else "n/a")
    d3.metric("Ground speed", f"{row.get('velocity', float('nan')):.0f} m/s")
    d4.metric("Naive ETA", f"{row.get('naive_eta_min', float('nan')):.0f} min")


def render_flight_map(row: pd.Series) -> None:
    """Small map for the selected flight only."""
    if pd.isna(row.get("latitude")) or pd.isna(row.get("longitude")):
        return
    lat, lon = float(row["latitude"]), float(row["longitude"])
    hub_icao, hub_lat, hub_lon = _hub_for_row(row)
    hdg = row.get("true_track")
    if pd.isna(hdg):
        hdg = _bearing_deg(lat, lon, hub_lat, hub_lon)
    tip_lat, tip_lon = _destination_point(lat, lon, float(hdg), 15.0)
    alt = float(row.get("altitude") or 0.0)
    dist = float(row.get("dist_to_arr_km") or 500.0)
    # Zoom in when on final; keep Europe-ish view for long haul.
    zoom = 9 if dist < 40 else (7 if dist < 200 else 5)

    fig = go.Figure()
    fig.add_trace(
        go.Scattermap(
            lat=[lat, hub_lat],
            lon=[lon, hub_lon],
            mode="lines",
            line={"width": 2, "color": "rgba(80,80,80,0.5)"},
            name=f"To {hub_icao}",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scattermap(
            lat=[lat, tip_lat],
            lon=[lon, tip_lon],
            mode="lines+markers",
            line={"width": 4, "color": "rgb(20,90,200)"},
            marker={"size": [10, 8], "color": "rgb(20,90,200)"},
            name=f"Heading {hdg:.0f}°",
        )
    )
    fig.add_trace(
        go.Scattermap(
            lat=[lat],
            lon=[lon],
            mode="markers",
            marker={
                "size": 18,
                "color": [alt],
                "colorscale": "Turbo",
                "cmin": 0,
                "cmax": 12000,
                "colorbar": {"title": "Alt (m)", "thickness": 10},
            },
            text=[
                f"{row.get('callsign', row['icao24'])}<br>"
                f"Alt {_alt_label(alt)}<br>ETA {row['pred_remaining_min']:.0f} min"
            ],
            hovertemplate="%{text}<extra></extra>",
            name="Aircraft",
        )
    )
    fig.add_trace(
        go.Scattermap(
            lat=[hub_lat],
            lon=[hub_lon],
            mode="markers+text",
            marker={"size": 14, "color": "crimson"},
            text=[hub_icao],
            textposition="top right",
            name="Hub",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=360,
        map={"style": "carto-positron", "center": {"lat": (lat + hub_lat) / 2, "lon": (lon + hub_lon) / 2}, "zoom": zoom},
        showlegend=False,
        uirevision=f"flight-{row['icao24']}",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_past_landings(past: pd.DataFrame) -> None:
    st.subheader("Previous landings & delays")
    st.caption(
        "Completed flights of this aircraft at the hub. "
        "**Delay vs naive** = actual remaining time at mid-flight minus distance/speed ETA "
        "(positive ⇒ slower than kinematics)."
    )
    if past.empty:
        st.info("No prior labelled landings for this aircraft in training history.")
        return

    k1, k2, k3 = st.columns(3)
    k1.metric("Past landings", len(past))
    k2.metric("Avg delay vs naive", f"{past['delay_vs_naive_min'].mean():+.0f} min")
    k3.metric("Last landing", str(past.iloc[0]["landing_local"]))

    show = past[
        [
            c
            for c in [
                "landing_local",
                "flight_duration_min",
                "delay_vs_naive_min",
                "mid_true_remaining_min",
                "mid_naive_eta_min",
                "mid_altitude_m",
                "mid_dist_km",
                "arr_airport",
            ]
            if c in past.columns
        ]
    ].rename(
        columns={
            "landing_local": "landed (local)",
            "flight_duration_min": "duration_min",
            "mid_true_remaining_min": "mid_true_rem_min",
            "mid_naive_eta_min": "mid_naive_eta_min",
            "mid_altitude_m": "mid_alt_m",
            "mid_dist_km": "mid_dist_km",
        }
    )
    for col, nd in (
        ("duration_min", 0),
        ("delay_vs_naive_min", 1),
        ("mid_true_rem_min", 0),
        ("mid_naive_eta_min", 0),
        ("mid_alt_m", 0),
        ("mid_dist_km", 0),
    ):
        if col in show.columns:
            show[col] = show[col].round(nd)
    st.dataframe(show, width="stretch", hide_index=True)

    if "delay_vs_naive_min" in past.columns and "landing_local" in past.columns:
        fig = px.bar(
            past.sort_values("landing_ts") if "landing_ts" in past.columns else past,
            x="landing_local",
            y="delay_vs_naive_min",
            title="Past delay vs naive ETA (minutes)",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(xaxis_tickangle=-30, height=320, margin={"t": 40, "b": 80})
        st.plotly_chart(fig, use_container_width=True)


def render_prediction_history(icao24: str) -> None:
    """Live Cassandra eta_history for this aircraft (prior inference snapshots)."""
    try:
        from viz.cassandra_source import load_eta_history
    except ImportError:
        return
    hist = load_eta_history(icao24)
    if hist.empty:
        st.caption("No prior live predictions in Cassandra for this aircraft yet.")
        return
    st.subheader("Recent live predictions")
    show = hist[
        [c for c in [
            "pred_at_local", "arr_airport", "pred_remaining_min",
            "dist_to_arr_km", "altitude", "latitude", "longitude", "pred_landing_time",
        ] if c in hist.columns]
    ].rename(columns={
        "pred_at_local": "predicted_at",
        "pred_remaining_min": "eta_min",
        "dist_to_arr_km": "dist_km",
    })
    if "eta_min" in show.columns:
        show["eta_min"] = show["eta_min"].round(0)
    if "dist_km" in show.columns:
        show["dist_km"] = show["dist_km"].round(0)
    if "altitude" in show.columns:
        show["alt_ft"] = (show["altitude"] * 3.28084).round(0)
        show["altitude"] = show["altitude"].round(0)
        show = show.rename(columns={"altitude": "alt_m"})
    st.dataframe(show, width="stretch", hide_index=True)


def render_lookup_tab(df: pd.DataFrame, landings: pd.DataFrame) -> None:
    options = df.copy()
    options["label"] = options.apply(_flight_label, axis=1)
    options = options.sort_values("pred_remaining_min")

    q = st.text_input(
        "Search callsign or icao24",
        placeholder="e.g. QTR28L or 3c6744",
        help="Filter the flight list, then pick one below.",
    )
    filtered = options
    if q.strip():
        needle = q.strip().lower()
        callsigns = (
            options["callsign"].astype(str)
            if "callsign" in options.columns
            else pd.Series("", index=options.index)
        )
        mask = options["icao24"].astype(str).str.lower().str.contains(
            needle, regex=False
        ) | callsigns.str.lower().str.contains(needle, regex=False)
        filtered = options.loc[mask]
        if filtered.empty:
            st.warning(f"No inbound flight matches “{q}”.")
            return

    labels = filtered["label"].tolist()
    choice = st.selectbox(
        "Flight",
        labels,
        index=0,
        help="Sorted by soonest predicted arrival.",
    )
    row = filtered.loc[filtered["label"] == choice].iloc[0]

    left, right = st.columns([1.1, 1])
    with left:
        render_flight_eta(row)
        if os.getenv("VIZ_SOURCE") == "cassandra" or os.getenv("CASSANDRA_HOST"):
            render_prediction_history(str(row["icao24"]))
        render_past_landings(history_for_aircraft(landings, str(row["icao24"])))
    with right:
        st.markdown("#### Position")
        render_flight_map(row)


def render_fleet_map(df: pd.DataFrame, highlight: str | None = None) -> None:
    plot_df = df.dropna(subset=["latitude", "longitude"]).copy()
    if plot_df.empty:
        st.warning("No positioned flights.")
        return
    coords = _airport_coords(DEFAULT_AIRPORTS)
    plot_df["altitude_m"] = plot_df["altitude"].fillna(0.0)
    plot_df["hub_icao"] = plot_df.get("arr_airport", HUB_ICAO).fillna(HUB_ICAO).astype(str).str.upper()
    plot_df["hub_lat"] = plot_df["hub_icao"].map(lambda c: coords.get(c, (HUB_LAT, HUB_LON))[0])
    plot_df["hub_lon"] = plot_df["hub_icao"].map(lambda c: coords.get(c, (HUB_LAT, HUB_LON))[1])
    if "true_track" in plot_df.columns and plot_df["true_track"].notna().any():
        plot_df["heading_deg"] = plot_df["true_track"].astype(float)
    else:
        plot_df["heading_deg"] = [
            _bearing_deg(a, b, hl, hn)
            for a, b, hl, hn in zip(
                plot_df["latitude"], plot_df["longitude"], plot_df["hub_lat"], plot_df["hub_lon"]
            )
        ]
    plot_df["label"] = plot_df.apply(_flight_label, axis=1)

    fig = go.Figure()
    line_lat: list[float | None] = []
    line_lon: list[float | None] = []
    for lat, lon, hl, hn in zip(
        plot_df["latitude"], plot_df["longitude"], plot_df["hub_lat"], plot_df["hub_lon"]
    ):
        line_lat.extend([float(lat), float(hl), None])
        line_lon.extend([float(lon), float(hn), None])
    fig.add_trace(
        go.Scattermap(
            lat=line_lat,
            lon=line_lon,
            mode="lines",
            line={"width": 1, "color": "rgba(80,80,80,0.25)"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    alt = plot_df["altitude_m"].clip(lower=0)
    alt_max = max(float(alt.max()), 1.0)
    hover = (
        plot_df["label"]
        + "<br>→ "
        + plot_df["hub_icao"]
        + "<br>Alt "
        + plot_df["altitude_m"].map(_alt_label)
        + "<br>HDG "
        + plot_df["heading_deg"].map(lambda x: f"{x:.0f}°")
        + "<br>ETA "
        + plot_df["pred_remaining_min"].map(lambda x: f"{x:.0f} min")
    )
    fig.add_trace(
        go.Scattermap(
            lat=plot_df["latitude"],
            lon=plot_df["longitude"],
            mode="markers",
            marker={
                "size": 9 + 14 * (alt / alt_max),
                "color": plot_df["altitude_m"],
                "colorscale": "Turbo",
                "cmin": 0,
                "cmax": max(alt_max, 12000.0),
                "colorbar": {"title": "Altitude (m)", "thickness": 12},
            },
            customdata=np.stack([hover.to_numpy()], axis=-1),
            hovertemplate="%{customdata[0]}<extra></extra>",
            name="Aircraft",
        )
    )
    if highlight:
        hi = plot_df.loc[plot_df["icao24"] == highlight]
        if not hi.empty:
            fig.add_trace(
                go.Scattermap(
                    lat=hi["latitude"],
                    lon=hi["longitude"],
                    mode="markers",
                    marker={"size": 22, "color": "crimson"},
                    name="Selected",
                    hoverinfo="skip",
                )
            )
    hubs = (
        plot_df[["hub_icao", "hub_lat", "hub_lon"]]
        .drop_duplicates("hub_icao")
        .dropna()
    )
    fig.add_trace(
        go.Scattermap(
            lat=hubs["hub_lat"],
            lon=hubs["hub_lon"],
            mode="markers+text",
            marker={"size": 12, "color": "crimson"},
            text=hubs["hub_icao"],
            textposition="top right",
            name="Hubs",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=520,
        map={"style": "carto-positron", "center": EUROPE_CENTER, "zoom": EUROPE_ZOOM},
        uirevision="europe-hubs",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_overview_tab(df: pd.DataFrame) -> None:
    m = summary_metrics(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inbound flights", m["n_flights"])
    c2.metric("Avg remaining", f"{m['avg_remaining_min']:.0f} min")
    c3.metric("MAE vs truth", f"{m['mae_min']:.0f} min" if "mae_min" in m else "n/a")
    c4.metric(
        "Avg delay vs naive",
        f"{m['avg_delay_vs_naive_min']:+.0f} min" if "avg_delay_vs_naive_min" in m else "n/a",
    )
    render_fleet_map(df)

    board = df.sort_values("pred_remaining_min").copy()
    board["landing_local"] = (
        pd.to_datetime(board["pred_landing_time"], utc=True)
        .dt.tz_convert(DISPLAY_TZ)
        .dt.strftime("%Y-%m-%d %H:%M")
    )
    cols = [
        c
        for c in [
            "callsign",
            "icao24",
            "pred_remaining_min",
            "landing_local",
            "delay_vs_naive_min",
            "altitude",
            "true_track",
            "dist_to_arr_km",
            "err_min",
        ]
        if c in board.columns
    ]
    show = board[cols].rename(
        columns={
            "pred_remaining_min": "eta_min",
            "delay_vs_naive_min": "delay_vs_naive_min",
            "true_track": "hdg_deg",
            "dist_to_arr_km": "dist_km",
            "err_min": "error_min",
        }
    )
    for col, nd in (
        ("eta_min", 0),
        ("delay_vs_naive_min", 0),
        ("altitude", 0),
        ("hdg_deg", 0),
        ("dist_km", 0),
        ("error_min", 0),
    ):
        if col in show.columns:
            show[col] = show[col].round(nd)
    st.subheader("Arrival board")
    st.dataframe(show, width="stretch", height=360, hide_index=True)


def _load_cassandra_frame(arr_airport: str | None) -> pd.DataFrame:
    from viz.cassandra_source import load_eta_latest

    return load_eta_latest(arr_airport=arr_airport or None)


def main() -> None:
    viz_source = os.getenv("VIZ_SOURCE", "auto")
    use_cassandra = viz_source == "cassandra" or (
        viz_source == "auto" and bool(os.getenv("CASSANDRA_HOST"))
    )

    st.title("Flight ETA & delay history")
    st.caption(
        "Search real (or demo) inbound flights for predicted landing times. "
        "Delay vs naive = model ETA − distance/speed baseline."
    )

    with st.sidebar:
        st.header("Source")
        if use_cassandra:
            st.success("Live source: Cassandra `flights.eta_latest`")
            refresh = st.button("Refresh from Cassandra", type="primary")
            if refresh:
                st.cache_data.clear()
        else:
            st.info("Parquet / Spark batch source")
        input_path = st.text_input("Prediction input", DEFAULT_INPUT)
        model_path = st.text_input("Model path", DEFAULT_MODEL)
        airports_path = st.text_input("Airports CSV", DEFAULT_AIRPORTS)
        history_path = st.text_input("Historical flights", DEFAULT_HISTORY)

    if use_cassandra:
        # Auto-refresh so ETAs track the per-minute stream upserts.
        try:
            from streamlit_autorefresh import st_autorefresh

            st_autorefresh(interval=60_000, key="cassandra_live_refresh")
        except ImportError:
            st.caption("Tip: `pip install streamlit-autorefresh` for automatic 60s UI refresh.")

        try:
            from viz.cassandra_source import list_arrival_airports

            airports = list_arrival_airports()
        except Exception as exc:  # noqa: BLE001
            st.error(
                f"Cannot read Cassandra: {exc}\n\n"
                "Run `make pipeline`, then `make producer` + `make stream-live`."
            )
            st.stop()
        choices = ["(all European hubs)"] + airports
        pick = st.selectbox("Arrival airport", choices, index=0)
        arr = None if pick.startswith("(all") else pick
        try:
            df = _load_cassandra_frame(arr)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Cassandra query failed: {exc}")
            st.stop()
        if df.empty:
            st.warning(
                "No live predictions yet. Start `make producer` and `make stream-live`, "
                "wait ~1 minute, then refresh."
            )
            st.stop()
        st.info(
            f"Live Cassandra: **{len(df)}** aircraft"
            + (f" inbound to **{arr}**" if arr else " across European hubs")
            + ". Rescored about **every minute** from OpenSky→Kafka while airborne. "
            "Search callsign / icao24 below."
        )
        landings = pd.DataFrame()
    else:
        df = load_predictions(input_path, model_path, airports_path)
        if "hist_bust" not in st.session_state:
            st.session_state.hist_bust = 0
        landings = (
            _cached_landings(history_path, st.session_state.hist_bust)
            if Path(history_path).exists()
            else pd.DataFrame()
        )

    tab_lookup, tab_fleet = st.tabs(["Flight lookup", "Fleet overview"])
    with tab_lookup:
        render_lookup_tab(df, landings)
    with tab_fleet:
        render_overview_tab(df)


if __name__ == "__main__":
    main()
