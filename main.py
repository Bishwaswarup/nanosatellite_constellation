"""Streamlit command center for the AI-Optimized Nanosatellite Constellation Manager."""

from datetime import datetime, timezone
from pathlib import Path

import plotly.express as px
import streamlit as st

from src.models import StationKeepingConfig
from src.simulation import simulate_tle_constellation

PROJECT_ROOT = Path(__file__).parent
DEFAULT_TLE_FILE = PROJECT_ROOT / "data" / "sample_constellation.tle"

st.set_page_config(page_title="Nanosatellite Constellation Manager", layout="wide")
st.title("Nanosatellite Constellation Manager")
st.caption("Phase 2 — TLE propagation, virtual orbital decay, and deterministic station-keeping")

with st.sidebar:
    st.header("Simulation")
    duration_hours = st.slider("Duration (hours)", 1, 24, 24)
    step_seconds = st.select_slider("Timestep (seconds)", options=[60, 120, 300, 600], value=300)
    st.header("Station-keeping")
    decay_rate = st.slider("Virtual decay (km/day)", 0.1, 2.0, 0.5, 0.1)
    altitude_tolerance = st.slider("Altitude tolerance (km)", 0.05, 1.0, 0.25, 0.05)
    initial_propellant = st.slider("Initial propellant (kg)", 0.1, 10.0, 2.0, 0.1)
    run = st.button("Run propagation", type="primary")

if run or "telemetry" not in st.session_state:
    with st.spinner("Propagating constellation with SGP4…"):
        st.session_state.telemetry = simulate_tle_constellation(
            DEFAULT_TLE_FILE,
            start=datetime.now(timezone.utc),
            duration_hours=duration_hours,
            step_seconds=step_seconds,
            station_keeping=StationKeepingConfig(
                decay_rate_km_per_day=decay_rate,
                altitude_tolerance_km=altitude_tolerance,
                initial_propellant_kg=initial_propellant,
            ),
        )

telemetry = st.session_state.telemetry
latest = telemetry[telemetry["timestamp"] == telemetry["timestamp"].max()].copy()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Satellites", latest["satellite"].nunique())
col2.metric("Telemetry samples", f"{len(telemetry):,}")
col3.metric("Mean altitude", f"{latest['altitude_km'].mean():.1f} km")
col4.metric("Station-keeping burns", int(telemetry["station_keeping_event"].sum()))

st.subheader("Ground tracks")
ground_track = px.line_geo(
    telemetry,
    lat="latitude_deg",
    lon="longitude_deg",
    color="satellite",
    hover_data={"altitude_km": ":.1f", "timestamp": True},
    projection="equirectangular",
)
ground_track.update_geos(showland=True, landcolor="#e8eef3", showocean=True, oceancolor="#d6efff")
st.plotly_chart(ground_track, use_container_width=True)

st.subheader("Latest ECEF constellation position")
orbital_view = px.scatter_3d(
    latest,
    x="ecef_x_km",
    y="ecef_y_km",
    z="ecef_z_km",
    color="satellite",
    hover_data={"altitude_km": ":.1f", "latitude_deg": ":.2f", "longitude_deg": ":.2f"},
    labels={"ecef_x_km": "ECEF X (km)", "ecef_y_km": "ECEF Y (km)", "ecef_z_km": "ECEF Z (km)"},
)
orbital_view.update_layout(height=600)
st.plotly_chart(orbital_view, use_container_width=True)

st.subheader("Latest telemetry and resources")
st.dataframe(
    latest[[
        "satellite", "timestamp", "altitude_km", "controlled_mean_altitude_km", "fuel_remaining_kg",
        "fuel_percent", "cumulative_delta_v_m_s", "maneuver_count", "station_keeping_status",
    ]],
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Control altitude is a simplified mean-orbit model. SGP4 ground tracks remain TLE-derived and are not "
    "physically re-propagated after a virtual maneuver."
)

st.download_button(
    "Download telemetry CSV",
    telemetry.to_csv(index=False).encode("utf-8"),
    file_name="constellation_telemetry.csv",
    mime="text/csv",
)


