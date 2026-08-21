"""Phase 4 Streamlit command center for the nanosatellite constellation manager."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.avoidance import CollisionAvoidanceConfig
from src.models import StationKeepingConfig
from src.simulation import simulate_tle_constellation
from src.test_scenario import synthetic_safe_pass_figure, synthetic_scenario_risks

PROJECT_ROOT = Path(__file__).parent
FLEET_OPTIONS = {
    "Official 10-satellite LEO fleet": PROJECT_ROOT / "data" / "official_fleet_10.tle",
    "Sample 5-satellite fleet": PROJECT_ROOT / "data" / "sample_constellation.tle",
}
EARTH_RADIUS_KM = 6378.137


def earth_surface() -> go.Surface:
    """Create a lightweight Earth reference sphere for the inertial 3D view."""
    azimuth, elevation = np.mgrid[0 : 2 * np.pi : 48j, 0 : np.pi : 24j]
    return go.Surface(
        x=EARTH_RADIUS_KM * np.cos(azimuth) * np.sin(elevation),
        y=EARTH_RADIUS_KM * np.sin(azimuth) * np.sin(elevation),
        z=EARTH_RADIUS_KM * np.cos(elevation),
        colorscale=[[0, "#0b3558"], [1, "#1f6f9d"]],
        showscale=False,
        opacity=0.82,
        hoverinfo="skip",
        name="Earth",
    )


def ground_track_figure(telemetry, snapshot) -> go.Figure:
    """Show all ground tracks and the selected instant's satellite locations."""
    figure = go.Figure()
    for satellite, track in telemetry.groupby("satellite", sort=True):
        figure.add_trace(
            go.Scattergeo(
                lon=track["longitude_deg"], lat=track["latitude_deg"], mode="lines", name=satellite,
                line={"width": 1.5}, hovertemplate=f"{satellite}<br>Lat %{{lat:.2f}}°<br>Lon %{{lon:.2f}}°<extra></extra>",
            )
        )
    figure.add_trace(
        go.Scattergeo(
            lon=snapshot["longitude_deg"], lat=snapshot["latitude_deg"], mode="markers+text",
            text=snapshot["satellite"], textposition="top center", name="Selected time",
            marker={"size": 9, "color": "#d00000", "line": {"color": "white", "width": 1}},
            hovertemplate="%{text}<br>Altitude %{customdata[0]:.1f} km<extra></extra>",
            customdata=snapshot[["altitude_km"]],
        )
    )
    figure.update_geos(showland=True, landcolor="#e8eef3", showocean=True, oceancolor="#ccecff", projection_type="equirectangular")
    figure.update_layout(height=570, margin={"l": 0, "r": 0, "t": 20, "b": 0}, legend={"orientation": "h", "y": -0.08})
    return figure


def orbital_web_figure(telemetry, snapshot) -> go.Figure:
    """Show SGP4 TEME trajectories, Earth, and selected-time satellite markers."""
    figure = go.Figure(data=[earth_surface()])
    for satellite, track in telemetry.groupby("satellite", sort=True):
        figure.add_trace(
            go.Scatter3d(
                x=track["teme_x_km"], y=track["teme_y_km"], z=track["teme_z_km"], mode="lines",
                name=satellite, line={"width": 3}, hoverinfo="skip",
            )
        )
    figure.add_trace(
        go.Scatter3d(
            x=snapshot["teme_x_km"], y=snapshot["teme_y_km"], z=snapshot["teme_z_km"], mode="markers+text",
            name="Selected time", text=snapshot["satellite"], textposition="top center",
            marker={"size": 5, "color": "#ffcb77"},
            customdata=snapshot[["altitude_km", "fuel_percent"]],
            hovertemplate="%{text}<br>Altitude %{customdata[0]:.1f} km<br>Fuel %{customdata[1]:.2f}%<extra></extra>",
        )
    )
    figure.update_layout(
        height=680, margin={"l": 0, "r": 0, "t": 20, "b": 0},
        scene={
            "xaxis": {"title": "TEME X (km)", "showbackground": False},
            "yaxis": {"title": "TEME Y (km)", "showbackground": False},
            "zaxis": {"title": "TEME Z (km)", "showbackground": False},
            "aspectmode": "data", "bgcolor": "#081c2c",
        },
    )
    return figure


def risk_queue_figure(alerts) -> go.Figure:
    """Plot the current deterministic alerts by their advisory review priority."""
    ordered = alerts.sort_values("ai_risk_probability", ascending=True)
    colors = {"CRITICAL": "#b42318", "HIGH": "#d97706", "ELEVATED": "#ca8a04", "LOW": "#2563eb"}
    return go.Figure(
        go.Bar(
            x=ordered["ai_risk_probability"], y=ordered["satellite"], orientation="h",
            marker_color=[colors.get(label, "#64748b") for label in ordered["ai_risk_label"]],
            customdata=ordered[["ai_risk_label", "ai_primary_driver", "predicted_closest_separation_km"]],
            hovertemplate="%{y}<br>Risk %{x:.1%}<br>Label %{customdata[0]}<br>Primary driver: %{customdata[1]}<br>Predicted separation: %{customdata[2]:.2f} km<extra></extra>",
        )
    ).update_layout(
        height=260, margin={"l": 0, "r": 0, "t": 10, "b": 0}, xaxis={"title": "Advisory risk probability", "tickformat": ".0%", "range": [0, 1]}, yaxis={"title": ""}, showlegend=False,
    )


st.set_page_config(page_title="Nanosatellite Constellation Manager", layout="wide")
st.markdown("""<style>
    .block-container {max-width: 1500px; padding-top: 2rem;}
    [data-testid="stMetric"] {background: #171923; border: 1px solid #3b4252; border-radius: 7px; padding: .65rem .8rem;}
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {color: #f8fafc;}
    h1, h2, h3 {color: inherit;}
</style>""", unsafe_allow_html=True)
st.title("Nanosatellite Constellation Manager")
st.caption("SGP4 tracking | deterministic safety screening | advisory risk prioritization | resource telemetry")

with st.sidebar:
    st.header("Simulation controls")
    fleet_name = st.selectbox("Fleet scenario", options=list(FLEET_OPTIONS), help="The official fleet uses archived CelesTrak TLEs. It is not a live operational feed.")
    tle_file = FLEET_OPTIONS[fleet_name]
    duration_hours = st.slider("Duration (hours)", 1, 24, 24)
    step_seconds = st.select_slider("Timestep (seconds)", options=[60, 120, 300, 600], value=300)
    st.divider()
    st.subheader("Station-keeping")
    decay_rate = st.slider("Virtual decay (km/day)", 0.1, 2.0, 0.5, 0.1)
    altitude_tolerance = st.slider("Altitude tolerance (km)", 0.05, 1.0, 0.25, 0.05)
    initial_propellant = st.slider("Initial propellant (kg)", 0.1, 10.0, 2.0, 0.1)
    st.subheader("Collision avoidance")
    keep_out_distance = st.slider("Keep-out distance (km)", 1.0, 2_000.0, 2.0, 1.0)
    run = st.button("Run simulation", type="primary", width="stretch")

if run or st.session_state.get("fleet_scenario") != fleet_name or "telemetry" not in st.session_state:
    with st.spinner("Propagating constellation and screening conjunctions…"):
        st.session_state.telemetry = simulate_tle_constellation(
            tle_file, start=datetime.now(timezone.utc), duration_hours=duration_hours,
            step_seconds=step_seconds,
            station_keeping=StationKeepingConfig(
                decay_rate_km_per_day=decay_rate, altitude_tolerance_km=altitude_tolerance,
                initial_propellant_kg=initial_propellant,
            ),
            collision_avoidance=CollisionAvoidanceConfig(keep_out_distance_km=keep_out_distance),
        )
        st.session_state.fleet_scenario = fleet_name

telemetry = st.session_state.telemetry
timestamps = list(telemetry["timestamp"].drop_duplicates().sort_values())
selected_time = st.select_slider("Mission time", options=timestamps, value=timestamps[-1], format_func=lambda value: value.strftime("%Y-%m-%d %H:%M UTC"))
snapshot = telemetry[telemetry["timestamp"] == selected_time].copy()
active_alerts = snapshot[snapshot["conjunction_alert"]]

with st.sidebar:
    st.divider()
    st.subheader("Fleet status at selected time")
    st.metric("Mean fuel", f"{snapshot['fuel_percent'].mean():.2f}%")
    st.metric("Active alerts", len(active_alerts))
    st.dataframe(
        snapshot[["satellite", "altitude_km", "fuel_percent", "ai_risk_label", "station_keeping_status", "avoidance_status"]],
        hide_index=True, width="stretch",
        column_config={"altitude_km": st.column_config.NumberColumn("Altitude (km)", format="%.1f"), "fuel_percent": st.column_config.NumberColumn("Fuel (%)", format="%.2f")},
    )

metric_columns = st.columns(5)
metric_columns[0].metric("Satellites", snapshot["satellite"].nunique())
metric_columns[1].metric("Mean altitude", f"{snapshot['altitude_km'].mean():.1f} km")
metric_columns[2].metric("Minimum fuel", f"{snapshot['fuel_percent'].min():.2f}%")
metric_columns[3].metric("Station-keeping burns", int(telemetry["station_keeping_event"].sum()))
metric_columns[4].metric("Avoidance commands", int(telemetry["avoidance_event"].sum()))
st.caption(f"Scenario: {fleet_name}. {snapshot['satellite'].nunique() * (snapshot['satellite'].nunique() - 1) // 2:,} pairwise comparisons per prediction timestamp.")

overview, ground_tracks, orbital_web, telemetry_tab, test_lab = st.tabs(["Overview", "Ground tracks", "3D orbital web", "Telemetry & alerts", "Synthetic test lab"])

with overview:
    left, right = st.columns((3, 2))
    with left:
        st.subheader("Fleet altitude at selected time")
        st.bar_chart(snapshot.set_index("satellite")["altitude_km"], color="#0077b6")
    with right:
        st.subheader("Operational status")
        if active_alerts.empty:
            st.success("No conjunction alert at the selected mission time.")
        else:
            st.error(f"{len(active_alerts)} satellite-side conjunction alert(s) active.")
            st.dataframe(active_alerts[["satellite", "closest_approach_km", "ai_risk_probability", "ai_risk_label", "ai_primary_driver", "avoidance_status"]], hide_index=True, width="stretch")
        recent_actions = snapshot[(snapshot["station_keeping_event"]) | (snapshot["avoidance_event"])]
        if recent_actions.empty:
            st.info("No maneuver was commanded at the selected time.")
        else:
            st.dataframe(recent_actions[["satellite", "maneuver_delta_v_m_s", "avoidance_delta_v_m_s", "fuel_remaining_kg"]], hide_index=True, width="stretch")

with ground_tracks:
    st.subheader("Ground-track map")
    st.caption("Lines show the simulated duration; red markers show the spacecraft position at the selected mission time.")
    st.plotly_chart(ground_track_figure(telemetry, snapshot), width="stretch")

with orbital_web:
    st.subheader("TEME orbital web")
    st.caption("This inertial view uses SGP4 TEME coordinates. Coloured paths show the full simulation window; gold markers show the selected time.")
    st.plotly_chart(orbital_web_figure(telemetry, snapshot), width="stretch")

with telemetry_tab:
    st.subheader("Fleet telemetry at selected time")
    st.dataframe(
        snapshot[[
            "satellite", "altitude_km", "controlled_mean_altitude_km", "latitude_deg", "longitude_deg",
            "fuel_remaining_kg", "fuel_percent", "cumulative_delta_v_m_s", "maneuver_count",
            "station_keeping_status", "conjunction_alert", "closest_approach_km", "avoidance_status",
            "relative_position_km", "relative_velocity_km_s", "time_to_closest_approach_s",
            "predicted_closest_separation_km", "keep_out_distance_km", "ai_risk_probability", "ai_risk_label",
            "ai_primary_driver", "ai_model_version",
        ]], hide_index=True, width="stretch",
    )
    st.subheader("Conjunction and avoidance event log")
    event_log = telemetry[telemetry["conjunction_alert"]][
        ["timestamp", "satellite", "closest_approach_km", "predicted_closest_separation_km", "ai_risk_probability", "ai_risk_label", "ai_primary_driver", "avoidance_status", "avoidance_delta_v_m_s", "predicted_post_burn_separation_km", "avoidance_effectiveness"]
    ]
    if event_log.empty:
        st.success("No close approaches crossed the selected keep-out distance.")
    else:
        st.warning("AI risk is advisory only. Deterministic keep-out screening and numerical post-burn re-screening remain the safety authority.")
        risk_column, log_column = st.columns((2, 3))
        with risk_column:
            st.caption("Advisory risk queue")
            st.plotly_chart(risk_queue_figure(event_log), width="stretch")
        with log_column:
            st.dataframe(event_log, hide_index=True, width="stretch")
    st.download_button("Download full telemetry CSV", telemetry.to_csv(index=False).encode("utf-8"), file_name="constellation_telemetry.csv", mime="text/csv")

with test_lab:
    st.subheader("Synthetic close-approach validation")
    st.caption("This local-frame animation is a controlled software test only; it does not represent or command a real satellite.")
    scenario_figure, unsafe_distance, safe_distance = synthetic_safe_pass_figure()
    unsafe_risk, safe_risk = synthetic_scenario_risks()
    scenario_metrics = st.columns(4)
    scenario_metrics[0].metric("Unmitigated closest approach", f"{unsafe_distance:.2f} km")
    scenario_metrics[1].metric("Post-maneuver closest approach", f"{safe_distance:.2f} km")
    scenario_metrics[2].metric("AI risk before", f"{unsafe_risk.probability:.1%}")
    scenario_metrics[3].metric("AI risk after", f"{safe_risk.probability:.1%}")
    st.caption(f"Before: {unsafe_risk.label}, primarily driven by {unsafe_risk.primary_driver}. After: {safe_risk.label}, primarily driven by {safe_risk.primary_driver}.")
    st.caption("The score prioritizes the test case for review; safe/unsafe status still comes from the deterministic 2 km keep-out and verified separation.")
    st.plotly_chart(scenario_figure, width="stretch")
