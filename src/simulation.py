"""Simulation orchestration for a TLE-defined satellite constellation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import numpy as np

from .avoidance import CollisionAvoidanceConfig, plan_avoidance_maneuvers, screen_conjunctions
from .control import advance_station_keeping, initialize_state, propellant_required_kg
from .models import StationKeepingConfig
from .numerical import apply_velocity_burn, propagate_two_body
from .propagation import load_tles, make_state_function, propagate_constellation


def build_timestamps(start: datetime, duration_hours: int, step_seconds: int) -> list[datetime]:
    """Create inclusive, timezone-aware simulation timestamps."""
    if start.tzinfo is None:
        raise ValueError("Simulation start time must be timezone-aware.")
    if duration_hours <= 0 or step_seconds <= 0:
        raise ValueError("Duration and step size must be positive.")
    start = start.astimezone(timezone.utc)
    steps = duration_hours * 3600 // step_seconds
    return [start + timedelta(seconds=step * step_seconds) for step in range(steps + 1)]


def simulate_tle_constellation(
    tle_path: str | Path,
    start: datetime,
    duration_hours: int = 24,
    step_seconds: int = 300,
    station_keeping: StationKeepingConfig | None = None,
    collision_avoidance: CollisionAvoidanceConfig | None = None,
) -> pd.DataFrame:
    """Generate SGP4 telemetry plus optional station-keeping and avoidance state."""
    satellites = load_tles(tle_path)
    timestamps = build_timestamps(start, duration_hours, step_seconds)
    telemetry = propagate_constellation(satellites, timestamps)
    frame = pd.DataFrame(telemetry).sort_values(["satellite", "timestamp"]).reset_index(drop=True)
    if station_keeping is None and collision_avoidance is None:
        return frame.sort_values(["timestamp", "satellite"]).reset_index(drop=True)

    control_config = station_keeping or StationKeepingConfig()
    state_fn = make_state_function(satellites)
    plans = (
        plan_avoidance_maneuvers(screen_conjunctions(frame, collision_avoidance, state_fn=state_fn), collision_avoidance)
        if collision_avoidance else pd.DataFrame()
    )
    if not plans.empty:
        available_times = list(frame["timestamp"].drop_duplicates().sort_values())
        for index, plan in plans.iterrows():
            tca = plan["tca_time"]
            desired_time = tca - timedelta(seconds=collision_avoidance.planning_lead_seconds)
            maneuver_time = max((time for time in available_times if time <= desired_time), default=available_times[0])
            duration_s = (tca - maneuver_time).total_seconds()
            position, velocity = state_fn(plan["maneuver_satellite"], maneuver_time)
            burned_velocity = apply_velocity_burn(velocity, float(plan["avoidance_delta_v_m_s"]), plan["maneuver_direction"])
            # Differential re-screen: the burn's effect is the difference between two-body
            # arcs with and without the burn, added to the SGP4 nominal at TCA. Comparing a
            # two-body arc directly with SGP4 would mix in 100+ km of model difference.
            burned_tca, _ = propagate_two_body(position, burned_velocity, duration_s)
            nominal_tca, _ = propagate_two_body(position, velocity, duration_s)
            displacement = burned_tca - nominal_tca
            target_tca = state_fn(plan["maneuver_satellite"], tca)[0] + displacement
            protected_tca = state_fn(plan["satellite_a"], tca)[0]
            separation = float(np.linalg.norm(target_tca - protected_tca))
            plans.loc[index, "maneuver_timestamp"] = maneuver_time
            plans.loc[index, "burn_displacement_at_tca_km"] = float(np.linalg.norm(displacement))
            plans.loc[index, "predicted_post_burn_separation_km"] = separation
            plans.loc[index, "avoidance_effectiveness"] = "CLEARED" if separation >= collision_avoidance.keep_out_distance_km else "INSUFFICIENT"
    alert_map: dict[tuple[object, str], dict[str, object]] = {}
    maneuver_map: dict[tuple[object, str], dict[str, object]] = {}
    for _, plan in plans.iterrows():
        for satellite in (plan["satellite_a"], plan["satellite_b"]):
            alert_map[(plan["timestamp"], satellite)] = plan.to_dict()
        maneuver_map[(plan.get("maneuver_timestamp", plan["timestamp"]), plan["maneuver_satellite"])] = plan.to_dict()

    controlled_rows: list[dict[str, object]] = []
    for _, satellite_rows in frame.groupby("satellite", sort=False):
        state = initialize_state(float(satellite_rows.iloc[0]["mean_orbit_altitude_km"]), control_config)
        previous_timestamp = None
        for _, row in satellite_rows.iterrows():
            timestamp = row["timestamp"]
            elapsed_seconds = 0.0 if previous_timestamp is None else (timestamp - previous_timestamp).total_seconds()
            result = advance_station_keeping(state, elapsed_seconds, control_config)
            alert = alert_map.get((timestamp, row["satellite"]))
            plan = maneuver_map.get((timestamp, row["satellite"]))
            result.update(
                {
                    "conjunction_alert": alert is not None,
                    "tca_time": alert["tca_time"] if alert else None,
                    "closest_approach_km": float(alert["closest_approach_km"]) if alert else None,
                    "relative_position_km": float(alert["relative_position_km"]) if alert and pd.notna(alert.get("relative_position_km")) else None,
                    "relative_velocity_km_s": float(alert["relative_velocity_km_s"]) if alert and pd.notna(alert.get("relative_velocity_km_s")) else None,
                    "time_to_closest_approach_s": float(alert["time_to_closest_approach_s"]) if alert and pd.notna(alert.get("time_to_closest_approach_s")) else None,
                    "predicted_closest_separation_km": float(alert["predicted_closest_separation_km"]) if alert and pd.notna(alert.get("predicted_closest_separation_km")) else None,
                    "keep_out_distance_km": float(alert["keep_out_distance_km"]) if alert and pd.notna(alert.get("keep_out_distance_km")) else collision_avoidance.keep_out_distance_km if collision_avoidance else None,
                    "ai_risk_probability": float(alert["ai_risk_probability"]) if alert and pd.notna(alert.get("ai_risk_probability")) else None,
                    "ai_risk_label": alert.get("ai_risk_label", "UNAVAILABLE") if alert else "CLEAR",
                    "ai_primary_driver": alert.get("ai_primary_driver", "NONE") if alert else "NONE",
                    "ai_model_version": alert.get("ai_model_version", "synthetic-logistic-v1") if alert else "synthetic-logistic-v1",
                    "avoidance_event": False,
                    "avoidance_delta_v_m_s": 0.0,
                    "avoidance_status": "CLEAR",
                    "predicted_post_burn_separation_km": float(alert["predicted_post_burn_separation_km"]) if alert else None,
                    "avoidance_effectiveness": alert.get("avoidance_effectiveness", "NOT_APPLICABLE") if alert else "NOT_APPLICABLE",
                }
            )
            if plan is not None:
                delta_v = float(plan["avoidance_delta_v_m_s"])
                propellant_used = propellant_required_kg(delta_v, state.propellant_remaining_kg, control_config)
                if propellant_used <= state.propellant_remaining_kg:
                    state.propellant_remaining_kg -= propellant_used
                    state.cumulative_delta_v_m_s += delta_v
                    result.update(
                        {
                            "fuel_remaining_kg": state.propellant_remaining_kg,
                            "fuel_percent": 100 * state.propellant_remaining_kg / control_config.initial_propellant_kg,
                            "cumulative_delta_v_m_s": state.cumulative_delta_v_m_s,
                            "avoidance_event": True,
                            "avoidance_delta_v_m_s": delta_v,
                            "avoidance_status": "MANEUVER_PLANNED",
                            "relative_position_km": float(plan["relative_position_km"]) if pd.notna(plan.get("relative_position_km")) else None,
                            "relative_velocity_km_s": float(plan["relative_velocity_km_s"]) if pd.notna(plan.get("relative_velocity_km_s")) else None,
                            "time_to_closest_approach_s": float(plan["time_to_closest_approach_s"]) if pd.notna(plan.get("time_to_closest_approach_s")) else None,
                            "predicted_closest_separation_km": float(plan["predicted_closest_separation_km"]) if pd.notna(plan.get("predicted_closest_separation_km")) else None,
                            "keep_out_distance_km": float(plan["keep_out_distance_km"]) if pd.notna(plan.get("keep_out_distance_km")) else collision_avoidance.keep_out_distance_km,
                            "ai_risk_probability": float(plan["ai_risk_probability"]) if pd.notna(plan.get("ai_risk_probability")) else None,
                            "ai_risk_label": plan.get("ai_risk_label", "UNAVAILABLE"),
                            "ai_primary_driver": plan.get("ai_primary_driver", "NONE"),
                            "ai_model_version": plan.get("ai_model_version", "synthetic-logistic-v1"),
                            "predicted_post_burn_separation_km": float(plan["predicted_post_burn_separation_km"]),
                            "avoidance_effectiveness": plan["avoidance_effectiveness"],
                        }
                    )
                else:
                    result["avoidance_status"] = "OUT_OF_FUEL"
            controlled_rows.append({**row.to_dict(), **result})
            previous_timestamp = timestamp
    return pd.DataFrame(controlled_rows).sort_values(["timestamp", "satellite"]).reset_index(drop=True)
