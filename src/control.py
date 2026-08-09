"""Deterministic station-keeping, orbit-raise, and propellant calculations."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

from .models import StationKeepingConfig

EARTH_MU_KM3_S2 = 398600.4418
EARTH_EQUATORIAL_RADIUS_KM = 6378.137
STANDARD_GRAVITY_M_S2 = 9.80665


@dataclass
class StationKeepingState:
    """Mutable resource and altitude state for one virtual spacecraft."""

    nominal_altitude_km: float
    controlled_altitude_km: float
    propellant_remaining_kg: float
    cumulative_delta_v_m_s: float = 0.0
    maneuver_count: int = 0
    out_of_fuel: bool = False


def validate_config(config: StationKeepingConfig) -> None:
    if config.decay_rate_km_per_day < 0 or config.altitude_tolerance_km <= 0:
        raise ValueError("Decay rate must be non-negative and altitude tolerance positive.")
    if config.initial_propellant_kg < 0 or config.dry_mass_kg <= 0 or config.specific_impulse_s <= 0:
        raise ValueError("Masses and specific impulse must be physically valid.")


def circular_orbit_raise_delta_v_m_s(current_altitude_km: float, target_altitude_km: float) -> float:
    """Return ideal two-burn Hohmann-transfer delta-v for a circular orbit raise."""
    if target_altitude_km <= current_altitude_km:
        return 0.0
    r1 = EARTH_EQUATORIAL_RADIUS_KM + current_altitude_km
    r2 = EARTH_EQUATORIAL_RADIUS_KM + target_altitude_km
    first_burn_km_s = sqrt(EARTH_MU_KM3_S2 / r1) * (sqrt(2 * r2 / (r1 + r2)) - 1)
    second_burn_km_s = sqrt(EARTH_MU_KM3_S2 / r2) * (1 - sqrt(2 * r1 / (r1 + r2)))
    return (first_burn_km_s + second_burn_km_s) * 1_000


def propellant_required_kg(delta_v_m_s: float, propellant_remaining_kg: float, config: StationKeepingConfig) -> float:
    """Calculate propellant use with the Tsiolkovsky rocket equation."""
    if delta_v_m_s <= 0:
        return 0.0
    initial_mass_kg = config.dry_mass_kg + propellant_remaining_kg
    final_mass_kg = initial_mass_kg / exp(delta_v_m_s / (config.specific_impulse_s * STANDARD_GRAVITY_M_S2))
    return initial_mass_kg - final_mass_kg


def initialize_state(nominal_altitude_km: float, config: StationKeepingConfig) -> StationKeepingState:
    validate_config(config)
    return StationKeepingState(
        nominal_altitude_km=nominal_altitude_km,
        controlled_altitude_km=nominal_altitude_km,
        propellant_remaining_kg=config.initial_propellant_kg,
    )


def advance_station_keeping(
    state: StationKeepingState, elapsed_seconds: float, config: StationKeepingConfig
) -> dict[str, float | int | str | bool]:
    """Apply virtual decay and perform an altitude-restoration maneuver if needed."""
    state.controlled_altitude_km -= config.decay_rate_km_per_day * elapsed_seconds / 86_400
    minimum_altitude_km = state.nominal_altitude_km - config.altitude_tolerance_km
    maneuver_delta_v_m_s = 0.0
    propellant_used_kg = 0.0
    maneuver_performed = False

    if state.controlled_altitude_km < minimum_altitude_km and not state.out_of_fuel:
        maneuver_delta_v_m_s = circular_orbit_raise_delta_v_m_s(
            state.controlled_altitude_km, state.nominal_altitude_km
        )
        propellant_used_kg = propellant_required_kg(maneuver_delta_v_m_s, state.propellant_remaining_kg, config)
        if propellant_used_kg <= state.propellant_remaining_kg:
            state.propellant_remaining_kg -= propellant_used_kg
            state.controlled_altitude_km = state.nominal_altitude_km
            state.cumulative_delta_v_m_s += maneuver_delta_v_m_s
            state.maneuver_count += 1
            maneuver_performed = True
        else:
            state.out_of_fuel = True

    status = "OUT_OF_FUEL" if state.out_of_fuel else ("MANEUVER" if maneuver_performed else "NOMINAL")
    fuel_percent = 100 * state.propellant_remaining_kg / config.initial_propellant_kg if config.initial_propellant_kg else 0.0
    return {
        "controlled_mean_altitude_km": state.controlled_altitude_km,
        "fuel_remaining_kg": state.propellant_remaining_kg,
        "fuel_percent": fuel_percent,
        "maneuver_delta_v_m_s": maneuver_delta_v_m_s if maneuver_performed else 0.0,
        "propellant_used_kg": propellant_used_kg if maneuver_performed else 0.0,
        "cumulative_delta_v_m_s": state.cumulative_delta_v_m_s,
        "maneuver_count": state.maneuver_count,
        "station_keeping_event": maneuver_performed,
        "station_keeping_status": status,
    }
