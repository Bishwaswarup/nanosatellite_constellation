"""Small maneuver-aware two-body numerical propagator in TEME-compatible Cartesian coordinates."""

from __future__ import annotations

import numpy as np

EARTH_MU_KM3_S2 = 398600.4418


def apply_velocity_burn(velocity_km_s: np.ndarray, delta_v_m_s: float, direction: str = "RETROGRADE") -> np.ndarray:
    """Apply an instantaneous prograde or retrograde burn along the velocity vector."""
    velocity = np.asarray(velocity_km_s, dtype=float)
    speed = np.linalg.norm(velocity)
    if speed == 0:
        raise ValueError("Cannot apply a burn to a zero-velocity state.")
    sign = -1 if direction.upper() == "RETROGRADE" else 1
    if direction.upper() not in {"RETROGRADE", "PROGRADE"}:
        raise ValueError("Direction must be PROGRADE or RETROGRADE.")
    return velocity + sign * (delta_v_m_s / 1_000) * velocity / speed


def _derivative(state: np.ndarray) -> np.ndarray:
    position, velocity = state[:3], state[3:]
    radius = np.linalg.norm(position)
    acceleration = -EARTH_MU_KM3_S2 * position / radius**3
    return np.concatenate((velocity, acceleration))


def propagate_two_body(position_km: np.ndarray, velocity_km_s: np.ndarray, duration_s: float, step_s: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a Cartesian state with RK4 under central Earth gravity only."""
    if duration_s < 0 or step_s <= 0:
        raise ValueError("Duration must be non-negative and step size positive.")
    state = np.concatenate((np.asarray(position_km, dtype=float), np.asarray(velocity_km_s, dtype=float)))
    remaining = float(duration_s)
    while remaining > 0:
        dt = min(step_s, remaining)
        k1 = _derivative(state)
        k2 = _derivative(state + dt * k1 / 2)
        k3 = _derivative(state + dt * k2 / 2)
        k4 = _derivative(state + dt * k3)
        state += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        remaining -= dt
    return state[:3], state[3:]
