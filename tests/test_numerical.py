import numpy as np

from src.numerical import apply_velocity_burn, propagate_two_body


def test_retrograde_burn_reduces_speed():
    velocity = np.array([0.0, 7.6, 0.0])
    burned = apply_velocity_burn(velocity, 10.0, "RETROGRADE")
    assert np.linalg.norm(burned) == np.linalg.norm(velocity) - 0.01


def test_two_body_propagation_preserves_orbit_radius_approximately():
    position = np.array([6778.137, 0.0, 0.0])
    velocity = np.array([0.0, 7.6686, 0.0])
    propagated_position, _ = propagate_two_body(position, velocity, duration_s=600, step_s=10)
    assert abs(np.linalg.norm(propagated_position) - np.linalg.norm(position)) < 1.0
