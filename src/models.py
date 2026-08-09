"""Typed domain models shared by propagation and simulation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SatelliteDefinition:
    """A satellite initialized from a name and two-line element set."""

    name: str
    line1: str
    line2: str


@dataclass(frozen=True)
class StationKeepingConfig:
    """Parameters for a simplified deterministic station-keeping model."""

    decay_rate_km_per_day: float = 0.5
    altitude_tolerance_km: float = 0.25
    initial_propellant_kg: float = 2.0
    dry_mass_kg: float = 10.0
    specific_impulse_s: float = 220.0
