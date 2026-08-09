"""TLE ingestion, SGP4 propagation, and TEME-to-ECEF transformations."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, ITRS, TEME
from astropy.time import Time
from sgp4.api import SGP4_ERRORS, Satrec, jday

from .models import SatelliteDefinition

WGS84_EQUATORIAL_RADIUS_KM = 6378.137
EARTH_MU_KM3_S2 = 398600.4418


def load_tles(path: str | Path) -> list[SatelliteDefinition]:
    """Load a standard three-line (name + two TLE lines) constellation file."""
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) % 3:
        raise ValueError("TLE file must contain repeating name, line 1, line 2 records.")

    satellites: list[SatelliteDefinition] = []
    for index in range(0, len(lines), 3):
        name, line1, line2 = lines[index : index + 3]
        if not line1.startswith("1 ") or not line2.startswith("2 "):
            raise ValueError(f"Invalid TLE record for {name!r}.")
        satellites.append(SatelliteDefinition(name=name, line1=line1, line2=line2))
    return satellites


def propagate_satellite(satellite: SatelliteDefinition, when: datetime) -> dict[str, float | str | datetime]:
    """Return TEME and ECEF state-derived telemetry for one satellite at UTC time."""
    if when.tzinfo is None:
        raise ValueError("Propagation timestamps must be timezone-aware.")
    when_utc = when.astimezone(timezone.utc)
    satrec = Satrec.twoline2rv(satellite.line1, satellite.line2)
    jd, fraction = jday(
        when_utc.year,
        when_utc.month,
        when_utc.day,
        when_utc.hour,
        when_utc.minute,
        when_utc.second + when_utc.microsecond / 1_000_000,
    )
    error, position_km, velocity_km_s = satrec.sgp4(jd, fraction)
    if error:
        raise RuntimeError(f"SGP4 failed for {satellite.name}: {SGP4_ERRORS[error]}")

    time = Time(when_utc)
    teme = TEME(
        CartesianRepresentation(np.asarray(position_km) * u.km),
        obstime=time,
    )
    itrs = teme.transform_to(ITRS(obstime=time))
    ecef_km = itrs.cartesian.xyz.to_value(u.km)
    radius_km = float(np.linalg.norm(ecef_km))
    latitude_deg = float(itrs.earth_location.lat.to_value(u.deg))
    longitude_deg = float(itrs.earth_location.lon.to_value(u.deg))
    altitude_km = float(itrs.earth_location.height.to_value(u.km))
    mean_motion_rad_s = satrec.no_kozai / 60
    semi_major_axis_km = (EARTH_MU_KM3_S2 / mean_motion_rad_s**2) ** (1 / 3)

    return {
        "timestamp": when_utc,
        "satellite": satellite.name,
        "teme_x_km": position_km[0],
        "teme_y_km": position_km[1],
        "teme_z_km": position_km[2],
        "teme_vx_km_s": velocity_km_s[0],
        "teme_vy_km_s": velocity_km_s[1],
        "teme_vz_km_s": velocity_km_s[2],
        "ecef_x_km": ecef_km[0],
        "ecef_y_km": ecef_km[1],
        "ecef_z_km": ecef_km[2],
        "latitude_deg": latitude_deg,
        "longitude_deg": longitude_deg,
        "altitude_km": altitude_km,
        "mean_orbit_altitude_km": semi_major_axis_km - WGS84_EQUATORIAL_RADIUS_KM,
        "geocentric_radius_km": radius_km,
    }


def propagate_constellation(
    satellites: Iterable[SatelliteDefinition], timestamps: Iterable[datetime]
) -> list[dict[str, float | str | datetime]]:
    """Propagate every satellite at every supplied timestamp."""
    return [propagate_satellite(satellite, when) for when in timestamps for satellite in satellites]
