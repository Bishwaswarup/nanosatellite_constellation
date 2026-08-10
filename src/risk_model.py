"""Reproducible, advisory collision-risk scoring for simulated encounters.

The coefficients below are intentionally fixed and calibrated only against
synthetic relative-motion cases.  This makes the demonstration deterministic
and inspectable; it does *not* turn the result into an operational collision
probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log1p

import numpy as np


@dataclass(frozen=True)
class CollisionRiskFeatures:
    """Relative state and closest-approach inputs for one candidate pair."""

    relative_position_km: tuple[float, float, float]
    relative_velocity_km_s: tuple[float, float, float]
    time_to_closest_approach_s: float
    predicted_closest_separation_km: float
    keep_out_distance_km: float


@dataclass(frozen=True)
class CollisionRiskAssessment:
    """A calibrated-for-simulation score and a readable priority label."""

    probability: float
    label: str


class CollisionRiskModel:
    """Small fixed logistic model, calibrated for synthetic simulation data.

    It deliberately has no training-time randomness or external dependencies.
    Its output is advisory: deterministic keep-out screening remains the
    system's safety authority.
    """

    def assess(self, features: CollisionRiskFeatures) -> CollisionRiskAssessment:
        self._validate(features)
        position_norm = float(np.linalg.norm(features.relative_position_km))
        velocity_norm = float(np.linalg.norm(features.relative_velocity_km_s))
        predicted_ratio = features.predicted_closest_separation_km / features.keep_out_distance_km
        current_ratio = position_norm / features.keep_out_distance_km
        urgency = exp(-features.time_to_closest_approach_s / 900.0)

        # Fixed, synthetic-data calibration: proximity is dominant, while
        # closing speed and short time-to-approach increase prioritization.
        logit = (
            -1.55
            + 4.25 * (1.0 - min(predicted_ratio, 2.0))
            + 0.45 * (1.0 - min(current_ratio, 2.0))
            + 0.30 * log1p(velocity_norm / 0.01)
            + 0.55 * urgency
        )
        probability = 1.0 / (1.0 + exp(-np.clip(logit, -30.0, 30.0)))
        label = "CRITICAL" if probability >= 0.80 else "HIGH" if probability >= 0.55 else "ELEVATED" if probability >= 0.30 else "LOW"
        return CollisionRiskAssessment(probability=float(probability), label=label)

    @staticmethod
    def _validate(features: CollisionRiskFeatures) -> None:
        vectors = (features.relative_position_km, features.relative_velocity_km_s)
        if any(len(vector) != 3 for vector in vectors):
            raise ValueError("Relative position and velocity must each contain exactly three values.")
        values = [*features.relative_position_km, *features.relative_velocity_km_s, features.time_to_closest_approach_s, features.predicted_closest_separation_km, features.keep_out_distance_km]
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("Collision-risk features must be finite numbers.")
        if features.time_to_closest_approach_s < 0:
            raise ValueError("Time to closest approach cannot be negative.")
        if features.predicted_closest_separation_km < 0 or features.keep_out_distance_km <= 0:
            raise ValueError("Closest separation must be non-negative and keep-out distance positive.")
