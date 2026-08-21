"""Synthetic, non-operational close-approach scenario for dashboard validation."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .risk_model import CollisionRiskAssessment, CollisionRiskFeatures, CollisionRiskModel


def synthetic_safe_pass_figure() -> tuple[go.Figure, float, float]:
    """Return an animated local-frame intercept and its maneuvered safe pass."""
    time_s = np.linspace(-300, 300, 61)
    reference = np.zeros((len(time_s), 3))
    test_before = np.column_stack((0.04 * time_s, np.zeros_like(time_s), np.zeros_like(time_s)))
    # A smooth phasing/avoidance displacement starts 120 seconds before closest approach.
    blend = np.clip((time_s + 120) / 180, 0, 1)
    test_after = test_before.copy()
    test_after[:, 1] = 3.0 * blend
    before_distance = float(np.linalg.norm(test_before[len(time_s) // 2] - reference[len(time_s) // 2]))
    after_distance = float(np.linalg.norm(test_after[len(time_s) // 2] - reference[len(time_s) // 2]))

    def traces(index: int):
        return [
            go.Scatter3d(x=[0], y=[0], z=[0], mode="markers+text", text=["Reference satellite"], textposition="top center", marker={"size": 7, "color": "#ffcb77"}, name="Reference"),
            go.Scatter3d(x=test_before[: index + 1, 0], y=test_before[: index + 1, 1], z=test_before[: index + 1, 2], mode="lines+markers", marker={"size": 4}, line={"color": "#d00000", "width": 5}, name="No maneuver"),
            go.Scatter3d(x=test_after[: index + 1, 0], y=test_after[: index + 1, 1], z=test_after[: index + 1, 2], mode="lines+markers", marker={"size": 4}, line={"color": "#0077b6", "width": 5}, name="Avoidance maneuver"),
        ]

    figure = go.Figure(data=traces(0), frames=[go.Frame(data=traces(i), name=str(i)) for i in range(len(time_s))])
    figure.update_layout(
        height=600, title="Synthetic test satellite: unsafe intercept vs. maneuvered safe pass",
        scene={"xaxis_title": "Along-track (km)", "yaxis_title": "Cross-track (km)", "zaxis_title": "Radial (km)", "aspectmode": "data"},
        updatemenus=[{"type": "buttons", "buttons": [{"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}]}, {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate"}]}]}],
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    return figure, before_distance, after_distance


def synthetic_scenario_risks() -> tuple[CollisionRiskAssessment, CollisionRiskAssessment]:
    """Score the controlled before/after encounter used by the dashboard test lab."""
    model = CollisionRiskModel()
    before = model.assess(CollisionRiskFeatures((0.0, 0.0, 0.0), (0.04, 0.0, 0.0), 0.0, 0.0, 2.0))
    after = model.assess(CollisionRiskFeatures((0.0, 2.0, 0.0), (0.04, 0.0, 0.0), 0.0, 2.0, 2.0))
    return before, after
