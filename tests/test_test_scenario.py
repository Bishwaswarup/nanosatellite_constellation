from src.test_scenario import synthetic_safe_pass_figure, synthetic_scenario_risks


def test_synthetic_maneuver_increases_closest_approach():
    _, unsafe_distance, safe_distance = synthetic_safe_pass_figure()
    assert unsafe_distance < safe_distance
    assert safe_distance == 2.0


def test_synthetic_maneuver_lowers_advisory_ai_risk():
    before, after = synthetic_scenario_risks()
    assert before.probability > after.probability
