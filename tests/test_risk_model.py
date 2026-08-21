import pytest

from src.risk_model import CollisionRiskFeatures, CollisionRiskModel


def test_high_risk_synthetic_encounter_scores_above_low_risk_encounter():
    model = CollisionRiskModel()
    high = model.assess(CollisionRiskFeatures((0.1, 0.0, 0.0), (-0.02, 0.0, 0.0), 5.0, 0.05, 2.0))
    low = model.assess(CollisionRiskFeatures((20.0, 0.0, 0.0), (0.01, 0.0, 0.0), 3_600.0, 12.0, 2.0))
    assert high.probability > low.probability
    assert high.label in {"HIGH", "CRITICAL"}
    assert low.label == "LOW"
    assert high.primary_driver == "predicted separation"
    assert high.model_version == "synthetic-logistic-v1"


def test_risk_score_is_reproducible():
    model = CollisionRiskModel()
    features = CollisionRiskFeatures((1.0, -0.5, 0.25), (0.01, 0.02, -0.01), 120.0, 0.8, 2.0)
    assert model.assess(features) == model.assess(features)


@pytest.mark.parametrize(
    "features",
    [
        CollisionRiskFeatures((1.0, 2.0), (0.0, 0.0, 0.0), 0.0, 1.0, 2.0),
        CollisionRiskFeatures((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), -1.0, 1.0, 2.0),
        CollisionRiskFeatures((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, 1.0, 0.0),
    ],
)
def test_invalid_risk_input_is_rejected_clearly(features):
    with pytest.raises(ValueError):
        CollisionRiskModel().assess(features)
