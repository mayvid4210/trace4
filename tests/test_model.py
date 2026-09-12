import numpy as np
import pandas as pd
import pytest

from trace.features import TYRE_POSITIONS
from trace.model import (
    FORBIDDEN_PREDICTORS,
    SUPPORTED_TYRE_POSITIONS,
    add_tyre_model_predictions,
    fit_tyre_degradation_model,
    predict_tyre_residual,
    temporal_train_validation_split,
    tyre_regime,
)


def _model_laps() -> pd.DataFrame:
    rows = []
    for lap in range(1, 13):
        for driver, offset, compound in (("A", 0.0, "SOFT"), ("B", 0.4, "MEDIUM")):
            tyre_life = float(lap)
            residual = 0.2 * tyre_life + offset
            rows.append(
                {
                    "Driver": driver,
                    "LapNumber": float(lap),
                    "TyreLife": tyre_life,
                    "TyreAgeSquared": tyre_life**2,
                    "Compound": compound,
                    "Stint": 1.0,
                    "RaceProgress": lap / 12,
                    "TrackTemp": 30.0 + lap / 10,
                    "AirTemp": 20.0,
                    "Humidity": 50.0,
                    "Rainfall": False,
                    "PrevAvgSpeed": 200.0 + lap,
                    "PrevAvgThrottle": 60.0 + lap,
                    "PrevBrakeUsage": 0.2,
                    "PrevBrakeDuration": 4.0,
                    "PrevBrakingIntensity": 0.8,
                    "PrevBrakingFrequency": 5.0,
                    "PrevThrottleAggressiveness": 70.0,
                    "PrevSpeedVariation": 30.0,
                    "PrevTrackEvolutionProxy": 91.0 - lap / 10,
                    "PrevPosition": 1.0 if driver == "A" else 2.0,
                    "PrevRelativePaceToAhead": np.nan if driver == "A" else 0.4,
                    "PrevRelativePaceToBehind": -0.4 if driver == "A" else np.nan,
                    "PrevOpponentAheadCompound": np.nan if driver == "A" else "SOFT",
                    "PrevOpponentAheadTyreLife": np.nan if driver == "A" else tyre_life - 1,
                    "ExpectedLapTime": 90.0,
                    "TyreResidual": residual,
                    # Deliberately tempting leakage fields, which must be ignored.
                    "LapTime": pd.Timedelta(seconds=90 + residual),
                    "RollingTyreResidual": residual,
                    "AvgSpeed": 999.0,
                    "TrackEvolutionProxy": 999.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("model_type", ["xgboost", "ridge", "random_forest"])
def test_model_trains_predicts_finitely_and_preserves_input(model_type: str) -> None:
    laps = _model_laps()
    original = laps.copy()

    model = fit_tyre_degradation_model(laps, model_type=model_type, n_estimators=10)
    predicted = predict_tyre_residual(model, laps)

    assert predicted.shape == (len(laps),)
    assert np.isfinite(predicted).all()
    assert "Driver" in model["predictors"]
    assert "Compound" in model["predictors"]
    assert model["model_type"] == model_type
    assert not set(model["predictors"]).intersection(FORBIDDEN_PREDICTORS)
    pd.testing.assert_frame_equal(laps, original)


def test_temporal_split_preserves_chronology_and_has_no_overlap() -> None:
    laps = _model_laps()
    train, validation = temporal_train_validation_split(laps, validation_fraction=0.25)

    assert train["LapNumber"].max() < validation["LapNumber"].min()
    assert set(train.index).isdisjoint(validation.index)
    assert len(train) + len(validation) == len(laps)


def test_final_prediction_uses_expected_plus_predicted_residual() -> None:
    laps = _model_laps()
    model = fit_tyre_degradation_model(laps, n_estimators=10)
    result = add_tyre_model_predictions(model, laps)

    assert result["PredictedLapTime"].to_numpy() == pytest.approx(
        result["ExpectedLapTime"].to_numpy() + result["PredictedTyreResidual"].to_numpy()
    )
    assert "TyrePosition" not in result
    pd.testing.assert_frame_equal(laps, _model_laps())


def test_prediction_supports_non_contiguous_validation_indexes() -> None:
    laps = _model_laps()
    model = fit_tyre_degradation_model(laps.iloc[:12], n_estimators=10)

    result = add_tyre_model_predictions(model, laps.iloc[12:])

    assert result.index.tolist() == list(range(12, len(laps)))
    assert result["PredictedLapTime"].notna().all()


def test_validation_only_categories_do_not_enter_training_encoder() -> None:
    laps = _model_laps()
    train, validation = laps.iloc[:12], laps.iloc[12:].copy()
    validation["Driver"] = "VALIDATION_ONLY"
    validation["Compound"] = "WET"

    model = fit_tyre_degradation_model(train, n_estimators=10)
    predictions = predict_tyre_residual(model, validation)

    assert "Driver_VALIDATION_ONLY" not in model["encoded_columns"]
    assert "Compound_WET" not in model["encoded_columns"]
    assert np.isfinite(predictions).all()


def test_model_does_not_need_fabricated_per_tyre_labels() -> None:
    model = fit_tyre_degradation_model(_model_laps(), n_estimators=10)

    assert SUPPORTED_TYRE_POSITIONS == TYRE_POSITIONS == ("FL", "FR", "RL", "RR")
    assert "TyrePosition" not in model["predictors"]


def test_feature_ablation_sets_and_regimes_use_only_real_fields() -> None:
    laps = _model_laps()
    laps.loc[0, "Rainfall"] = True
    laps.loc[1, "Compound"] = "INTERMEDIATE"

    core = fit_tyre_degradation_model(laps, feature_set="tyre_core", n_estimators=10)
    environment = fit_tyre_degradation_model(laps, feature_set="tyre_environment", n_estimators=10)

    assert set(core["predictors"]) == {"TyreLife", "TyreAgeSquared", "Compound", "Stint", "RaceProgress"}
    assert "TrackTemp" in environment["predictors"]
    assert tyre_regime(laps).iloc[:2].tolist() == ["wet_mixed", "wet_mixed"]


def test_model_rejects_missing_target_and_invalid_numeric_values() -> None:
    laps = _model_laps()
    with pytest.raises(ValueError, match="TyreResidual"):
        fit_tyre_degradation_model(laps.drop(columns="TyreResidual"))

    laps["TyreLife"] = laps["TyreLife"].astype(object)
    laps.loc[0, "TyreLife"] = "not-a-number"
    with pytest.raises(ValueError, match="TyreLife"):
        fit_tyre_degradation_model(laps)


@pytest.mark.parametrize("fraction", [0, 1, -0.1])
def test_temporal_split_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        temporal_train_validation_split(_model_laps(), validation_fraction=fraction)
