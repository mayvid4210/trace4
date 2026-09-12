"""Tests for non-degradation lap-time baselines."""

import pandas as pd
import pytest

from trace.physics import (
    apply_lagged_context_physics_model,
    expected_lap_time,
    fit_improved_physics_baseline,
    fit_lagged_context_physics_baseline,
    fit_lagged_context_physics_model,
    fit_physics_baseline,
    predict_lagged_context_physics_model,
)


def _improved_baseline_laps() -> pd.DataFrame:
    rows = []
    for index in range(40):
        driver = "A" if index % 2 == 0 else "B"
        compound = "SOFT" if index % 3 else "HARD"
        track_status = "1" if index % 4 else "12"
        rainfall = index % 5 == 0
        lap_time_seconds = (
            90
            + index * 0.1
            + (1 if driver == "B" else 0)
            + (0.5 if compound == "HARD" else 0)
            + (2 if rainfall else 0)
        )
        rows.append(
            {
                "LapTime": pd.Timedelta(seconds=lap_time_seconds),
                "RaceProgress": (index % 20 + 1) / 20,
                "TrackTemp": 20 + index % 5,
                "AirTemp": 15 + index % 4,
                "Humidity": 50 + index % 6,
                "Rainfall": rainfall,
                "Driver": driver,
                "Compound": compound,
                "TrackStatus": track_status,
                "TrackEvolutionProxy": 95 - index * 0.05,
                "AvgSpeed": 200 + index % 7,
                "MaxSpeed": 300 + index % 9,
                "AvgThrottle": 60 + index % 8,
                "BrakeUsage": 0.2 + (index % 3) * 0.01,
                "BrakeDuration": 20 + index % 5,
                "BrakingIntensity": 50 + index % 4,
                "BrakingFrequency": 8 + index % 3,
                "ThrottleAggressiveness": 80 + index % 6,
                "SpeedVariation": 65 + index % 5,
                "Position": float(index % 20 + 1),
                "RelativePaceToAhead": None if index % 7 == 0 else index * 0.01,
                "RelativePaceToBehind": None if index % 7 == 0 else -index * 0.01,
                "TyreLife": float(index % 10 + 1),
                "TyreAgeSquared": float((index % 10 + 1) ** 2),
            }
        )
    return pd.DataFrame(rows)


def test_expected_lap_time_returns_additive_baseline() -> None:
    lap_time = expected_lap_time(
        base_lap_time_seconds=90.0,
        estimated_fuel_load=10.0,
        fuel_time_per_unit_seconds=0.03,
        track_adjustment_seconds=0.5,
        weather_adjustment_seconds=1.0,
        driver_adjustment_seconds=-0.2,
        traffic_adjustment_seconds=0.7,
    )

    assert lap_time == pytest.approx(92.3)
    assert lap_time > 0


def test_more_estimated_fuel_increases_lap_time() -> None:
    low_fuel = expected_lap_time(90.0, 5.0, 0.03)
    high_fuel = expected_lap_time(90.0, 15.0, 0.03)

    assert high_fuel > low_fuel
    assert high_fuel - low_fuel == pytest.approx(0.3)


def test_adjustments_change_lap_time_consistently() -> None:
    baseline = expected_lap_time(90.0, 0.0, 0.03)
    conditions = expected_lap_time(
        90.0,
        0.0,
        0.03,
        track_adjustment_seconds=0.4,
        weather_adjustment_seconds=0.8,
        driver_adjustment_seconds=-0.3,
        traffic_adjustment_seconds=0.6,
    )

    assert conditions - baseline == pytest.approx(1.5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"base_lap_time_seconds": 0.0}, "base_lap_time_seconds"),
        ({"estimated_fuel_load": -1.0}, "estimated_fuel_load"),
        ({"fuel_time_per_unit_seconds": -0.01}, "fuel_time_per_unit_seconds"),
        ({"traffic_adjustment_seconds": -0.1}, "traffic_adjustment_seconds"),
        ({"weather_adjustment_seconds": float("inf")}, "weather_adjustment_seconds"),
    ],
)
def test_invalid_inputs_raise_value_error(kwargs: dict[str, float], message: str) -> None:
    inputs = {
        "base_lap_time_seconds": 90.0,
        "estimated_fuel_load": 10.0,
        "fuel_time_per_unit_seconds": 0.03,
    }
    inputs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        expected_lap_time(**inputs)


def test_fit_physics_baseline_fits_and_returns_expected_lap_times() -> None:
    laps = pd.DataFrame(
        {
            "LapTime": pd.to_timedelta(
                [89.5, 91.05, 92.85, 94.4, 91.5, 93.05, 94.85, 96.4], unit="s"
            ),
            "RaceProgress": [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
            "TrackTemp": [20, 22, 21, 23, 20, 22, 21, 23],
            "AirTemp": [15, 16, 15, 17, 15, 16, 15, 17],
            "Humidity": [60, 61, 63, 62, 60, 61, 63, 62],
            "Rainfall": [False, False, True, True, False, False, True, True],
            "Driver": ["A", "A", "A", "A", "B", "B", "B", "B"],
        }
    )
    original = laps.copy()

    result = fit_physics_baseline(laps)

    assert len(result["ExpectedLapTime"]) == len(laps)
    assert result["ExpectedLapTime"].tolist() == pytest.approx(
        laps["LapTime"].dt.total_seconds().tolist()
    )
    assert result["TyreResidual"].tolist() == pytest.approx([0.0] * len(laps))
    pd.testing.assert_frame_equal(laps, original)


def test_fit_physics_baseline_does_not_require_tyre_age_columns() -> None:
    laps = pd.DataFrame(
        {
            "LapTime": pd.to_timedelta([100, 101, 102, 103, 104, 105], unit="s"),
            "RaceProgress": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "TrackTemp": [20, 21, 19, 22, 20, 23],
            "AirTemp": [15, 16, 14, 17, 15, 18],
            "Humidity": [60, 61, 59, 62, 60, 63],
            "Rainfall": [False, True, False, True, False, True],
            "Driver": ["A", "A", "A", "A", "A", "A"],
        }
    )

    result = fit_physics_baseline(laps)

    assert len(result) == len(laps)


@pytest.mark.parametrize("missing_column", ["LapTime", "RaceProgress", "Driver"])
def test_fit_physics_baseline_requires_predictor_columns(missing_column: str) -> None:
    laps = pd.DataFrame(
        {
            "LapTime": pd.to_timedelta([100], unit="s"),
            "RaceProgress": [0.1],
            "TrackTemp": [20.0],
            "AirTemp": [15.0],
            "Humidity": [60.0],
            "Rainfall": [False],
            "Driver": ["A"],
        }
    ).drop(columns=missing_column)

    with pytest.raises(ValueError, match=missing_column):
        fit_physics_baseline(laps)


@pytest.mark.parametrize(
    "laps",
    [
        pd.DataFrame(
            columns=[
                "LapTime",
                "RaceProgress",
                "TrackTemp",
                "AirTemp",
                "Humidity",
                "Rainfall",
                "Driver",
            ]
        ),
        pd.DataFrame(
            {
                "LapTime": pd.to_timedelta([0], unit="s"),
                "RaceProgress": [0.1],
                "TrackTemp": [20.0],
                "AirTemp": [15.0],
                "Humidity": [60.0],
                "Rainfall": [False],
                "Driver": ["A"],
            }
        ),
    ],
)
def test_fit_physics_baseline_rejects_invalid_or_empty_data(laps: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        fit_physics_baseline(laps)


def test_improved_baseline_uses_categorical_context_without_tyre_age() -> None:
    laps = _improved_baseline_laps()
    original = laps.copy()

    result = fit_improved_physics_baseline(laps)

    assert len(result) == len(laps)
    assert result["ExpectedLapTime"].notna().all()
    assert result["TyreResidual"].tolist() == pytest.approx(
        laps["LapTime"].dt.total_seconds() - result["ExpectedLapTime"]
    )
    assert any(name.startswith("Driver_") for name in result.attrs["predictors"])
    assert any(name.startswith("Compound_") for name in result.attrs["predictors"])
    assert any(name.startswith("TrackStatus_") for name in result.attrs["predictors"])
    assert not any("TyreLife" in name or "TyreAge" in name for name in result.attrs["predictors"])
    pd.testing.assert_frame_equal(laps, original)


def test_diagnostic_baseline_handles_missing_opponent_relative_pace() -> None:
    result = fit_improved_physics_baseline(_improved_baseline_laps(), diagnostic=True)

    assert len(result) == 40
    assert result["ExpectedLapTime"].notna().all()
    assert "RelativePaceToAheadAvailable" in result.attrs["predictors"]
    assert any(
        name.startswith("RelativePaceToBehindAvailable")
        for name in result.attrs["predictors"] + result.attrs["rejected_predictors"]
    )


def test_production_baseline_excludes_diagnostic_predictors() -> None:
    laps = _improved_baseline_laps()

    production = fit_improved_physics_baseline(laps)
    diagnostic = fit_improved_physics_baseline(laps, diagnostic=True)

    assert "AvgSpeed" not in production.attrs["predictors"]
    assert "RelativePaceToAhead" not in production.attrs["predictors"]
    assert "TrackEvolutionProxy" not in production.attrs["predictors"]
    assert "AvgSpeed" in diagnostic.attrs["predictors"]
    assert "RelativePaceToAhead" in diagnostic.attrs["predictors"]


def test_improved_baseline_rejects_non_finite_predictors() -> None:
    laps = _improved_baseline_laps()
    laps["TrackTemp"] = laps["TrackTemp"].astype(float)
    laps.loc[0, "TrackTemp"] = float("inf")

    with pytest.raises(ValueError, match="finite"):
        fit_improved_physics_baseline(laps)


def test_lagged_context_baseline_excludes_current_and_own_tyre_features() -> None:
    laps = _improved_baseline_laps()
    lagged_columns = [
        "PrevAvgSpeed",
        "PrevAvgThrottle",
        "PrevBrakeUsage",
        "PrevBrakeDuration",
        "PrevBrakingIntensity",
        "PrevBrakingFrequency",
        "PrevThrottleAggressiveness",
        "PrevSpeedVariation",
        "PrevPosition",
        "PrevRelativePaceToAhead",
        "PrevRelativePaceToBehind",
        "PrevOpponentAheadTyreLife",
        "PrevTrackEvolutionProxy",
    ]
    for index, column in enumerate(lagged_columns):
        laps[column] = [None if row % 7 == 0 else row + index for row in range(len(laps))]
    laps["PrevOpponentAheadCompound"] = [
        None if row % 7 == 0 else "SOFT" for row in range(len(laps))
    ]
    original = laps.copy()

    result = fit_lagged_context_physics_baseline(laps)

    assert len(result) == len(laps)
    assert result["TyreResidual"].tolist() == pytest.approx(
        laps["LapTime"].dt.total_seconds() - result["ExpectedLapTime"]
    )
    assert "PrevAvgSpeed" in result.attrs["predictors"]
    assert "AvgSpeed" not in result.attrs["predictors"]
    assert "TrackEvolutionProxy" not in result.attrs["predictors"]
    assert any(
        name.startswith("PrevTrackEvolutionProxy")
        for name in result.attrs["predictors"] + result.attrs["rejected_predictors"]
    )
    assert "TyreLife" not in result.attrs["predictors"]
    assert "TyreAgeSquared" not in result.attrs["predictors"]
    pd.testing.assert_frame_equal(laps, original)


def test_train_fitted_lagged_physics_model_does_not_use_validation_lap_times() -> None:
    laps = _improved_baseline_laps()
    lagged_columns = [
        "PrevAvgSpeed", "PrevAvgThrottle", "PrevBrakeUsage", "PrevBrakeDuration",
        "PrevBrakingIntensity", "PrevBrakingFrequency", "PrevThrottleAggressiveness",
        "PrevSpeedVariation", "PrevPosition", "PrevRelativePaceToAhead",
        "PrevRelativePaceToBehind", "PrevOpponentAheadTyreLife", "PrevTrackEvolutionProxy",
    ]
    for offset, column in enumerate(lagged_columns):
        laps[column] = [None if row % 6 == 0 else float(row + offset) for row in range(len(laps))]
    laps["PrevOpponentAheadCompound"] = [None if row % 6 == 0 else "SOFT" for row in range(len(laps))]
    train, validation = laps.iloc[:30].copy(), laps.iloc[30:].copy()

    model = fit_lagged_context_physics_model(train)
    expected = predict_lagged_context_physics_model(model, validation)
    changed_validation = validation.copy()
    changed_validation["LapTime"] += pd.Timedelta(seconds=1000)
    same_train_model = fit_lagged_context_physics_model(train)
    changed_expected = predict_lagged_context_physics_model(same_train_model, changed_validation)
    scored = apply_lagged_context_physics_model(model, changed_validation)

    assert expected == pytest.approx(changed_expected)
    assert scored["ExpectedLapTime"].to_numpy() == pytest.approx(expected)
    assert scored["TyreResidual"].to_numpy() == pytest.approx(
        changed_validation["LapTime"].dt.total_seconds().to_numpy() - expected
    )
