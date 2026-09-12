import pandas as pd
import pytest

import trace.features as feature_module
from trace.data import aggregate_telemetry
from trace.features import add_telemetry_features, build_tyre_features


def test_build_tyre_features_preserves_real_lap_columns_and_adds_features() -> None:
    laps = pd.DataFrame(
        {
            "LapNumber": [1.0, 5.0, 10.0],
            "TyreLife": [1.0, 3.0, 4.0],
            "Stint": [1.0, 1.0, 2.0],
        }
    )
    original = laps.copy()

    features = build_tyre_features(laps, total_race_laps=10)

    assert features["LapNumber"].tolist() == [1.0, 5.0, 10.0]
    assert features["TyreLife"].tolist() == [1.0, 3.0, 4.0]
    assert features["Stint"].tolist() == [1.0, 1.0, 2.0]
    assert features["RaceProgress"].tolist() == [0.1, 0.5, 1.0]
    assert features["TyreAgeSquared"].tolist() == [1.0, 9.0, 16.0]
    pd.testing.assert_frame_equal(laps, original)


@pytest.mark.parametrize("missing_column", ["LapNumber", "TyreLife", "Stint"])
def test_build_tyre_features_requires_real_source_columns(missing_column: str) -> None:
    laps = pd.DataFrame(
        {"LapNumber": [1], "TyreLife": [1], "Stint": [1]}
    ).drop(columns=missing_column)

    with pytest.raises(ValueError, match=missing_column):
        build_tyre_features(laps, total_race_laps=10)


@pytest.mark.parametrize("total_race_laps", [0, -1, float("nan"), float("inf"), True])
def test_build_tyre_features_rejects_invalid_total_race_laps(
    total_race_laps: float,
) -> None:
    laps = pd.DataFrame({"LapNumber": [1], "TyreLife": [1], "Stint": [1]})

    with pytest.raises(ValueError, match="total_race_laps"):
        build_tyre_features(laps, total_race_laps)


def test_add_telemetry_features_uses_real_telemetry_aggregates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    laps = pd.DataFrame(
        {"Driver": ["VER"], "LapNumber": [1.0], "TyreLife": [1.0], "Stint": [1.0]}
    )
    original = laps.copy()
    telemetry = pd.DataFrame(
        {
            "Time": pd.to_timedelta([0, 1, 2, 3], unit="s"),
            "Speed": [100, 120, 140, 160],
            "Throttle": [50, 60, 70, 80],
            "Brake": [False, True, True, False],
        }
    )
    monkeypatch.setattr(
        feature_module,
        "get_lap_telemetry_features",
        lambda session, driver, lap_number: aggregate_telemetry(telemetry),
    )

    features = add_telemetry_features(laps, session=object())

    assert features.loc[0, "AvgSpeed"] == 130.0
    assert features.loc[0, "MaxSpeed"] == 160.0
    assert features.loc[0, "AvgThrottle"] == 65.0
    assert features.loc[0, "BrakeUsage"] == 0.5
    assert features.loc[0, "BrakeDuration"] == 2.0
    pd.testing.assert_frame_equal(laps, original)


def test_add_telemetry_features_keeps_missing_values_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    laps = pd.DataFrame({"Driver": ["VER"], "LapNumber": [1.0]})
    monkeypatch.setattr(
        feature_module,
        "get_lap_telemetry_features",
        lambda session, driver, lap_number: (_ for _ in ()).throw(
            ValueError("telemetry unavailable")
        ),
    )

    features = add_telemetry_features(laps, session=object())

    assert features[["AvgSpeed", "MaxSpeed", "AvgThrottle", "BrakeUsage", "BrakeDuration"]].isna().all(axis=None)
