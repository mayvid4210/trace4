import pandas as pd
import pytest

import trace.features as feature_module
from trace.data import aggregate_telemetry
from trace.features import (
    TYRE_POSITIONS,
    add_telemetry_features,
    add_opponent_context_features,
    add_lagged_context_features,
    add_track_evolution_proxy,
    add_tyre_residual_features,
    add_weather_track_features,
    build_tyre_features,
)


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
    assert features.loc[0, "BrakingFrequency"] == 1.0
    assert features.loc[0, "ThrottleAggressiveness"] == 70.0
    assert len(features) == len(laps)
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

    assert features[
        [
            "AvgSpeed",
            "MaxSpeed",
            "AvgThrottle",
            "BrakeUsage",
            "BrakeDuration",
            "BrakingIntensity",
            "BrakingFrequency",
            "ThrottleAggressiveness",
            "SpeedVariation",
        ]
    ].isna().all(axis=None)


def test_add_telemetry_features_does_not_share_values_between_laps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    laps = pd.DataFrame({"Driver": ["A", "A"], "LapNumber": [1.0, 2.0]})

    def lap_values(session: object, driver: str, lap_number: float) -> dict[str, float]:
        return {
            "average_speed": lap_number,
            "maximum_speed": lap_number,
            "average_throttle": lap_number,
            "brake_usage_fraction": lap_number,
            "brake_duration_seconds": lap_number,
            "braking_intensity": lap_number,
            "braking_frequency": lap_number,
            "throttle_aggressiveness": lap_number,
            "speed_variation": lap_number,
        }

    monkeypatch.setattr(feature_module, "get_lap_telemetry_features", lap_values)

    features = add_telemetry_features(laps, session=object())

    assert features["AvgSpeed"].tolist() == [1.0, 2.0]
    assert features["BrakingFrequency"].tolist() == [1.0, 2.0]


def test_add_weather_track_features_preserves_real_source_values() -> None:
    features = pd.DataFrame(
        {
            "LapNumber": [1.0, 2.0],
            "RaceProgress": [0.1, 0.2],
            "AvgSpeed": [200.0, 201.0],
        }
    )
    cleaned_laps = pd.DataFrame(
        {
            "TrackTemp": [20.0, 21.0],
            "AirTemp": [15.0, 16.0],
            "Humidity": [70.0, 71.0],
            "Rainfall": [True, False],
            "TrackStatus": ["12", "1"],
        }
    )
    original_features = features.copy()
    original_laps = cleaned_laps.copy()

    result = add_weather_track_features(features, cleaned_laps)

    assert result["TrackTemp"].tolist() == [20.0, 21.0]
    assert result["AirTemp"].tolist() == [15.0, 16.0]
    assert result["Humidity"].tolist() == [70.0, 71.0]
    assert result["Rainfall"].tolist() == [True, False]
    assert pd.api.types.is_bool_dtype(result["Rainfall"])
    assert result["TrackStatus"].tolist() == ["12", "1"]
    assert result["TrackStatus"].dtype == object
    pd.testing.assert_frame_equal(result[original_features.columns], original_features)
    pd.testing.assert_frame_equal(features, original_features)
    pd.testing.assert_frame_equal(cleaned_laps, original_laps)


def test_add_weather_track_features_requires_all_source_columns() -> None:
    features = pd.DataFrame({"LapNumber": [1.0]})
    cleaned_laps = pd.DataFrame(
        {
            "TrackTemp": [20.0],
            "AirTemp": [15.0],
            "Humidity": [70.0],
            "Rainfall": [False],
        }
    )

    with pytest.raises(ValueError, match="TrackStatus"):
        add_weather_track_features(features, cleaned_laps)


def test_tyre_positions_are_supported_without_fabricating_lap_values() -> None:
    assert TYRE_POSITIONS == ("FL", "FR", "RL", "RR")


def test_add_tyre_residual_features_uses_only_current_and_previous_laps() -> None:
    laps = pd.DataFrame(
        {
            "Driver": ["A", "A", "A", "A", "A", "B", "B"],
            "Stint": [1, 1, 1, 2, 2, 1, 1],
            "LapNumber": [2, 1, 3, 1, 2, 1, 2],
            "TyreResidual": [20.0, 10.0, 30.0, 100.0, 200.0, 1000.0, 2000.0],
            "ExistingFeature": list(range(7)),
        }
    )
    original = laps.copy()

    result = add_tyre_residual_features(laps)

    assert result["PreviousTyreResidual"].tolist() == pytest.approx(
        [10.0, float("nan"), 20.0, float("nan"), 100.0, float("nan"), 1000.0],
        nan_ok=True,
    )
    assert result["RollingTyreResidual"].tolist() == pytest.approx(
        [15.0, 10.0, 20.0, 100.0, 150.0, 1000.0, 1500.0]
    )
    assert pd.isna(result.loc[1, "PreviousTyreResidual"])
    assert pd.isna(result.loc[3, "PreviousTyreResidual"])
    assert pd.isna(result.loc[5, "PreviousTyreResidual"])
    assert result.loc[1, "RollingTyreResidual"] == 10.0
    pd.testing.assert_frame_equal(result[original.columns], original)
    pd.testing.assert_frame_equal(laps, original)


def test_add_tyre_residual_features_rejects_missing_residuals() -> None:
    laps = pd.DataFrame({"Driver": ["A"], "Stint": [1], "LapNumber": [1]})

    with pytest.raises(ValueError, match="TyreResidual"):
        add_tyre_residual_features(laps)


@pytest.mark.parametrize("window", [0, -1, 1.5, True])
def test_add_tyre_residual_features_rejects_invalid_window(window: object) -> None:
    laps = pd.DataFrame(
        {"Driver": ["A"], "Stint": [1], "LapNumber": [1], "TyreResidual": [1.0]}
    )

    with pytest.raises(ValueError, match="window"):
        add_tyre_residual_features(laps, window=window)


def test_add_opponent_context_features_matches_adjacent_positions() -> None:
    laps = pd.DataFrame(
        {
            "Driver": ["LEADER", "MIDDLE", "LAST"],
            "LapNumber": [1.0, 1.0, 1.0],
            "Position": [1.0, 2.0, 3.0],
            "Compound": ["SOFT", "MEDIUM", "HARD"],
            "TyreLife": [2.0, 3.0, 4.0],
            "LapTime": pd.to_timedelta([90.0, 91.0, 92.0], unit="s"),
            "ExistingFeature": [1, 2, 3],
        }
    )
    original = laps.copy()

    result = add_opponent_context_features(laps)

    middle = result.loc[result["Driver"] == "MIDDLE"].iloc[0]
    leader = result.loc[result["Driver"] == "LEADER"].iloc[0]
    last = result.loc[result["Driver"] == "LAST"].iloc[0]
    assert result["Driver"].tolist() == ["LEADER", "MIDDLE", "LAST"]
    assert result["Position"].tolist() == [1.0, 2.0, 3.0]
    assert middle["OpponentAheadDriver"] == "LEADER"
    assert middle["OpponentBehindDriver"] == "LAST"
    assert middle["OpponentAheadPosition"] == 1.0
    assert middle["OpponentBehindPosition"] == 3.0
    assert middle["OpponentAheadCompound"] == "SOFT"
    assert middle["OpponentBehindCompound"] == "HARD"
    assert middle["OpponentAheadTyreLife"] == 2.0
    assert middle["OpponentBehindTyreLife"] == 4.0
    assert middle["RelativePaceToAhead"] == 1.0
    assert middle["RelativePaceToBehind"] == -1.0
    assert pd.isna(leader["OpponentAheadDriver"])
    assert pd.isna(last["OpponentBehindDriver"])
    pd.testing.assert_frame_equal(result[original.columns], original)
    pd.testing.assert_frame_equal(laps, original)


def test_add_opponent_context_features_keeps_missing_opponents_missing() -> None:
    laps = pd.DataFrame(
        {
            "Driver": ["LEADER", "THIRD"],
            "LapNumber": [1.0, 1.0],
            "Position": [1.0, 3.0],
            "Compound": ["SOFT", "HARD"],
            "TyreLife": [2.0, 4.0],
            "LapTime": pd.to_timedelta([90.0, 92.0], unit="s"),
        }
    )

    result = add_opponent_context_features(laps)

    assert result["OpponentBehindDriver"].isna().all()
    assert result["OpponentAheadDriver"].isna().all()


def test_add_track_evolution_proxy_is_causal_and_session_specific() -> None:
    laps = pd.DataFrame(
        {
            "LapNumber": [1.0, 1.0, 2.0, 2.0],
            "LapTime": pd.to_timedelta([100.0, 102.0, 98.0, 104.0], unit="s"),
        }
    )
    original = laps.copy()

    result = add_track_evolution_proxy(laps)
    later_laps = pd.concat(
        [
            laps,
            pd.DataFrame(
                {"LapNumber": [3.0], "LapTime": pd.to_timedelta([200.0], unit="s")}
            ),
        ],
        ignore_index=True,
    )
    with_future = add_track_evolution_proxy(later_laps)
    new_session = add_track_evolution_proxy(
        pd.DataFrame(
            {"LapNumber": [1.0], "LapTime": pd.to_timedelta([80.0], unit="s")}
        )
    )

    assert result["TrackEvolutionProxy"].tolist() == [101.0, 101.0, 101.0, 101.0]
    assert with_future.loc[:3, "TrackEvolutionProxy"].tolist() == result[
        "TrackEvolutionProxy"
    ].tolist()
    assert new_session["TrackEvolutionProxy"].tolist() == [80.0]
    pd.testing.assert_frame_equal(laps, original)


def test_add_lagged_context_features_stays_within_driver_and_prior_laps() -> None:
    laps = pd.DataFrame(
        {
            "Driver": ["A", "A", "A", "B", "B"],
            "LapNumber": [2.0, 1.0, 3.0, 1.0, 2.0],
            "TrackEvolutionProxy": [200.0, 100.0, 300.0, 100.0, 200.0],
            "AvgSpeed": [20.0, 10.0, 30.0, 100.0, 200.0],
            "AvgThrottle": [2.0, 1.0, 3.0, 10.0, 20.0],
            "BrakeUsage": [0.2, 0.1, 0.3, 0.4, 0.5],
            "BrakeDuration": [2.0, 1.0, 3.0, 4.0, 5.0],
            "BrakingIntensity": [20.0, 10.0, 30.0, 40.0, 50.0],
            "BrakingFrequency": [2.0, 1.0, 3.0, 4.0, 5.0],
            "ThrottleAggressiveness": [20.0, 10.0, 30.0, 40.0, 50.0],
            "SpeedVariation": [2.0, 1.0, 3.0, 4.0, 5.0],
            "Position": [2.0, 1.0, 3.0, 1.0, 2.0],
            "RelativePaceToAhead": [0.2, 0.1, 0.3, None, 0.5],
            "RelativePaceToBehind": [-0.2, -0.1, -0.3, -0.4, -0.5],
            "OpponentAheadCompound": ["SOFT", None, "HARD", None, "MEDIUM"],
            "OpponentAheadTyreLife": [2.0, None, 4.0, None, 5.0],
        }
    )
    original = laps.copy()

    result = add_lagged_context_features(laps)
    with_future = add_lagged_context_features(
        pd.concat(
            [
                laps,
                pd.DataFrame({column: [value] for column, value in laps.iloc[-1].items()}),
            ],
            ignore_index=True,
        ).assign(LapNumber=lambda frame: frame["LapNumber"].where(frame.index != 5, 4.0))
    )

    assert result["PrevAvgSpeed"].tolist() == pytest.approx(
        [10.0, float("nan"), 20.0, float("nan"), 100.0], nan_ok=True
    )
    assert result["PrevPosition"].tolist() == pytest.approx(
        [1.0, float("nan"), 2.0, float("nan"), 1.0], nan_ok=True
    )
    assert result["PrevTrackEvolutionProxy"].tolist() == pytest.approx(
        [100.0, float("nan"), 200.0, float("nan"), 100.0], nan_ok=True
    )
    assert pd.isna(result.loc[1, "PrevOpponentAheadCompound"])
    pd.testing.assert_series_equal(
        with_future.loc[:4, "PrevAvgSpeed"].reset_index(drop=True),
        result["PrevAvgSpeed"].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(result[original.columns], original)
    pd.testing.assert_frame_equal(laps, original)
