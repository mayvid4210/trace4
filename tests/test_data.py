import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import trace.data as data
from trace.data import LAP_COLUMNS, aggregate_telemetry, clean_laps


@pytest.mark.parametrize("event", ["Bahrain", "Canada"])
def test_load_race_uses_supplied_session_details(
    monkeypatch: pytest.MonkeyPatch, event: str
) -> None:
    calls: list[tuple[int, str, str]] = []
    session = SimpleNamespace(
        laps=SimpleNamespace(get_weather_data=lambda: "weather"), load=lambda: None
    )

    def get_session(year: int, selected_event: str, session_type: str) -> object:
        calls.append((year, selected_event, session_type))
        return session

    monkeypatch.setitem(sys.modules, "fastf1", SimpleNamespace(get_session=get_session))
    monkeypatch.setattr(data, "clean_laps", lambda laps, weather: pd.DataFrame())

    result = data.load_race(2024, event, "R")

    assert result.empty
    assert calls == [(2024, event, "R")]


def test_clean_laps_returns_expected_schema_and_removes_invalid_times() -> None:
    laps = pd.DataFrame(
        {
            "Driver": ["VER", "PER", "SAI"],
            "LapNumber": [1, 1, 1],
            "LapTime": [pd.Timedelta(seconds=91), pd.NaT, pd.Timedelta(0)],
            "Compound": ["SOFT", "SOFT", "MEDIUM"],
            "TyreLife": [1.0, 1.0, 1.0],
            "Stint": [1.0, 1.0, 1.0],
            "TrackStatus": ["1", "1", "1"],
        }
    )
    weather = pd.DataFrame(
        {
            "TrackTemp": [35.0, 36.0, 37.0],
            "AirTemp": [25.0, 25.0, 25.0],
            "Humidity": [45.0, 46.0, 47.0],
            "Rainfall": [False, False, False],
        },
        index=[60, 62, 63],
    )

    clean = clean_laps(laps, weather)

    assert list(clean.columns) == LAP_COLUMNS
    assert len(clean) == 1
    assert clean.loc[0, "Driver"] == "VER"
    assert clean.loc[0, "TrackTemp"] == 35.0
    assert clean.loc[0, "Stint"] == 1.0
    assert clean.loc[0, "LapTime"] == pd.Timedelta(seconds=91)


def test_clean_laps_requires_core_fastf1_lap_fields() -> None:
    laps = pd.DataFrame({"Driver": ["VER"], "LapNumber": [1]})

    with pytest.raises(ValueError, match="LapTime"):
        clean_laps(laps)


def test_aggregate_telemetry_calculates_basic_features() -> None:
    telemetry = pd.DataFrame(
        {
            "Time": pd.to_timedelta([0, 1, 2, 3], unit="s"),
            "Speed": [100, 120, 140, 160],
            "Throttle": [50, 60, 70, 80],
            "Brake": [False, True, True, False],
            "RPM": [10_000, 11_000, 12_000, 13_000],
            "DRS": [0, 8, 10, 1],
        }
    )

    features = aggregate_telemetry(telemetry)

    assert features == {
        "average_speed": 130.0,
        "maximum_speed": 160.0,
        "average_throttle": 65.0,
        "average_brake": 0.5,
        "brake_usage_fraction": 0.5,
        "brake_duration_seconds": 2.0,
        "braking_intensity": None,
        "braking_frequency": 1.0,
        "throttle_aggressiveness": 70.0,
        "speed_variation": pytest.approx(22.360679774997898),
        "average_rpm": 11_500.0,
        "drs_usage_fraction": 0.5,
    }


def test_aggregate_telemetry_allows_missing_optional_fields() -> None:
    features = aggregate_telemetry(pd.DataFrame({"Speed": [100, 120]}))

    assert features["average_speed"] == 110.0
    assert features["maximum_speed"] == 120.0
    assert features["average_throttle"] is None
    assert features["brake_duration_seconds"] is None
    assert features["drs_usage_fraction"] is None


def test_aggregate_telemetry_calculates_driver_style_features() -> None:
    telemetry = pd.DataFrame(
        {
            "Time": pd.to_timedelta([0, 1, 2, 3, 4], unit="s"),
            "Speed": [200, 180, 160, 170, 150],
            "Throttle": [10, 20, 30, 40, 50],
            "Brake": [False, True, True, False, True],
        }
    )

    features = aggregate_telemetry(telemetry)

    assert features["braking_intensity"] == 20.0
    assert features["braking_frequency"] == 2.0
    assert features["throttle_aggressiveness"] == 40.0
    assert features["speed_variation"] == pytest.approx(17.204650534085253)


def test_aggregate_telemetry_rejects_invalid_numeric_values() -> None:
    telemetry = pd.DataFrame({"Speed": [100, "not-a-speed"]})

    with pytest.raises(ValueError, match="Speed"):
        aggregate_telemetry(telemetry)


def test_aggregate_telemetry_rejects_empty_telemetry() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        aggregate_telemetry(pd.DataFrame())
