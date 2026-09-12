"""Tests for the non-degradation lap-time baseline."""

import pytest

from trace.physics import expected_lap_time


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
