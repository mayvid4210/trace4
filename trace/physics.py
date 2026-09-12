"""Simple, non-degradation lap-time physics baseline."""

from math import isfinite
from numbers import Real


def expected_lap_time(
    base_lap_time_seconds: float,
    estimated_fuel_load: float,
    fuel_time_per_unit_seconds: float,
    *,
    track_adjustment_seconds: float = 0.0,
    weather_adjustment_seconds: float = 0.0,
    driver_adjustment_seconds: float = 0.0,
    traffic_adjustment_seconds: float = 0.0,
) -> float:
    """Estimate a lap time in seconds without a tyre-degradation term.

    ``estimated_fuel_load`` is a user-provided proxy, not a claimed fuel mass.
    Positive adjustment values make a lap slower; negative values make it faster.
    Traffic is constrained to zero or greater because it represents delay.
    """
    values = {
        "base_lap_time_seconds": base_lap_time_seconds,
        "estimated_fuel_load": estimated_fuel_load,
        "fuel_time_per_unit_seconds": fuel_time_per_unit_seconds,
        "track_adjustment_seconds": track_adjustment_seconds,
        "weather_adjustment_seconds": weather_adjustment_seconds,
        "driver_adjustment_seconds": driver_adjustment_seconds,
        "traffic_adjustment_seconds": traffic_adjustment_seconds,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError(f"{name} must be a finite number")

    if base_lap_time_seconds <= 0:
        raise ValueError("base_lap_time_seconds must be greater than zero")
    if estimated_fuel_load < 0:
        raise ValueError("estimated_fuel_load must be zero or greater")
    if fuel_time_per_unit_seconds < 0:
        raise ValueError("fuel_time_per_unit_seconds must be zero or greater")
    if traffic_adjustment_seconds < 0:
        raise ValueError("traffic_adjustment_seconds must be zero or greater")

    lap_time = (
        base_lap_time_seconds
        + estimated_fuel_load * fuel_time_per_unit_seconds
        + track_adjustment_seconds
        + weather_adjustment_seconds
        + driver_adjustment_seconds
        + traffic_adjustment_seconds
    )
    if lap_time <= 0:
        raise ValueError("adjustments must result in a positive lap time")

    return lap_time
