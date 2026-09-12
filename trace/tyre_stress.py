"""Causal, telemetry-derived per-tyre stress proxies.

These estimates redistribute a shared degradation signal. They are not
measurements of wheel load, tyre wear, temperature, pressure, or blistering.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, isfinite, pi, sin, cos
from typing import Mapping

import numpy as np
import pandas as pd

from trace.features import TYRE_POSITIONS, TyrePosition
from trace.tyre_state import TyreState, redistribute_degradation_by_stress


# Heuristic modelling assumptions, deliberately centralised and uncalibrated.
# Position units are assumed to be FastF1 circuit-coordinate metres; values are
# bounded only to stabilize this dimensionless proxy, not to represent measured
# forces. The weights and stress-to-degradation adjustment are not constants of
# vehicle physics.
HEADING_SMOOTHING_WINDOW = 3
TURN_ANGLE_THRESHOLD_RADIANS = 0.02
MIN_POSITION_STEP = 1.0
LATERAL_DEMAND_SCALE = 15.0
BRAKING_DECELERATION_SCALE = 8.0
TRACTION_ACCELERATION_SCALE = 6.0
LATERAL_STRESS_WEIGHT = 1.0
BRAKING_STRESS_WEIGHT = 1.0
TRACTION_STRESS_WEIGHT = 1.0
STRESS_DEGRADATION_ADJUSTMENT = 0.5


@dataclass(frozen=True)
class PerTyreStress:
    """Lap-average TRACE-estimated stress proxies, one value per tyre."""

    FL: float
    FR: float
    RL: float
    RR: float

    def as_dict(self) -> dict[TyrePosition, float]:
        return {position: getattr(self, position) for position in TYRE_POSITIONS}


@dataclass(frozen=True)
class LapTyreStress:
    stress: PerTyreStress
    dominant_tyre: TyrePosition | None
    left_right_balance: float
    front_rear_balance: float
    left_corner_count: int
    right_corner_count: int


def estimate_lap_tyre_stress(telemetry: pd.DataFrame) -> LapTyreStress:
    """Estimate lap stress from current and earlier telemetry samples only.

    Required real telemetry fields are ``Time``, ``X``, ``Y``, ``Speed``,
    ``Brake``, and ``Throttle``. Speed is interpreted as FastF1 km/h. The
    returned values are normalized, dimensionless TRACE stress proxies.
    """
    required = {"Time", "X", "Y", "Speed", "Brake", "Throttle"}
    missing = required.difference(telemetry.columns)
    if missing:
        raise ValueError(f"telemetry is missing required columns: {', '.join(sorted(missing))}")
    if len(telemetry) < 3:
        raise ValueError("telemetry must contain at least three samples")
    frame = _validated_telemetry(telemetry)
    times = frame["Time"].dt.total_seconds().to_numpy()
    x, y = frame["X"].to_numpy(), frame["Y"].to_numpy()
    speed = frame["Speed"].to_numpy() / 3.6
    brake, throttle = frame["Brake"].to_numpy(), frame["Throttle"].to_numpy() / 100.0

    stress = {position: 0.0 for position in TYRE_POSITIONS}
    corner_directions: list[int] = []
    heading_changes: list[float] = []
    for index in range(len(frame)):
        lateral, direction = _lateral_demand(index, x, y, speed, heading_changes)
        corner_directions.append(direction)
        if direction > 0:
            stress["FR"] += LATERAL_STRESS_WEIGHT * lateral
            stress["RR"] += LATERAL_STRESS_WEIGHT * lateral
        elif direction < 0:
            stress["FL"] += LATERAL_STRESS_WEIGHT * lateral
            stress["RL"] += LATERAL_STRESS_WEIGHT * lateral

        braking, traction = _longitudinal_demand(index, times, speed, brake, throttle)
        stress["FL"] += BRAKING_STRESS_WEIGHT * braking
        stress["FR"] += BRAKING_STRESS_WEIGHT * braking
        stress["RL"] += TRACTION_STRESS_WEIGHT * traction
        stress["RR"] += TRACTION_STRESS_WEIGHT * traction

    values = {position: value / len(frame) for position, value in stress.items()}
    per_tyre = PerTyreStress(**values)
    total = sum(values.values())
    left = values["FL"] + values["RL"]
    right = values["FR"] + values["RR"]
    front = values["FL"] + values["FR"]
    rear = values["RL"] + values["RR"]
    maximum = max(values.values())
    dominant = [position for position, value in values.items() if value == maximum]
    return LapTyreStress(
        stress=per_tyre,
        dominant_tyre=dominant[0] if len(dominant) == 1 else None,
        left_right_balance=0.0 if total == 0 else (right - left) / total,
        front_rear_balance=0.0 if total == 0 else (front - rear) / total,
        left_corner_count=_corner_count(corner_directions, 1),
        right_corner_count=_corner_count(corner_directions, -1),
    )


def apply_lap_stress_to_states(
    states: Mapping[TyrePosition, TyreState],
    lap_stress: LapTyreStress,
    *,
    adjustment_scale: float = STRESS_DEGRADATION_ADJUSTMENT,
) -> dict[TyrePosition, TyreState]:
    """Mean-preservingly distribute a shared state by estimated lap stress."""
    return redistribute_degradation_by_stress(
        states, lap_stress.stress.as_dict(), adjustment_scale=adjustment_scale
    )


def combine_lap_telemetry(car_data: pd.DataFrame, position_data: pd.DataFrame) -> pd.DataFrame:
    """Causally align FastF1 car and position samples by most recent position."""
    required_car = {"Time", "Speed", "Brake", "Throttle"}
    required_position = {"Time", "X", "Y"}
    if required_car.difference(car_data.columns) or required_position.difference(position_data.columns):
        raise ValueError("car and position data must contain the required FastF1 fields")
    car = car_data[["Time", "Speed", "Brake", "Throttle"]].sort_values("Time")
    position = position_data[["Time", "X", "Y"]].sort_values("Time")
    # The earliest car samples can precede the first position sample. Dropping
    # unmatched rows avoids fabricating coordinates or forward-looking matches.
    return pd.merge_asof(car, position, on="Time", direction="backward").dropna(
        subset=["X", "Y"]
    )


def _validated_telemetry(telemetry: pd.DataFrame) -> pd.DataFrame:
    frame = telemetry[["Time", "X", "Y", "Speed", "Brake", "Throttle"]].copy()
    frame["Time"] = pd.to_timedelta(frame["Time"], errors="coerce")
    for column in ("X", "Y", "Speed", "Brake", "Throttle"):
        source = frame[column]
        frame[column] = pd.to_numeric(source, errors="coerce")
        if (source.notna() & frame[column].isna()).any():
            raise ValueError(f"{column} contains invalid numeric values")
    if frame.isna().any().any() or not np.isfinite(
        frame.drop(columns="Time").to_numpy(dtype=float)
    ).all():
        raise ValueError("telemetry must contain complete finite values")
    if not frame["Time"].is_monotonic_increasing or frame["Time"].duplicated().any():
        raise ValueError("Time must be strictly increasing")
    if (frame["Speed"] < 0).any() or (frame["Throttle"] < 0).any() or (frame["Brake"] < 0).any():
        raise ValueError("Speed, Brake, and Throttle must be non-negative")
    return frame.reset_index(drop=True)


def _lateral_demand(
    index: int, x: np.ndarray, y: np.ndarray, speed: np.ndarray, changes: list[float]
) -> tuple[float, int]:
    if index < 2:
        return 0.0, 0
    previous_heading = atan2(y[index - 1] - y[index - 2], x[index - 1] - x[index - 2])
    heading = atan2(y[index] - y[index - 1], x[index] - x[index - 1])
    change = atan2(sin(heading - previous_heading), cos(heading - previous_heading))
    changes.append(change)
    smoothed = sum(changes[-HEADING_SMOOTHING_WINDOW:]) / min(len(changes), HEADING_SMOOTHING_WINDOW)
    if abs(smoothed) < TURN_ANGLE_THRESHOLD_RADIANS:
        return 0.0, 0
    distance = max(float(np.hypot(x[index] - x[index - 1], y[index] - y[index - 1])), MIN_POSITION_STEP)
    raw = speed[index] ** 2 * abs(smoothed) / distance
    # Smoothly bound the proxy without flattening every fast/tight corner to
    # the same value, and without using a future-lap maximum for normalization.
    return raw / (LATERAL_DEMAND_SCALE + raw), 1 if smoothed > 0 else -1


def _longitudinal_demand(
    index: int, times: np.ndarray, speed: np.ndarray, brake: np.ndarray, throttle: np.ndarray
) -> tuple[float, float]:
    if index == 0:
        return min(brake[index], 1.0), 0.0
    elapsed = times[index] - times[index - 1]
    acceleration = 0.0 if elapsed <= 0 else (speed[index] - speed[index - 1]) / elapsed
    braking = max(min(brake[index], 1.0), min(max(-acceleration, 0.0) / BRAKING_DECELERATION_SCALE, 1.0))
    traction = min(max(acceleration, 0.0) / TRACTION_ACCELERATION_SCALE, 1.0) * min(throttle[index], 1.0)
    return braking, traction


def _corner_count(directions: list[int], target: int) -> int:
    count, previous = 0, 0
    for direction in directions:
        if direction == target and previous != target:
            count += 1
        previous = direction
    return count
