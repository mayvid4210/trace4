import pandas as pd
import pytest

from trace.features import TYRE_POSITIONS
from trace.tyre_state import build_four_tyre_states
from trace.tyre_stress import (
    apply_lap_stress_to_states,
    combine_lap_telemetry,
    estimate_lap_tyre_stress,
)


def _telemetry(points, speeds, *, brake=None, throttle=None):
    count = len(points)
    return pd.DataFrame({
        "Time": pd.to_timedelta(range(count), unit="s"),
        "X": [point[0] for point in points],
        "Y": [point[1] for point in points],
        "Speed": speeds,
        "Brake": [0.0] * count if brake is None else brake,
        "Throttle": [0.0] * count if throttle is None else throttle,
    })


def test_straight_constant_speed_is_balanced_without_a_fake_dominant_tyre() -> None:
    telemetry = _telemetry([(0, 0), (10, 0), (20, 0), (30, 0)], [100] * 4)
    result = estimate_lap_tyre_stress(telemetry)

    assert result.stress.as_dict() == {position: 0.0 for position in TYRE_POSITIONS}
    assert result.dominant_tyre is None


def test_left_and_right_corners_load_the_outside_tyres() -> None:
    left = _telemetry([(0, 0), (10, 0), (10, 10), (0, 10)], [180] * 4)
    right = _telemetry([(0, 0), (10, 0), (10, -10), (0, -10)], [180] * 4)

    left_stress = estimate_lap_tyre_stress(left).stress.as_dict()
    right_stress = estimate_lap_tyre_stress(right).stress.as_dict()

    assert left_stress["FR"] > left_stress["FL"]
    assert left_stress["RR"] > left_stress["RL"]
    assert right_stress["FL"] > right_stress["FR"]
    assert right_stress["RL"] > right_stress["RR"]


def test_braking_loads_front_and_acceleration_loads_rear() -> None:
    braking = _telemetry([(0, 0), (10, 0), (20, 0), (30, 0)], [180, 140, 100, 60], brake=[0, 1, 1, 1])
    accelerating = _telemetry([(0, 0), (10, 0), (20, 0), (30, 0)], [40, 80, 120, 160], throttle=[0, 100, 100, 100])

    brake_stress = estimate_lap_tyre_stress(braking).stress.as_dict()
    traction_stress = estimate_lap_tyre_stress(accelerating).stress.as_dict()

    assert brake_stress["FL"] > brake_stress["RL"]
    assert brake_stress["FR"] > brake_stress["RR"]
    assert traction_stress["RL"] > traction_stress["FL"]
    assert traction_stress["RR"] > traction_stress["FR"]


def test_faster_tighter_corner_has_more_lateral_stress() -> None:
    gentle = _telemetry([(0, 0), (20, 0), (20, 20), (0, 20)], [80] * 4)
    tight_fast = _telemetry([(0, 0), (5, 0), (5, 5), (0, 5)], [240] * 4)

    assert sum(estimate_lap_tyre_stress(tight_fast).stress.as_dict().values()) > sum(estimate_lap_tyre_stress(gentle).stress.as_dict().values())


def test_stress_adjustment_preserves_shared_mean_and_can_identify_limiting_tyre() -> None:
    states = build_four_tyre_states(2.0, "physics_fallback")
    stress = estimate_lap_tyre_stress(_telemetry(
        [(0, 0), (10, 0), (10, 10), (0, 10)], [180] * 4,
        brake=[0, 0, 1, 1],
    ))
    adjusted = apply_lap_stress_to_states(states, stress)

    assert sum(state.estimated_degradation for state in adjusted.values()) / 4 == pytest.approx(2.0)
    assert adjusted["FR"].is_limiting or adjusted["RR"].is_limiting
    assert not any(state.is_limiting for state in apply_lap_stress_to_states(states, estimate_lap_tyre_stress(_telemetry([(0, 0), (10, 0), (20, 0)], [100] * 3))).values())


def test_inputs_are_not_mutated_and_invalid_telemetry_is_rejected() -> None:
    telemetry = _telemetry([(0, 0), (10, 0), (20, 0)], [100] * 3)
    original = telemetry.copy(deep=True)
    estimate_lap_tyre_stress(telemetry)
    pd.testing.assert_frame_equal(telemetry, original)
    with pytest.raises(ValueError, match="X"):
        estimate_lap_tyre_stress(telemetry.drop(columns="X"))
    with pytest.raises(ValueError, match="increasing"):
        estimate_lap_tyre_stress(telemetry.assign(Time=pd.to_timedelta([0, 2, 1], unit="s")))


def test_combining_telemetry_drops_unmatched_position_samples_without_filling() -> None:
    car = pd.DataFrame({
        "Time": pd.to_timedelta([0, 1, 2], unit="s"),
        "Speed": [100, 100, 100], "Brake": [0, 0, 0], "Throttle": [0, 0, 0],
    })
    position = pd.DataFrame({
        "Time": pd.to_timedelta([1, 2], unit="s"), "X": [0, 10], "Y": [0, 0],
    })

    combined = combine_lap_telemetry(car, position)

    assert combined["Time"].tolist() == pd.to_timedelta([1, 2], unit="s").tolist()
