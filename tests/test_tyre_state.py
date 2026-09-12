import pytest

from trace.features import TYRE_POSITIONS
from trace.tyre_state import (
    TyreState,
    build_four_tyre_states,
    classify_severity,
    select_base_degradation,
)


def test_four_positions_copy_one_shared_base_estimate_without_differences() -> None:
    states = build_four_tyre_states(1.2, "dry_ml")

    assert tuple(states) == TYRE_POSITIONS
    assert {state.estimated_degradation for state in states.values()} == {1.2}
    assert {state.trend for state in states.values()} == {0.0}
    assert not any(state.is_limiting for state in states.values())


def test_trend_uses_only_previous_estimates_for_its_own_position() -> None:
    history = {"FL": [1.0, 2.0], "FR": [10.0], "RL": [], "RR": [3.0]}
    original = {position: values.copy() for position, values in history.items()}

    states = build_four_tyre_states(3.0, "physics_fallback", previous_estimates=history)
    later = build_four_tyre_states(100.0, "physics_fallback", previous_estimates=history)

    assert states["FL"].trend == pytest.approx(1.5)
    assert states["FR"].trend == pytest.approx(-7.0)
    assert states["RL"].trend == 0.0
    assert later["FL"].trend == pytest.approx(98.5)
    assert history == original


def test_severity_thresholds_are_interpretable() -> None:
    assert classify_severity(0.5, 0.1) == "NORMAL"
    assert classify_severity(1.0, 0.1) == "WATCH"
    assert classify_severity(0.1, 0.5) == "WATCH"
    assert classify_severity(3.0, 0.1) == "CRITICAL"
    assert classify_severity(0.1, 1.0) == "CRITICAL"


def test_estimation_path_distinguishes_ml_from_physics_fallback_without_probability() -> None:
    dry = build_four_tyre_states(0.2, "dry_ml")
    fallback = build_four_tyre_states(0.2, "physics_fallback")

    assert dry["FL"].estimation_path == "dry_ml"
    assert fallback["FL"].estimation_path == "physics_fallback"
    assert "confidence" not in TyreState.__dataclass_fields__


def test_base_signal_uses_ml_only_for_supported_dry_conditions() -> None:
    assert select_base_degradation(1.0, is_dry=True, dry_ml_residual=0.6) == (0.6, "dry_ml")
    assert select_base_degradation(1.0, is_dry=False, dry_ml_residual=0.6) == (1.0, "physics_fallback")
    assert select_base_degradation(1.0, is_dry=True) == (1.0, "physics_fallback")


def test_external_position_specific_estimate_can_identify_a_limiting_tyre() -> None:
    states = build_four_tyre_states(
        0.5,
        "physics_fallback",
        external_degradation={"RR": 4.0},
    )

    assert states["RR"].is_limiting
    assert not any(state.is_limiting for position, state in states.items() if position != "RR")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_invalid_estimates_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="base_degradation"):
        build_four_tyre_states(value, "dry_ml")
