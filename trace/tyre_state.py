"""Minimal shared four-tyre state representation.

FastF1 has no per-corner tyre-health labels. Consequently, all four positions
receive the same supplied estimate unless a later caller provides an external,
position-specific estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Literal, Mapping, Sequence

from trace.features import TYRE_POSITIONS, TyrePosition


Severity = Literal["NORMAL", "WATCH", "CRITICAL"]
EstimationPath = Literal["dry_ml", "physics_fallback"]

# Heuristic interpretation thresholds, not calibrated physical constants or
# tyre-health ground truth. They are expressed in residual-signal seconds and
# change per estimate only to make the three labels interpretable.
WATCH_DEGRADATION_THRESHOLD = 1.0
CRITICAL_DEGRADATION_THRESHOLD = 3.0
WATCH_TREND_THRESHOLD = 0.5
CRITICAL_TREND_THRESHOLD = 1.0
@dataclass(frozen=True)
class TyreState:
    """One tyre position's estimated state, without claiming a true label.

    ``estimation_path`` identifies the method used; it is not a calibrated
    confidence, probability, or uncertainty estimate.
    """

    position: TyrePosition
    estimated_degradation: float
    trend: float
    severity: Severity
    estimation_path: EstimationPath
    is_limiting: bool = False


def select_base_degradation(
    physics_residual: float,
    *,
    is_dry: bool,
    dry_ml_residual: float | None = None,
) -> tuple[float, EstimationPath]:
    """Choose the permitted degradation signal for a known track regime.

    The dry ML residual is used only when the caller identifies a dry lap and
    supplies an accepted ML correction. Wet/mixed or unavailable ML always use
    the physics residual. No weather state is inferred here.
    """
    _require_finite("physics_residual", physics_residual)
    if is_dry and dry_ml_residual is not None:
        _require_finite("dry_ml_residual", dry_ml_residual)
        return float(dry_ml_residual), "dry_ml"
    return float(physics_residual), "physics_fallback"


def build_four_tyre_states(
    base_degradation: float,
    estimation_path: EstimationPath,
    *,
    previous_estimates: Mapping[TyrePosition, Sequence[float]] | None = None,
    external_degradation: Mapping[TyrePosition, float] | None = None,
    trend_window: int = 3,
) -> dict[TyrePosition, TyreState]:
    """Build four causal state estimates without inventing corner differences.

    ``base_degradation`` should be the accepted dry ML correction for dry laps,
    or the physics-only residual signal for wet/mixed or unseen conditions.
    ``external_degradation`` is an optional future hook for genuinely supplied
    per-position information; omitted positions retain the shared base value.
    """
    _require_finite("base_degradation", base_degradation)
    if estimation_path not in {"dry_ml", "physics_fallback"}:
        raise ValueError("estimation_path must be 'dry_ml' or 'physics_fallback'")
    if isinstance(trend_window, bool) or not isinstance(trend_window, int) or trend_window <= 0:
        raise ValueError("trend_window must be a positive integer")

    supplied = external_degradation or {}
    unknown = set(supplied).difference(TYRE_POSITIONS)
    if unknown:
        raise ValueError(f"external_degradation has unknown positions: {', '.join(sorted(unknown))}")
    states: dict[TyrePosition, TyreState] = {}
    for position in TYRE_POSITIONS:
        estimate = supplied.get(position, base_degradation)
        _require_finite(f"external_degradation[{position}]", estimate)
        history = () if previous_estimates is None else previous_estimates.get(position, ())
        trend = _causal_trend(history, estimate, trend_window)
        states[position] = TyreState(
            position=position,
            estimated_degradation=float(estimate),
            trend=trend,
            severity=classify_severity(estimate, trend),
            estimation_path=estimation_path,
        )
    return _mark_limiting(states)


def classify_severity(estimated_degradation: float, trend: float) -> Severity:
    """Classify a residual estimate using documented, configurable thresholds."""
    _require_finite("estimated_degradation", estimated_degradation)
    _require_finite("trend", trend)
    magnitude = abs(estimated_degradation)
    if magnitude >= CRITICAL_DEGRADATION_THRESHOLD or trend >= CRITICAL_TREND_THRESHOLD:
        return "CRITICAL"
    if magnitude >= WATCH_DEGRADATION_THRESHOLD or trend >= WATCH_TREND_THRESHOLD:
        return "WATCH"
    return "NORMAL"


def redistribute_degradation_by_stress(
    states: Mapping[TyrePosition, TyreState],
    stress: Mapping[TyrePosition, float],
    *,
    adjustment_scale: float,
) -> dict[TyrePosition, TyreState]:
    """Redistribute existing estimates by relative stress without changing mean.

    ``stress`` is a TRACE-estimated proxy, not measured wheel load. Its mean is
    removed before applying the configurable scale, preserving the car-level
    shared degradation estimate.
    """
    if set(states) != set(TYRE_POSITIONS) or set(stress) != set(TYRE_POSITIONS):
        raise ValueError("states and stress must contain FL, FR, RL, and RR")
    _require_finite("adjustment_scale", adjustment_scale)
    stress_values = {position: float(stress[position]) for position in TYRE_POSITIONS}
    for position, value in stress_values.items():
        _require_finite(f"stress[{position}]", value)
    mean_stress = sum(stress_values.values()) / len(stress_values)
    adjusted = {
        position: replace(
            states[position],
            estimated_degradation=(
                states[position].estimated_degradation
                + adjustment_scale * (stress_values[position] - mean_stress)
            ),
        )
        for position in TYRE_POSITIONS
    }
    classified = {
        position: replace(
            state,
            severity=classify_severity(state.estimated_degradation, state.trend),
        )
        for position, state in adjusted.items()
    }
    return _mark_limiting(classified)


def _causal_trend(history: Sequence[float], estimate: float, window: int) -> float:
    """Current estimate minus the mean of the prior estimates only."""
    recent = list(history)[-window:]
    for value in recent:
        _require_finite("previous estimate", value)
    return 0.0 if not recent else float(estimate - sum(recent) / len(recent))


def _mark_limiting(
    states: Mapping[TyrePosition, TyreState],
) -> dict[TyrePosition, TyreState]:
    """Mark one tyre only when the supplied state has a unique worst value."""
    ranking = {
        "NORMAL": 0,
        "WATCH": 1,
        "CRITICAL": 2,
    }
    scores = {
        position: (ranking[state.severity], abs(state.estimated_degradation))
        for position, state in states.items()
    }
    highest = max(scores.values())
    limiting = [position for position, score in scores.items() if score == highest]
    limiting_position = limiting[0] if len(limiting) == 1 else None
    return {
        position: replace(state, is_limiting=position == limiting_position)
        for position, state in states.items()
    }


def _require_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
