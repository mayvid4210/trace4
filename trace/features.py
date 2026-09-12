"""Small, reusable feature transformations for cleaned race laps."""

from math import isfinite
from numbers import Real
from typing import Literal

import pandas as pd

from trace.data import get_lap_telemetry_features


# FastF1 does not identify a physical tyre position for a lap. This vocabulary
# is available for future, externally supplied per-tyre state only.
TyrePosition = Literal["FL", "FR", "RL", "RR"]
TYRE_POSITIONS: tuple[TyrePosition, ...] = ("FL", "FR", "RL", "RR")

_LAGGED_DRIVER_CONTEXT_COLUMNS = [
    "AvgSpeed",
    "AvgThrottle",
    "BrakeUsage",
    "BrakeDuration",
    "BrakingIntensity",
    "BrakingFrequency",
    "ThrottleAggressiveness",
    "SpeedVariation",
    "Position",
    "RelativePaceToAhead",
    "RelativePaceToBehind",
    "OpponentAheadCompound",
    "OpponentAheadTyreLife",
]


def build_tyre_features(laps: pd.DataFrame, total_race_laps: float) -> pd.DataFrame:
    """Add tyre/stint and race-progression features without mutating ``laps``."""
    required_columns = {"LapNumber", "TyreLife", "Stint"}
    missing_columns = required_columns.difference(laps.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"laps is missing required columns: {names}")
    if (
        isinstance(total_race_laps, bool)
        or not isinstance(total_race_laps, Real)
        or not isfinite(total_race_laps)
        or total_race_laps <= 0
    ):
        raise ValueError("total_race_laps must be a positive finite number")

    features = laps.copy()
    features["RaceProgress"] = features["LapNumber"] / total_race_laps
    features["TyreAgeSquared"] = features["TyreLife"] ** 2
    return features


def add_telemetry_features(laps: pd.DataFrame, session: object) -> pd.DataFrame:
    """Add available FastF1 car-telemetry aggregates without mutating ``laps``.

    A lap with unavailable or invalid telemetry keeps missing feature values;
    no values are estimated or filled.
    """
    required_columns = {"Driver", "LapNumber"}
    missing_columns = required_columns.difference(laps.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"laps is missing required columns: {names}")

    feature_names = {
        "AvgSpeed": "average_speed",
        "MaxSpeed": "maximum_speed",
        "AvgThrottle": "average_throttle",
        "BrakeUsage": "brake_usage_fraction",
        "BrakeDuration": "brake_duration_seconds",
        "BrakingIntensity": "braking_intensity",
        "BrakingFrequency": "braking_frequency",
        "ThrottleAggressiveness": "throttle_aggressiveness",
        "SpeedVariation": "speed_variation",
    }
    values_by_feature = {name: [] for name in feature_names}

    for lap in laps.itertuples(index=False):
        values = None
        try:
            values = get_lap_telemetry_features(session, lap.Driver, lap.LapNumber)
        except ValueError:
            pass

        for output_name, source_name in feature_names.items():
            value = None if values is None else values[source_name]
            values_by_feature[output_name].append(value)

    features = laps.copy()
    for name, values in values_by_feature.items():
        features[name] = values
    return features


def add_weather_track_features(
    features: pd.DataFrame, cleaned_laps: pd.DataFrame
) -> pd.DataFrame:
    """Copy real weather and track fields from aligned cleaned lap data."""
    source_columns = [
        "TrackTemp",
        "AirTemp",
        "Humidity",
        "Rainfall",
        "TrackStatus",
    ]
    missing_columns = set(source_columns).difference(cleaned_laps.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"cleaned_laps is missing required columns: {names}")
    if len(features) != len(cleaned_laps):
        raise ValueError("features and cleaned_laps must have the same number of rows")

    result = features.copy()
    for column in source_columns:
        result[column] = cleaned_laps[column].to_numpy()
    return result


def add_tyre_residual_features(laps: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Add strictly current-or-past residual features within driver stints."""
    required_columns = {"Driver", "Stint", "LapNumber", "TyreResidual"}
    missing_columns = required_columns.difference(laps.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"laps is missing required columns: {names}")
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer")

    residuals = pd.to_numeric(laps["TyreResidual"], errors="coerce")
    if residuals.isna().any() or not pd.Series(residuals).map(isfinite).all():
        raise ValueError("TyreResidual must contain finite values")

    ordered = laps.copy()
    ordered["TyreResidual"] = residuals
    ordered["_trace_original_order"] = range(len(ordered))
    ordered = ordered.sort_values(
        ["Driver", "Stint", "LapNumber"], kind="stable"
    )
    grouped = ordered.groupby(["Driver", "Stint"], sort=False)["TyreResidual"]
    ordered["PreviousTyreResidual"] = grouped.shift(1)
    ordered["RollingTyreResidual"] = grouped.transform(
        lambda values: values.rolling(window=window, min_periods=1).mean()
    )
    return (
        ordered.sort_values("_trace_original_order", kind="stable")
        .drop(columns="_trace_original_order")
    )


def add_opponent_context_features(laps: pd.DataFrame) -> pd.DataFrame:
    """Add same-lap opponent context by adjacent recorded race position.

    Relative pace is current LapTime minus the opponent's LapTime in seconds;
    a positive value means the current lap was slower. No on-track time gaps
    are inferred from these lap-level records.
    """
    required_columns = {
        "Driver",
        "LapNumber",
        "Position",
        "Compound",
        "TyreLife",
        "LapTime",
    }
    missing_columns = required_columns.difference(laps.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"laps is missing required columns: {names}")

    result = laps.copy()
    position = pd.to_numeric(laps["Position"], errors="coerce")
    lap_time = pd.to_timedelta(laps["LapTime"], errors="coerce")
    valid_opponents = laps.assign(_position=position).dropna(
        subset=["LapNumber", "_position"]
    )
    valid_opponents = valid_opponents.loc[
        ~valid_opponents.duplicated(["LapNumber", "_position"], keep=False)
    ]

    opponent_columns = ["Driver", "Position", "Compound", "TyreLife", "LapTime"]
    lookup = valid_opponents.set_index(["LapNumber", "_position"])[opponent_columns]
    for direction, offset in (("Ahead", -1), ("Behind", 1)):
        values = []
        for lap_number, current_position in zip(laps["LapNumber"], position):
            key = (lap_number, current_position + offset)
            values.append(lookup.loc[key] if key in lookup.index else None)

        for source_column in opponent_columns:
            name = f"Opponent{direction}{source_column}"
            result[name] = [
                None if value is None else value[source_column] for value in values
            ]

        opponent_seconds = pd.to_timedelta(
            result[f"Opponent{direction}LapTime"], errors="coerce"
        ).dt.total_seconds()
        result[f"RelativePaceTo{direction}"] = lap_time.dt.total_seconds() - opponent_seconds

    return result


def add_track_evolution_proxy(laps: pd.DataFrame) -> pd.DataFrame:
    """Add a causal per-session cumulative median lap-time proxy in seconds.

    For race lap ``L``, the proxy is the median of valid lap times from every
    driver whose LapNumber is less than or equal to ``L`` in this input session.
    """
    required_columns = {"LapNumber", "LapTime"}
    missing_columns = required_columns.difference(laps.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"laps is missing required columns: {names}")

    lap_numbers = pd.to_numeric(laps["LapNumber"], errors="coerce")
    lap_times = pd.to_timedelta(laps["LapTime"], errors="coerce")
    if lap_numbers.isna().any() or lap_times.isna().any() or (lap_times <= pd.Timedelta(0)).any():
        raise ValueError("LapNumber and LapTime must contain valid values")

    result = laps.copy()
    lap_seconds = lap_times.dt.total_seconds()
    proxy_by_lap: dict[float, float] = {}
    history: list[float] = []
    for lap_number in sorted(lap_numbers.unique()):
        history.extend(lap_seconds[lap_numbers == lap_number].tolist())
        proxy_by_lap[lap_number] = float(pd.Series(history).median())
    result["TrackEvolutionProxy"] = lap_numbers.map(proxy_by_lap)
    return result


def add_lagged_context_features(laps: pd.DataFrame) -> pd.DataFrame:
    """Add driver-lagged context and a prior-race-lap track proxy.

    Driver context is shifted by chronological LapNumber within Driver. The
    track proxy for lap L is mapped from the most recent earlier race lap, so
    it never uses TrackEvolutionProxy from L itself.
    """
    required_columns = {
        "Driver",
        "LapNumber",
        "TrackEvolutionProxy",
        *_LAGGED_DRIVER_CONTEXT_COLUMNS,
    }
    missing_columns = required_columns.difference(laps.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"laps is missing required columns: {names}")

    result = laps.copy()
    result["_trace_original_order"] = range(len(result))
    ordered = result.sort_values(["Driver", "LapNumber"], kind="stable")
    grouped = ordered.groupby("Driver", sort=False)
    for column in _LAGGED_DRIVER_CONTEXT_COLUMNS:
        ordered[f"Prev{column}"] = grouped[column].shift(1)

    lap_numbers = pd.to_numeric(laps["LapNumber"], errors="coerce")
    if lap_numbers.isna().any():
        raise ValueError("LapNumber must contain numeric values")
    proxy_by_lap = laps.assign(_lap_number=lap_numbers).groupby(
        "_lap_number", sort=True
    )["TrackEvolutionProxy"].first()
    previous_proxy = proxy_by_lap.shift(1)
    ordered["PrevTrackEvolutionProxy"] = pd.to_numeric(
        ordered["LapNumber"], errors="coerce"
    ).map(previous_proxy)

    return (
        ordered.sort_values("_trace_original_order", kind="stable")
        .drop(columns="_trace_original_order")
    )
