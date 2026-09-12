"""Small, reusable feature transformations for cleaned race laps."""

from math import isfinite
from numbers import Real

import pandas as pd

from trace.data import get_lap_telemetry_features


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
