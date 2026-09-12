"""Simple, non-degradation lap-time physics baseline."""

from math import isfinite
from numbers import Real

import numpy as np
import pandas as pd


_BASELINE_COLUMNS = {
    "LapTime",
    "RaceProgress",
    "TrackTemp",
    "AirTemp",
    "Humidity",
    "Rainfall",
    "Driver",
}

_CAUSAL_BASELINE_COLUMNS = {
    "LapTime",
    "RaceProgress",
    "TrackTemp",
    "AirTemp",
    "Humidity",
    "Rainfall",
    "Driver",
    "Compound",
    "TrackStatus",
}
_DIAGNOSTIC_NUMERIC_COLUMNS = [
    "TrackEvolutionProxy",
    "AvgSpeed",
    "MaxSpeed",
    "AvgThrottle",
    "BrakeUsage",
    "BrakeDuration",
    "BrakingIntensity",
    "BrakingFrequency",
    "ThrottleAggressiveness",
    "SpeedVariation",
    "Position",
]
_DIAGNOSTIC_OPPONENT_COLUMNS = ["RelativePaceToAhead", "RelativePaceToBehind"]
_LAGGED_CONTEXT_NUMERIC_COLUMNS = [
    "PrevAvgSpeed",
    "PrevAvgThrottle",
    "PrevBrakeUsage",
    "PrevBrakeDuration",
    "PrevBrakingIntensity",
    "PrevBrakingFrequency",
    "PrevThrottleAggressiveness",
    "PrevSpeedVariation",
    "PrevPosition",
    "PrevRelativePaceToAhead",
    "PrevRelativePaceToBehind",
    "PrevOpponentAheadTyreLife",
    "PrevTrackEvolutionProxy",
]
_LAGGED_CONTEXT_CATEGORICAL_COLUMNS = ["PrevOpponentAheadCompound"]


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


def fit_physics_baseline(laps: pd.DataFrame) -> pd.DataFrame:
    """Fit a per-session OLS lap-time baseline and return its residuals.

    RaceProgress is used only as a race-progression/fuel-burn proxy. Tyre age
    and tyre-life fields are intentionally excluded from the regression.
    """
    missing_columns = _BASELINE_COLUMNS.difference(laps.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"laps is missing required columns: {names}")
    if laps.empty:
        raise ValueError("laps must contain at least one row")
    if not pd.api.types.is_bool_dtype(laps["Rainfall"]):
        raise ValueError("Rainfall must be boolean")
    if laps["Driver"].isna().any() or laps["Driver"].eq("").any():
        raise ValueError("Driver must contain non-empty values")

    actual_lap_time = pd.to_timedelta(laps["LapTime"], errors="coerce")
    if actual_lap_time.isna().any() or (actual_lap_time <= pd.Timedelta(0)).any():
        raise ValueError("LapTime must contain positive timedeltas")
    actual_seconds = actual_lap_time.dt.total_seconds()

    numeric_columns = ["RaceProgress", "TrackTemp", "AirTemp", "Humidity"]
    numeric_values = laps[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric_values.isna().any().any() or not np.isfinite(numeric_values.to_numpy()).all():
        raise ValueError("baseline predictors must contain finite numeric values")

    driver_effects = pd.get_dummies(laps["Driver"], drop_first=True, dtype=float)
    design = np.column_stack(
        [
            np.ones(len(laps)),
            numeric_values.to_numpy(dtype=float),
            laps["Rainfall"].to_numpy(dtype=float),
            driver_effects.to_numpy(dtype=float),
        ]
    )
    if len(laps) < design.shape[1]:
        raise ValueError("laps does not contain enough rows to fit the baseline")

    coefficients, _, _, _ = np.linalg.lstsq(design, actual_seconds.to_numpy(), rcond=None)
    expected_seconds = design @ coefficients

    result = laps.copy()
    result["ExpectedLapTime"] = expected_seconds
    result["TyreResidual"] = actual_seconds - expected_seconds
    return result


def fit_improved_physics_baseline(
    laps: pd.DataFrame, *, diagnostic: bool = False
) -> pd.DataFrame:
    """Fit an interpretable OLS baseline with causal or diagnostic predictors.

    The production default excludes current-lap telemetry, relative pace,
    Position, and TrackEvolutionProxy because they can describe the outcome of
    the lap being predicted. Diagnostic mode includes them only for analysis.
    """
    required_columns = set(_CAUSAL_BASELINE_COLUMNS)
    if diagnostic:
        required_columns.update(_DIAGNOSTIC_NUMERIC_COLUMNS)
        required_columns.update(_DIAGNOSTIC_OPPONENT_COLUMNS)
    missing_columns = required_columns.difference(laps.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"laps is missing required columns: {names}")
    if laps.empty:
        raise ValueError("laps must contain at least one row")
    if not pd.api.types.is_bool_dtype(laps["Rainfall"]):
        raise ValueError("Rainfall must be boolean")

    actual_lap_time = pd.to_timedelta(laps["LapTime"], errors="coerce")
    if actual_lap_time.isna().any() or (actual_lap_time <= pd.Timedelta(0)).any():
        raise ValueError("LapTime must contain positive timedeltas")

    numeric_columns = ["RaceProgress", "TrackTemp", "AirTemp", "Humidity"]
    if diagnostic:
        numeric_columns.extend(_DIAGNOSTIC_NUMERIC_COLUMNS)
    numeric_values = laps[numeric_columns].apply(pd.to_numeric, errors="coerce")
    invalid_numeric = numeric_values.isna() & laps[numeric_columns].notna()
    if invalid_numeric.any().any():
        raise ValueError("baseline predictors must contain numeric values")
    if not np.isfinite(numeric_values.dropna().to_numpy(dtype=float)).all():
        raise ValueError("baseline predictors must contain finite values")
    if numeric_values[["RaceProgress", "TrackTemp", "AirTemp", "Humidity"]].isna().any().any():
        raise ValueError("causal baseline predictors must be available")

    design_parts = [np.ones(len(laps))]
    labels = ["intercept"]
    rejected_predictors: list[str] = []
    for column in numeric_columns:
        values = numeric_values[column]
        if values.isna().any():
            raise ValueError(f"{column} must be available for this baseline")
        if values.nunique() <= 1:
            rejected_predictors.append(f"{column} (no variation)")
            continue
        design_parts.append(values.to_numpy(dtype=float))
        labels.append(column)

    rainfall = laps["Rainfall"].to_numpy(dtype=float)
    if np.unique(rainfall).size <= 1:
        rejected_predictors.append("Rainfall (no variation)")
    else:
        design_parts.append(rainfall)
        labels.append("Rainfall")

    for column in ("Driver", "Compound", "TrackStatus"):
        if laps[column].isna().any() or laps[column].eq("").any():
            raise ValueError(f"{column} must contain non-empty categorical values")
        encoded = pd.get_dummies(laps[column], prefix=column, drop_first=True, dtype=float)
        if encoded.empty:
            rejected_predictors.append(f"{column} (one category)")
            continue
        design_parts.append(encoded.to_numpy(dtype=float))
        labels.extend(encoded.columns.tolist())

    if diagnostic:
        for column in _DIAGNOSTIC_OPPONENT_COLUMNS:
            values = pd.to_numeric(laps[column], errors="coerce")
            if (laps[column].notna() & values.isna()).any():
                raise ValueError(f"{column} must contain numeric values")
            if not np.isfinite(values.dropna().to_numpy(dtype=float)).all():
                raise ValueError(f"{column} must contain finite values")
            available = values.notna().to_numpy(dtype=float)
            design_parts.append(values.fillna(0.0).to_numpy(dtype=float))
            design_parts.append(available)
            labels.extend([column, f"{column}Available"])

    design = np.column_stack(design_parts)
    independent_columns: list[int] = []
    for index, label in enumerate(labels):
        candidate_columns = independent_columns + [index]
        candidate = design[:, candidate_columns]
        if np.linalg.matrix_rank(candidate) > len(independent_columns):
            independent_columns.append(index)
        else:
            rejected_predictors.append(f"{label} (collinear)")
    design = design[:, independent_columns]
    labels = [labels[index] for index in independent_columns]
    if len(laps) < design.shape[1]:
        raise ValueError("laps does not contain enough rows to fit the baseline")

    actual_seconds = actual_lap_time.dt.total_seconds().to_numpy()
    coefficients, _, _, _ = np.linalg.lstsq(design, actual_seconds, rcond=None)
    expected_seconds = design @ coefficients

    result = laps.copy()
    result["ExpectedLapTime"] = expected_seconds
    result["TyreResidual"] = actual_seconds - expected_seconds
    result.attrs["predictors"] = labels
    result.attrs["coefficients"] = dict(zip(labels, coefficients))
    result.attrs["rejected_predictors"] = rejected_predictors
    result.attrs["condition_number"] = float(np.linalg.cond(design))
    result.attrs["diagnostic"] = diagnostic
    return result


def fit_lagged_context_physics_baseline(laps: pd.DataFrame) -> pd.DataFrame:
    """Fit a production OLS baseline using only prior-lap context features.

    Missing lagged numeric context is represented in the design matrix by an
    availability indicator. This does not fill or alter missing source values.
    """
    required_columns = set(_CAUSAL_BASELINE_COLUMNS)
    required_columns.update(_LAGGED_CONTEXT_NUMERIC_COLUMNS)
    required_columns.update(_LAGGED_CONTEXT_CATEGORICAL_COLUMNS)
    missing_columns = required_columns.difference(laps.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"laps is missing required columns: {names}")
    if laps.empty:
        raise ValueError("laps must contain at least one row")
    if not pd.api.types.is_bool_dtype(laps["Rainfall"]):
        raise ValueError("Rainfall must be boolean")

    actual_lap_time = pd.to_timedelta(laps["LapTime"], errors="coerce")
    if actual_lap_time.isna().any() or (actual_lap_time <= pd.Timedelta(0)).any():
        raise ValueError("LapTime must contain positive timedeltas")

    core_numeric_columns = ["RaceProgress", "TrackTemp", "AirTemp", "Humidity"]
    all_numeric_columns = core_numeric_columns + _LAGGED_CONTEXT_NUMERIC_COLUMNS
    numeric_values = laps[all_numeric_columns].apply(pd.to_numeric, errors="coerce")
    invalid_numeric = numeric_values.isna() & laps[all_numeric_columns].notna()
    if invalid_numeric.any().any():
        raise ValueError("baseline predictors must contain numeric values")
    if not np.isfinite(numeric_values.dropna().to_numpy(dtype=float)).all():
        raise ValueError("baseline predictors must contain finite values")
    if numeric_values[core_numeric_columns].isna().any().any():
        raise ValueError("causal baseline predictors must be available")

    design_parts = [np.ones(len(laps))]
    labels = ["intercept"]
    rejected_predictors: list[str] = []
    scaling: dict[str, tuple[float, float]] = {}
    for column in core_numeric_columns:
        values = numeric_values[column]
        if values.nunique() <= 1:
            rejected_predictors.append(f"{column} (no variation)")
            continue
        mean = float(values.mean())
        scale = float(values.std(ddof=0))
        design_parts.append(((values - mean) / scale).to_numpy(dtype=float))
        labels.append(column)
        scaling[column] = (mean, scale)

    rainfall = laps["Rainfall"].to_numpy(dtype=float)
    if np.unique(rainfall).size <= 1:
        rejected_predictors.append("Rainfall (no variation)")
    else:
        design_parts.append(rainfall)
        labels.append("Rainfall")

    for column in ("Driver", "Compound", "TrackStatus"):
        if laps[column].isna().any() or laps[column].eq("").any():
            raise ValueError(f"{column} must contain non-empty categorical values")
        encoded = pd.get_dummies(laps[column], prefix=column, drop_first=True, dtype=float)
        if encoded.empty:
            rejected_predictors.append(f"{column} (one category)")
            continue
        design_parts.append(encoded.to_numpy(dtype=float))
        labels.extend(encoded.columns.tolist())

    for column in _LAGGED_CONTEXT_NUMERIC_COLUMNS:
        values = numeric_values[column]
        available = values.notna()
        if not available.any():
            rejected_predictors.append(f"{column} (unavailable)")
            continue
        if values.nunique(dropna=True) > 1:
            mean = float(values[available].mean())
            scale = float(values[available].std(ddof=0))
            design_parts.append(
                ((values - mean) / scale).fillna(0.0).to_numpy(dtype=float)
            )
            labels.append(column)
            scaling[column] = (mean, scale)
        else:
            rejected_predictors.append(f"{column} (no variation)")
        if available.nunique() > 1:
            design_parts.append(available.to_numpy(dtype=float))
            labels.append(f"{column}Available")

    for column in _LAGGED_CONTEXT_CATEGORICAL_COLUMNS:
        categories = laps[column].astype("string").fillna("__MISSING__")
        encoded = pd.get_dummies(categories, prefix=column, drop_first=True, dtype=float)
        if encoded.empty:
            rejected_predictors.append(f"{column} (one category)")
            continue
        design_parts.append(encoded.to_numpy(dtype=float))
        labels.extend(encoded.columns.tolist())

    design = np.column_stack(design_parts)
    independent_columns: list[int] = []
    for index, label in enumerate(labels):
        candidate_columns = independent_columns + [index]
        candidate = design[:, candidate_columns]
        if np.linalg.matrix_rank(candidate) > len(independent_columns):
            independent_columns.append(index)
        else:
            rejected_predictors.append(f"{label} (collinear)")
    design = design[:, independent_columns]
    labels = [labels[index] for index in independent_columns]
    if len(laps) < design.shape[1]:
        raise ValueError("laps does not contain enough rows to fit the baseline")

    actual_seconds = actual_lap_time.dt.total_seconds().to_numpy()
    coefficients, _, _, _ = np.linalg.lstsq(design, actual_seconds, rcond=None)
    expected_seconds = design @ coefficients

    result = laps.copy()
    result["ExpectedLapTime"] = expected_seconds
    result["TyreResidual"] = actual_seconds - expected_seconds
    result.attrs["predictors"] = labels
    result.attrs["coefficients"] = dict(zip(labels, coefficients))
    result.attrs["rejected_predictors"] = rejected_predictors
    result.attrs["condition_number"] = float(np.linalg.cond(design))
    result.attrs["diagnostic"] = False
    result.attrs["numeric_scaling"] = scaling
    return result


def fit_lagged_context_physics_model(laps: pd.DataFrame) -> dict[str, object]:
    """Fit the production causal baseline for later, unseen-lap prediction.

    The returned dictionary contains only training-derived coefficients,
    standardisation values, and categorical columns. It lets temporal ML
    validation keep validation laps out of the physics fit as well.
    """
    actual_seconds = _lagged_actual_seconds(laps)
    design, labels, rejected, scaling = _lagged_physics_design(laps)
    independent: list[int] = []
    for index, label in enumerate(labels):
        candidate = design[:, independent + [index]]
        if np.linalg.matrix_rank(candidate) > len(independent):
            independent.append(index)
        else:
            rejected.append(f"{label} (collinear)")
    design = design[:, independent]
    labels = [labels[index] for index in independent]
    if len(laps) < design.shape[1]:
        raise ValueError("laps does not contain enough rows to fit the baseline")
    coefficients, _, _, _ = np.linalg.lstsq(design, actual_seconds, rcond=None)
    return {
        "coefficients": coefficients,
        "predictors": tuple(labels),
        "rejected_predictors": tuple(rejected),
        "numeric_scaling": scaling,
        "condition_number": float(np.linalg.cond(design)),
    }


def predict_lagged_context_physics_model(
    model: dict[str, object], laps: pd.DataFrame
) -> np.ndarray:
    """Predict expected lap time using only a train-fitted causal baseline."""
    try:
        labels = model["predictors"]
        coefficients = model["coefficients"]
        scaling = model["numeric_scaling"]
    except KeyError as error:
        raise ValueError("model is missing fitted physics metadata") from error
    if not isinstance(labels, tuple) or not isinstance(scaling, dict):
        raise ValueError("model has invalid fitted physics metadata")
    coefficients = np.asarray(coefficients, dtype=float)
    if len(labels) != len(coefficients):
        raise ValueError("model coefficients do not match predictors")
    _validate_lagged_context_laps(laps)
    design = _lagged_prediction_design(laps, labels, scaling)
    expected = design @ coefficients
    if not np.isfinite(expected).all():
        raise ValueError("physics model produced non-finite predictions")
    return expected


def apply_lagged_context_physics_model(
    model: dict[str, object], laps: pd.DataFrame
) -> pd.DataFrame:
    """Return expected lap times and residuals from a train-fitted model."""
    actual_seconds = _lagged_actual_seconds(laps)
    expected = predict_lagged_context_physics_model(model, laps)
    result = laps.copy()
    result["ExpectedLapTime"] = expected
    result["TyreResidual"] = actual_seconds - expected
    return result


def _lagged_actual_seconds(laps: pd.DataFrame) -> np.ndarray:
    _validate_lagged_context_laps(laps)
    lap_time = pd.to_timedelta(laps["LapTime"], errors="coerce")
    if lap_time.isna().any() or (lap_time <= pd.Timedelta(0)).any():
        raise ValueError("LapTime must contain positive timedeltas")
    return lap_time.dt.total_seconds().to_numpy()


def _validate_lagged_context_laps(laps: pd.DataFrame) -> None:
    required = set(_CAUSAL_BASELINE_COLUMNS)
    required.update(_LAGGED_CONTEXT_NUMERIC_COLUMNS)
    required.update(_LAGGED_CONTEXT_CATEGORICAL_COLUMNS)
    missing = required.difference(laps.columns)
    if missing:
        raise ValueError(f"laps is missing required columns: {', '.join(sorted(missing))}")
    if laps.empty:
        raise ValueError("laps must contain at least one row")
    if not pd.api.types.is_bool_dtype(laps["Rainfall"]):
        raise ValueError("Rainfall must be boolean")
    core = ["RaceProgress", "TrackTemp", "AirTemp", "Humidity"]
    numeric = core + _LAGGED_CONTEXT_NUMERIC_COLUMNS
    values = laps[numeric].apply(pd.to_numeric, errors="coerce")
    if (values.isna() & laps[numeric].notna()).any().any():
        raise ValueError("baseline predictors must contain numeric values")
    if not np.isfinite(values.dropna().to_numpy(dtype=float)).all():
        raise ValueError("baseline predictors must contain finite values")
    if values[core].isna().any().any():
        raise ValueError("causal baseline predictors must be available")
    for column in ("Driver", "Compound", "TrackStatus"):
        if laps[column].isna().any() or laps[column].eq("").any():
            raise ValueError(f"{column} must contain non-empty categorical values")


def _lagged_physics_design(
    laps: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[str], dict[str, tuple[float, float]]]:
    _validate_lagged_context_laps(laps)
    numeric = laps[["RaceProgress", "TrackTemp", "AirTemp", "Humidity", *_LAGGED_CONTEXT_NUMERIC_COLUMNS]].apply(pd.to_numeric)
    parts = [np.ones(len(laps))]
    labels = ["intercept"]
    rejected: list[str] = []
    scaling: dict[str, tuple[float, float]] = {}
    for column in ("RaceProgress", "TrackTemp", "AirTemp", "Humidity"):
        values = numeric[column]
        if values.nunique() <= 1:
            rejected.append(f"{column} (no variation)")
            continue
        mean, scale = float(values.mean()), float(values.std(ddof=0))
        parts.append(((values - mean) / scale).to_numpy(dtype=float))
        labels.append(column)
        scaling[column] = (mean, scale)
    rainfall = laps["Rainfall"].to_numpy(dtype=float)
    if np.unique(rainfall).size > 1:
        parts.append(rainfall)
        labels.append("Rainfall")
    else:
        rejected.append("Rainfall (no variation)")
    for column in ("Driver", "Compound", "TrackStatus"):
        encoded = pd.get_dummies(laps[column], prefix=column, drop_first=True, dtype=float)
        if encoded.empty:
            rejected.append(f"{column} (one category)")
        else:
            parts.append(encoded.to_numpy(dtype=float))
            labels.extend(encoded.columns.tolist())
    for column in _LAGGED_CONTEXT_NUMERIC_COLUMNS:
        values = numeric[column]
        available = values.notna()
        if not available.any():
            rejected.append(f"{column} (unavailable)")
            continue
        if values.nunique(dropna=True) > 1:
            mean, scale = float(values[available].mean()), float(values[available].std(ddof=0))
            parts.append(((values - mean) / scale).fillna(0.0).to_numpy(dtype=float))
            labels.append(column)
            scaling[column] = (mean, scale)
        else:
            rejected.append(f"{column} (no variation)")
        if available.nunique() > 1:
            parts.append(available.to_numpy(dtype=float))
            labels.append(f"{column}Available")
    categories = laps["PrevOpponentAheadCompound"].astype("string").fillna("__MISSING__")
    encoded = pd.get_dummies(categories, prefix="PrevOpponentAheadCompound", drop_first=True, dtype=float)
    if encoded.empty:
        rejected.append("PrevOpponentAheadCompound (one category)")
    else:
        parts.append(encoded.to_numpy(dtype=float))
        labels.extend(encoded.columns.tolist())
    return np.column_stack(parts), labels, rejected, scaling


def _lagged_prediction_design(
    laps: pd.DataFrame, labels: tuple[str, ...], scaling: dict[str, tuple[float, float]]
) -> np.ndarray:
    frame = pd.DataFrame({"intercept": np.ones(len(laps))}, index=laps.index)
    numeric = laps[["RaceProgress", "TrackTemp", "AirTemp", "Humidity", *_LAGGED_CONTEXT_NUMERIC_COLUMNS]].apply(pd.to_numeric)
    for column, (mean, scale) in scaling.items():
        frame[column] = ((numeric[column] - mean) / scale).fillna(0.0)
    frame["Rainfall"] = laps["Rainfall"].to_numpy(dtype=float)
    for column in ("Driver", "Compound", "TrackStatus"):
        frame = pd.concat([frame, pd.get_dummies(laps[column], prefix=column, drop_first=True, dtype=float)], axis=1)
    for column in _LAGGED_CONTEXT_NUMERIC_COLUMNS:
        frame[f"{column}Available"] = numeric[column].notna().to_numpy(dtype=float)
    categories = laps["PrevOpponentAheadCompound"].astype("string").fillna("__MISSING__")
    frame = pd.concat([frame, pd.get_dummies(categories, prefix="PrevOpponentAheadCompound", drop_first=True, dtype=float)], axis=1)
    return frame.reindex(columns=labels, fill_value=0.0).to_numpy(dtype=float)
