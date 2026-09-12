"""Leakage-safe XGBoost model for the tyre-residual learning unit."""

from __future__ import annotations

from collections.abc import Iterable
from math import ceil

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from trace.features import TYRE_POSITIONS, TyrePosition


# These are deliberately not model inputs. They either contain the quantity the
# model is meant to predict, or summarize information from the same lap.
FORBIDDEN_PREDICTORS = frozenset(
    {
        "LapTime",
        "ExpectedLapTime",
        "TyreResidual",
        "PreviousTyreResidual",
        "RollingTyreResidual",
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
        "RelativePaceToAhead",
        "RelativePaceToBehind",
        "TrackEvolutionProxy",
    }
)

TYRE_CORE_PREDICTORS = (
    "TyreLife",
    "TyreAgeSquared",
    "Compound",
    "Stint",
    "RaceProgress",
)
TYRE_ENVIRONMENT_PREDICTORS = TYRE_CORE_PREDICTORS + (
    "TrackTemp",
    "AirTemp",
    "Humidity",
    "Rainfall",
)
# This deliberately small production set avoids adding every correlated lagged
# telemetry aggregate merely because it exists. Every added value is known from
# a prior lap.
PRODUCTION_PREDICTORS = TYRE_ENVIRONMENT_PREDICTORS + (
    "Driver",
    "PrevAvgSpeed",
    "PrevBrakeUsage",
    "PrevTrackEvolutionProxy",
    "PrevPosition",
    "PrevOpponentAheadCompound",
    "PrevOpponentAheadTyreLife",
)
_CATEGORICAL_PREDICTORS = {"Compound", "Driver", "PrevOpponentAheadCompound"}
FEATURE_SETS = {
    "tyre_core": TYRE_CORE_PREDICTORS,
    "tyre_environment": TYRE_ENVIRONMENT_PREDICTORS,
    "tyre_environment_lagged": PRODUCTION_PREDICTORS,
}

# A shared vocabulary only: FastF1 does not provide per-corner tyre labels.
SUPPORTED_TYRE_POSITIONS: tuple[TyrePosition, ...] = TYRE_POSITIONS


def temporal_train_validation_split(
    laps: pd.DataFrame, validation_fraction: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a single race into early training laps and later validation laps.

    Whole race-lap numbers stay together, so no driver record from a lap can be
    in training while another is in validation. Rows retain their original order.
    """
    if "LapNumber" not in laps:
        raise ValueError("laps is missing required column: LapNumber")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    lap_numbers = pd.to_numeric(laps["LapNumber"], errors="coerce")
    if lap_numbers.isna().any() or not np.isfinite(lap_numbers.to_numpy()).all():
        raise ValueError("LapNumber must contain finite numeric values")
    unique_laps = np.sort(lap_numbers.unique())
    if len(unique_laps) < 2:
        raise ValueError("laps needs at least two distinct lap numbers")

    validation_laps = max(1, ceil(len(unique_laps) * validation_fraction))
    if validation_laps >= len(unique_laps):
        raise ValueError("validation split leaves no training laps")
    first_validation_lap = unique_laps[-validation_laps]
    training = laps.loc[lap_numbers < first_validation_lap].copy()
    validation = laps.loc[lap_numbers >= first_validation_lap].copy()
    if training.empty or validation.empty:
        raise ValueError("temporal split must produce non-empty partitions")
    return training, validation


def fit_tyre_degradation_model(
    laps: pd.DataFrame,
    *,
    model_type: str = "xgboost",
    feature_set: str = "tyre_environment_lagged",
    n_estimators: int = 150,
    max_depth: int = 2,
    learning_rate: float = 0.05,
    random_state: int = 0,
) -> dict[str, object]:
    """Fit a residual model to causal-baseline residuals without mutation.

    ``TyreResidual`` must come from the production causal physics baseline.
    Missing prior-lap context is represented as missing data; XGBoost handles it
    directly. Categorical fields are one-hot encoded using training categories.
    """
    if "TyreResidual" not in laps:
        raise ValueError("laps is missing required column: TyreResidual")
    if laps.empty:
        raise ValueError("laps must contain at least one row")
    if model_type not in {"xgboost", "ridge", "random_forest"}:
        raise ValueError(f"unknown model_type: {model_type}")
    _validate_model_parameters(n_estimators, max_depth, learning_rate)

    target = pd.to_numeric(laps["TyreResidual"], errors="coerce")
    if target.isna().any() or not np.isfinite(target.to_numpy()).all():
        raise ValueError("TyreResidual must contain finite numeric values")

    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature_set: {feature_set}")
    predictors = _available_predictors(laps.columns, FEATURE_SETS[feature_set])
    if not predictors:
        raise ValueError("laps contains none of the supported causal predictors")
    design, encoded_columns = _make_design_matrix(laps, predictors)
    if design.empty:
        raise ValueError("laps contains no usable predictor values")

    estimator = _build_estimator(
        model_type, n_estimators, max_depth, learning_rate, random_state
    )
    if model_type == "ridge":
        # RidgeCV's time-ordered folds select alpha from training rows only.
        order = np.argsort(pd.to_numeric(laps["LapNumber"]), kind="stable")
        estimator.fit(design.iloc[order], target.iloc[order])
    else:
        estimator.fit(design, target)
    return {
        "estimator": estimator,
        "model_type": model_type,
        "predictors": predictors,
        "feature_set": feature_set,
        "encoded_columns": encoded_columns,
        "feature_importance": _feature_importance(model_type, estimator, encoded_columns),
    }


def predict_tyre_residual(model: dict[str, object], laps: pd.DataFrame) -> np.ndarray:
    """Predict residual correction using only the model's causal inputs."""
    try:
        estimator = model["estimator"]
        predictors = model["predictors"]
        encoded_columns = model["encoded_columns"]
    except KeyError as error:
        raise ValueError("model is missing fitted model metadata") from error
    if not hasattr(estimator, "predict"):
        raise ValueError("model does not contain a fitted estimator")
    if not isinstance(predictors, tuple) or not isinstance(encoded_columns, tuple):
        raise ValueError("model has invalid predictor metadata")
    missing = set(predictors).difference(laps.columns)
    if missing:
        raise ValueError(f"laps is missing model predictors: {', '.join(sorted(missing))}")

    design, _ = _make_design_matrix(laps, predictors, encoded_columns)
    predictions = estimator.predict(design)
    if not np.isfinite(predictions).all():
        raise ValueError("model produced non-finite predictions")
    return predictions


def add_tyre_model_predictions(model: dict[str, object], laps: pd.DataFrame) -> pd.DataFrame:
    """Add residual correction and corrected lap-time prediction to a copy."""
    if "ExpectedLapTime" not in laps:
        raise ValueError("laps is missing required column: ExpectedLapTime")
    expected = pd.to_numeric(laps["ExpectedLapTime"], errors="coerce")
    if expected.isna().any() or not np.isfinite(expected.to_numpy()).all():
        raise ValueError("ExpectedLapTime must contain finite numeric values")

    result = laps.copy()
    result["PredictedTyreResidual"] = pd.Series(
        predict_tyre_residual(model, laps), index=result.index
    )
    result["PredictedLapTime"] = expected + result["PredictedTyreResidual"]
    return result


def tyre_regime(laps: pd.DataFrame) -> pd.Series:
    """Classify only observed dry versus wet/mixed tyre/weather regimes.

    Wet/mixed means an INTERMEDIATE/WET compound or recorded Rainfall=True;
    all other rows are dry. This is not a wetness score.
    """
    required = {"Compound", "Rainfall"}
    missing = required.difference(laps.columns)
    if missing:
        raise ValueError(f"laps is missing required columns: {', '.join(sorted(missing))}")
    if not pd.api.types.is_bool_dtype(laps["Rainfall"]):
        raise ValueError("Rainfall must be boolean")
    wet_compound = laps["Compound"].astype("string").isin(["INTERMEDIATE", "WET"])
    return pd.Series(
        np.where(wet_compound | laps["Rainfall"], "wet_mixed", "dry"),
        index=laps.index,
        dtype="string",
    )


def _available_predictors(
    columns: Iterable[str], candidates: tuple[str, ...]
) -> tuple[str, ...]:
    available = tuple(column for column in candidates if column in columns)
    if set(available).intersection(FORBIDDEN_PREDICTORS):  # Defensive future guard.
        raise ValueError("forbidden leakage predictors cannot be used")
    return available


def _make_design_matrix(
    laps: pd.DataFrame,
    predictors: tuple[str, ...],
    encoded_columns: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    numeric = [column for column in predictors if column not in _CATEGORICAL_PREDICTORS]
    design = pd.DataFrame(index=laps.index)
    for column in numeric:
        values = pd.to_numeric(laps[column], errors="coerce")
        if (laps[column].notna() & values.isna()).any():
            raise ValueError(f"{column} must contain numeric values")
        if not np.isfinite(values.dropna().to_numpy()).all():
            raise ValueError(f"{column} must contain finite values")
        design[column] = values.astype(float)

    categorical = [column for column in predictors if column in _CATEGORICAL_PREDICTORS]
    if categorical:
        categories = laps[categorical].astype("string").fillna("__MISSING__")
        encoded = pd.get_dummies(categories, prefix=categorical, dtype=float)
        design = pd.concat([design, encoded], axis=1)

    if encoded_columns is not None:
        design = design.reindex(columns=encoded_columns, fill_value=0.0)
        return design, encoded_columns
    return design, tuple(design.columns)


def _build_estimator(
    model_type: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    random_state: int,
) -> object:
    if model_type == "xgboost":
        return XGBRegressor(
            objective="reg:squarederror", n_estimators=n_estimators,
            max_depth=max_depth, learning_rate=learning_rate, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3, reg_lambda=5.0,
            random_state=random_state, n_jobs=1,
        )
    if model_type == "random_forest":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("forest", RandomForestRegressor(
                    n_estimators=200, max_depth=6, min_samples_leaf=7,
                    max_features="sqrt", random_state=random_state, n_jobs=1,
                )),
            ]
        )
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=(0.1, 1.0, 10.0), cv=TimeSeriesSplit(n_splits=3))),
        ]
    )


def _feature_importance(
    model_type: str, estimator: object, encoded_columns: tuple[str, ...]
) -> dict[str, float]:
    if model_type == "xgboost":
        values = estimator.feature_importances_
    elif model_type == "random_forest":
        values = estimator.named_steps["forest"].feature_importances_[:len(encoded_columns)]
    else:
        values = np.abs(estimator.named_steps["ridge"].coef_[:len(encoded_columns)])
    return dict(sorted(zip(encoded_columns, values, strict=True), key=lambda item: item[1], reverse=True))


def _validate_model_parameters(
    n_estimators: int, max_depth: int, learning_rate: float
) -> None:
    if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or n_estimators <= 0:
        raise ValueError("n_estimators must be a positive integer")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth <= 0:
        raise ValueError("max_depth must be a positive integer")
    if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
