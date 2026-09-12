"""Minimal FastF1 session loading and lap-level preprocessing."""

from typing import Any

import pandas as pd


LAP_COLUMNS = [
    "Driver",
    "LapNumber",
    "LapTime",
    "Compound",
    "TyreLife",
    "PitInTime",
    "PitOutTime",
    "TrackStatus",
    "TrackTemp",
    "AirTemp",
    "Humidity",
    "Rainfall",
]
WEATHER_COLUMNS = ["TrackTemp", "AirTemp", "Humidity", "Rainfall"]


def clean_laps(laps: pd.DataFrame, weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return valid laps with the small, stable schema used by TRACE.

    Weather rows should be in the same order as the lap rows, as returned by
    ``FastF1.Laps.get_weather_data()``. Missing optional FastF1 fields remain
    missing instead of being estimated.
    """
    required_columns = {"Driver", "LapNumber", "LapTime"}
    missing_columns = required_columns.difference(laps.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"laps is missing required columns: {names}")

    clean = laps.reindex(columns=LAP_COLUMNS).copy()
    if weather is not None:
        if len(weather) != len(laps):
            raise ValueError("weather must have one row for each lap")
        for column in WEATHER_COLUMNS:
            if column in weather:
                clean[column] = weather[column].to_numpy()

    clean["LapTime"] = pd.to_timedelta(clean["LapTime"], errors="coerce")
    valid_lap_time = clean["LapTime"].notna() & (
        clean["LapTime"] > pd.Timedelta(0)
    )
    return clean.loc[valid_lap_time].reset_index(drop=True)


def load_race(year: int, event: str, session_type: str = "R") -> pd.DataFrame:
    """Load a FastF1 session and return its cleaned lap-level data."""
    try:
        import fastf1
    except ImportError as error:  # pragma: no cover - dependency configuration
        raise RuntimeError("FastF1 must be installed to load session data") from error

    session = fastf1.get_session(year, event, session_type)
    session.load()
    return clean_laps(session.laps, session.laps.get_weather_data())


def aggregate_telemetry(telemetry: pd.DataFrame) -> dict[str, float | None]:
    """Calculate basic available telemetry summaries for one lap."""
    if telemetry.empty:
        raise ValueError("telemetry must contain at least one sample")

    def average(column: str) -> float | None:
        if column not in telemetry:
            return None
        values = pd.to_numeric(telemetry[column], errors="coerce").dropna()
        return None if values.empty else float(values.mean())

    def maximum(column: str) -> float | None:
        if column not in telemetry:
            return None
        values = pd.to_numeric(telemetry[column], errors="coerce").dropna()
        return None if values.empty else float(values.max())

    brake_usage = None
    brake_duration_seconds = None
    if "Brake" in telemetry:
        brake_samples = pd.to_numeric(telemetry["Brake"], errors="coerce")
        braking = brake_samples.fillna(0).gt(0)
        brake_usage = float(braking.mean())
        if "Time" in telemetry:
            times = pd.to_timedelta(telemetry["Time"], errors="coerce")
            intervals = times.shift(-1) - times
            brake_duration_seconds = float(
                intervals.where(braking).dropna().dt.total_seconds().sum()
            )

    drs_usage_fraction = None
    if "DRS" in telemetry:
        drs = pd.to_numeric(telemetry["DRS"], errors="coerce")
        available = drs.dropna()
        if not available.empty:
            # FastF1 documents even DRS states as enabled and odd states as disabled.
            drs_usage_fraction = float(
                (available.gt(0) & available.mod(2).eq(0)).mean()
            )

    return {
        "average_speed": average("Speed"),
        "maximum_speed": maximum("Speed"),
        "average_throttle": average("Throttle"),
        "average_brake": brake_usage,
        "brake_usage_fraction": brake_usage,
        "brake_duration_seconds": brake_duration_seconds,
        "average_rpm": average("RPM"),
        "drs_usage_fraction": drs_usage_fraction,
    }


def get_lap_telemetry_features(
    session: Any, driver: str, lap_number: int
) -> dict[str, float | None]:
    """Retrieve one driver's FastF1 lap telemetry and aggregate it."""
    driver_laps = session.laps.pick_drivers(driver)
    matching_laps = driver_laps[driver_laps["LapNumber"] == lap_number]
    if matching_laps.empty:
        raise ValueError(f"no lap {lap_number} found for driver {driver}")

    return aggregate_telemetry(matching_laps.iloc[0].get_telemetry())
