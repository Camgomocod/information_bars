"""Validation and safe persistence for training parquet datasets."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.storage.schema_contract import BAR_COLUMNS, FEATURE_COLUMNS, POSITIONING_COLUMNS


class TrainingDataValidationError(ValueError):
    """Raised when a training frame is unsafe to persist or train on."""


def validate_training_frame(
    df: pd.DataFrame,
    *,
    symbol: str | None = None,
    year: int | None = None,
    min_bars: int = 1,
    allow_partial: bool = True,
) -> dict:
    """Validate structural, numerical, and feature invariants.

    ``allow_partial`` only controls calendar coverage. It never relaxes the
    OHLCV, timestamp, schema, or feature checks.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(df, pd.DataFrame) or df.empty:
        errors.append("frame is empty")
        return {"valid": False, "errors": errors, "warnings": warnings}

    missing = [c for c in BAR_COLUMNS + FEATURE_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"missing canonical columns: {missing}")

    if len(df) < min_bars:
        errors.append(f"only {len(df)} bars; minimum is {min_bars}")

    if "open_time" in df:
        times = pd.to_datetime(df["open_time"], errors="coerce")
        if times.isna().any():
            errors.append("open_time contains invalid timestamps")
        else:
            if not times.is_monotonic_increasing:
                errors.append("open_time is not sorted")
            if times.duplicated().any():
                warnings.append(f"{int(times.duplicated().sum())} timestamp duplicates")
            if year is not None:
                outside = (times.dt.year != year).sum()
                if outside:
                    errors.append(f"{int(outside)} bars outside year {year}")
                if not allow_partial:
                    expected_start = pd.Timestamp(f"{year}-01-01")
                    expected_end = pd.Timestamp(f"{year}-12-31 23:59:59")
                    if times.min() > expected_start + pd.Timedelta(days=31):
                        errors.append("dataset starts too late for a full year")
                    if times.max() < expected_end - pd.Timedelta(days=31):
                        errors.append("dataset ends too early for a full year")

    numeric_ohlcv = [c for c in BAR_COLUMNS if c not in {"open_time", "close_time"}]
    present_ohlcv = [c for c in numeric_ohlcv if c in df.columns]
    if present_ohlcv:
        if df[present_ohlcv].isna().any().any():
            errors.append("OHLCV contains NaN values")
        if not np.isfinite(df[present_ohlcv].to_numpy(dtype=float)).all():
            errors.append("OHLCV contains non-finite values")

    if all(c in df.columns for c in ("open", "high", "low", "close", "volume")):
        bad_ohlcv = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
            | (df["volume"] < 0)
        ).sum()
        if bad_ohlcv:
            errors.append(f"{int(bad_ohlcv)} OHLCV consistency violations")

    bar_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    if bar_features:
        feature_values = df[bar_features].apply(pd.to_numeric, errors="coerce")
        if feature_values.isna().any().any():
            errors.append("bar-level features contain NaN or non-numeric values")
        if not np.isfinite(feature_values.to_numpy(dtype=float)).all():
            errors.append("bar-level features contain non-finite values")

        # A varying price/volume series must not collapse all engineered
        # features to one scalar, which is the failure seen in ETH 2026.
        core_varies = any(
            c in df.columns and df[c].nunique(dropna=True) > 1
            for c in ("open", "high", "low", "close", "volume", "n_ticks")
        )
        if core_varies:
            dynamic_features = [
                "log_return",
                "frac_diff_return",
                "rolling_volatility",
                "atr_pct",
                "vwap",
                "rsi",
                "bb_pct_b",
                "macd_hist",
            ]
            constant = [
                c for c in dynamic_features
                if c in df.columns and df[c].nunique(dropna=True) <= 1
            ]
            if constant:
                errors.append(f"constant bar-level features: {constant}")

    if "funding_rate_mean" in df.columns:
        funding = pd.to_numeric(df["funding_rate_mean"], errors="coerce")
        if funding.isna().any():
            errors.append("funding_rate_mean contains NaN values")
        elif not np.isfinite(funding.to_numpy(dtype=float)).all():
            errors.append("funding_rate_mean contains non-finite values")

    if "completion_rate" in df.columns:
        completion = pd.to_numeric(df["completion_rate"], errors="coerce")
        if completion.isna().any() or not completion.between(0.0, 1.0).all():
            errors.append("completion_rate must be finite and within [0, 1]")

    if "sample_weight" in df.columns:
        weights = pd.to_numeric(df["sample_weight"], errors="coerce")
        if weights.isna().any() or not weights.isin([0.5, 1.0]).all():
            errors.append("sample_weight contains unsupported values")

    if "recovery_status" in df.columns:
        allowed = {"normal", "recovered", "failed_irrecoverable", "missing_ticks", "excluded_partial"}
        statuses = set(df["recovery_status"].astype(str).unique())
        invalid = statuses - allowed
        if invalid:
            errors.append(f"unsupported recovery_status values: {sorted(invalid)}")

    if symbol is not None and "symbol" in df.columns:
        symbols = set(df["symbol"].astype(str).unique())
        if symbols != {symbol}:
            errors.append(f"unexpected symbols: {sorted(symbols)}")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "rows": int(len(df)),
        "feature_unique": {c: int(df[c].nunique(dropna=True)) for c in bar_features},
    }


def assert_valid_training_frame(df: pd.DataFrame, **kwargs) -> dict:
    """Validate a frame and raise a useful error when it is unsafe."""
    report = validate_training_frame(df, **kwargs)
    if not report["valid"]:
        raise TrainingDataValidationError("; ".join(report["errors"]))
    return report


def write_validated_parquet(
    df: pd.DataFrame,
    path: str | Path,
    *,
    symbol: str | None = None,
    year: int | None = None,
    min_bars: int = 1,
    allow_partial: bool = True,
) -> dict:
    """Validate, round-trip, and atomically replace a parquet file."""
    path = Path(path)
    report = assert_valid_training_frame(
        df, symbol=symbol, year=year, min_bars=min_bars, allow_partial=allow_partial
    )
    tmp_path = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(tmp_path, index=False, engine="pyarrow", compression="snappy")
        round_trip = pd.read_parquet(tmp_path)
        assert_valid_training_frame(
            round_trip, symbol=symbol, year=year, min_bars=min_bars, allow_partial=allow_partial
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return report
