"""
Tests for the canonical schema contract (src/storage/schema_contract).
"""

import pandas as pd

from src.storage.schema_contract import (
    FEATURE_COLUMNS,
    INIT_EXP_T,
    normalize_columns,
    normalize_params,
    validate_bars,
    validate_features,
)


def test_normalize_params_renames_db_style_key():
    raw = {"exp_lambda": 0.99, "init_exp_t": 500, "source": "bayesian_2024_01_w2m"}
    out = normalize_params(raw)
    assert INIT_EXP_T in out
    assert out[INIT_EXP_T] == 500
    assert "init_exp_t" not in out
    assert raw["init_exp_t"] == 500  # input not mutated


def test_normalize_params_keeps_canonical_key():
    out = normalize_params({"exp_lambda": 0.99, "init_exp_T": 500})
    assert out == {"exp_lambda": 0.99, "init_exp_T": 500}


def test_normalize_columns_maps_old_feature_aliases():
    cols = normalize_columns(["open", "close", "atr", "volume_zscore", "volatility", "macd"])
    assert cols == ["open", "close", "atr_pct", "volume_z", "rolling_volatility", "macd_hist"]


def test_validate_bars_ok_on_full_frame():
    df = pd.DataFrame(
        columns=[
            "open_time",
            "close_time",
            "open",
            "high",
            "low",
            "close",
            "n_ticks",
            "volume",
            "dollar_value",
        ]
    )
    assert validate_bars(df) == []


def test_validate_bars_reports_missing():
    df = pd.DataFrame(columns=["open", "high"])
    missing = validate_bars(df)
    assert "close" in missing
    assert "dollar_value" in missing


def test_validate_features_reports_missing():
    df = pd.DataFrame(columns=["open", "close"])
    missing = validate_features(df)
    assert "rsi" in missing
    assert "log_return" in missing


def test_feature_columns_are_canonical():
    # No alias names may live in the canonical feature list.
    for bad in ["atr", "volume_zscore", "volatility", "macd"]:
        assert bad not in FEATURE_COLUMNS
