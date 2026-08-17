"""
schema_contract.py — Single canonical column contract for bars, features, and studies.

Centralizes the canonical column names (single source of truth) plus helpers to
validate and normalize DataFrames/records against the contract. Any drift between
parquet, TimescaleDB, and in-memory frames must be caught here.

Canonical hyperparameter column name: init_exp_T (uppercase T), matching
everything in src/pipeline/hyperparam_loader.py and config/bars_config.yaml.
DB row keys historically used the lowercase init_exp_t; DBReader translates.
"""

from typing import Dict, List

# Canonical hyperparameter key (used everywhere in Python land).
EXP_LAMBDA = "exp_lambda"
INIT_EXP_T = "init_exp_T"

# Canonical bar columns (OHLCV + sampling + metadata).
BAR_COLUMNS: List[str] = [
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

# Canonical feature columns (bar-level, non-positioning).
FEATURE_COLUMNS: List[str] = [
    "log_return",
    "frac_diff_return",
    "rolling_volatility",
    "bar_range",
    "atr_pct",
    "volume_z",
    "dollar_value_z",
    "n_ticks_z",
    "vwap",
    "price_to_vwap",
    "bar_duration_secs",
    "rsi",
    "bb_pct_b",
    "macd_hist",
]

# Canonical positioning feature columns.
POSITIONING_COLUMNS: List[str] = ["funding_rate_mean"]

# Canonical metadata columns added to training/inference parquets.
METADATA_COLUMNS: List[str] = [
    "symbol",
    "year",
    "month",
    "study_source",
    "completion_rate",
    "failed_day",
    "recovered_day",
    "recovery_status",
    "partial_month",
    "sample_weight",
    EXP_LAMBDA,
    INIT_EXP_T,
]

# Full canonical training schema.
FULL_SCHEMA: List[str] = (
    BAR_COLUMNS + FEATURE_COLUMNS + POSITIONING_COLUMNS + METADATA_COLUMNS
)

# Aliases we accept on read and normalize away (old naming drift).
_ALIASES: Dict[str, str] = {
    # lowercase T variant from TimescaleDB/old code
    "init_exp_t": INIT_EXP_T,
    # older feature renames (see tests/test_data_integrity.py CURRENT_FEATURES)
    "atr": "atr_pct",
    "volume_zscore": "volume_z",
    "volatility": "rolling_volatility",
    "macd": "macd_hist",
}


def normalize_columns(columns: List[str]) -> List[str]:
    """Map any known alias to the canonical name, preserving order."""
    return [_ALIASES.get(c, c) for c in columns]


def normalize_params(params: Dict) -> Dict:
    """Return a copy of params with the canonical hyperparameter key names.

    Accepts dicts carrying 'init_exp_t' (DB style) and returns them as
    'init_exp_T'. Does not mutate the input.
    """
    out = dict(params)
    if "init_exp_t" in out and INIT_EXP_T not in out:
        out[INIT_EXP_T] = out.pop("init_exp_t")
    return out


def validate_bars(df, required: List[str] = BAR_COLUMNS) -> List[str]:
    """Return the list of required columns missing from df (empty = valid)."""
    present = set(df.columns)
    return [c for c in required if c not in present]


def validate_features(df, required: List[str] = FEATURE_COLUMNS) -> List[str]:
    """Return the list of required feature columns missing from df."""
    present = set(df.columns)
    return [c for c in required if c not in present]
