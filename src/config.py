"""
Configuration loader for trading-core.
Single source of truth for all config/*.yaml values.
"""

import os
from pathlib import Path
from typing import Any

import yaml

_repo_config_dir = Path(__file__).parent.parent / "config"
_configured_dir = os.environ.get("TRADING_CORE_CONFIG_DIR")
_config_candidates = [
    Path(_configured_dir) if _configured_dir else None,
    Path.cwd() / "config",
    _repo_config_dir,
]
CONFIG_DIR = next((path for path in _config_candidates if path and path.is_dir()), _repo_config_dir)

_cache: dict[str, Any] = {}


def _load_yaml(filename: str) -> dict[str, Any]:
    """Load and cache a YAML config file."""
    if filename in _cache:
        return _cache[filename]

    config_path = CONFIG_DIR / filename
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    _cache[filename] = data
    return data


def get_symbols() -> list[str]:
    config = _load_yaml("symbols.yaml")
    return config.get("symbols", ["BTCUSDT"])


def get_default_symbol() -> str:
    config = _load_yaml("symbols.yaml")
    return config.get("default", "BTCUSDT")


def get_exchange_config(exchange: str = "binance") -> dict[str, Any]:
    config = _load_yaml("exchanges.yaml")
    return config.get(exchange, {})


def get_bar_config(bar_type: str = "drb") -> dict[str, Any]:
    config = _load_yaml("bars_config.yaml")
    return config.get(bar_type, {})


def get_default_hyperparams(symbol: str = None) -> dict[str, float]:
    """Get default DRB hyperparams (per-symbol or global fallback from bars_config.yaml)."""
    config = _load_yaml("bars_config.yaml")
    drb_config = config.get("drb", {})
    fallback = {
        "exp_lambda": drb_config.get("exp_lambda", 0.9975),
        "init_exp_T": drb_config.get("init_exp_T", 2000),
    }
    per_symbol = drb_config.get("per_symbol", {})
    if symbol and symbol in per_symbol:
        return {
            "exp_lambda": per_symbol[symbol].get("exp_lambda", fallback["exp_lambda"]),
            "init_exp_T": per_symbol[symbol].get("init_exp_T", fallback["init_exp_T"]),
        }
    return dict(fallback)


def get_feature_config() -> dict[str, Any]:
    return _load_yaml("features_config.yaml")


def get_feature_params(calculator: str = None) -> dict[str, Any]:
    """Get params for a specific feature calculator, or full features_config."""
    config = _load_yaml("features_config.yaml")
    if calculator:
        return config.get(calculator) or {}
    return config


def get_cleaning_config() -> dict[str, Any]:
    """Get data cleaning params (MAD filter, winsorization) from bars_config.yaml."""
    config = _load_yaml("bars_config.yaml")
    return config.get("cleaning", {})


def get_quality_thresholds() -> dict[str, float]:
    config = _load_yaml("bars_config.yaml")
    return config.get("quality", {})
