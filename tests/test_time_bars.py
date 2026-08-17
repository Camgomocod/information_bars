"""
test_time_bars.py — Test time bar generation from synthetic ticks.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_ticks(n_ticks: int = 50000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    base_price = 50000.0
    timestamps = pd.date_range("2024-01-01", periods=n_ticks, freq="100ms")
    price_changes = np.random.randn(n_ticks) * 10
    prices = base_price + np.cumsum(price_changes)
    prices = np.maximum(prices, 1000)
    quantities = np.random.exponential(1.0, n_ticks)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices.astype(np.float32),
            "quantity": quantities.astype(np.float32),
            "dollar_value": (prices * quantities).astype(np.float32),
        }
    )


def test_resample_to_time_bars_1min():
    from scripts.generate_time_bars import resample_to_time_bars

    ticks = generate_synthetic_ticks(n_ticks=50000)
    bars = resample_to_time_bars(ticks, interval_min=1)
    assert bars is not None
    assert len(bars) > 0
    assert "open_time" in bars.columns
    assert "close_time" in bars.columns
    assert "open" in bars.columns
    assert "high" in bars.columns
    assert "low" in bars.columns
    assert "close" in bars.columns
    assert "volume" in bars.columns
    assert "dollar_value" in bars.columns
    assert "n_ticks" in bars.columns
    assert bars["open"].dtype == np.float32
    assert bars["n_ticks"].dtype == np.int32
    assert bars["open_time"].dtype == np.dtype("datetime64[ns]")
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["volume"] >= 0).all()
    assert (bars["n_ticks"] > 0).all()


def test_resample_to_time_bars_intervals():
    from scripts.generate_time_bars import resample_to_time_bars

    ticks = generate_synthetic_ticks(n_ticks=100000)
    bars_1 = resample_to_time_bars(ticks, interval_min=1)
    bars_5 = resample_to_time_bars(ticks, interval_min=5)
    bars_60 = resample_to_time_bars(ticks, interval_min=60)
    assert len(bars_1) > len(bars_5) > len(bars_60)
    assert len(bars_60) > 0


def test_resample_to_time_bars_empty():
    from scripts.generate_time_bars import resample_to_time_bars

    empty = pd.DataFrame(columns=["timestamp", "price", "quantity", "dollar_value"])
    bars = resample_to_time_bars(empty, interval_min=1)
    assert bars.empty


def test_load_monthly_ticks_missing(tmp_path):
    from scripts.generate_time_bars import load_monthly_ticks

    df = load_monthly_ticks("FAKE", 2020, 1, tmp_path)
    assert df.empty


def test_ohlcv_consistency():
    from scripts.generate_time_bars import resample_to_time_bars

    ticks = generate_synthetic_ticks(n_ticks=20000, seed=99)
    bars = resample_to_time_bars(ticks, interval_min=5)
    assert not bars.empty
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["high"] >= bars["open"]).all()
    assert (bars["high"] >= bars["close"]).all()
    assert (bars["low"] <= bars["open"]).all()
    assert (bars["low"] <= bars["close"]).all()
    assert (bars["open_time"] < bars["close_time"]).all()
    delta = (bars["close_time"] - bars["open_time"]).dt.total_seconds()
    assert (delta == 300).all()


def test_cli_dry_run():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.generate_time_bars import main

    test_args = [
        "generate_time_bars.py",
        "--symbols",
        "BTCUSDT",
        "--year",
        "2020",
        "--month",
        "01",
        "--dry-run",
    ]
    sys.argv = test_args
    import contextlib

    with contextlib.suppress(SystemExit):
        main()
