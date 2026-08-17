"""
test_compare_bar_types.py — Test bar type comparison functionality.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_bars():
    np.random.seed(42)
    n = 1000
    base_price = 50000.0
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1min")
    ret = np.random.randn(n) * 0.001
    close = base_price * np.exp(np.cumsum(ret))
    return pd.DataFrame(
        {
            "open_time": timestamps,
            "open": close * (1 + np.random.randn(n) * 0.0005),
            "high": close * (1 + np.abs(np.random.randn(n) * 0.001)),
            "low": close * (1 - np.abs(np.random.randn(n) * 0.001)),
            "close": close,
            "volume": np.random.exponential(100, n).astype(np.float32),
            "dollar_value": (close * np.random.exponential(100, n)).astype(np.float32),
            "n_ticks": np.random.randint(100, 5000, n).astype(np.int32),
        }
    )


def test_load_bars_missing_dir(tmp_path):
    from scripts.compare_bar_types import load_bars

    result = load_bars(tmp_path / "nonexistent", "BTCUSDT", "*.parquet", date(2024, 1, 1), date(2024, 1, 31))
    assert result is None


def test_normalize_columns():
    from scripts.compare_bar_types import normalize_columns

    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, freq="1h"), "close": [100.0] * 5})
    result = normalize_columns(df)
    assert "open_time" in result.columns
    assert result["n_ticks"].sum() == 0
    assert result["volume"].sum() == 0
    assert result["dollar_value"].sum() == 0


def test_compute_comparison(synthetic_bars):
    from scripts.compare_bar_types import compute_comparison
    from src.bars.bars_statistics import BarsStatistics

    stats_calc = BarsStatistics()
    bars_dict = {"TEST": synthetic_bars}
    comp = compute_comparison(bars_dict, stats_calc)
    assert not comp.empty
    assert "TEST" in comp.index
    assert comp.loc["TEST", "n_bars"] == 1000
    assert comp.loc["TEST", "quality_score"] > 0
    assert comp.loc["TEST", "is_stationary"] is True or comp.loc["TEST", "is_stationary"] is np.True_


def test_compute_comparison_empty():
    from scripts.compare_bar_types import compute_comparison
    from src.bars.bars_statistics import BarsStatistics

    stats_calc = BarsStatistics()
    comp = compute_comparison({}, stats_calc)
    assert comp.empty


def test_cli_dry_run():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.compare_bar_types import main

    test_args = [
        "compare_bar_types.py",
        "--symbol",
        "BTCUSDT",
        "--start",
        "2020-01-01",
        "--end",
        "2020-01-31",
        "--intervals",
        "1",
        "60",
        "--dry-run",
    ]
    sys.argv = test_args
    import contextlib

    with contextlib.suppress(SystemExit):
        main()


def test_csv_output(tmp_path, synthetic_bars):
    from scripts.compare_bar_types import compute_comparison
    from src.bars.bars_statistics import BarsStatistics

    stats_calc = BarsStatistics()
    comp = compute_comparison({"TEST": synthetic_bars}, stats_calc)
    csv_path = tmp_path / "test.csv"
    comp.to_csv(csv_path)
    assert csv_path.exists()
    loaded = pd.read_csv(csv_path, index_col=0)
    assert loaded.loc["TEST", "n_bars"] == 1000
