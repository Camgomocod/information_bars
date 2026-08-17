"""
Test VectorBT feature calculation with synthetic bars data.
"""

import numpy as np
import pandas as pd


def generate_synthetic_bars(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic DRB bars for feature testing."""
    np.random.seed(seed)

    base_price = 50000.0
    timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="1min")

    returns = np.random.randn(n_bars) * 0.02
    close_prices = base_price * np.exp(np.cumsum(returns))

    high_prices = close_prices * (1 + np.abs(np.random.randn(n_bars) * 0.01))
    low_prices = close_prices * (1 - np.abs(np.random.randn(n_bars) * 0.01))
    open_prices = close_prices * (1 + np.random.randn(n_bars) * 0.005)

    volumes = np.random.exponential(100, n_bars)
    dollar_values = close_prices * volumes
    n_ticks = np.random.randint(100, 5000, n_bars)

    durations = np.random.randint(60, 600, n_bars)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_prices.astype(np.float32),
            "high": high_prices.astype(np.float32),
            "low": low_prices.astype(np.float32),
            "close": close_prices.astype(np.float32),
            "volume": volumes.astype(np.float32),
            "dollar_value": dollar_values.astype(np.float32),
            "n_ticks": n_ticks.astype(np.int32),
            "bar_duration_secs": durations.astype(np.int32),
        }
    )


def test_feature_calculation():
    """Test that all features can be calculated."""
    from src.features.base_features import compute_all_features
    from src.normalizers.data_normalizer import Winsorizer

    bars_df = generate_synthetic_bars(n_bars=200)
    # Fit winsorizer on the same bars to exercise the causal path.
    wz = Winsorizer(limits=(0.05, 0.05)).fit(bars_df.select_dtypes(include=[np.number]))
    features_df = compute_all_features(bars_df, drop_warmup=True, winsorizer=wz)

    assert features_df is not None
    assert len(features_df) > 0

    expected_features = [
        "log_return",
        "rolling_volatility",
        "rsi",
        "bb_pct_b",
        "macd_hist",
    ]

    for feat in expected_features:
        assert feat in features_df.columns, f"Missing feature: {feat}"


def test_rsi_calculation():
    """Test RSI feature specifically."""
    from src.features.microstructure_calculator import MicrostructureCalculator

    bars_df = generate_synthetic_bars(n_bars=100)
    calc = MicrostructureCalculator(rsi_window=14, bb_window=20)
    result = calc.compute(bars_df)

    assert "rsi" in result.columns
    rsi = result["rsi"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_volatility_calculation():
    """Test volatility features."""
    from src.features.volatility_calculator import VolatilityCalculator

    bars_df = generate_synthetic_bars(n_bars=100)
    calc = VolatilityCalculator(window=20, atr_window=14)
    result = calc.compute(bars_df)

    assert "rolling_volatility" in result.columns
    assert "atr_pct" in result.columns


def test_volume_zscore():
    """Test volume z-score calculation."""
    from src.features.volume_calculator import VolumeCalculator

    bars_df = generate_synthetic_bars(n_bars=100)
    calc = VolumeCalculator(window=20)
    result = calc.compute(bars_df)

    assert "volume_z" in result.columns
    assert "dollar_value_z" in result.columns
