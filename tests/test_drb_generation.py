"""
Test DRB generation with synthetic tick data.
"""

import tempfile

import numpy as np
import pandas as pd


def generate_synthetic_ticks(n_ticks: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic tick data for testing."""
    np.random.seed(seed)

    base_price = 50000.0
    timestamps = pd.date_range("2024-01-01", periods=n_ticks, freq="ms")

    price_changes = np.random.randn(n_ticks) * 10
    prices = base_price + np.cumsum(price_changes)
    prices = np.maximum(prices, 1000)

    quantities = np.random.exponential(1.0, n_ticks)
    dollar_values = prices * quantities

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices.astype(np.float32),
            "quantity": quantities.astype(np.float32),
            "dollar_value": dollar_values.astype(np.float32),
            "is_buyer_maker": np.random.choice([True, False], n_ticks),
        }
    )


def test_drb_generation():
    """Test that DRB generation works with synthetic data."""
    from src.bars.info_bars import OptimizedInfoRunBars

    df = generate_synthetic_ticks(n_ticks=50000)

    with tempfile.TemporaryDirectory() as tmpdir:
        drb = OptimizedInfoRunBars(save_path=tmpdir)
        bars = drb.get_drbs(df, exp_lambda=0.9975, init_exp_T=2000)

    assert bars is not None
    assert len(bars) > 0
    assert "open" in bars.columns
    assert "high" in bars.columns
    assert "low" in bars.columns
    assert "close" in bars.columns
    assert "volume" in bars.columns

    assert bars["high"].max() >= bars["low"].min()
    assert (bars["close"] > 0).all()


def test_drb_with_different_params():
    """Test DRB with different hyperparameter combinations."""
    from src.bars.info_bars import OptimizedInfoRunBars

    df = generate_synthetic_ticks(n_ticks=30000, seed=123)

    with tempfile.TemporaryDirectory() as tmpdir:
        drb = OptimizedInfoRunBars(save_path=tmpdir)

        bars1 = drb.get_drbs(df, exp_lambda=0.99, init_exp_T=1000)
        bars2 = drb.get_drbs(df, exp_lambda=0.999, init_exp_T=5000)

    assert bars1 is not None
    assert bars2 is not None

    assert len(bars1) != len(bars2)
