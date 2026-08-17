"""
Tests for the effect-size-based quality score.

Key property: the score must reward the MAGNITUDE of effects (low ACF, near-1
variance ratio, low CV of ticks, low skew/kurtosis) and NOT reward simply
"not rejecting" a null hypothesis.
"""

import numpy as np
import pandas as pd

from src.bars.bars_statistics import BarsStatistics


def make_bars(returns: np.ndarray, n_ticks: np.ndarray) -> pd.DataFrame:
    """Build a minimal bars frame from a returns path (n_ticks per bar)."""
    close = np.exp(np.cumsum(returns))
    close = np.concatenate([[1.0], close])
    return pd.DataFrame(
        {
            "open": close[:-1],
            "high": close[1:] * 1.001,
            "low": close[1:] * 0.999,
            "close": close[1:],
            "volume": np.ones(len(returns)),
            "dollar_value": close[1:] * np.ones(len(returns)),
            "n_ticks": n_ticks,
        }
    )


def test_white_noise_scores_high():
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 5000)
    bars = make_bars(returns, np.full(5000, 1000))
    score = BarsStatistics().compute_quality_score_fast(bars)
    # IID returns → near-ideal score.
    assert score >= 70.0


def test_autocorrelated_returns_score_lower():
    rng = np.random.default_rng(42)
    # AR(1) with rho=0.5 → strong positive autocorrelation.
    e = rng.normal(0, 0.01, 5000)
    returns = np.zeros_like(e)
    rho = 0.5
    for i in range(1, len(e)):
        returns[i] = rho * returns[i - 1] + e[i]
    bars = make_bars(returns, np.full(5000, 1000))
    score = BarsStatistics().compute_quality_score_fast(bars)
    assert score < 50.0


def test_heteroscedastic_scores_lower():
    rng = np.random.default_rng(7)
    # First half calm, second half volatile → clear variance heterogeneity.
    returns = np.concatenate(
        [
            rng.normal(0, 0.001, 2500),
            rng.normal(0, 0.05, 2500),
        ]
    )
    bars = make_bars(returns, np.full(5000, 1000))
    score = BarsStatistics().compute_quality_score_fast(bars)
    assert score < 60.0


def test_score_stable_and_bounded():
    rng = np.random.default_rng(1)
    for _ in range(5):
        returns = rng.normal(0, 0.01, 3000)
        bars = make_bars(returns, rng.integers(200, 3000, 3000))
        score = BarsStatistics().compute_quality_score_fast(bars)
        assert 0.0 <= score <= 100.0


def test_stationarity_component_bounded_when_non_stationary():
    """Regression: the ADF component must stay in [0,10] even when adf_stat >
    crit_1pct. A bug divided by max(crit_1pct, 1e-9) with a negative crit,
    exploding the score to ~1e9 and crushing the other quality components."""
    rng = np.random.default_rng(5)
    # Random walk in log-returns → strongly non-stationary close path.
    e = rng.normal(0, 0.01, 4000)
    returns = np.cumsum(e) * 0.01 + e  # persistent, non-stationary
    bars = make_bars(returns, rng.integers(200, 3000, len(returns)))
    score = BarsStatistics().compute_quality_score_fast(bars)
    assert 0.0 <= score <= 100.0, f"score exploded to {score}"
    assert score < 1e6
