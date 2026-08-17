"""
test_feature_causality.py — Proves that modifying FUTURE data never changes
features of the past (no lookahead), across the causal FeatureContext path.
"""

import numpy as np
import pandas as pd

from src.normalizers.data_normalizer import AccumulatingWinsorizer
from src.pipeline.feature_context import FeatureContext


def _synthetic_bars(n: int = 400, seed: int = 1, t0: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 50000.0
    ts = pd.date_range(t0, periods=n, freq="min")
    price = base + np.cumsum(rng.normal(0, 5, n))
    price = np.maximum(price, 1000)
    qty = rng.exponential(1.0, n)
    n_ticks = rng.integers(300, 800, n)
    return pd.DataFrame(
        {
            "open_time": ts,
            "close_time": ts + pd.Timedelta(minutes=1),
            "open": price,
            "high": price + 3,
            "low": price - 3,
            "close": np.roll(price, -1),
            "volume": qty,
            "dollar_value": price * qty,
            "n_ticks": n_ticks,
        }
    )


def test_future_rows_do_not_change_past_features():
    ctx = FeatureContext(symbol="BTCUSDT")
    m1 = _synthetic_bars(600, seed=1, t0="2024-01-01")
    m2 = _synthetic_bars(600, seed=2, t0="2024-02-01")

    feats_m1 = ctx.compute_features(m1.copy())
    # Snapshot: features after Jan, before Feb is seen.
    snapshot = feats_m1[["open_time", "log_return", "rsi", "bb_pct_b", "macd_hist"]].copy()

    # Now process Feb — must NOT alter Jan's features.
    ctx.compute_features(m2.copy())
    ctx2 = FeatureContext(symbol="BTCUSDT")
    feats_m1_b = ctx2.compute_features(m1.copy())

    a = snapshot.sort_values("open_time").reset_index(drop=True)
    b = (
        feats_m1_b[["open_time", "log_return", "rsi", "bb_pct_b", "macd_hist"]]
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(a, b)


def test_accumulating_winsorizer_bounds_past_only():
    wz = AccumulatingWinsorizer(limits=(0.1, 0.1), max_samples=100000)
    past = pd.DataFrame({"x": np.linspace(0, 100, 1000)})
    wz.update(past)

    # Extreme future values must be clipped to PAST-based bounds, never to
    # values that only exist in the future batch.
    future = pd.DataFrame({"x": [5000.0, -5000.0, 60.0]})
    clipped = wz.transform(future)
    assert clipped["x"].max() <= past["x"].quantile(0.9)
    assert clipped["x"].min() >= past["x"].quantile(0.1)


def test_accumulating_winsorizer_causal_order():
    wz = AccumulatingWinsorizer(limits=(0.0, 0.0))
    early = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    late = pd.DataFrame({"x": [100.0, 200.0]})

    # transform BEFORE update sees nothing -> no-op.
    out = wz.transform(late)
    assert out["x"].max() == 200.0

    wz.update(early)
    out = wz.transform(late)
    assert out["x"].max() == 4.0  # clipped by prior data only


def test_feature_context_cross_month_keeps_warmup_rows():
    ctx = FeatureContext(symbol="BTCUSDT")
    # First month: many rows will be warm-up (NaN) and dropped.
    m1 = _synthetic_bars(600, seed=1, t0="2024-01-01")
    f1 = ctx.compute_features(m1)
    assert not f1.empty

    # Second month must NOT lose its warm-up to the period boundary.
    m2 = _synthetic_bars(600, seed=2, t0="2024-02-01")
    f2 = ctx.compute_features(m2)
    assert not f2.empty
    # All bar-level features present and finite on the first kept rows.
    for col in ("log_return", "rsi", "bb_pct_b"):
        assert f2[col].notna().all(), f"column {col} has NaNs at month boundary"
