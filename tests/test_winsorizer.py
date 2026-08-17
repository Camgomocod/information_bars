"""
Tests for the causal Winsorizer — verifies that future rows cannot influence
how past rows are clipped (no lookahead leakage).
"""

import numpy as np
import pandas as pd

from src.normalizers.data_normalizer import DataNormalizer, Winsorizer


def _series() -> pd.Series:
    # Deterministic series with an extreme value in the "future".
    rng = np.random.default_rng(7)
    base = rng.normal(0, 1, 200)
    base[180] = 999.0  # future spike
    return pd.Series(base)


def test_winsorizer_fit_uses_only_train_rows():
    s = _series()
    train = s.iloc[:100]
    future = s.iloc[100:]

    wz = Winsorizer(limits=(0.05, 0.05)).fit(train.to_frame("x"))
    # The future spike (999) must NOT be part of the fitted bounds.
    assert wz.bounds["x"][1] < 999.0

    out = wz.transform(future.to_frame("x"))
    assert out["x"].max() <= wz.bounds["x"][1]


def test_transform_preserves_bounds_regardless_of_future():
    s = _series()
    wz = Winsorizer(limits=(0.05, 0.05)).fit(s.iloc[:100].to_frame("x"))

    # Transforming different future slices yields identical clipping thresholds.
    a = wz.transform(s.iloc[100:140].to_frame("x"))
    b = wz.transform(s.iloc[140:].to_frame("x"))
    assert a["x"].max() == b["x"].max()
    assert a["x"].min() == b["x"].min()


def test_fit_transform_matches_fit_then_transform():
    s = _series()
    df = s.to_frame("x")
    wz1 = Winsorizer(limits=(0.05, 0.05)).fit(df).transform(df)
    wz2 = Winsorizer(limits=(0.05, 0.05)).fit_transform(df)
    pd.testing.assert_frame_equal(wz1, wz2)


def test_legacy_winsorize_leaks_future_into_past():
    # Document why the legacy path is dangerous: full-series quantiles shift
    # past rows when a future extreme value exists.
    s = _series()
    past = s.iloc[:100]

    global_out = DataNormalizer.winsorize(pd.concat([past, s.iloc[100:]])).iloc[:100]
    isolated_out = DataNormalizer.winsorize(past)

    # The past rows get clipped differently depending on the future spike.
    assert not np.allclose(global_out.values, isolated_out.values)


def test_from_fitted_roundtrip():
    wz = Winsorizer(limits=(0.05, 0.05)).fit(pd.Series(np.arange(100.0)).to_frame("x"))
    wz2 = Winsorizer.from_fitted(wz.bounds, limits=(0.05, 0.05))
    assert wz2.bounds == wz.bounds
