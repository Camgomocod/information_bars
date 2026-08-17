"""
test_drb_reference.py — Cross-validation: numba DRB/DIB cores must match the
pure-Python reference implementation (same bars, same residual behavior).
"""

import numpy as np
import pandas as pd

from src.bars.info_bars import (
    _compute_dibs_core,
    _compute_drbs_core,
    _compute_drbs_core_audited,
)
from src.bars.reference_info_bars import (
    compute_dibs_reference,
    compute_drbs_reference,
    drbs_from_indices,
    residual_count,
)


def _synthetic(seed: int, n: int = 8000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 50000.0
    ts = pd.date_range("2024-01-01", periods=n, freq="ms")
    price = base + np.cumsum(rng.normal(0, 5, n))
    price = np.maximum(price, 1000)
    qty = rng.exponential(1.0, n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "price": price.astype(np.float32),
            "quantity": qty.astype(np.float32),
        }
    )


def _inputs(df):
    price = df["price"].to_numpy()
    b_t = np.sign(np.diff(price, prepend=price[0]))
    b_t = np.where(b_t == 0, 1, b_t).astype(np.int8)
    dv = df["price"].to_numpy() * df["quantity"].to_numpy()
    return b_t.astype(np.float64), dv.astype(np.float64)


def _params():
    return [
        (0.9975, 2000),
        (0.99, 1000),
        (0.95, 500),
        (0.999, 5000),
        (0.9, 100),
    ]


def test_drb_numba_matches_reference():
    for seed in (1, 2, 3):
        df = _synthetic(seed)
        b_t, dv = _inputs(df)
        for lam, T in _params():
            ref = compute_drbs_reference(b_t, dv, lam, T, 400)
            num = _compute_drbs_core(b_t, dv, lam, T, 400)
            assert ref == num, f"seed={seed} λ={lam} T={T}"


def test_dib_numba_matches_reference():
    for seed in (1, 2):
        df = _synthetic(seed)
        b_t, dv = _inputs(df)
        for lam, T in _params():
            ref = compute_dibs_reference(b_t, dv, lam, T, 400)
            num = _compute_dibs_core(b_t, dv, lam, T, 400)
            assert ref == num, f"seed={seed} λ={lam} T={T}"


def test_residual_matches_reference():
    for seed in (4, 5):
        df = _synthetic(seed)
        b_t, dv = _inputs(df)
        for lam, T in _params():
            num_indices, num_residual = _compute_drbs_core_audited(b_t, dv, lam, T, 400)
            ref_residual = residual_count(len(b_t), num_indices)
            assert num_residual == ref_residual
            assert num_residual + num_indices[-1][1] == len(b_t) if num_indices else num_residual == len(b_t)


def test_formed_bars_reference_match():
    df = _synthetic(7)
    b_t, dv = _inputs(df)
    num_indices = _compute_drbs_core(b_t, dv, 0.9975, 2000, 400)
    ref_bars = drbs_from_indices(df, num_indices)
    assert len(ref_bars) > 0
    assert (ref_bars["high"] >= ref_bars[["open", "close"]].max(axis=1)).all()
    assert (ref_bars["low"] <= ref_bars[["open", "close"]].min(axis=1)).all()
