"""
test_optimizer_reproducibility.py — The DRB generation + quality pipeline is
deterministic for fixed (data, params, seed): identical inputs → identical bars.
"""

import numpy as np
import pandas as pd

from src.bars.info_bars import OptimizedInfoRunBars
from src.bars.reference_info_bars import compute_drbs_reference


def _ticks(seed: int = 11, n: int = 6000) -> pd.DataFrame:
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


def _sign_inputs(df):
    price = df["price"].to_numpy()
    b_t = np.sign(np.diff(price, prepend=price[0]))
    b_t = np.where(b_t == 0, 1, b_t).astype(np.float64)
    dv = (df["price"] * df["quantity"]).to_numpy().astype(np.float64)
    return b_t, dv


def test_drb_generation_deterministic_across_runs():
    df = _ticks()
    drb = OptimizedInfoRunBars(save_path=None)
    b1 = drb.get_drbs(df, exp_lambda=0.9975, init_exp_T=2000)
    b2 = drb.get_drbs(df, exp_lambda=0.9975, init_exp_T=2000)
    pd.testing.assert_frame_equal(b1, b2)


def test_reference_core_deterministic():
    df = _ticks(seed=13)
    b_t, dv = _sign_inputs(df)
    a = compute_drbs_reference(b_t, dv, 0.9975, 2000, 400)
    b = compute_drbs_reference(b_t, dv, 0.9975, 2000, 400)
    assert a == b


def test_direction_modes_are_consistent_repeatable():
    df = _ticks(seed=21)
    df["is_buyer_maker"] = np.random.default_rng(5).choice([True, False], len(df))
    drb = OptimizedInfoRunBars(save_path=None)
    for mode in ("tick_rule", "is_buyer_maker"):
        x1 = drb.get_drbs(df, exp_lambda=0.99, init_exp_T=1000, direction_mode=mode)
        x2 = drb.get_drbs(df, exp_lambda=0.99, init_exp_T=1000, direction_mode=mode)
        pd.testing.assert_frame_equal(x1, x2)
