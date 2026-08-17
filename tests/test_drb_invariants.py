"""
test_drb_invariants.py — Invariant + tick-accounting tests for DRB generation.

Proves that generated bars satisfy: ordering, non-overlap, valid OHLC, no
silent tick loss (consumed + residual == clean), and that the audited numba
entry point reconciles perfectly.
"""

import numpy as np
import pandas as pd

from src.bars.bar_audit import check_bar_invariants
from src.bars.info_bars import OptimizedInfoRunBars


def make_ticks(n: int = 50000, seed: int = 7, maker_col: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 50000.0
    ts = pd.date_range("2024-01-01", periods=n, freq="ms")
    price = base + np.cumsum(rng.normal(0, 10, n))
    price = np.maximum(price, 1000)
    qty = rng.exponential(1.0, n)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "price": price.astype(np.float32),
            "quantity": qty.astype(np.float32),
            "dollar_value": (price * qty).astype(np.float32),
        }
    )
    if maker_col:
        df["is_buyer_maker"] = rng.choice([True, False], n)
    return df


def test_bars_ordered_and_non_overlapping():
    drb = OptimizedInfoRunBars(save_path=None)
    bars, audit = drb.get_drbs_audited(make_ticks(), exp_lambda=0.9975, init_exp_T=2000)
    assert len(bars) > 0
    assert audit.violations == []


def test_valid_ohlc():
    drb = OptimizedInfoRunBars(save_path=None)
    bars, audit = drb.get_drbs_audited(make_ticks(), exp_lambda=0.99, init_exp_T=1000)
    assert audit.violations == []
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1)).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1)).all()
    assert (bars["n_ticks"] > 0).all()
    assert (bars["volume"] > 0).all()
    assert (bars["dollar_value"] > 0).all()


def test_no_silent_tick_loss():
    for seed in (1, 2, 3):
        df = make_ticks(seed=seed)
        drb = OptimizedInfoRunBars(save_path=None)
        bars, audit = drb.get_drbs_audited(df, exp_lambda=0.9975, init_exp_T=2000)
        assert audit.no_silent_loss, audit.to_dict()
        assert audit.consumed_ticks + audit.residual_ticks == len(df)
        assert audit.clean_ticks == len(df)


def test_residual_ticks_reported():
    # A tiny input (below min_ticks * warm needs) should produce residual = all.
    df = make_ticks(n=50, seed=5)
    drb = OptimizedInfoRunBars(save_path=None)
    bars, audit = drb.get_drbs_audited(df, exp_lambda=0.9, init_exp_T=100)
    assert audit.residual_ticks + audit.consumed_ticks == len(df)
    # Nothing can be silently lost even with 0 bars.
    assert audit.no_silent_loss


def test_maker_mode_produces_valid_bars():
    df = make_ticks(maker_col=True)
    drb = OptimizedInfoRunBars(save_path=None)
    bars, audit = drb.get_drbs_audited(df, exp_lambda=0.9975, init_exp_T=2000, direction_mode="is_buyer_maker")
    assert audit.violations == []
    assert audit.no_silent_loss


def test_maker_mode_falls_back_without_column():
    df = make_ticks(maker_col=False)
    drb = OptimizedInfoRunBars(save_path=None)
    bars, audit = drb.get_drbs_audited(df, exp_lambda=0.9975, init_exp_T=2000, direction_mode="is_buyer_maker")
    assert audit.violations == []
    assert audit.no_silent_loss


def test_check_bar_invariants_detects_bad_bars():
    bad = pd.DataFrame(
        {
            "open_time": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "close_time": [pd.Timestamp("2024-01-01 00:01"), pd.Timestamp("2024-01-01 00:02")],
            "open": [100.0, 101.0],
            "high": [90.0, 103.0],
            "low": [80.0, 102.0],
            "close": [99.0, 102.5],
            "n_ticks": [400, 0],
            "volume": [10.0, -1.0],
            "dollar_value": [1000.0, 1200.0],
        }
    )
    v = check_bar_invariants(bad)
    assert any("overlap" in x for x in v)
    assert any("high < max" in x for x in v)
    assert any("n_ticks <= 0" in x for x in v)
    assert any("invalid volume" in x for x in v)
