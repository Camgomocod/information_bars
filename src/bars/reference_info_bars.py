"""
reference_info_bars.py — Pure-Python reference implementations of the DRB/DIB cores.

These functions are intentionally slow and readable: they mirror the numba
kernels in ``src/bars/info_bars.py`` line-by-line so tests can prove that the
optimized implementation matches the auditable reference (same thresholds,
same warm-up, same EWMA updates, same runs, same residual behavior).

Reference semantics (López de Prado, *Advances in Financial ML*, Ch. 2):

DRB — Dollar Run Bars
    θ_T = max{ Σ d_t|b_t=1 , Σ d_t|b_t=-1 }   (separate runs, no offsetting)
    E[θ_T] = E[T] · max{ P[buy]·E[d|buy] , P[sell]·E[d|sell] }

DIB — Dollar Imbalance Bars
    θ_T = Σ b_t·d_t
    E[θ_T] = E[T] · E[d] · |2·P[buy] − 1|
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.bars.base_bars import BaseBars


def compute_drbs_reference(
    b_t: np.ndarray,
    dollar_values: np.ndarray,
    exp_lambda: float,
    init_exp_T: int,
    min_ticks: int,
) -> list[tuple[int, int]]:
    """Reference DRB implementation. Returns [(start_idx, end_idx), ...]."""
    n = len(b_t)
    warm_up = min(100, n // 10)
    E_T = float(init_exp_T)

    buy_mask = b_t[:warm_up] == 1
    sell_mask = b_t[:warm_up] == -1
    buy_count = int(np.sum(buy_mask))
    sell_count = int(np.sum(sell_mask))
    P_buy = buy_count / warm_up if warm_up > 0 else 0.5

    E_d_buy = (
        float(np.mean(dollar_values[:warm_up][buy_mask])) if buy_count > 0 else float(np.mean(dollar_values[:warm_up]))
    )
    E_d_sell = (
        float(np.mean(dollar_values[:warm_up][sell_mask]))
        if sell_count > 0
        else float(np.mean(dollar_values[:warm_up]))
    )

    bar_indices: list[tuple[int, int]] = []
    buy_run = 0.0
    sell_run = 0.0
    last_index = 0

    for i in range(1, n):
        d_t = float(dollar_values[i])
        if b_t[i] == 1:
            buy_run += d_t
        else:
            sell_run += d_t

        theta = max(buy_run, sell_run)
        expected_buy = P_buy * E_d_buy
        expected_sell = (1.0 - P_buy) * E_d_sell
        expected_theta = E_T * max(expected_buy, expected_sell)

        ticks_in_bar = i - last_index + 1
        if theta >= expected_theta and ticks_in_bar >= min_ticks:
            bar_indices.append((last_index, i + 1))

            E_T = (1.0 - exp_lambda) * float(ticks_in_bar) + exp_lambda * E_T

            bar_b_t = b_t[last_index : i + 1]
            bar_dollars = dollar_values[last_index : i + 1]
            buy_mask_bar = bar_b_t == 1
            sell_mask_bar = bar_b_t == -1
            buy_count_bar = int(np.sum(buy_mask_bar))
            sell_count_bar = int(np.sum(sell_mask_bar))

            P_buy_new = buy_count_bar / ticks_in_bar
            P_buy = (1.0 - exp_lambda) * float(P_buy_new) + exp_lambda * P_buy

            if buy_count_bar > 0:
                E_d_buy_new = float(np.mean(bar_dollars[buy_mask_bar]))
                E_d_buy = (1.0 - exp_lambda) * E_d_buy_new + exp_lambda * E_d_buy
            if sell_count_bar > 0:
                E_d_sell_new = float(np.mean(bar_dollars[sell_mask_bar]))
                E_d_sell = (1.0 - exp_lambda) * E_d_sell_new + exp_lambda * E_d_sell

            buy_run = 0.0
            sell_run = 0.0
            last_index = i + 1

    return bar_indices


def compute_dibs_reference(
    b_t: np.ndarray,
    dollar_values: np.ndarray,
    exp_lambda: float,
    init_exp_T: int,
    min_ticks: int,
) -> list[tuple[int, int]]:
    """Reference DIB implementation. Returns [(start_idx, end_idx), ...]."""
    n = len(b_t)
    warm_up = min(100, n // 10)
    E_T = float(init_exp_T)
    E_v = float(np.mean(dollar_values[:warm_up])) if warm_up > 0 else 0.0
    P_buy = float(np.sum(b_t[:warm_up] == 1)) / warm_up if warm_up > 0 else 0.5

    bar_indices: list[tuple[int, int]] = []
    theta = 0.0
    last_index = 0

    for i in range(1, n):
        theta += float(b_t[i]) * float(dollar_values[i])
        expected_imbalance = E_T * E_v * abs(2.0 * P_buy - 1.0)

        ticks_in_bar = i - last_index + 1
        if abs(theta) >= expected_imbalance and ticks_in_bar >= min_ticks:
            bar_indices.append((last_index, i + 1))

            E_T = (1.0 - exp_lambda) * float(ticks_in_bar) + exp_lambda * E_T
            E_v_new = float(np.mean(dollar_values[last_index : i + 1]))
            E_v = (1.0 - exp_lambda) * E_v_new + exp_lambda * E_v

            buy_count = int(np.sum(b_t[last_index : i + 1] == 1))
            P_buy_new = buy_count / ticks_in_bar
            P_buy = (1.0 - exp_lambda) * float(P_buy_new) + exp_lambda * P_buy

            theta = 0.0
            last_index = i + 1

    return bar_indices


def drbs_from_indices(df: pd.DataFrame, bar_indices: list[tuple[int, int]]) -> pd.DataFrame:
    """Form OHLCV bars from (start, end) index ranges against the tick frame."""
    if "dollar_value" not in df.columns:
        df = df.copy()
        df["dollar_value"] = df["price"] * df["quantity"]
    bars = []
    base = BaseBars()
    for start, end in bar_indices:
        bars.append(base._form_bar(df.iloc[start:end]))
    return pd.DataFrame(bars)


def residual_count(n_ticks: int, bar_indices: list[tuple[int, int]]) -> int:
    """Number of ticks after the last closed bar (never silently dropped)."""
    if not bar_indices:
        return n_ticks
    return n_ticks - bar_indices[-1][1]
