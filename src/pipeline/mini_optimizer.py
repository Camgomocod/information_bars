"""
mini_optimizer.py — Single-Day Hyperparameter Optimizer

Lightweight Optuna study for a single day when the standard walk-forward
params produce insufficient bars (< 40 DRBs).

This is NOT a full Bayesian study — it's a focused search over a bounded
space designed to complete in ~60 seconds on a VPS.

The objective is simple: find (exp_lambda, init_exp_T) that produces >= 40 bars.
If no params achieve the threshold, return the one with the most bars.

Fallback chain:
    walk-forward params → default params → mini_opt → aggregated_bar

Usage:
    from src.pipeline.mini_optimizer import MiniOptimizer

    opt = MiniOptimizer(symbol="BTCUSDT", verbose=True)
    params = opt.optimize(
        tick_df,            # cleaned tick DataFrame
        trials=20,
        timeout_sec=60,
    )
"""

import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


MIN_BARS_THRESHOLD = 40
DEFAULT_PARAMS_BY_SYMBOL = {
    "BTCUSDT": {"exp_lambda": 0.9975, "init_exp_T": 14738},
    "ETHUSDT": {"exp_lambda": 0.9975, "init_exp_T": 5000},
    "BNBUSDT": {"exp_lambda": 0.9975, "init_exp_T": 500},
    "SOLUSDT": {"exp_lambda": 0.9975, "init_exp_T": 2000},
    "_default": {"exp_lambda": 0.9975, "init_exp_T": 5000},
}


def _build_bar_from_ticks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 1 aggregated OHLCV bar from a full day of ticks.
    Used as last resort when DRB generation fails completely.

    Returns DataFrame with 1 row:
    - open: first tick price
    - high: max tick price
    - low: min tick price
    - close: last tick price
    - volume: sum of quantities
    - dollar_value: sum of dollar_values
    - n_ticks: total tick count
    - open_time: timestamp of first tick
    - close_time: timestamp of last tick
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    prices = df["price"].values
    quantities = df["quantity"].values
    dollar_values = df["dollar_value"].values
    timestamps = df["timestamp"].values

    return pd.DataFrame([{
        "open_time": timestamps[0],
        "close_time": timestamps[-1],
        "open": float(prices[0]),
        "high": float(prices.max()),
        "low": float(prices.min()),
        "close": float(prices[-1]),
        "volume": float(quantities.sum()),
        "dollar_value": float(dollar_values.sum()),
        "n_ticks": len(df),
    }])


class MiniOptimizer:
    """
    Single-day DRB hyperparameter optimizer.

    Finds exp_lambda and init_exp_T that maximize bar count for a single
    day of tick data. Designed to run in ~60 seconds on a VPS.

    Search space is adaptive based on the symbol defaults:
    - exp_lambda: [0.97, 0.9999]  (default is usually ~0.9975)
    - init_exp_T: [100, default_T * 3]  (lower T = wider bars = fewer bars)

    The objective penalizes both too-few bars (failure) and encourages
    meeting the MIN_BARS_THRESHOLD.
    """

    def __init__(self, symbol: str, verbose: bool = True):
        self.symbol = symbol
        self.verbose = verbose
        self.defaults = DEFAULT_PARAMS_BY_SYMBOL.get(
            symbol, DEFAULT_PARAMS_BY_SYMBOL["_default"]
        )

    def _count_bars(
        self,
        tick_df,
        exp_lambda: float,
        init_exp_T: int,
    ) -> int:
        """
        Generate DRBs and count how many were produced.
        Uses OptimizedInfoRunBars.get_drbs().
        """
        from src.bars.info_bars import OptimizedInfoRunBars

        bars_gen = OptimizedInfoRunBars(save_path=None)
        bars = bars_gen.get_drbs(
            df=tick_df,
            exp_lambda=exp_lambda,
            init_exp_T=init_exp_T,
        )

        if bars is None or bars.empty:
            return 0
        return len(bars)

    def _objective(self, trial: optuna.Trial, tick_df) -> float:
        """
        Optuna objective: maximize bar count.
        - Returns raw bar count (higher is better)
        - Trial is pruned if it would produce < 5 bars (not worth completing)
        """
        exp_lambda = trial.suggest_float("exp_lambda", 0.9700, 0.9999)
        init_exp_T = trial.suggest_int("init_exp_T", 100, int(self.defaults["init_exp_T"] * 3))

        n_bars = self._count_bars(tick_df, exp_lambda, init_exp_T)

        if n_bars < 5:
            raise optuna.TrialPruned(f"Only {n_bars} bars — too few")

        return float(n_bars)

    def optimize(
        self,
        tick_df,
        trials: int = 20,
        timeout_sec: int = 60,
    ) -> Optional[Dict]:
        """
        Run the mini-optimization study.

        Args:
            tick_df: Cleaned tick DataFrame (must have timestamp, price, quantity, dollar_value)
            trials: Max number of Optuna trials (default 20)
            timeout_sec: Max time to spend (default 60s)

        Returns:
            Dict with best params: {"exp_lambda": float, "init_exp_T": int, "n_bars": int}
            None if no params produced >= MIN_BARS_THRESHOLD
        """
        if tick_df is None or len(tick_df) < 100:
            return None

        if self.verbose:
            print(f"   🔬 Mini-optimizer for {self.symbol}: {len(tick_df):,} ticks, "
                  f"{trials} trials, {timeout_sec}s timeout")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        study.set_user_attr("symbol", self.symbol)

        t0 = time.time()

        def early_stop_callback(study: optuna.Study, trial: optuna.Trial):
            elapsed = time.time() - t0
            if elapsed >= timeout_sec:
                study.stop()
            if trial.value is not None and trial.value >= MIN_BARS_THRESHOLD * 2:
                study.stop()

        try:
            study.optimize(
                lambda trial: self._objective(trial, tick_df),
                n_trials=trials,
                timeout=timeout_sec,
                callbacks=[early_stop_callback],
                show_progress_bar=False,
            )
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Mini-optimizer error: {e}")
            return None

        elapsed = time.time() - t0

        best = study.best_trial
        best_params = {
            "exp_lambda": best.params["exp_lambda"],
            "init_exp_T": best.params["init_exp_T"],
            "n_bars": int(best.value),
        }

        if self.verbose:
            meets = "✅" if best.value >= MIN_BARS_THRESHOLD else "⚠️"
            print(f"   {meets} Best: λ={best.params['exp_lambda']:.4f}, "
                  f"T={best.params['init_exp_T']}, {int(best.value)} bars "
                  f"({elapsed:.1f}s, {study.best_trial.number} trials)")

        if best.value < 1:
            return None

        return best_params