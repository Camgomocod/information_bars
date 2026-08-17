"""
VolatilityCalculator — Volatility features using VectorBT
"""

import numpy as np
import pandas as pd
import vectorbt as vbt


class VolatilityCalculator:
    """Computes volatility-based features on a bars DataFrame."""

    def __init__(self, window: int = None, atr_window: int = None):
        from src.config import get_feature_params
        cfg = get_feature_params("volatility")
        self.window = window if window is not None else cfg.get("window", 20)
        self.atr_window = atr_window if atr_window is not None else cfg.get("atr_window", 14)

    def compute(self, bars_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds to bars_df:
          - rolling_volatility : rolling std of log_return (window bars)
          - bar_range          : (high - low) / close  (normalized range)
          - atr_pct            : ATR(14) / close  (% average true range)
        """
        close = bars_df["close"].astype(np.float64)
        high = bars_df["high"].astype(np.float64)
        low = bars_df["low"].astype(np.float64)

        # Log returns (needed even if already computed; cheap)
        log_ret = np.log(close / close.shift(1))

        # --- Rolling volatility (std of log returns) via VectorBT ---
        rolling_vol = vbt.MA.run(
            log_ret ** 2, window=self.window, ewm=False
        ).ma.apply(np.sqrt)  # √(E[r²]) ≈ realized vol

        # --- Bar range ---
        bar_range = ((high - low) / close).astype(np.float32)

        # --- ATR% via VectorBT ---
        atr = vbt.ATR.run(high, low, close, window=self.atr_window).atr
        atr_pct = atr / close

        result = bars_df.copy()
        result["rolling_volatility"] = rolling_vol.reindex(result.index).astype(np.float32)
        result["bar_range"] = bar_range.astype(np.float32)
        result["atr_pct"] = atr_pct.reindex(result.index).astype(np.float32)
        return result
