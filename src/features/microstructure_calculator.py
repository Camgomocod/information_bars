"""
MicrostructureCalculator — Microstructure features using VectorBT
"""

import numpy as np
import pandas as pd
import vectorbt as vbt


class MicrostructureCalculator:
    """Computes microstructure features on a bars DataFrame."""

    def __init__(self, rsi_window: int = None, bb_window: int = None, bb_alpha: float = None):
        from src.config import get_feature_params
        cfg = get_feature_params("microstructure")
        self.rsi_window = rsi_window if rsi_window is not None else cfg.get("rsi_window", 14)
        self.bb_window = bb_window if bb_window is not None else cfg.get("bb_window", 20)
        self.bb_alpha = bb_alpha if bb_alpha is not None else cfg.get("bb_alpha", 2.0)

    def compute(self, bars_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds to bars_df:
          - vwap             : dollar_value / volume  (per-bar VWAP)
          - price_to_vwap    : close / vwap - 1
          - bar_duration_secs: seconds from open_time to close_time
          - rsi              : RSI(14) on close
          - bb_pct_b         : Bollinger %B  = (close - lower) / (upper - lower)
          - macd_hist        : MACD histogram (12, 26, 9)
        """
        close = bars_df["close"].astype(np.float64)
        result = bars_df.copy()

        # --- VWAP and price relative to VWAP ---
        if "dollar_value" in bars_df.columns and "volume" in bars_df.columns:
            vol = bars_df["volume"].astype(np.float64).replace(0, np.nan)
            vwap = bars_df["dollar_value"].astype(np.float64) / vol
            result["vwap"] = vwap.astype(np.float32)
            result["price_to_vwap"] = ((close / vwap) - 1.0).astype(np.float32).reindex(result.index)

        # --- Bar duration in seconds ---
        if "open_time" in bars_df.columns and "close_time" in bars_df.columns:
            open_t = pd.to_datetime(bars_df["open_time"])
            close_t = pd.to_datetime(bars_df["close_time"])
            duration = (close_t - open_t).dt.total_seconds().astype(np.float32)
            result["bar_duration_secs"] = duration

        # --- RSI via VectorBT ---
        rsi = vbt.RSI.run(close, window=self.rsi_window).rsi
        result["rsi"] = rsi.reindex(result.index).astype(np.float32)

        # --- Bollinger Bands %B via VectorBT ---
        bbands = vbt.BBANDS.run(close, window=self.bb_window, alpha=self.bb_alpha)
        upper = bbands.upper
        lower = bbands.lower
        band_width = upper - lower
        band_width = band_width.replace(0, np.nan)
        pct_b = (close - lower) / band_width
        result["bb_pct_b"] = pct_b.reindex(result.index).astype(np.float32)

        # --- MACD histogram via VectorBT ---
        macd = vbt.MACD.run(close, fast_window=12, slow_window=26, signal_window=9)
        macd_hist = macd.macd - macd.signal
        result["macd_hist"] = macd_hist.reindex(result.index).astype(np.float32)

        return result
