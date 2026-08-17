"""
ReturnsCalculator — Log returns and fractionally differenced returns using VectorBT
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

from src.normalizers.data_normalizer import DataNormalizer


class ReturnsCalculator:
    """Computes return-based features on a bars DataFrame."""

    def __init__(self, frac_diff_d: float = None, frac_diff_thres: float = None):
        from src.config import get_feature_params
        cfg = get_feature_params("returns")
        self.frac_diff_d = frac_diff_d if frac_diff_d is not None else cfg.get("frac_diff_d", 0.4)
        self.frac_diff_thres = frac_diff_thres if frac_diff_thres is not None else cfg.get("frac_diff_thres", 1e-4)

    def compute(self, bars_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds to bars_df:
          - log_return       : log(close[t] / close[t-1])
          - frac_diff_return : fractionally differenced log(close), d≈0.4
        """
        close = bars_df["close"].astype(np.float64)

        # --- log returns (via VectorBT returns accessor) ---
        # vbt.returns expects a price series; log_returns gives log(p[t]/p[t-1])
        log_ret = np.log(close / close.shift(1))

        # --- fractional differencing on log-price (stationarity preserving) ---
        log_close = np.log(close)
        frac_diff = DataNormalizer.frac_diff_fixed(
            log_close, d=self.frac_diff_d, thres=self.frac_diff_thres
        )

        result = bars_df.copy()
        result["log_return"] = log_ret.astype(np.float32)
        result["frac_diff_return"] = frac_diff.reindex(result.index).astype(np.float32)
        return result
