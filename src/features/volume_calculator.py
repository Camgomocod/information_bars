"""
VolumeCalculator — Volume/activity z-score features using VectorBT
"""

import numpy as np
import pandas as pd
import vectorbt as vbt


class VolumeCalculator:
    """Computes volume-based z-score features on a bars DataFrame."""

    def __init__(self, window: int = None):
        from src.config import get_feature_params
        cfg = get_feature_params("volume")
        self.window = window if window is not None else cfg.get("window", 20)

    def _rolling_zscore(self, series: pd.Series) -> pd.Series:
        """Rolling z-score: (x - rolling_mean) / rolling_std"""
        mean = vbt.MA.run(series, window=self.window, ewm=False).ma
        # Rolling std: use pandas (vbt doesn't expose rolling std directly)
        std = series.rolling(window=self.window).std()
        std = std.replace(0, np.nan)
        z = (series - mean) / std
        return z

    def compute(self, bars_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds to bars_df:
          - volume_z       : rolling z-score of volume
          - dollar_value_z : rolling z-score of dollar_value
          - n_ticks_z      : rolling z-score of n_ticks
        """
        result = bars_df.copy()

        for col, out_col in [
            ("volume", "volume_z"),
            ("dollar_value", "dollar_value_z"),
            ("n_ticks", "n_ticks_z"),
        ]:
            if col in bars_df.columns:
                series = bars_df[col].astype(np.float64)
                z = self._rolling_zscore(series)
                result[out_col] = z.reindex(result.index).astype(np.float32)

        return result
