"""
base_features.py — Feature Engineering Orchestrator

Computes all ML-ready features on a DRB bars DataFrame using VectorBT-based calculators.
Drops NaN rows from warm-up windows and returns a clean feature matrix ready for training.
"""

import numpy as np
import pandas as pd

from src.features.returns_calculator import ReturnsCalculator
from src.features.volatility_calculator import VolatilityCalculator
from src.features.volume_calculator import VolumeCalculator
from src.features.microstructure_calculator import MicrostructureCalculator
from src.features.positioning_calculator import PositioningCalculator, POSITIONING_COLUMNS


BAR_FEATURE_COLUMNS = [
    "log_return",
    "frac_diff_return",
    "rolling_volatility",
    "bar_range",
    "atr_pct",
    "volume_z",
    "dollar_value_z",
    "n_ticks_z",
    "vwap",
    "price_to_vwap",
    "bar_duration_secs",
    "rsi",
    "bb_pct_b",
    "macd_hist",
]

FEATURE_COLUMNS = BAR_FEATURE_COLUMNS + POSITIONING_COLUMNS


def compute_all_features(
    bars_df: pd.DataFrame,
    drop_warmup: bool = True,
    winsorize: bool = True,
    winsorize_limits: tuple = (0.01, 0.01),
    winsorizer: object = None,
    positioning_data: dict = None,
    symbol: str = None,
    positioning_data_dir: str = "data_raw/futures",
) -> pd.DataFrame:
    """
    Compute all ML features for a bars DataFrame (from build_monthly_bars or concat of months).

    Parameters
    ----------
    bars_df              : DRB bars DataFrame with OHLCV + n_ticks + open_time + close_time columns
    drop_warmup          : Drop rows where any bar-level feature is NaN (warm-up period)
    winsorize            : Apply winsorization to clip extreme feature values
    winsorize_limits     : (lower_pct, upper_pct) for winsorization
    winsorizer           : Optional pre-fitted Winsorizer (src.normalizers.data_normalizer.Winsorizer).
                           When provided, its limits (fitted on train only) are applied to these rows,
                           preventing future data from influencing past clips. If None and winsorize=True,
                           falls back to the legacy global winsorization (leakage-prone) with a warning.
    positioning_data     : Pre-downloaded positioning data dict (from PositioningCalculator.download_positioning_data).
                           If None and symbol is provided, downloads automatically.
    symbol               : Symbol for positioning data download (e.g. "BTCUSDT")
    positioning_data_dir : Directory for caching downloaded futures data

    Returns
    -------
    pd.DataFrame with all original bar columns + feature columns, NaN rows dropped.
    """
    from src.normalizers.data_normalizer import DataNormalizer

    df = bars_df.copy()

    # Reset index to ensure integer indexing is consistent across all calculators
    df = df.reset_index(drop=True)

    # 1. Returns
    returns_calc = ReturnsCalculator()
    df = returns_calc.compute(df)

    # 2. Volatility
    vol_calc = VolatilityCalculator()
    df = vol_calc.compute(df)

    # 3. Volume Z-scores
    vol_z_calc = VolumeCalculator()
    df = vol_z_calc.compute(df)

    # 4. Microstructure (RSI, Bollinger, MACD, VWAP)
    micro_calc = MicrostructureCalculator()
    df = micro_calc.compute(df)

    # 5. Positioning features (Binance Futures: funding rate, OI, taker ratio, L/S ratio)
    if positioning_data is not None or symbol is not None:
        pos_calc = PositioningCalculator(data_dir=positioning_data_dir)
        if positioning_data is None:
            # Extract year/month from bars_df to download the correct month
            if "open_time" in df.columns:
                first_time = pd.to_datetime(df["open_time"].iloc[0])
                year, month = first_time.year, first_time.month
            else:
                raise ValueError("Cannot infer year/month from bars_df — provide positioning_data or ensure open_time column exists")
            positioning_data = pos_calc.download_positioning_data(symbol, year, month)
        df = pos_calc.compute(df, positioning_data)

    # 6. Optional: Winsorize features to clip extreme values
    if winsorize:
        feature_cols_present = [c for c in FEATURE_COLUMNS if c in df.columns]
        if winsorizer is not None:
            df = winsorizer.transform(df)
        else:
            import warnings
            warnings.warn(
                "compute_all_features(winsorize=True) without a pre-fitted winsorizer "
                "uses global quantiles over the whole input, which leaks future information "
                "into past rows. Fit a Winsorizer on the training split and pass it here.",
                UserWarning,
            )
            for col in feature_cols_present:
                df[col] = DataNormalizer.winsorize(df[col], limits=winsorize_limits)

    # 7. Drop warm-up NaN rows (only check bar-level features — positioning may be sparse)
    if drop_warmup:
        bar_cols_present = [c for c in BAR_FEATURE_COLUMNS if c in df.columns]
        before = len(df)
        df = df.dropna(subset=bar_cols_present).reset_index(drop=True)
        dropped = before - len(df)
        if dropped > 0:
            print(f"   🧹 Dropped {dropped} warm-up rows (NaN in bar features), kept {len(df):,}")
        pos_cols_present = [c for c in POSITIONING_COLUMNS if c in df.columns]
        pos_nan_count = df[pos_cols_present].isna().any(axis=1).sum() if pos_cols_present else 0
        if pos_nan_count > 0:
            print(f"   ℹ️  {pos_nan_count} rows have NaN in positioning features (bars before first data point)")

    return df
