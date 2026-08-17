"""
PositioningCalculator — Funding rate feature from Binance Futures.

Only funding_rate is available historically from Binance Futures API.
The other metrics (OI, taker ratio, L/S ratio) require real-time collection
and are not available for historical queries.

Produced feature column:
  - funding_rate_mean : Rolling mean of funding rate (last 3 periods ≈ 24h)
"""

import numpy as np
import pandas as pd


POSITIONING_COLUMNS = [
    "funding_rate_mean",
]


class PositioningCalculator:
    def __init__(
        self,
        funding_rate_window: int = 3,
        data_dir: str = "data_raw/futures",
    ):
        self.funding_rate_window = funding_rate_window
        self.data_dir = data_dir

    def download_positioning_data(
        self, symbol: str, year: int, month: int
    ) -> dict[str, pd.DataFrame]:
        from src.connectors.binance_futures_connector import BinanceFuturesClient

        client = BinanceFuturesClient(symbol=symbol, data_dir=self.data_dir)
        fr = client.download_funding_rate(year, month)
        result = {}
        if fr is not None and not fr.empty:
            fr = fr.copy()
            fr["timestamp"] = pd.to_datetime(fr["timestamp"], utc=True)
            fr = fr.sort_values("timestamp").reset_index(drop=True)
            result["funding_rate"] = fr
        return result

    def compute(
        self,
        bars_df: pd.DataFrame,
        positioning_data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        result = bars_df.copy()
        result["_bar_time"] = pd.to_datetime(result["open_time"], utc=True)
        result["_bar_time"] = result["_bar_time"].dt.tz_localize(None)
        result = result.sort_values("_bar_time").reset_index(drop=True)

        fr = positioning_data.get("funding_rate")
        if fr is not None and not fr.empty and "funding_rate" in fr.columns:
            fr = fr.copy()
            fr["timestamp"] = pd.to_datetime(fr["timestamp"], utc=True).dt.tz_localize(None)
            fr = fr.sort_values("timestamp").reset_index(drop=True)
            fr["funding_rate_mean"] = (
                fr["funding_rate"]
                .rolling(window=self.funding_rate_window, min_periods=1)
                .mean()
            )
            fr_merge = fr[["timestamp", "funding_rate_mean"]].dropna(
                subset=["funding_rate_mean"]
            )
            result = pd.merge_asof(
                result,
                fr_merge,
                left_on="_bar_time",
                right_on="timestamp",
                direction="backward",
            )
            result = result.drop(columns=["timestamp"], errors="ignore")
        else:
            result["funding_rate_mean"] = np.float32(np.nan)

        result = result.drop(columns=["_bar_time"])

        if "funding_rate_mean" in result.columns:
            result["funding_rate_mean"] = result["funding_rate_mean"].astype(np.float32)

        return result.reset_index(drop=True)
