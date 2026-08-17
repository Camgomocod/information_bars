"""
BinanceFuturesClient — Downloads funding rate data from Binance Futures API.

Public endpoint (no API key required):
  - GET /fapi/v1/fundingRate  (funding rate every 8 hours)

All data is cached locally as monthly parquet files to avoid redundant downloads.
"""

import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

FUTURES_BASE_URL = "https://fapi.binance.com"

RATE_LIMIT_DELAY = 0.12  # ~8 req/sec, well within 1200/min
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3


class BinanceFuturesClient:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        data_dir: str = "data_raw/futures",
        rate_limit_delay: float = RATE_LIMIT_DELAY,
    ):
        self.symbol = symbol
        self.data_dir = Path(data_dir)
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()

    def _cache_path(self, metric: str, year: int, month: int) -> Path:
        p = self.data_dir / str(year) / f"{self.symbol}_{metric}_{year}_{month:02d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _load_cache(self, metric: str, year: int, month: int) -> Optional[pd.DataFrame]:
        p = self._cache_path(metric, year, month)
        if p.exists():
            try:
                df = pd.read_parquet(p)
                if not df.empty:
                    return df
            except Exception:
                pass
        return None

    def _save_cache(self, df: pd.DataFrame, metric: str, year: int, month: int) -> None:
        p = self._cache_path(metric, year, month)
        df.to_parquet(p, index=False, engine="pyarrow", compression="snappy")

    def _fetch_page(self, url: str, params: dict) -> Optional[list]:
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 60))
                    time.sleep(wait)
                    continue
                if resp.status_code == 418:
                    time.sleep(120)
                    continue
                if resp.status_code != 200:
                    return None
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    return data
                return None
            except requests.exceptions.RequestException:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                continue
        return None

    def download_funding_rate(self, year: int, month: int) -> Optional[pd.DataFrame]:
        cached = self._load_cache("funding_rate", year, month)
        if cached is not None:
            return cached

        from calendar import monthrange
        _, n_days = monthrange(year, month)
        start_ms = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime(year, month, n_days, 23, 59, 59, tzinfo=timezone.utc).timestamp() * 1000)

        all_records = []
        current_start = start_ms
        endpoint = "/fapi/v1/fundingRate"
        timestamp_col = "fundingTime"

        while current_start < end_ms:
            params = {
                "symbol": self.symbol,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": 1000,
            }
            page = self._fetch_page(FUTURES_BASE_URL + endpoint, params)
            if page is None:
                break

            all_records.extend(page)
            if len(page) < 1000:
                break

            last_ts = max(r[timestamp_col] for r in page)
            current_start = last_ts + 1
            time.sleep(self.rate_limit_delay)

        if not all_records:
            return None

        df = pd.DataFrame(all_records)
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df = df.rename(columns={"fundingRate": "funding_rate"})
        df["funding_rate"] = df["funding_rate"].astype(float)
        df = df[["fundingTime", "funding_rate"]].rename(columns={"fundingTime": "timestamp"})
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        self._save_cache(df, "funding_rate", year, month)
        return df

    def download_month(self, year: int, month: int) -> dict[str, Optional[pd.DataFrame]]:
        results = {}
        fr = self.download_funding_rate(year, month)
        if fr is not None and not fr.empty:
            results["funding_rate"] = fr
        return results
