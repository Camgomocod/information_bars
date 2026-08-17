"""
test_timescale_idempotency.py — Double-writing bars/studies must not duplicate.

Requires a reachable TimescaleDB. Uses a throwaway symbol/time range and
cleans up afterwards.
"""

import numpy as np
import pandas as pd
import psycopg2
import pytest

pytestmark = pytest.mark.db

from src.storage.db_config import get_db_url
from src.storage.timescale_client import TimescaleDBClient

DB_URL = get_db_url()
TEST_SYMBOL = "IDEMTEST"
TEST_START = "2019-01-01"
TEST_END = "2019-01-31"


def _reachable() -> bool:
    try:
        conn = psycopg2.connect(DB_URL, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


_needs_db = pytest.mark.skipif(not _reachable(), reason="TimescaleDB not reachable")


def _sample_bars(n: int = 50) -> pd.DataFrame:
    ts = pd.date_range(TEST_START, periods=n, freq="30min")
    return pd.DataFrame(
        {
            "open_time": ts,
            "close_time": ts + pd.Timedelta("29min"),
            "open": np.linspace(100, 150, n),
            "high": np.linspace(101, 151, n),
            "low": np.linspace(99, 149, n),
            "close": np.linspace(100.5, 150.5, n),
            "n_ticks": np.full(n, 400, dtype=int),
            "volume": np.linspace(1, 5, n),
            "dollar_value": np.linspace(1000, 5000, n),
            "failed_day": 0,
            "sample_weight": 1.0,
        }
    )


@_needs_db
def test_write_bars_is_idempotent():
    db = TimescaleDBClient(DB_URL)
    try:
        conn = db._get_conn()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM bars WHERE symbol = %s", (TEST_SYMBOL,))
        conn.commit()

        df = _sample_bars()
        first = db.write_bars(df, symbol=TEST_SYMBOL)
        # Re-run the same build window → replaces, not duplicates.
        second = db.write_bars(df, symbol=TEST_SYMBOL)

        count_sql = "SELECT COUNT(*) FROM bars WHERE symbol = %s AND open_time BETWEEN %s AND %s"
        with conn.cursor() as cur:
            cur.execute(count_sql, (TEST_SYMBOL, TEST_START, TEST_END))
            count = cur.fetchone()[0]
        assert first == len(df)
        assert second == len(df)
        assert count == len(df), f"expected {len(df)} bars after double write, got {count}"
    finally:
        with db._get_conn().cursor() as cur:
            cur.execute("DELETE FROM bars WHERE symbol = %s", (TEST_SYMBOL,))
        db._get_conn().commit()
        db.close()


@_needs_db
def test_upsert_study_is_idempotent():
    db = TimescaleDBClient(DB_URL)
    try:
        conn = db._get_conn()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM studies WHERE symbol = %s", (TEST_SYMBOL,))
        conn.commit()

        kwargs = {
            "symbol": TEST_SYMBOL,
            "period": "2019-01",
            "window_name": "bayesian_2018_12_w2m",
            "exp_lambda": 0.99,
            "init_exp_t": 2000,
            "composite_score": 60.0,
            "quality_component": 36.0,
            "coverage_component": 15.0,
            "stability_component": 9.0,
            "granularity_component": 0.0,
            "total_trials": 50,
            "completed_trials": 45,
            "pruned_trials": 5,
            "completion_rate": 0.9,
            "avg_bars_per_day": 200.0,
            "t_min": 100,
            "t_max": 5000,
            "lambda_min": 0.96,
            "lambda_max": 0.999,
            "sampler_type": "tpe",
            "device": "cpu",
        }
        db.upsert_study(**kwargs)
        db.upsert_study(**kwargs)

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM studies WHERE symbol = %s AND period = '2019-01'", (TEST_SYMBOL,))
            count = cur.fetchone()[0]
        assert count == 1, f"expected 1 study row, got {count}"
    finally:
        with db._get_conn().cursor() as cur:
            cur.execute("DELETE FROM studies WHERE symbol = %s", (TEST_SYMBOL,))
        db._get_conn().commit()
        db.close()
